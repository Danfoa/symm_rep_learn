"""Real World Experiment of G-Equivariant Regression in Robotics."""

from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
import math
import pathlib
import pickle
from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Iterable, List, Optional, Union

import escnn
import hydra
import numpy as np
import pandas as pd
import torch
from escnn.group import Representation, directsum, groups_dict
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, default_collate

from symm_rep_learn.models.equiv_ncp import ENCP
from symm_rep_learn.models.lightning_modules import SupervisedTrainingModule, TrainingModule
from symm_rep_learn.models.ncp import NCP
from symm_rep_learn.mysc.symm_algebra import invariant_orthogonal_projector
from symm_rep_learn.nn.equiv_layers import EMLP, EquivResidualEncoder
from symm_rep_learn.nn.layers import MLP, ResidualEncoder

log = logging.getLogger(__name__)


@dataclass
class DynamicsRecording:
    """Data structure to store recordings of a Markov Dynamics."""

    description: Optional[str] = None
    info: dict[str, object] = field(default_factory=dict)
    dynamics_parameters: dict = field(default_factory=lambda: {"dt": None})

    # Ordered list of observations composing to the state and action space of the Markov Process.
    state_obs: tuple[str, ...] = field(default_factory=list)
    action_obs: tuple[str, ...] = field(default_factory=list)

    # Map from observation name to the observation representation name. This name should be in `group_representations`.
    obs_representations: dict[str, Representation] = field(default_factory=dict)
    recordings: dict[str, Iterable] = field(default_factory=dict)
    # Map from observation name to the observation moments (mean, var) of the recordings.
    obs_moments: dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        # Check provided state observables are in the recordings
        for obs_name in self.state_obs:
            assert (
                obs_name in self.recordings.keys()
            ), f"State observable {obs_name} not in the provided recordings: {self.recordings.keys()}"
        for obs_name in self.recordings.keys():
            # Scalar observations should have shape (traj, time, 1)
            # obs_shape = self.recordings[obs_name].shape
            # if len(obs_shape) == 2:
            #     self.recordings[obs_name] = self.recordings[obs_name][..., None]
            # If group representation of observation is provided, check the representation has the same dimension
            if obs_name in self.obs_representations.keys():
                rep = self.obs_representations[obs_name]
                assert (
                    rep.size == self.recordings[obs_name].shape[-1]
                ), f"[{obs_name}] Invalid rep dim={rep.size} for {self.recordings[obs_name]}"

    @property
    def obs_dims(self):
        """Dictionary providing the map between observation name and observation dimension."""
        return {k: v.shape[-1] for k, v in self.recordings.items()}

    @property
    def state_dim(self):
        """Return the dimension of the state vector."""
        return sum([self.obs_dims[obs] for obs in self.state_obs])

    _obs_idx = None

    def obs_idx_in_state(self, obs_name) -> tuple[int]:
        """Returns a tuple of indices of each observation in the state vector."""
        assert obs_name in self.state_obs, f"Observation {obs_name} not in state observations"
        if self._obs_idx is None:
            self._obs_idx = {}
            # Iterate over all state observables in order and store the indices of each observation given its dimension
            curr_dim = 0
            for obs in self.state_obs:
                obs_dim = self.obs_dims[obs]
                self._obs_idx[obs] = tuple(range(curr_dim, curr_dim + obs_dim))
                curr_dim += obs_dim
        return self._obs_idx[obs_name]

    def state_representations(self) -> list[Representation]:
        """Return the ordered list of representations of the state vector."""
        return [self.obs_representations[m] for m in self.state_obs]

    def state_moments(self) -> [np.ndarray, np.ndarray]:
        """Compute the mean and standard deviation of the state observations."""
        mean, var = [], []
        for obs_name in self.state_obs:
            if obs_name not in self.obs_moments.keys():
                self.compute_obs_moments(obs_name)
            obs_mean, obs_var = self.obs_moments[obs_name]
            mean.append(obs_mean)
            var.append(obs_var)
        mean, var = np.concatenate(mean), np.concatenate(var)
        return mean, var

    def compute_obs_moments(self, obs_name: str) -> [np.ndarray, np.ndarray]:
        """Compute the mean and standard deviation of observations."""
        assert obs_name in self.recordings.keys(), f"Observation {obs_name} not found in recording"
        is_symmetric_obs = obs_name in self.obs_representations.keys()
        if is_symmetric_obs:
            rep_obs = self.obs_representations[obs_name]
            obs_original_basis = np.asarray(self.recordings[obs_name])
            G = rep_obs.group
            # Allocate the mean and variance arrays.
            mean, var = np.zeros(rep_obs.size), np.ones(rep_obs.size)
            # Change basis of the observation to expose the irrep G-stable subspaces
            Q_inv = rep_obs.change_of_basis_inv  # Orthogonal transformation to irrep basis (Q^T = Q^-1)
            Q = rep_obs.change_of_basis
            # Get the dimensions of each irrep.

            S = np.zeros((rep_obs.size, rep_obs.size))
            irreps_dimension = []
            cum_dim = 0
            for irrep_id in rep_obs.irreps:
                irrep = G.irrep(*irrep_id)
                # Get dimensions of the irrep in the original basis
                irrep_dims = range(cum_dim, cum_dim + irrep.size)
                irreps_dimension.append(irrep_dims)
                if irrep_id == G.trivial_representation.id:
                    S[irrep_dims, irrep_dims] = 1
                cum_dim += irrep.size

            # Compute the mean of the observation.
            # The mean of a symmetric random variable (rv) lives in the subspaces associated with the trivial irreps.
            has_trivial_irreps = G.trivial_representation.id in rep_obs.irreps
            if has_trivial_irreps:
                avg_projector = rep_obs.change_of_basis @ S @ rep_obs.change_of_basis_inv
                # Compute the mean in a single vectorized operation
                mean_empirical = np.mean(obs_original_basis, axis=(0, 1))
                mean = np.einsum("...ij,...j->...i", avg_projector, mean_empirical)

            # Compute the variance of the observable by computing a single variance per irrep G-stable subspace.
            # To do this, we project the observations to the basis exposing the irreps, compute the variance per
            # G-stable subspace, and map the variance back to the original basis.
            centered_obs_irrep_basis = np.einsum("...ij,...j->...i", Q_inv, obs_original_basis - mean)
            var_irrep_basis = np.ones_like(var)
            for irrep_id, irrep_dims in zip(rep_obs.irreps, irreps_dimension):
                irrep = G.irrep(*irrep_id)
                centered_obs_irrep = centered_obs_irrep_basis[..., irrep_dims]
                assert (
                    centered_obs_irrep.shape[-1] == irrep.size
                ), f"Obs irrep shape {centered_obs_irrep.shape} != {irrep.size}"

                # Since the irreps are unitary/orthogonal transformations, we are constrained compute a unique variance
                # for all dimensions of the irrep G-stable subspace, as scaling the dimensions independently would break
                # the symmetry of the rv. As a centered rv the variance is the expectation of the squared rv.
                var_irrep = np.mean(centered_obs_irrep**2)  # Single scalar variance per G-stable subspace
                # Store the irrep mean and variance in the entire representation mean and variance
                var_irrep_basis[irrep_dims] = var_irrep
            # Convert the variance from the irrep basis to the original basis
            Cov = Q @ np.diag(var_irrep_basis) @ Q_inv
            var = np.diagonal(Cov)

            # TODO: Move this check to Unit test as it is computationally demanding to check this at runtime.
            # Ensure the mean is equivalent to computing the mean of the orbit of the recording under the group action
            # aug_obs = []
            # for g in G.elements:
            #     g_obs = np.einsum('...ij,...j->...i', rep_obs(g), obs_original_basis)
            #     aug_obs.append(g_obs)
            #
            # aug_obs = np.concatenate(aug_obs, axis=0)   # Append over the trajectory dimension
            # mean_emp = np.mean(aug_obs, axis=(0, 1))
            # assert np.allclose(mean, mean_emp, rtol=1e-3, atol=1e-3), f"Mean {mean} != {mean_emp}"
            #
            # var_emp = np.var(aug_obs, axis=(0, 1))
            # assert np.allclose(var, var_emp, rtol=1e-2, atol=1e-2), f"Var {var} != {var_emp}"
        else:
            mean = np.mean(np.asarray(self.recordings[obs_name]), axis=(0, 1))
            var = np.var(np.asarray(self.recordings[obs_name]), axis=(0, 1))
        assert mean.shape == (
            self.obs_dims[obs_name],
        ), f"Obs {obs_name} dim ({self.obs_dims[obs_name]},) diff from estimated mean dim ({mean.shape},)!= "
        assert var.shape == (
            self.obs_dims[obs_name],
        ), f"Obs {obs_name} dim ({self.obs_dims[obs_name]},) diff from estimated var dim ({var.shape},)!= "

        self.obs_moments[obs_name] = mean, var

    def get_state_trajs(self, standardize: bool = False):
        """Returns a single array containing the concatenated state observations trajectories.

        Given the state observations `self.state_obs` this method concatenates the trajectories of each observation
        into a single array of shape [traj, time, state_dim]. If standardize is set to True, the state observations
        are standardized to have zero mean and unit variance.

        Returns:
            A single array containing the concatenated state observations trajectories of shape [traj, time, state_dim].
        """
        obs = [self.recordings[obs_name] for obs_name in self.state_obs]
        state_trajs = np.concatenate(obs, axis=-1)
        if standardize:
            state_mean, var = self.state_moments()
            state_std = np.sqrt(var)
            state_trajs = (state_trajs - state_mean) / state_std

        return state_trajs

    def get_state_dim_names(self):
        dim_names = []
        for obs_name in self.state_obs:
            obs_dim = self.obs_dims[obs_name]
            dim_names += [f"{obs_name}:{i}" for i in range(obs_dim)]
        return dim_names

    def get_obs_from_vector(self, vector: np.ndarray, obs_names: tuple[str, ...] = None) -> dict[str, np.ndarray]:
        """Extract the observation values from a vector given the ordered observation names."""
        if obs_names is None:
            obs_names = self.state_obs
        else:
            for name in obs_names:
                assert name in self.recordings.keys(), f"Observation {name} not in recordings {self.recordings.keys()}"
        obs_dims = [self.obs_dims[obs] for obs in obs_names]
        assert vector.shape[-1] == sum(
            obs_dims
        ), f"Vector dim {vector.shape[-1]} differs from expected dimension {sum(obs_dims)}"
        obs_idx = [0] + [sum(obs_dims[:i]) for i in range(1, len(obs_dims))]
        obs_values = {
            obs_name: vector[..., obs_idx[i] : obs_idx[i] + obs_dims[i]] for i, obs_name in enumerate(obs_names)
        }
        return obs_values

    def save_to_file(self, file_path: Path):
        # Store representations and groups without serializing
        if len(self.obs_representations) > 0:
            self._obs_rep_irreps = {}
            self._obs_rep_names = {}
            self._obs_rep_Q = {}
            for k, rep in self.obs_representations.items():
                self._obs_rep_irreps[k] = rep.irreps if rep is not None else None
                self._obs_rep_names[k] = rep.name if rep is not None else None
                self._obs_rep_Q[k] = rep.change_of_basis if rep is not None else None
            group = self.obs_representations[self.state_obs[0]].group
            self._group_keys = group._keys
            self._group_name = group.__class__.__name__
            # Remove non-serializable objects
            del self.obs_representations
            self.dynamics_parameters.pop("group", None)

        with file_path.with_suffix(".pkl").open("wb") as file:
            self._path = file_path.with_suffix(".pkl").absolute()
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(
        file_path: Path, only_metadata=False, obs_names: Optional[Iterable[str]] = None, ignore_other_obs: bool = True
    ) -> "DynamicsRecording":
        """Load a DynamicsRecording object from a file.

        Args:
            file_path: Path to the file containing the DynamicsRecording object.
            only_metadata: If True, the recordings are not loaded, only the metadata.
            obs_names: List of observation names to load. If None, all observations are loaded.
            ignore_other_obs: If True, observations not in `obs_names` are ignored.

        Returns:
            A DynamicsRecording object containing the loaded data.
        """
        with file_path.with_suffix(".pkl").open("rb") as file:
            data = pickle.load(file)
            if only_metadata:
                del data.recordings
            else:
                data_obs_names = list(data.recordings.keys())
                if obs_names is not None:
                    data.state_obs = tuple(obs_names)
                    if ignore_other_obs:
                        irrelevant_obs = [k for k in data_obs_names if k not in obs_names and obs_names is not None]
                        for k in irrelevant_obs:
                            del data.recordings[k]

            if hasattr(data, "_group_name"):
                group = groups_dict[data._group_name]._generator(**data._group_keys)  # Instanciate symmetry group
                data.dynamics_parameters["group"] = group
                data.obs_representations = {}
                for obs_name in data._obs_rep_irreps.keys():
                    irreps_ids = data._obs_rep_irreps[obs_name]
                    rep_name = data._obs_rep_names[obs_name]
                    rep_Q = data._obs_rep_Q[obs_name]
                    if rep_name is None:
                        data.obs_representations[obs_name] = None
                    elif rep_name in group.representations:
                        data.obs_representations[obs_name] = group.representations[rep_name]
                    else:
                        data.obs_representations[obs_name] = Representation(
                            group, name=rep_name, irreps=irreps_ids, change_of_basis=rep_Q
                        )
                    group.representations[rep_name] = data.obs_representations[obs_name]
        data.__post_init__()
        return data

    @staticmethod
    def load_data_generator(
        dynamics_recordings: list["DynamicsRecording"],
        frames_per_step: int = 1,
        prediction_horizon: Union[int, float] = 1,
        state_obs: Optional[list[str]] = None,
        action_obs: Optional[list[str]] = None,
    ):
        """Generator that yields observation samples of length `n_frames_per_state` from the Markov Dynamics recordings.

        Args:
            recordings (list[DynamicsRecording]): List of DynamicsRecordings.
            frames_per_step: Number of frames to compose a single observation sample at time `t`. E.g. if `f` is
            provided
            the state samples will be of shape [f, obs_dim].
            prediction_horizon (int, float): Number of future time steps to include in the next time samples.
                E.g: if `n` is an integer the samples will be of shape [n, frames_per_state, obs_dim]
                If `n` is a float, then the samples will be of shape [int(n*traj_length), frames_per_state, obs_dim]
            state_obs: Ordered list of observations names composing the state space.
            action_obs: Ordered list of observations names composing the action space.

        Returns:
            A dictionary containing the observations of shape (time_horizon, frames_per_step, obs_dim)
        """
        for file_data in dynamics_recordings:
            recordings = file_data.recordings
            relevant_obs = set(file_data.state_obs).union(set(file_data.action_obs))
            if state_obs is not None:
                relevant_obs = set(state_obs)
            if action_obs is not None:
                relevant_obs = relevant_obs.union(set(action_obs))

            # Get any observation list of trajectories and count the number of trajectories
            # We assume all observations have the same number of trajectories
            n_trajs = len(next(iter(recordings.values())))

            # Since we assume trajectories can have different lengths, we iterate over each trajectory
            # and generate samples of length `n_frames_per_state` from each trajectory.
            for traj_id in range(n_trajs):
                traj_length = next(iter(recordings.values()))[traj_id].shape[0]
                if isinstance(prediction_horizon, float):
                    steps_in_pred_horizon = floor((prediction_horizon * traj_length) // frames_per_step) - 1
                else:
                    steps_in_pred_horizon = prediction_horizon
                assert steps_in_pred_horizon > 0, f"Invalid prediction horizon {steps_in_pred_horizon}"

                remnant = traj_length % frames_per_step
                frames_in_pred_horizon = steps_in_pred_horizon * frames_per_step
                # Iterate over the frames of the trajectory
                for frame in range(traj_length - frames_per_step):
                    # Collect the next steps until the end of the trajectory. If the prediction horizon is larger than
                    # the remaining steps (prediction outside trajectory length), then we continue to the next traj.
                    # This is better computationally for avoiding copying while batch processing data later.
                    if frame + frames_per_step + frames_in_pred_horizon > (traj_length - remnant):
                        continue
                    sample = {}
                    for obs_name, trajs in recordings.items():
                        if obs_name not in relevant_obs:  # Do not process unrequested observations
                            continue
                        num_steps = (frames_in_pred_horizon // frames_per_step) + 1
                        # Compute the indices for the start and end of each "step" in the time horizon
                        start_indices = np.arange(0, num_steps) * frames_per_step + frame
                        end_indices = start_indices + frames_per_step
                        # Use these indices to slice the relevant portion of the trajectory
                        obs_time_horizon = trajs[traj_id][start_indices[0] : end_indices[-1]]
                        # Reshape the slice to have the desired shape (time, frames_per_step, obs_dim)
                        obs_dim = file_data.obs_dims[obs_name]
                        obs_time_horizon = obs_time_horizon.reshape((num_steps, frames_per_step, obs_dim))

                        # Test no copy is being made (too costly to do at runtime)
                        # assert np.shares_memory(obs_time_horizon, trajs[traj_id])
                        assert (
                            len(obs_time_horizon) == steps_in_pred_horizon + 1
                        ), f"{len(obs_time_horizon)} != {steps_in_pred_horizon + 1}"
                        sample[obs_name] = obs_time_horizon
                    # print(frame)
                    yield sample

    @staticmethod
    def map_state_next_state(
        sample: dict,
        state_observations: List[str],
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ) -> dict:
        """Map composing multiple frames of observations into a flat vectors `state` and `next_state` samples.

        This method constructs the state `s_t` and history of nex steps `s_t+1` of the Markov Process.
        The state is defined as a set of observations within a window of fps=`frames_per_state`.
        E.g.: Consider the state is defined by the observations [m=momentum, p=position] at `fps` consecutive frames.
            Then the state at time `t` is defined as `s_t = [m_f, m_f+1,..., m_f+fps, p_f, p_f+1, ..., p_f+fps]`.
            Where we use f to denote frame in time to make the distinction from the time index `t` of the Markov
            Process.
            Then, the next state is defined as `s_t+1 = [m_f+fps,..., m_fps+fps, p_f+fps, ..., p_f+fps+fps]`.

        Args:
            sample (dict): Dictionary containing the observations of the system of shape [state_time, f].
            state_observations: Ordered list of observations names composing the state space.

        Returns:
            A dictionary containing the MDP state `s_t` and the next_state/s `[s_t+1, s_t+2, ..., s_t+pred_horizon]`.
        """
        batch_size = len(sample[f"{state_observations[0]}"])
        time_horizon = len(sample[f"{state_observations[0]}"][0])
        # Flatten observations a_t = [a_f, a_f+1, af+2, ..., a_f+F] s.t. a_t in R^{F * dim(a)}, a_f in R^{dim(a)}
        state_obs = [sample[m] for m in state_observations]
        # Define the state at time t and the states at time [t+1, t+pred_horizon]
        state_traj = np.concatenate(state_obs, axis=-1).reshape(batch_size, time_horizon, -1)
        if state_mean is not None and state_std is not None:
            state_traj = (state_traj - state_mean) / state_std
        return dict(state=state_traj[:, 0], next_state=state_traj[:, 1:])

    @staticmethod
    def map_state_action_state(sample, state_observations: List[str], action_observations: List[str]) -> dict:
        """Map composing multiple observations to single state, action, next_state samples."""
        flat_sample = DynamicsRecording.map_state_next_state(sample, state_observations)
        # Reuse the same function to for flattening action and next_action
        action_sample = DynamicsRecording.map_state_next_state(sample, action_observations)
        action_sample["action"] = action_sample.pop("state")
        action_sample["next_action"] = action_sample.pop("next_state")
        flat_sample.update(action_sample)
        return flat_sample


def get_model(cfg: DictConfig, x_type, y_type) -> torch.nn.Module:
    embedding_dim = cfg.architecture.embedding_dim
    dim_x = x_type.size
    dim_y = y_type.size

    if cfg.model.lower() == "encp":  # Equivariant NCP
        from escnn.nn import FieldType

        from symm_rep_learn.models.equiv_ncp import ENCP
        from symm_rep_learn.nn.equiv_layers import EMLP

        G = x_type.representation.group

        reg_rep = G.regular_representation

        kwargs = dict(
            hidden_layers=cfg.architecture.hidden_layers,
            activation=cfg.architecture.activation,
            hidden_units=cfg.architecture.hidden_units,
            bias=False,
        )
        if cfg.architecture.residual_encoder:
            lat_rep = [reg_rep] * max(1, math.ceil(cfg.architecture.embedding_dim // reg_rep.size))
            lat_x_type = FieldType(
                gspace=escnn.gspaces.no_base_space(G), representations=list(y_type.representations) + lat_rep
            )
            lat_y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=lat_rep)
            x_embedding = EMLP(in_type=x_type, out_type=lat_x_type, **kwargs)
            y_embedding = EquivResidualEncoder(
                encoder=EMLP(in_type=y_type, out_type=lat_y_type, **kwargs), in_type=y_type
            )
            assert (
                y_embedding.out_type.size == x_embedding.out_type.size
            ), f"{y_embedding.out_type.size} != {x_embedding.out_type.size}"
        else:
            lat_type = FieldType(
                gspace=escnn.gspaces.no_base_space(G),
                representations=[reg_rep] * max(1, math.ceil(cfg.architecture.embedding_dim // reg_rep.size)),
            )
            x_embedding = EMLP(in_type=x_type, out_type=lat_type, **kwargs)
            y_embedding = EMLP(in_type=y_type, out_type=lat_type, **kwargs)
        eNCPop = ENCP(
            embedding_x=x_embedding, embedding_y=y_embedding, gamma=cfg.gamma, truncated_op_bias=cfg.truncated_op_bias
        )

        return eNCPop
    elif cfg.model.lower() == "ncp":  # NCP
        from symm_rep_learn.models.ncp import NCP
        from symm_rep_learn.mysc.utils import class_from_name
        from symm_rep_learn.nn.layers import MLP

        activation = class_from_name("torch.nn", cfg.architecture.activation)
        kwargs = dict(
            output_shape=embedding_dim,
            n_hidden=cfg.architecture.hidden_layers,
            layer_size=cfg.architecture.hidden_units,
            activation=activation,
            bias=False,
            iterative_whitening=cfg.architecture.iter_whitening,
        )
        if cfg.architecture.residual_encoder_x:
            dim_free_embedding = embedding_dim - dim_x
            fx = ResidualEncoder(
                encoder=MLP(input_shape=dim_x, **kwargs | {"output_shape": dim_free_embedding}), in_dim=dim_x
            )
        else:
            fx = MLP(input_shape=dim_x, **kwargs)
        if cfg.architecture.residual_encoder:
            dim_free_embedding = embedding_dim - dim_y
            fy = ResidualEncoder(
                encoder=MLP(input_shape=dim_y, **kwargs | {"output_shape": dim_free_embedding}), in_dim=dim_y
            )
        else:
            fy = MLP(input_shape=dim_y, **kwargs)
        ncp = NCP(
            embedding_x=fx,
            embedding_y=fy,
            embedding_dim=embedding_dim,
            gamma=cfg.gamma,
            truncated_op_bias=cfg.truncated_op_bias,
        )
        return ncp

    elif cfg.model.lower() == "mlp":
        from symm_rep_learn.mysc.utils import class_from_name
        from symm_rep_learn.nn.layers import MLP

        activation = class_from_name("torch.nn", cfg.architecture.activation)
        n_h_layers = cfg.architecture.hidden_layers
        mlp = MLP(
            input_shape=dim_x,
            output_shape=dim_y,
            n_hidden=cfg.architecture.hidden_layers,
            layer_size=[cfg.architecture.hidden_units] * (n_h_layers - 1) + [cfg.architecture.embedding_dim],
            activation=activation,
            bias=False,
        )
        return mlp
    elif cfg.model.lower() == "emlp":
        from symm_rep_learn.nn.equiv_layers import EMLP

        n_h_layers = cfg.architecture.hidden_layers
        emlp = EMLP(
            in_type=x_type,
            out_type=y_type,
            hidden_layers=cfg.architecture.hidden_layers,
            activation=cfg.architecture.activation,
            hidden_units=[cfg.architecture.hidden_units] * (n_h_layers - 1) + [cfg.architecture.embedding_dim],
            bias=False,
        )
        return emlp
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


def com_momentum_dataset(cfg):
    """Loads dataset for 'Center of Mass Momentum Regression' experiment from https://arxiv.org/pdf/2302.10433."""
    device = cfg.device
    # TODO: Y is currently 6 dimension, without 'KinE'.
    dr = DynamicsRecording.load_from_file(Path(cfg.path_ds).absolute())
    X_obs = [dr.recordings[obs_name].squeeze(1) for obs_name in cfg.x_obs_names]
    Y_obs = [dr.recordings[obs_name].squeeze(1) for obs_name in cfg.y_obs_names]
    X = np.concatenate(X_obs, axis=-1)
    Y = np.concatenate(Y_obs, axis=-1)

    # Compute the symmetry aware meand and variance of each observable
    X_mean_obs, X_var_obs = [], []
    for obs_name in cfg.x_obs_names:
        dr.compute_obs_moments(obs_name)
        mean, var = dr.obs_moments[obs_name]
        X_mean_obs.append(mean)
        X_var_obs.append(var)
    X_mean = np.concatenate(X_mean_obs, axis=-1)
    X_var = np.concatenate(X_var_obs, axis=-1)
    Y_mean_obs, Y_var_obs = [], []
    y_obs_dims = {}
    start_dim = 0
    for obs_name in cfg.y_obs_names:
        dr.compute_obs_moments(obs_name)
        mean, var = dr.obs_moments[obs_name]
        Y_mean_obs.append(mean)
        Y_var_obs.append(var)
        y_obs_dims[obs_name] = slice(start_dim, start_dim + mean.shape[-1])
        start_dim += mean.shape[-1]
    Y_mean = np.concatenate(Y_mean_obs, axis=-1)
    Y_var = np.concatenate(Y_var_obs, axis=-1)
    # Get the group representations per obs
    rep_X_obs = [dr.obs_representations[obs_name] for obs_name in cfg.x_obs_names]
    rep_Y_obs = [dr.obs_representations[obs_name] for obs_name in cfg.y_obs_names]
    rep_X = directsum(rep_X_obs, name="X")
    rep_Y = directsum(rep_Y_obs, name="Y")

    # Split data into train, validation, and test subsets
    assert 0 < cfg.optim.train_sample_ratio <= 0.7, f"Invalid train ratio {cfg.optim.train_sample_ratio}"
    train_ratio, val_ratio, test_ratio = cfg.optim.train_sample_ratio, 0.15, 0.15
    n_samples = X.shape[0]
    n_train, n_val, n_test = np.asarray(np.array([train_ratio, val_ratio, test_ratio]) * n_samples, dtype=int)

    X_c = (X - X_mean) / np.sqrt(X_var)
    Y_c = (Y - Y_mean) / np.sqrt(Y_var)
    x_train, x_val, x_test = (X_c[:n_train], X_c[n_train : n_train + n_val], X_c[n_train + n_val :])
    y_train, y_val, y_test = (Y_c[:n_train], Y_c[n_train : n_train + n_val], Y_c[n_train + n_val :])

    # Moving data to gpu
    X_train = torch.atleast_2d(torch.from_numpy(x_train).float()).to(device)
    Y_train = torch.atleast_2d(torch.from_numpy(y_train).float()).to(device)
    X_val = torch.atleast_2d(torch.from_numpy(x_val).float())
    Y_val = torch.atleast_2d(torch.from_numpy(y_val).float())
    X_test = torch.atleast_2d(torch.from_numpy(x_test).float())
    Y_test = torch.atleast_2d(torch.from_numpy(y_test).float())
    y_moments = (torch.from_numpy(Y_mean).float().to(device), torch.from_numpy(Y_var).float().to(device))

    # Define data samples
    train_samples = X_train, Y_train
    val_samples = X_val, Y_val
    test_samples = X_test, Y_test

    # Define datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    return (
        (train_samples, val_samples, test_samples),
        (train_dataset, val_dataset, test_dataset),
        rep_X,
        rep_Y,
        y_obs_dims,
        y_moments,
    )


@torch.no_grad()
def regression_metrics(model, x, y, y_train, x_type, y_type, y_obs_dims, y_moments):
    """Predicts CoM Momenta from test sample (x_test, y_test)."""
    device = next(model.parameters()).device
    prev_data_device = x.device

    x = x.to(device)
    y = y.to(device)
    # Introduce the entire group orbit of the testing set, to appropriately compute the equivariant error.
    rep_X, rep_Y = x_type.representation, y_type.representation
    G = rep_X.group
    G_loss, G_metrics = [], []
    for g in G.elements:
        rep_X_g = torch.from_numpy(rep_X(g)).float().to(device)
        rep_Y_g = torch.from_numpy(rep_Y(g)).float().to(device)

        gx = torch.einsum("ij,kj->ki", rep_X_g, x)
        gy = torch.einsum("ij,kj->ki", rep_Y_g, y)

        if isinstance(model.embedding_x, EquivariantModule):
            fgx = model.embedding_x(x_type(gx)).tensor
        else:
            fgx = model.embedding_x(gx)  # shape: (n_test, embedding_dim)
        if isinstance(model.embedding_y, EquivariantModule):
            hy_train = model.embedding_y(y_type(y_train)).tensor  # shape: (n_train, embedding_dim)
        else:
            hy_train = model.embedding_y(y_train)  # shape: (n_train, embedding_dim)

        n_train = y_train.shape[0]

        if isinstance(model.embedding_y, ResidualEncoder):
            y_dims_in_hy = model.embedding_y.residual_dims
            mask = torch.zeros(hy_train.shape[-1], device=device)
            mask[y_dims_in_hy] = 1
            hy_train = hy_train * mask

        if isinstance(model.embedding_y, escnn.nn.SequentialModule):
            # Get last module of the sequetial mod
            if isinstance(model.embedding_y[0], EquivResidualEncoder):
                res_encoder = model.embedding_y[0]
                y_dims_in_hy = res_encoder.residual_dims
                change2iso_module = model.embedding_y[-1]
                Qy2iso = change2iso_module.Qin2iso
                mask = torch.zeros(hy_train.shape[-1], device=device)
                mask[y_dims_in_hy] = 1
                # mask_projector = Qy2iso @ mask @ Qy2iso.T
                mask_projector = torch.einsum("ij,j,jk->ik", Qy2iso, mask, Qy2iso.T)
                hy_train = hy_train * mask
                hy_train = torch.einsum("ij,...j->...i", mask_projector, hy_train)
            print("")

        E_y = y_train.mean(axis=0)  # shape: (6,)
        if rep_Y is not None:
            inv_projector = invariant_orthogonal_projector(rep_Y).to(y_train.device)
            E_y = inv_projector @ E_y  # Mean of symmetric RV lives in the invariant subspace.

        # shape: (n_test, 6). Check formula 12 from https://arxiv.org/pdf/2407.01171
        Dr = model.truncated_operator
        gy_deflated_basis_expansion = (1 / n_train) * torch.einsum("ij,ki,lj,lm->km", Dr, fgx, hy_train, y_train)
        gy_pred = E_y + gy_deflated_basis_expansion  # TODO: experiment with linear regression instead of einsums

        gy_mse, metrics = proprioceptive_regression_metrics(gy, gy_pred, y_obs_dims, y_moments)
        G_loss.append(gy_mse)
        G_metrics.append(metrics)

    # Compute average over the orbit for all metrics
    metrics_names = G_metrics[0].keys()
    metrics = {name: torch.stack([m[name] for m in G_metrics]).mean() for name in metrics_names}

    x = x.to(prev_data_device)
    y = y.to(prev_data_device)
    return metrics


def proprioceptive_regression_metrics(y, y_pred, y_obs_dims: dict, y_moments):
    # Compute MSE with standarized data
    if isinstance(y, GeometricTensor):
        assert y.type == y_pred.type
        y = y.tensor
        y_pred = y_pred.tensor

    mse = torch.nn.MSELoss(reduce=False)
    y_mse = mse(y_pred, y)
    loss = y_mse.mean()
    metrics = {"y_mse": loss}
    with torch.no_grad():
        # Unstandardie data and compute MSE in the original data scale
        y_mean, y_var = y_moments
        y_mse_un = y_mse * y_var

        metrics["hg_mse"] = y_mse_un.mean()
        for obs_name, dims in y_obs_dims.items():
            obs_mse = y_mse_un[..., dims]
            metrics[obs_name] = obs_mse.mean()

    return loss, metrics


@hydra.main(config_path="cfg", config_name="eNCP_regression", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    seed_everything(seed)

    # Load dataset______________________________________________________________________
    samples, datasets, rep_X, rep_Y, y_obs_dims, y_moments = com_momentum_dataset(cfg)
    train_samples, val_samples, test_samples = samples
    train_ds, val_ds, test_ds = datasets
    x_train, y_train = train_samples
    x_val, y_val = val_samples
    x_test, y_test = test_samples

    G = rep_X.group
    # lat_rep = G.regular_representation
    x_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_X])
    y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_Y])
    # lat_type = FieldType(
    #     gspace=escnn.gspaces.no_base_space(G),
    #     representations=[lat_rep] * max(1, math.ceil(cfg.architecture.embedding_dim // lat_rep.size))
    #     )

    # Get the model_____________________________________________________________________
    model = get_model(cfg, x_type, y_type)
    print(model)
    # Print the number of trainable paramerters
    n_trainable_params = sum(p.numel() for p in model.parameters())
    log.info(f"No. trainable parameters: {n_trainable_params}")

    # Define the dataloaders_____________________________________________________________
    # ESCNN equivariant models expect GeometricTensors.
    def geom_tensor_collate_fn(batch) -> [GeometricTensor, GeometricTensor]:
        x_batch, y_batch = default_collate(batch)
        return GeometricTensor(x_batch, x_type), GeometricTensor(y_batch, y_type)

    is_equiv_model = isinstance(model, ENCP) or isinstance(model, EMLP)
    collate_fn = geom_tensor_collate_fn if is_equiv_model else default_collate
    train_dataloader = DataLoader(train_ds, batch_size=cfg.optim.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=cfg.optim.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=cfg.optim.batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the Lightning module ______________________________________________________
    if isinstance(model, MLP) or isinstance(model, EMLP):
        lightning_module = SupervisedTrainingModule(
            model=model,
            optimizer_fn=Adam,
            optimizer_kwargs={"lr": cfg.optim.lr},
            loss_fn=lambda x, y: proprioceptive_regression_metrics(x, y, y_obs_dims, y_moments),
        )
    else:  # NCP / ENCP models
        lightning_module = TrainingModule(
            model=model,
            optimizer_fn=Adam,
            optimizer_kwargs={"lr": cfg.optim.lr},
            loss_fn=model.loss if hasattr(model, "loss") else None,
            val_metrics=lambda _: regression_metrics(
                model=model,
                x=x_val,
                y=y_val,
                y_train=y_train,
                x_type=x_type,
                y_type=y_type,
                y_obs_dims=y_obs_dims,
                y_moments=y_moments,
            ),
            test_metrics=lambda _: regression_metrics(
                model=model,
                x=x_test,
                y=y_test,
                y_train=y_train,
                x_type=x_type,
                y_type=y_type,
                y_obs_dims=y_obs_dims,
                y_moments=y_moments,
            ),
        )

    # Define the logger and callbacks
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.proj_name, log_model=False, config=run_cfg)
    n_total_samples = int(train_samples[0].shape[0] / cfg.optim.train_sample_ratio)
    scaled_saved_freq = int(5 * n_total_samples // cfg.optim.batch_size)
    BEST_CKPT_NAME, LAST_CKPT_NAME = "best", ModelCheckpoint.CHECKPOINT_NAME_LAST
    ckpt_call = ModelCheckpoint(
        dirpath=run_path,
        filename=BEST_CKPT_NAME,
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
        every_n_epochs=scaled_saved_freq,
    )

    # Fix for all runs independent on the train_ratio chosen. This way we compare on effective number of "epochs"
    max_steps = int(n_total_samples * cfg.optim.max_epochs // cfg.optim.batch_size)
    scaled_patience = int(cfg.optim.patience * n_total_samples * 0.7 // cfg.optim.batch_size)

    metric_to_monitor = "||k(x,y) - k_r(x,y)||/val" if isinstance(model, ENCP) or isinstance(model, NCP) else "loss/val"
    early_call = EarlyStopping(monitor=metric_to_monitor, patience=scaled_patience, mode="min")

    trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
        max_epochs=max_steps,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=25,
        check_val_every_n_epoch=5,
        callbacks=[ckpt_call, early_call],
        fast_dev_run=25 if cfg.debug else False,
        num_sanity_val_steps=5,
    )

    torch.set_float32_matmul_precision("medium")
    last_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / LAST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=last_ckpt_path if last_ckpt_path.exists() else None,
    )

    # Loads the best model.
    best_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / BEST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
    test_logs = trainer.test(
        lightning_module,
        dataloaders=test_dataloader,
        ckpt_path=best_ckpt_path if best_ckpt_path.exists() else None,
    )
    test_metrics = test_logs[0]  # dict: metric_name -> value
    # Save the testing matrics in a csv file using pandas.
    test_metrics_path = pathlib.Path(run_path) / "test_metrics.csv"
    pd.DataFrame(test_metrics, index=[0]).to_csv(test_metrics_path, index=False)

    # Flush the logger.
    logger.finalize(trainer.state)
    # Wand sync
    logger.experiment.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
