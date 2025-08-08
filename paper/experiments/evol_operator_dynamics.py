from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
import pathlib

import hydra
import numpy as np
import pandas as pd
import torch
from escnn.nn import FieldType, GeometricTensor
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from symm_learning.models.emlp import EMLP
from torch.optim import Adam
from torch.utils.data import DataLoader, default_collate

from paper.experiments.dynamics.dynamics_dataset import TrajectoryDataset
from paper.experiments.dynamics.thomas_attractor import ThomasAttractor
from paper.experiments.train_utils import get_train_logger_and_callbacks
from symm_rep_learn.models.evol_op import EvolOp1D
from symm_rep_learn.models.lightning_modules import TrainingModule

log = logging.getLogger(__name__)


def get_model(cfg: DictConfig, state_type: FieldType) -> torch.nn.Module:
    if cfg.model == "eEvolOp":  # Equivariant NCP
        raise NotImplementedError("Equivariant NCP is not implemented yet.")
    elif cfg.model == "EvolOp":  # NCP
        from symm_rep_learn.models.evol_op import EvolOp1D
        from symm_learning.models import MLP
        from symm_rep_learn.mysc.utils import act_name_to_torch
        from symm_rep_learn.nn.layers import ResidualEncoder

        embedding_dim = cfg.architecture.embedding_dim
        fx = MLP(
            in_dim=state_type.size,
            out_dim=embedding_dim,
            hidden_units=cfg.architecture.hidden_units,
            activation=act_name_to_torch(cfg.architecture.activation),
        )

        if cfg.architecture.residual_encoder:
            fx = ResidualEncoder(fx, in_dim=1)  # Append state to the embedding
            embedding_dim += state_type.size

        ncp = EvolOp1D(
            embedding_state=fx,
            state_embedding_dim=embedding_dim,
            state_dim=state_type.size,
            orth_reg=cfg.gamma,
            centering_reg=cfg.gamma_centering,
            momentum=cfg.momentum,
            self_adjoint=cfg.architecture.self_adjoint,
        )
        return ncp
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


def get_dataset(cfg: DictConfig):
    data_desc = (
        f"thomas_b={cfg.dataset.b}_noise={cfg.dataset.noise_scale}_"
        f"trajs={cfg.dataset.n_trajectories}_frames={cfg.dataset.traj_time_horizon}_dt={cfg.dataset.dt}"
    )

    thomas = ThomasAttractor(b=cfg.dataset.b, noise_scale=cfg.dataset.noise_scale)
    rep_state = thomas.get_state_symmetry_rep()

    data_path = pathlib.Path(cfg.dataset.path) / f"{data_desc}.npy"
    if not data_path.exists():
        log.info(f"Dataset {data_desc} not found, generating new dataset...")
        _, trajectories = thomas.generate_dataset(  # (n_trajs, time, n_features)
            n_trajectories=cfg.dataset.n_trajectories,
            trajectory_length=cfg.dataset.traj_time_horizon,
            dt=cfg.dataset.dt,
            seed=cfg.dataset.seed,
        )
        trajectories = np.asarray(trajectories, dtype=np.float32).transpose(0, 2, 1)  #  (n_trajs, n_features, time)
        # Save the trajectories to disk
        np.save(data_path.absolute(), trajectories)
        log.info(f"Generated dataset and saved to {data_path}")
    else:
        log.info(f"Loading dataset from {data_path}")
        trajectories = np.load(data_path, allow_pickle=False)

    print(trajectories.shape)
    n_trajs, state_dim, time_horizon = trajectories.shape

    assert state_dim == rep_state.size, f"{state_dim} != {rep_state.size}"

    # Split into train, validation and test datasets splitting on trajectories ----------------------------------------
    n_trajectories = len(trajectories)
    n_train, n_val = int(n_trajectories * cfg.dataset.train_ratio), int(n_trajectories * cfg.dataset.val_ratio)
    train_trajs = trajectories[range(n_train)]
    val_trajs = trajectories[range(n_train, n_train + n_val)]
    test_trajs = trajectories[range(n_train + n_val, n_trajectories)]

    torch_kwargs = dict(dtype=torch.float32, device=cfg.device)
    ds_kwargs = dict(
        past_frames=cfg.dataset.past_frames,
        future_frames=cfg.dataset.future_frames,
        time_lag=cfg.dataset.time_lag,
        shuffle=True,
        **torch_kwargs,
    )
    # TrajectoryDataset expects [n_trajs, time_horizon, state_dim] shape
    train_ds = TrajectoryDataset(trajectories=np.moveaxis(train_trajs, 2, 1), **ds_kwargs)
    val_ds = TrajectoryDataset(trajectories=np.moveaxis(val_trajs, 2, 1), **ds_kwargs)
    test_ds = TrajectoryDataset(trajectories=np.moveaxis(test_trajs, 2, 1), **ds_kwargs)

    # Configure the observation space representations ------------------------------------------------------------------
    from escnn.gspaces import no_base_space
    from symm_learning.nn.conv import GSpace1D
    from symm_learning.stats import var_mean

    G = rep_state.group
    if cfg.dataset.past_frames == 1 and cfg.dataset.future_frames == 1:
        state_type = FieldType(no_base_space(G), [rep_state])
    else:
        state_type = FieldType(GSpace1D(G), [rep_state])

    # Compute the mean and variance of the observations ---------------------------------------------------------------

    train_samples = np.moveaxis(train_trajs, 1, -1).reshape(-1, state_dim)  # (n_trajs * time, state_dim)

    state_var, state_mean = var_mean(torch.tensor(train_samples), rep_x=rep_state)

    return (
        (train_ds, val_ds, test_ds),
        state_type,
        (state_mean, state_var),
    )


def symmetric_collate(
    batch,
    split: str,
    past_frames: int,
    augment: bool,
    state_type: FieldType,
    state_moments: tuple[torch.Tensor, torch.Tensor],
):
    batch = default_collate(batch)
    traj = batch
    assert traj.ndim == 3 and traj.shape[2] == state_type.size, (
        f"Expected tensor of shape (batch, time, state_dim), got {traj.shape}"
    )
    traj = traj.transpose(1, 2)  # (batch, state_dim, time)

    # Standardize the trajectory data
    state_mean, state_var = state_moments
    traj_c = traj - state_mean[None, :, None].to(traj.device)  # Center the data
    traj_c = traj_c / torch.sqrt(state_var[None, :, None]).to(traj_c.device)  # Normalize the

    # Check if augmentation should be applied for this split
    if augment:
        G = state_type.fibergroup
        g = G.sample()  # Uniformly sample a group element.

        if g != G.identity:
            traj_aug = state_type.transform_fibers(traj_c, g)
            if split == "train":
                # For training, replace with augmented data
                traj_c = traj_aug
            else:
                # For test/val, append augmented data to original
                traj_c = torch.cat([traj_c, traj_aug], dim=0)

    past = traj_c[..., :past_frames]  # (batch, state_dim, past_frames)
    future = traj_c[..., past_frames:]  # (batch, state_dim, future_frames)

    # Squeeze the time dimension if it is 1
    if past.shape[-1] == 1:
        past = past.squeeze(-1)
    if future.shape[-1] == 1:
        future = future.squeeze(-1)

    return past, future


def decoder_collect_fn(batch, **dl_kwargs):
    past, future = symmetric_collate(batch, **dl_kwargs)
    y = torch.cat([past, future], dim=0)
    return y, y


@torch.no_grad()
def linear_reconstruction_metrics(
    ncp_model: EvolOp1D,
    train_ds: torch.utils.data.Dataset,
    eval_dl: torch.utils.data.DataLoader,
    dl_kwargs: dict = None,
    **kwargs,
):
    """Compute the NCP future forcasting capability"""

    ncp_device = next(ncp_model.parameters()).device
    ncp_model.eval()

    # Compute the training data for the linear regressor.
    train_dl = DataLoader(
        train_ds,
        batch_size=max(len(train_ds) // 4, 128),
        shuffle=False,
        collate_fn=lambda x: decoder_collect_fn(x, split="train", **dl_kwargs),
    )

    lin_dec = ncp_model.fit_linear_decoder(train_dataloader=train_dl)

    metrics = {}
    for batch_idx, (x, x_next) in enumerate(eval_dl):
        batch_metrics = {}
        z, z_next_gt = ncp_model(x=x.to(device=ncp_device), y=x_next.to(device=ncp_device))
        z_next_pred = ncp_model.conditional_expectation(x=x.to(device=ncp_device))
        x_rec = lin_dec(z)
        x_next_pred = lin_dec(z_next_pred)

        batch_metrics["latent_lin_pred_err"] = torch.nn.functional.mse_loss(
            input=z_next_pred.cpu(), target=z_next_gt.cpu(), reduction="mean"
        ).item()
        batch_metrics["lin_rec_err"] = torch.nn.functional.mse_loss(
            input=x_rec.cpu(), target=x.cpu(), reduction="mean"
        ).item()
        batch_metrics["lin_pred_err"] = torch.nn.functional.mse_loss(
            input=x_next_pred.cpu(), target=x_next.cpu(), reduction="mean"
        ).item()

        for key, value in batch_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)

    # Average the metrics over the batches
    for key, value in metrics.items():
        metrics[key] = np.mean(value).item()

    return metrics


@hydra.main(config_path="cfg", config_name="evol_dynamics", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed > 0 else np.random.randint(0, 1000)
    seed_everything(seed)
    batch_size = cfg.optim.batch_size

    # Load dataset______________________________________________________________________
    datasets, state_type, (state_mean, state_var) = get_dataset(cfg)

    train_ds, val_ds, test_ds = datasets

    # Get the model_____________________________________________________________________
    model = get_model(cfg, state_type=state_type)
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters())
    log.info(f"No. trainable parameters: {n_trainable_params}")

    # Define the dataloaders_____________________________________________________________
    dl_kwargs = dict(
        augment=cfg.dataset.augment,
        state_type=state_type,
        state_moments=(state_mean, state_var),
        past_frames=cfg.dataset.past_frames,
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        collate_fn=lambda batch_data: symmetric_collate(batch_data, split="train", **dl_kwargs),
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size,
        shuffle=True,
        collate_fn=lambda batch_data: symmetric_collate(batch_data, split="val", **dl_kwargs),
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size,
        shuffle=False,
        collate_fn=lambda batch_data: symmetric_collate(batch_data, split="test", **dl_kwargs),
    )

    # Define the Lightning module ______________________________________________________
    lightning_module = TrainingModule(
        model=model,
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": cfg.optim.lr},
        loss_fn=model.loss if not cfg.optim.regression_loss else model.regression_loss,
        val_metrics=lambda **kwargs: linear_reconstruction_metrics(
            ncp_model=model, train_ds=train_ds, eval_dl=val_dataloader, dl_kwargs=dl_kwargs, **kwargs
        ),
        test_metrics=lambda **kwargs: linear_reconstruction_metrics(
            ncp_model=model, train_ds=train_ds, eval_dl=test_dataloader, dl_kwargs=dl_kwargs, **kwargs
        ),
    )

    # Define Lightning Trainer  ____________________________________________________
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    VAL_METRIC = "||k(x,y) - k_r(x,y)||/val" if not cfg.optim.regression_loss else "loss/val"
    ckpt_call, early_call, logger = get_train_logger_and_callbacks(run_path, cfg, VAL_METRIC)
    BEST_CKPT_NAME, LAST_CKPT_NAME = "best", ModelCheckpoint.CHECKPOINT_NAME_LAST
    last_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / LAST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
    best_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / BEST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)

    trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
        max_epochs=cfg.optim.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=25,
        check_val_every_n_epoch=5,
        callbacks=[ckpt_call, early_call],
        fast_dev_run=25 if cfg.debug else False,
        num_sanity_val_steps=5,
    )

    # Train the model ________________________________________________________________
    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=last_ckpt_path if last_ckpt_path.exists() else None,
    )

    # Test the model _________________________________________________________________
    test_logs = trainer.test(
        lightning_module,
        dataloaders=test_dataloader,
        ckpt_path=best_ckpt_path if best_ckpt_path.exists() else None,
    )

    test_metrics = test_logs[0]  # dict: metric_name -> value
    test_metrics_path = pathlib.Path(run_path) / "test_metrics.csv"
    pd.DataFrame(test_metrics, index=[0]).to_csv(test_metrics_path, index=False)

    # Put model in cpu
    model.cpu()
    model.eval()
    lightning_module.cpu()

    # Wand sync
    logger.experiment.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
