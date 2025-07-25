"""Real World Experiment of G-Equivariant Regression in Robotics."""

from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
import pathlib
from pyexpat import model

import escnn
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
from paper.experiments.train_utils import get_model, inference_metrics
from symm_rep_learn.inference.ncp import NCPConditionalCDF, NCPRegressor
from symm_rep_learn.models.equiv_ncp import ENCP
from symm_rep_learn.models.lightning_modules import TrainingModule
from symm_rep_learn.models.ncp import NCP

log = logging.getLogger(__name__)


def symmetric_collate(
    batch,
    split: str,
    ds_cfg: DictConfig,
    past_frames: int,
    state_type: FieldType,
    state_moments: tuple[torch.Tensor, torch.Tensor],
    geometric_tensor=False,
):
    batch = default_collate(batch)

    traj = batch

    assert traj.ndim == 3, f"Expected tensor of shape (batch, state_dim, time), got {traj.shape}"

    # Standardize the trajectory data
    state_mean, state_var = state_moments
    traj = traj - state_mean[None, :, None].to(traj.device)  # Center the data
    traj = traj / torch.sqrt(state_var[None, :, None]).to(traj.device)  # Normalize the

    # Check if augmentation should be applied for this split
    should_augment = (
        (split == "train" and ds_cfg.augment_train)
        or (split == "test" and ds_cfg.augment_test)
        or (split == "val" and ds_cfg.augment_val)
    )

    if should_augment:
        G = state_type.fibergroup
        g = G.sample()  # Uniformly sample a group element.

        if g != G.identity:
            traj_aug = state_type.transform_fibers(traj, g)
            if split == "train":
                # For training, replace with augmented data
                traj = traj_aug
            else:
                # For test/val, append augmented data to original
                traj = torch.cat([traj, traj_aug], dim=0)

    past = traj[..., :past_frames]  # (batch, state_dim, past_frames)
    future = traj[..., past_frames:]  # (batch, state_dim, future_frames)

    # Squeeze the time dimension if it is 1
    if past.shape[-1] == 1:
        past = past.squeeze(-1)
    if future.shape[-1] == 1:
        future = future.squeeze(-1)

    if geometric_tensor:
        # Convert to GeometricTensor if required
        past = GeometricTensor(past, state_type)
        future = GeometricTensor(future, state_type)

    return past, future


def get_dataset(cfg: DictConfig):
    data_desc = (
        f"thomas_b={cfg.dataset.b}_noise={cfg.dataset.noise_scale}_"
        f"trajs={cfg.dataset.n_trajectories}_frames={cfg.dataset.traj_time_horizon}_dt={cfg.dataset.dt}"
    )

    thomas = ThomasAttractor(b=cfg.dataset.b, noise_scale=cfg.dataset.noise_scale)
    rep_state = thomas.get_state_symmetry_rep()

    data_path = pathlib.Path(cfg.dataset.path) / data_desc
    if not data_path.exists():
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
    train_ds = TrajectoryDataset(trajectories=train_trajs, **ds_kwargs)
    val_ds = TrajectoryDataset(trajectories=val_trajs, **ds_kwargs)
    test_ds = TrajectoryDataset(trajectories=test_trajs, **ds_kwargs)

    # Configure the observation space representations ------------------------------------------------------------------
    from symm_learning.nn.conv import GSpace1D
    from symm_learning.stats import var_mean
    from escnn.gspaces import no_base_space

    G = rep_state.group
    if cfg.dataset.past_frames == 1 and cfg.dataset.future_frames == 1:
        state_type = FieldType(no_base_space(G), [rep_state])
    else:
        state_type = FieldType(GSpace1D(G), [rep_state])

    # Compute the mean and variance of the observations ---------------------------------------------------------------
    p_train, f_train = train_ds.get_all_past_windows(), train_ds.get_all_future_windows()
    p_val, f_val = val_ds.get_all_past_windows(), val_ds.get_all_future_windows()
    p_test, f_test = test_ds.get_all_past_windows(), test_ds.get_all_future_windows()

    state_var, state_mean = var_mean(p_train[..., 0], rep_x=rep_state)

    return (
        ((p_train, f_train), (p_val, f_val), (p_test, f_test)),
        (train_ds, val_ds, test_ds),
        state_type,
        (state_mean, state_var),
    )


@hydra.main(config_path="cfg", config_name="evol_dynamics", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed > 0 else np.random.randint(0, 1000)
    seed_everything(seed)
    batch_size = cfg.optim.batch_size

    # Load dataset______________________________________________________________________
    samples, datasets, state_type, (state_mean, state_var) = get_dataset(cfg)
    (past_train, future_train), (past_val, future_val), (past_test, future_test) = samples
    # TODO: Only for now.
    future_train = future_train.to(device="cpu").squeeze(-1)  # Remove the last dimension if it is 1
    future_val = future_val.to(device="cpu").squeeze(-1)  # Remove the last dimension if it is 1
    future_test = future_test.to(device="cpu").squeeze(-1)  # Remove the last dimension if it is 1
    past_train = past_train.to(device="cpu").squeeze(-1)  # Remove the last dimension if it is 1
    past_val = past_val.to(device="cpu").squeeze(-1)  # Remove the last dimension if it is 1
    past_test = past_test.to(device="cpu").squeeze(-1)  # Remove the last dimension if it is 1

    train_ds, val_ds, test_ds = datasets

    # Get the model_____________________________________________________________________
    model = get_model(cfg, x_type=state_type, y_type=state_type)
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters())
    log.info(f"No. trainable parameters: {n_trainable_params}")
    is_equiv_model = isinstance(model, ENCP) or isinstance(model, EMLP)

    # Define the dataloaders_____________________________________________________________
    dl_kwargs = dict(
        ds_cfg=cfg.dataset,
        state_type=state_type,
        state_moments=(state_mean, state_var),
        past_frames=cfg.dataset.past_frames,
        geometric_tensor=is_equiv_model,
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
        loss_fn=model.loss if hasattr(model, "loss") else None,
        val_metrics=lambda _: inference_metrics(
            model,
            x_cond=past_val,
            y_gt=future_val,
            y_train=future_train,
            x_type=state_type,
            y_type=state_type,
            alpha=cfg.alpha,
            lstsq=cfg.lstsq,
        ),
        test_metrics=lambda _: inference_metrics(
            model,
            x_cond=past_test,
            y_gt=future_test,
            y_train=future_train,
            x_type=state_type,
            y_type=state_type,
            alpha=cfg.alpha,
            lstsq=cfg.lstsq,
        ),
    )

    # Define the logger and callbacks
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.proj_name, log_model=False, config=run_cfg)
    logger.watch(model, log="gradients")
    n_total_samples = int(len(train_ds) / cfg.dataset.train_ratio)
    scaled_saved_freq = int(5 * n_total_samples // batch_size)
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
    metric_to_monitor = "||k(x,y) - k_r(x,y)||/val" if isinstance(model, ENCP) or isinstance(model, NCP) else "loss/val"
    early_call = EarlyStopping(metric_to_monitor, patience=cfg.optim.patience, mode="min")

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
    # Save the testing matrices in a csv file using pandas.
    test_metrics_path = pathlib.Path(run_path) / "test_metrics.csv"
    pd.DataFrame(test_metrics, index=[0]).to_csv(test_metrics_path, index=False)

    # Put model in cpu
    model.cpu()
    model.eval()
    lightning_module.cpu()

    if isinstance(model, NCP):
        ncp_state_reg = NCPRegressor(model=model, y_train=future_train, zy_train=future_train, lstsq=cfg.lstsq)
        ncp_ccdf = NCPConditionalCDF(model=model, y_train=future_train, support_discretization_points=500, lstsq=cfg.lstsq)
        y_test_pred = ncp_state_reg(x_cond=past_test)
        q_low, q_high = ncp_ccdf.conditional_quantiles(x_cond=past_test, alpha=cfg.alpha)
    else:
        raise ValueError(f"Model type {type(model)} not supported.")

    y_test_pred = y_test_pred.to(future_train.device, future_train.dtype)
    q_low = torch.tensor(q_low).to(future_train.device, future_train.dtype)
    q_high = torch.tensor(q_high).to(future_train.device, future_train.dtype)

    y_test_un = future_test * torch.sqrt(state_var) + state_mean
    y_test_pred_un = y_test_pred * torch.sqrt(state_var) + state_mean
    q_loq_un = (q_low * torch.sqrt(state_var) + state_mean).detach()
    q_upq_un = (q_high * torch.sqrt(state_var) + state_mean).detach()

    for obs_name, obs_dims in y_obs_dims.items():
        obs = y_test_un[:, y_obs_dims[obs_name]].detach().cpu().numpy()  # (time, 12)
        q_low_obs = q_loq_un[:, y_obs_dims[obs_name]].detach().cpu().numpy()
        q_high_obs = q_upq_un[:, y_obs_dims[obs_name]].detach().cpu().numpy()
        view_range = slice(600, 1000)
        fig = plot_gt_and_quantiles(
            obs[view_range], q_low_obs[view_range], q_high_obs[view_range], title_prefix="grf", title="Contact Forces"
        )
        fig_path = pathlib.Path(run_path) / f"{obs_name}_uncertainty_quantification.png"
        fig.write_image(str(fig_path))
        fig_path = pathlib.Path(run_path) / f"{obs_name}_uncertainty_quantification.html"
        fig.write_html(str(fig_path))
        log.info(f"Saved figure to {fig_path}")

    # Wand sync
    logger.experiment.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
