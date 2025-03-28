"""Real World Experiment of G-Equivariant Regression in Robotics."""

from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
import pathlib
from pathlib import Path

import escnn
import hydra
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from escnn.group import directsum
from escnn.nn import FieldType, GeometricTensor
from gym_quadruped.utils.quadruped_utils import configure_observation_space_representations
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from plotly.subplots import make_subplots
from torch.optim import Adam
from torch.utils.data import DataLoader, default_collate

from paper.experiments.CoM_regression import (
    get_model,
    ncp_regression,
    proprioceptive_regression_metrics,
    regression_metrics,
)
from paper.experiments.grf_regression.proprioceptive_datasets import ProprioceptiveDataset
from symm_rep_learn.models.equiv_ncp import ENCP
from symm_rep_learn.models.lightning_modules import SupervisedTrainingModule, TrainingModule
from symm_rep_learn.models.ncp import NCP
from symm_rep_learn.nn.equiv_layers import EMLP
from symm_rep_learn.nn.layers import MLP

log = logging.getLogger(__name__)


def plot_predictions_vs_gt(gt, pred, title_prefix="Dim", subtitles=None, title="Observables"):
    """Plots predictions vs ground truth per dimension with shared legend group using Plotly.

    Args:
        gt (torch.Tensor or np.ndarray): Ground truth (time, dim)
        pred (torch.Tensor or np.ndarray): Predictions (time, dim)
        title_prefix (str): Fallback prefix for subplot titles.
        subtitles (list of str): Optional list of titles per dimension.
        title (str): Title of the entire figure.
    """
    gt = gt.cpu().numpy() if hasattr(gt, "cpu") else gt
    pred = pred.cpu().numpy() if hasattr(pred, "cpu") else pred

    time = np.arange(gt.shape[0])
    dim = gt.shape[1]
    n_cols = min(3, dim)
    n_rows = int(np.ceil(dim / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        subplot_titles=[
            subtitles[i] if subtitles and i < len(subtitles) else f"{title_prefix} {i}" for i in range(dim)
        ],
    )

    for i in range(dim):
        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(
            go.Scatter(
                x=time,
                y=gt[:, i],
                mode="lines",
                name="GT",
                legendgroup="GT",
                showlegend=(i == 0),
                line=dict(color="blue"),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=pred[:, i],
                mode="lines",
                name="Pred",
                legendgroup="Pred",
                showlegend=(i == 0),
                line=dict(color="red"),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(height=300 * n_rows, width=350 * n_cols, title_text=title)
    return fig


def symmetric_collate(
    batch,
    split: str,
    ds_cfg: DictConfig,
    x_moments: tuple,
    y_moments: tuple,
    x_type: FieldType,
    y_type: FieldType,
    geometric_tensor=False,
):
    batch = default_collate(batch)
    x_obs, y_obs = batch
    x = torch.cat([x_obs[obs_name] for obs_name in ds_cfg.x_obs], dim=-1).to(dtype=torch.float32)
    y = torch.cat([y_obs[obs_name] for obs_name in ds_cfg.y_obs], dim=-1).to(dtype=torch.float32)

    x_batch = torch.squeeze(x, dim=1)
    y_batch = torch.squeeze(y, dim=1)

    x_mean, x_var = x_moments
    y_mean, y_var = y_moments

    # Standardize the data
    x_batch = (x_batch - x_mean) / torch.sqrt(x_var)
    y_batch = (y_batch - y_mean) / torch.sqrt(y_var)

    if (
        (split == "train" and ds_cfg.augment_train)
        or (split == "test" and ds_cfg.augment_test)
        or (split == "val" and ds_cfg.augment_val)
    ):
        G = x_type.fibergroup
        g = G.sample()  # Uniformly sample a group element.
        if g != G.identity:
            x_batch = x_type.transform_fibers(x_batch, g)
            y_batch = y_type.transform_fibers(y_batch, g)

    if geometric_tensor:
        x_batch = GeometricTensor(x_batch, x_type)
        y_batch = GeometricTensor(y_batch, y_type)

    return x_batch, y_batch


def get_proprioceptive_data(cfg: DictConfig):
    #
    data_path = Path(cfg.dataset.path)
    dataset = ProprioceptiveDataset(
        data_path,
        x_obs_names=cfg.dataset.x_obs,
        y_obs_names=cfg.dataset.y_obs,
        x_frames=cfg.dataset.x_frames,
        y_frames=cfg.dataset.y_frames,
        mode=cfg.dataset.mode,
        load_to_memory=cfg.dataset.load_to_memory,
        device=cfg.device if cfg.dataset.device == "cuda" else "cpu",
    )

    # Split into train, validation and test datasets splitting on trajectories ----------------------------------------
    n_trajectories = dataset.n_trajectories
    n_train, n_val = int(n_trajectories * cfg.dataset.train_ratio), int(n_trajectories * cfg.dataset.val_ratio)
    train_ds = dataset.subset_dataset(trajectory_ids=range(n_train))
    val_ds = dataset.subset_dataset(trajectory_ids=range(n_train, n_train + n_val))
    test_ds = dataset.subset_dataset(trajectory_ids=range(n_train + n_val, n_trajectories))
    # Configure the observation space representations ------------------------------------------------------------------
    obs_reps = configure_observation_space_representations(
        robot_name=cfg.robot_name, obs_names=cfg.dataset.x_obs + cfg.dataset.y_obs
    )
    for obs_name, rep in obs_reps.items():
        print(f"- {obs_name}: {rep}")

    # Hack required to ensure mujoco joint configuration convention is used.
    obs_reps["qpos_js"] = obs_reps["qvel_js"]

    # Compute the mean and variance of the observations ---------------------------------------------------------------
    is_equiv_model = cfg.model.lower() == "emlp" or cfg.model.lower() == "encp"
    train_ds.compute_obs_moments(obs_reps=obs_reps if is_equiv_model else None)
    train_ds.shuffle()

    y_obs_dims = {}
    start_dim = 0
    for obs_name in cfg.dataset.y_obs:
        mean, _ = train_ds.mean_vars[obs_name]
        y_obs_dims[obs_name] = slice(start_dim, start_dim + mean.shape[-1])
        start_dim += mean.shape[-1]

    x_mean = torch.tensor(np.concatenate([train_ds.mean_vars[obs_name][0] for obs_name in cfg.dataset.x_obs]))
    x_var = torch.tensor(np.concatenate([train_ds.mean_vars[obs_name][1] for obs_name in cfg.dataset.x_obs]))
    y_mean = torch.tensor(np.concatenate([train_ds.mean_vars[obs_name][0] for obs_name in cfg.dataset.y_obs]))
    y_var = torch.tensor(np.concatenate([train_ds.mean_vars[obs_name][1] for obs_name in cfg.dataset.y_obs]))
    x_mean = x_mean.to(dtype=train_ds.dtype, device=train_ds.device)
    x_var = x_var.to(dtype=train_ds.dtype, device=train_ds.device)
    y_mean = y_mean.to(dtype=train_ds.dtype, device=train_ds.device)
    y_var = y_var.to(dtype=train_ds.dtype, device=train_ds.device)

    train_samples = train_ds.numpy_arrays
    val_samples = val_ds.numpy_arrays
    test_samples = test_ds.numpy_arrays
    # Flatten time and trajectory dimensions
    train_samples = {k: np.concatenate(v, axis=0) for k, v in train_samples.items()}
    val_samples = {k: np.concatenate(v, axis=0) for k, v in val_samples.items()}
    test_samples = {k: np.concatenate(v, axis=0) for k, v in test_samples.items()}

    assert cfg.dataset.x_frames == 1 and cfg.dataset.y_frames == 1, "Only single frame supported for now."
    x_train = torch.tensor(np.concatenate([train_samples[obs_name] for obs_name in cfg.dataset.x_obs], axis=1))
    y_train = torch.tensor(np.concatenate([train_samples[obs_name] for obs_name in cfg.dataset.y_obs], axis=1))
    x_val = torch.tensor(np.concatenate([val_samples[obs_name] for obs_name in cfg.dataset.x_obs], axis=1))
    y_val = torch.tensor(np.concatenate([val_samples[obs_name] for obs_name in cfg.dataset.y_obs], axis=1))
    x_test = torch.tensor(np.concatenate([test_samples[obs_name] for obs_name in cfg.dataset.x_obs], axis=1))
    y_test = torch.tensor(np.concatenate([test_samples[obs_name] for obs_name in cfg.dataset.y_obs], axis=1))

    # Standardized data for inference tasks.
    train_samples = (x_train - x_mean) / torch.sqrt(x_var), (y_train - y_mean) / torch.sqrt(y_var)
    val_samples = (x_val - x_mean) / torch.sqrt(x_var), (y_val - y_mean) / torch.sqrt(y_var)
    test_samples = (x_test - x_mean) / torch.sqrt(x_var), (y_test - y_mean) / torch.sqrt(y_var)

    # Group representations ___________________________________________________________________________________________
    rep_x = directsum([obs_reps[obs_name] for obs_name in cfg.dataset.x_obs], name="rep_X")
    rep_y = directsum([obs_reps[obs_name] for obs_name in cfg.dataset.y_obs], name="rep_Y")
    assert rep_x.size == x_mean.size(0), f"{rep_x.size} != {x_mean.size(0)}"
    assert rep_y.size == y_mean.size(0), f"{rep_y.size} != {y_mean.size(0)}"

    return (
        (train_samples, val_samples, test_samples),
        (train_ds, val_ds, test_ds),
        (rep_x, rep_y),
        (x_mean, x_var),
        (y_mean, y_var),
        y_obs_dims,
    )


@torch.no_grad()
def uncertainty_metrics(model, x, y, y_train, x_type, y_type, y_obs_dims, y_moments):
    """Args:
        model: NCP or ENCP model.
        x: (batch, x_dim) tensor of input data to evaluate.
        y: (batch, y_dim) tensor of target data to regress with uncertainty quantification.
        y_train: (n_train, y_dim) tensor of Y training data (standardized).
        x_type: (FieldType) input field type.
        y_type: (FieldType) output field type.
        y_obs_dims: (dict[str, slice]) dictionary with names of observables in the Y vector (e.g., "force": slice(0, 3)).)
        y_moments: Mean and variance of the Y data, used to standardize the data.

    Returns:

    """
    pass


@hydra.main(config_path="cfg", config_name="GRF_regression", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    seed_everything(seed)

    # Load dataset______________________________________________________________________
    samples, datasets, (rep_x, rep_y), x_moments, y_moments, y_obs_dims = get_proprioceptive_data(cfg)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = samples
    train_ds, val_ds, test_ds = datasets

    G = rep_x.group
    # lat_rep = G.regular_representation
    x_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_x])
    y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_y])

    # Get the model_____________________________________________________________________
    model = get_model(cfg, x_type, y_type)
    print(model)
    # Print the number of trainable parameters
    n_trainable_params = sum(p.numel() for p in model.parameters())
    log.info(f"No. trainable parameters: {n_trainable_params}")

    # Define the dataloaders_____________________________________________________________
    # ESCNN equivariant models expect GeometricTensors.
    is_equiv_model = isinstance(model, ENCP) or isinstance(model, EMLP)
    dl_kwargs = dict(
        ds_cfg=cfg.dataset,
        x_moments=x_moments,
        y_moments=y_moments,
        x_type=x_type,
        y_type=y_type,
        geometric_tensor=is_equiv_model,
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        collate_fn=lambda x: symmetric_collate(x, split="train", **dl_kwargs),
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        collate_fn=lambda x: symmetric_collate(x, split="val", **dl_kwargs),
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        collate_fn=lambda x: symmetric_collate(x, split="test", **dl_kwargs),
    )

    # Define the Lightning module ______________________________________________________
    if isinstance(model, MLP) or isinstance(model, EMLP):
        lightning_module = SupervisedTrainingModule(
            model=model,
            optimizer_fn=Adam,
            optimizer_kwargs={"lr": cfg.optim.lr},
            loss_fn=lambda x, y: proprioceptive_regression_metrics(x, y, y_obs_dims, y_moments),
        )
    else:  # NCP / ENCP models
        metrics_args = (y_train, x_type, y_type, y_obs_dims, y_moments, cfg.lstsq, cfg.analytic_residual)
        lightning_module = TrainingModule(
            model=model,
            optimizer_fn=Adam,
            optimizer_kwargs={"lr": cfg.optim.lr},
            loss_fn=model.loss if hasattr(model, "loss") else None,
            val_metrics=lambda _: regression_metrics(model, x_val, y_val, *metrics_args),
            test_metrics=lambda _: regression_metrics(model, x_test, y_test, *metrics_args),
        )

    # Define the logger and callbacks
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.proj_name, log_model=False, config=run_cfg)
    n_total_samples = int(len(train_ds) / cfg.dataset.train_ratio)
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
    check_val_every_n_epochs = max(5, int(n_total_samples // cfg.optim.batch_size / 8))
    metric_to_monitor = "||k(x,y) - k_r(x,y)||/val" if isinstance(model, ENCP) or isinstance(model, NCP) else "loss/val"
    early_call = EarlyStopping(metric_to_monitor, patience=50, mode="min")

    trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
        max_epochs=max_steps,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=25,
        check_val_every_n_epoch=check_val_every_n_epochs,
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
    lightning_module.cpu()

    x_mean, x_var, y_mean, y_var = x_moments[0], x_moments[1], y_moments[0], y_moments[1]
    if isinstance(model, NCP):
        y_test_pred = ncp_regression(model, x_test, y_test, y_train, x_type, y_type, cfg.lstsq, cfg.analytic_residual)
    else:
        y_test_pred = model(x_test)
    y_test_un = y_test * torch.sqrt(y_var) + y_mean
    y_test_pred = y_test_pred * torch.sqrt(y_var) + y_mean

    grf = y_test_un[:, y_obs_dims["contact_forces:base"]].detach().cpu().numpy()  # (time, 12)
    grf_pred = y_test_pred[:, y_obs_dims["contact_forces:base"]].detach().cpu().numpy()  # (time, 12)

    fig = plot_predictions_vs_gt(grf[600:3000], grf_pred[600:3000], title_prefix="grf", title="Contact Forces")
    # Save the figure in high resolution
    # fig_path = pathlib.Path(run_path) / "grf_predictions.html"
    # fig.write_html(str(fig_path))
    fig_path = pathlib.Path(run_path) / "grf_predictions.png"
    fig.write_image(str(fig_path))

    # Flush the logger.
    logger.finalize(trainer.state)
    # Wand sync
    logger.experiment.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
