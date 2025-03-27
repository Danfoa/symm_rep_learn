"""Real World Experiment of G-Equivariant Regression in Robotics."""

from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
import math
import pathlib
from pathlib import Path

import escnn
import hydra
import numpy as np
import pandas as pd
import torch
from escnn.group import directsum
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from morpho_symm.data.DynamicsRecording import DynamicsRecording
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, default_collate

from symm_rep_learn.models.equiv_ncp import ENCP
from symm_rep_learn.models.lightning_modules import SupervisedTrainingModule, TrainingModule
from symm_rep_learn.models.ncp import NCP
from symm_rep_learn.mysc.symm_algebra import invariant_orthogonal_projector
from symm_rep_learn.nn.equiv_layers import EMLP
from symm_rep_learn.nn.layers import MLP

log = logging.getLogger(__name__)


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
            from symm_rep_learn.nn.equiv_layers import EMLP, ResidualEncoder

            lat_rep = [reg_rep] * max(1, math.ceil(cfg.architecture.embedding_dim // reg_rep.size))
            lat_x_type = FieldType(
                gspace=escnn.gspaces.no_base_space(G), representations=list(y_type.representations) + lat_rep
            )
            lat_y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=lat_rep)
            x_embedding = EMLP(in_type=x_type, out_type=lat_x_type, **kwargs)
            y_embedding = ResidualEncoder(encoder=EMLP(in_type=y_type, out_type=lat_y_type, **kwargs), in_type=y_type)
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
            embedding_x=x_embedding,
            embedding_y=y_embedding,
            gamma=cfg.gamma,
            truncated_op_bias=cfg.truncated_op_bias,
            learnable_change_of_basis=cfg.learnable_change_basis,
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
            from symm_rep_learn.nn.layers import MLP, ResidualEncoder

            dim_free_embedding = embedding_dim - dim_x
            fx = ResidualEncoder(
                encoder=MLP(input_shape=dim_x, **kwargs | {"output_shape": dim_free_embedding}), in_dim=dim_x
            )
        else:
            fx = MLP(input_shape=dim_x, **kwargs)
        if cfg.architecture.residual_encoder:
            from symm_rep_learn.nn.layers import MLP, ResidualEncoder

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
            learnable_change_basis=cfg.learnable_change_basis,
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


def symmetric_collate(batch, split: str, x_type: FieldType, y_type: FieldType, geometric_tensor=False):
    x_batch, y_batch = default_collate(batch)

    if split == "test" or split == "val":
        G = x_type.fibergroup
        g = G.sample()  # Uniformly sample a group element.
        if g != G.identity:
            x_batch = x_type.transform_fibers(x_batch, g)
            y_batch = y_type.transform_fibers(y_batch, g)

    if geometric_tensor:
        x_batch = GeometricTensor(x_batch, x_type)
        y_batch = GeometricTensor(y_batch, y_type)

    return x_batch, y_batch


def com_momentum_dataset(cfg):
    """Loads dataset for 'Center of Mass Momentum Regression' experiment from https://arxiv.org/pdf/2302.10433."""
    device = cfg.device
    # TODO: Y is currently 6 dimension, without 'KinE'.
    dr = DynamicsRecording.load_from_file(Path(cfg.path_ds).absolute())
    X_obs = [dr.recordings[obs_name].squeeze(1) for obs_name in cfg.x_obs_names]
    Y_obs = [dr.recordings[obs_name].squeeze(1) for obs_name in cfg.y_obs_names]
    X = np.concatenate(X_obs, axis=-1)
    Y = np.concatenate(Y_obs, axis=-1)

    # Compute the symmetry aware mean and variance of each observable
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
def regression_metrics(
    model, x, y, y_train, x_type, y_type, y_obs_dims, y_moments, lstsq=False, analytic_residual=False
):
    """Predicts CoM Momenta from test sample (x_test, y_test)."""
    device = next(model.parameters()).device
    prev_data_device = x.device

    x = x.to(device)
    y = y.to(device)
    y_train = y_train.to(device)

    rep_Y, rep_X = y_type.representation, x_type.representation
    G = rep_X.group

    # Compute the expectation of the r.v `y` from the training dataset.
    mean_y = y_train.mean(axis=0)
    if rep_Y is not None:
        inv_projector = invariant_orthogonal_projector(rep_Y).to(y_train.device)
        mean_y = inv_projector @ mean_y  # Mean of symmetric RV lives in the invariant subspace.
    y_train_c = y_train - mean_y

    # Compute the embeddings of the entire y training dataset. And the linear regression between y and h(y)
    Cyhy = torch.zeros((y_train.shape[-1], model.embedding_dim), device=device)
    n_train = y_train.shape[0]
    if isinstance(model.embedding_y, EquivariantModule):  # Symmetry aware models.
        hy_train = model.embedding_y(y_type(y_train)).tensor  # shape: (n_train, embedding_dim)
        from symm_rep_learn.nn.equiv_layers import ResidualEncoder

        if analytic_residual and isinstance(model.embedding_y[0], ResidualEncoder):
            # Y is embedded in the encoded vector h(y), we can get the prediction using indexing.
            res_encoder = model.embedding_y[0]
            change2iso_module = model.embedding_y[-1]
            Qiso2y = change2iso_module.Qin2iso.T
            Cyhy = Qiso2y[res_encoder.residual_dims, :]
        else:  # Compute the symmetry aware linear regression from h(y) to y
            rep_Hy = model.embedding_y.out_type.representation
            if lstsq:  # TODO: symmetry aware lstsq
                import linear_operator_learning as lol
                Cyhy = lol.nn.symmetric.linalg.lstsq(X=hy_train, Y=y_train_c, rep_X=rep_Hy, rep_Y=rep_Y)
            else:  # Symmetry aware basis expansion coefficients.
                import linear_operator_learning as lol
                Cyhy = lol.nn.symmetric.stats.covariance(X=hy_train, Y=y_train_c, rep_X=rep_Hy, rep_Y=rep_Y)
    else:  # Symmetry agnostic models.
        hy_train = model.embedding_y(y_train)  # shape: (n_train, embedding_dim)
        from symm_rep_learn.nn.layers import ResidualEncoder

        if analytic_residual and isinstance(model.embedding_y, ResidualEncoder):
            y_dims_in_hy = model.embedding_y.residual_dims
            mask = torch.zeros(hy_train.shape[-1], device=device)
            mask[y_dims_in_hy] = 1
            for dim in range(y_dims_in_hy.start, y_dims_in_hy.stop):
                Cyhy[dim, dim] = 1
        else:  # Compute the linear regression from h(y) to y
            if lstsq:
                out = torch.linalg.lstsq(hy_train, y_train_c)
                Cyhy = out.solution.T
                assert Cyhy.shape == (y_train.shape[-1], hy_train.shape[-1]), f"Invalid shape {Cyhy.shape}"
            else:
                Cyhy = (1 / n_train) * torch.einsum("by,bh->yh", y_train_c, hy_train)

    # Introduce the entire group orbit of the testing set, to appropriately compute the equivariant error.
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

        # shape: (n_test, 6). Check formula 12 from https://arxiv.org/pdf/2407.01171
        Dr = model.truncated_operator
        gy_deflated_basis_expansion = torch.einsum("bf,fh,yh->by", fgx, Dr, Cyhy)
        gy_pred = mean_y + gy_deflated_basis_expansion

        gy_mse, metrics = proprioceptive_regression_metrics(gy, gy_pred, y_obs_dims, y_moments)
        G_loss.append(gy_mse)
        G_metrics.append(metrics)

    # Compute average over the orbit for all metrics
    metrics_names = G_metrics[0].keys()
    metrics = {name: torch.stack([m[name] for m in G_metrics]).mean() for name in metrics_names}

    x = x.to(prev_data_device)
    y = y.to(prev_data_device)
    y_train = y_train.to(prev_data_device)
    return metrics


def proprioceptive_regression_metrics(y, y_pred, y_obs_dims: dict, y_moments: tuple):
    # Compute MSE with standarized data
    if isinstance(y, GeometricTensor):
        assert y.type == y_pred.type
        y = y.tensor
        y_pred = y_pred.tensor

    mse = torch.nn.MSELoss(reduce=True)
    y_mse = mse(y_pred, y)
    loss = y_mse
    metrics = {"y_mse": loss}
    with torch.no_grad():
        # Un-standardize data and compute MSE in the original data scale
        y_mean, y_var = y_moments
        y_err_un = (y - y_pred.to(y.device)) * torch.sqrt(y_var.to(y.device))

        for obs_name, dims in y_obs_dims.items():
            obs_err_in = y_err_un[..., dims]
            metrics[obs_name] = (obs_err_in**2).mean()

    return loss, metrics


@hydra.main(config_path="cfg", config_name="CoM_regression", version_base="1.3")
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
    is_equiv_model = isinstance(model, ENCP) or isinstance(model, EMLP)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        collate_fn=lambda x: symmetric_collate(x, "train", x_type, y_type, geometric_tensor=is_equiv_model),
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        collate_fn=lambda x: symmetric_collate(x, "val", x_type, y_type, geometric_tensor=is_equiv_model),
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        collate_fn=lambda x: symmetric_collate(x, "test", x_type, y_type, geometric_tensor=is_equiv_model),
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
                lstsq=cfg.lstsq,
                analytic_residual=cfg.analytic_residual,
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
                lstsq=cfg.lstsq,
                analytic_residual=cfg.analytic_residual,
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
    check_val_every_n_epochs = int(n_total_samples // cfg.optim.batch_size)
    scaled_patience = int(cfg.optim.patience * n_total_samples * 0.7 // cfg.optim.batch_size / check_val_every_n_epochs)
    metric_to_monitor = "||k(x,y) - k_r(x,y)||/val" if isinstance(model, ENCP) or isinstance(model, NCP) else "loss/val"
    early_call = EarlyStopping(metric_to_monitor, patience=int(scaled_patience), mode="min")

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
    # Save the testing matrics in a csv file using pandas.
    test_metrics_path = pathlib.Path(run_path) / "test_metrics.csv"
    pd.DataFrame(test_metrics, index=[0]).to_csv(test_metrics_path, index=False)

    # Put model in cpu
    model.cpu()
    lightning_module.cpu()

    # Flush the logger.
    logger.finalize(trainer.state)
    # Wand sync
    logger.experiment.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
