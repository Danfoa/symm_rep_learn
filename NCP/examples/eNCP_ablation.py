"""Ablation Study for G-Equivariant Regression with UQ."""

import logging

import hydra
import lightning as L
import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, default_collate

from NCP.models.ncp_lightning_module import TrainingModule

log = logging.getLogger(__name__)


def get_model2(cfg: DictConfig) -> torch.nn.Module:
    # TODO: This function should be refactored into a utils file
    embedding_dim = cfg.architecture.embedding_dim
    dim_x = cfg.experiment.dim_x
    dim_y = dim_x  # dim_y = dim_x

    if cfg.model.lower() == "encp":  # Equivariant NCP
        raise NotImplementedError("This experiment currently doesn't support eNCP.")
    elif cfg.model.lower() == "ncp":  # NCP
        from NCP.models.ncp import NCP
        from NCP.mysc.utils import class_from_name
        from NCP.nn.layers import MLP

        activation = class_from_name("torch.nn", cfg.architecture.activation)
        kwargs = dict(
            output_shape=embedding_dim,
            n_hidden=cfg.architecture.hidden_layers,
            layer_size=cfg.architecture.hidden_units,
            activation=activation,
            bias=False,
            iterative_whitening=cfg.architecture.iter_whitening,
        )
        fx = MLP(input_shape=dim_x, **kwargs)
        fy = MLP(input_shape=dim_y, **kwargs)
        ncp = NCP(
            embedding_x=fx,
            embedding_y=fy,
            embedding_dim=embedding_dim,
            gamma=cfg.optim.regularization,
            truncated_op_bias=cfg.architecture.last_layer,
        )
        return ncp
    elif cfg.model.lower() == "drf":  # Density Ratio Fitting
        from NCP.models.density_ratio_fitting import DRF
        from NCP.mysc.utils import class_from_name
        from NCP.nn.layers import MLP

        activation = class_from_name("torch.nn", cfg.architecture.activation)
        embedding = MLP(
            input_shape=dim_x + dim_y,  # z = (x,y)
            output_shape=1,
            n_hidden=cfg.architecture.hidden_layers,
            layer_size=cfg.architecture.hidden_units * 2,
            activation=activation,
            bias=False,
        )
        drf = DRF(embedding=embedding, gamma=cfg.optim.regularization)
        return drf
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


def equiv_regression_dataset(n_samples, dim, noise_var, alpha):
    """Y = f(X) + eps, where f is g-equivariant and eps is white noise."""
    # TODO: The scale of X could be changed without breaking symmetries
    X = stats.norm.rvs(loc=0, scale=1, size=(n_samples, dim))
    eps = stats.norm.rvs(loc=0, scale=noise_var, size=(n_samples, dim))
    Y = alpha * X + eps

    # Split data into train, validation, and test subsets
    # TODO: Split rations could be params
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    n_samples = X.shape[0]
    n_train, n_val, n_test = np.asarray(
        np.array([train_ratio, val_ratio, test_ratio]) * n_samples, dtype=int
    )
    train_samples = X[:n_train], Y[:n_train]
    val_samples = X[n_train : n_train + n_val], Y[n_train : n_train + n_val]
    test_samples = X[n_train + n_val :], Y[n_train + n_val :]

    # Standardize data
    X_c = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_c = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    x_train, x_val, x_test = (
        X_c[:n_train],
        X_c[n_train : n_train + n_val],
        X_c[n_train + n_val :],
    )
    y_train, y_val, y_test = (
        Y_c[:n_train],
        Y_c[n_train : n_train + n_val],
        Y_c[n_train + n_val :],
    )

    # Moving data to gpu
    X_train = torch.atleast_2d(torch.from_numpy(x_train).float()).to("cuda")
    Y_train = torch.atleast_2d(torch.from_numpy(y_train).float()).to("cuda")
    X_val = torch.atleast_2d(torch.from_numpy(x_val).float())
    Y_val = torch.atleast_2d(torch.from_numpy(y_val).float())
    X_test = torch.atleast_2d(torch.from_numpy(x_test).float())
    Y_test = torch.atleast_2d(torch.from_numpy(y_test).float())

    # Define datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    return (
        (train_samples, val_samples, test_samples),
        (train_dataset, val_dataset, test_dataset),
    )


@hydra.main(config_path="cfg", config_name="eNCP_ablation", version_base="1.3")
def main(cfg: DictConfig):
    seed = (
        cfg.experiment.seed if cfg.experiment.seed >= 0 else np.random.randint(0, 1000)
    )

    # Sample from data model____________________________________________________________
    seed_everything(seed)
    (train_samples, val_samples, test_samples), datasets = equiv_regression_dataset(
        n_samples=cfg.experiment.n_samples,
        dim=cfg.experiment.dim_x,
        noise_var=cfg.experiment.noise_variance,
        alpha=cfg.experiment.alpha,
    )
    train_ds, val_ds, test_ds = datasets

    collate_fn = default_collate
    train_dataloader = DataLoader(
        train_ds, batch_size=cfg.optim.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=cfg.optim.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_ds, batch_size=cfg.optim.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Get the model_____________________________________________________________________
    model = get_model2(cfg)
    print(model)

    # Define the Lightning module ______________________________________________________
    lightning_module = TrainingModule(
        model=model,
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": cfg.optim.lr},
        loss_fn=model.loss,
        # val_metrics=lambda x: estimated_mse_loss_indep_gaussian(
        #     nnPME, dim_x=cfg.indep_gauss.dim_x, dim_y=cfg.indep_gauss.dim_y
        # ),
    )

    # Define the logger and callbacks
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(
        save_dir=run_path, project=cfg.proj_name, log_model=False, config=run_cfg
    )
    ckpt_call = ModelCheckpoint(
        dirpath=run_path,
        filename="best",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    # TODO: What is this metric?
    # NCP seems to saturate MI mse when "||E - E_r||_HS" is minimized
    early_call = EarlyStopping(
        monitor="||k(x,y) - k_r(x,y)||/val", patience=cfg.optim.patience, mode="min"
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[cfg.env.device]
        if cfg.env.device != -1
        else cfg.env.device,  # -1 for all available GPUs
        max_epochs=cfg.optim.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=25,
        check_val_every_n_epoch=20,
        callbacks=[ckpt_call, early_call],
        fast_dev_run=10 if cfg.experiment.debug else False,
    )

    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Loads the best model.
    trainer.test(lightning_module, dataloaders=test_dataloader)
    # Flush the logger.
    logger.finalize(trainer.state)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
