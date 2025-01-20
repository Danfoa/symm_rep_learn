import logging

import hydra
import lightning
import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset, default_collate

from NCP.models.ncp_lightning_module import TrainingModule

log = logging.getLogger(__name__)


def indep_gaussian_dataset(n_samples, dim_x, dim_y, device):
    """Z = (X,Y), where Z ~ N(0,Id_{dim_x+dim_y})."""
    x_samples = stats.norm.rvs(loc=1, scale=1, size=(n_samples, dim_x))
    y_samples = stats.norm.rvs(loc=1, scale=1, size=(n_samples, dim_y))

    # Train, val, test splitting
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    n_samples = len(x_samples)
    n_train, n_val, n_test = np.asarray(np.array([train_ratio, val_ratio, test_ratio]) * n_samples, dtype=int)
    train_samples = x_samples[:n_train], y_samples[:n_train]
    val_samples = x_samples[n_train : n_train + n_val], y_samples[n_train : n_train + n_val]
    test_samples = x_samples[n_train + n_val :], y_samples[n_train + n_val :]

    X_c = x_samples
    Y_c = y_samples
    x_train, x_val, x_test = X_c[:n_train], X_c[n_train : n_train + n_val], X_c[n_train + n_val :]
    y_train, y_val, y_test = Y_c[:n_train], Y_c[n_train : n_train + n_val], Y_c[n_train + n_val :]

    X_train = torch.atleast_2d(torch.from_numpy(x_train).float()).to(device=device)
    Y_train = torch.atleast_2d(torch.from_numpy(y_train).float()).to(device=device)
    X_val = torch.atleast_2d(torch.from_numpy(x_val).float())
    Y_val = torch.atleast_2d(torch.from_numpy(y_val).float())
    X_test = torch.atleast_2d(torch.from_numpy(x_test).float())
    Y_test = torch.atleast_2d(torch.from_numpy(y_test).float())

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    return (
        (train_samples, val_samples, test_samples),
        (train_dataset, val_dataset, test_dataset),
    )


def estimated_mse_loss_indep_gaussian(nn_model, dim_x, dim_y, n_samples=10000):
    """TODO:"""
    x_samples = stats.norm.rvs(loc=0, scale=1, size=(n_samples, dim_x))
    y_samples = stats.norm.rvs(loc=0, scale=1, size=(n_samples, dim_y))
    X = torch.atleast_2d(torch.from_numpy(x_samples).float()).to("cuda")
    Y = torch.atleast_2d(torch.from_numpy(y_samples).float()).to("cuda")

    return {"PMD/mse_naive": ((nn_model.pointwise_mutual_dependency(X, Y) - 1) ** 2).mean()}


def analytic_mse_loss_indep_gaussian(nn_model):
    """Numerical expectation of |F(Z)|^2, where Z is standard gaussian."""
    pass


def get_model2(cfg: DictConfig, dim_x, dim_y, embedding_dim) -> torch.nn.Module:
    # TODO: UNIFY GET_MODEL
    embedding_dim = embedding_dim
    if cfg.model.lower() == "encp":  # Equivariant NCP
        raise NotImplementedError("This experiment currently doesn't support eNCP.")
    elif cfg.model.lower() == "ncp":  # NCP
        from NCP.models.ncp import NCP
        from NCP.mysc.utils import class_from_name
        from NCP.nn.layers import MLP

        activation = class_from_name("torch.nn", cfg.embedding.activation)
        kwargs = dict(
            output_shape=embedding_dim,
            n_hidden=cfg.embedding.hidden_layers,
            layer_size=cfg.embedding.hidden_units,
            activation=activation,
            bias=False,
            iterative_whitening=cfg.iter_whitening,
        )
        fx = MLP(input_shape=dim_x, **kwargs)
        fy = MLP(input_shape=dim_y, **kwargs)
        ncp = NCP(
            embedding_x=fx,
            embedding_y=fy,
            embedding_dim=embedding_dim,
            gamma=cfg.gamma,
            truncated_op_bias=cfg.truncated_op_bias,
        )
        return ncp
    elif cfg.model.lower() == "drf":  # Density Ratio Fitting
        from NCP.models.density_ratio_fitting import DRF
        from NCP.mysc.utils import class_from_name
        from NCP.nn.layers import MLP

        activation = class_from_name("torch.nn", cfg.embedding.activation)
        embedding = MLP(
            input_shape=dim_x + dim_y,  # z = (x,y)
            output_shape=1,
            n_hidden=cfg.embedding.hidden_layers,
            layer_size=cfg.embedding.hidden_units * 2,
            activation=activation,
            bias=False,
        )
        drf = DRF(embedding=embedding, gamma=cfg.gamma)
        return drf
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


@hydra.main(config_path="cfg", config_name="indep_gauss_config", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    # TODO: Add Symmetries

    # GENERATE the training data _______________________________________________________
    seed_everything(seed)
    (train_samples, val_samples, test_samples), datasets = indep_gaussian_dataset(
        n_samples=cfg.indep_gauss.n_samples,
        dim_x=cfg.indep_gauss.dim_x,
        dim_y=cfg.indep_gauss.dim_y,
        device=cfg.device if cfg.data_on_device else "cpu",
    )

    # x_train, y_train = train_samples
    x_val, y_val = val_samples
    x_test, y_test = test_samples

    train_ds, val_ds, test_ds = datasets

    # Get the model ______________________________________________________________________
    nnPME = get_model2(cfg, cfg.indep_gauss.dim_x, cfg.indep_gauss.dim_y, cfg.embedding.embedding_dim)
    print(nnPME)

    collate_fn = default_collate
    train_dataloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the Lightning module ________________________________________________________
    lightning_module = TrainingModule(
        model=nnPME,
        optimizer_fn=torch.optim.Adam,
        optimizer_kwargs={"lr": cfg.lr},
        loss_fn=nnPME.loss,
        val_metrics=lambda x: estimated_mse_loss_indep_gaussian(
            nnPME, dim_x=cfg.indep_gauss.dim_x, dim_y=cfg.indep_gauss.dim_y
        ),
    )

    # Define the logger and callbacks
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.exp_name, log_model=False, config=run_cfg)
    ckpt_call = ModelCheckpoint(
        dirpath=run_path, filename="best", monitor="loss/val", save_top_k=1, save_last=True, mode="min"
    )

    # TODO: What is this metric?
    # NCP seems to saturate MI mse when "||E - E_r||_HS" is minimized
    early_call = EarlyStopping(monitor="||k(x,y) - k_r(x,y)||/val", patience=cfg.patience, mode="min")

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
        max_epochs=cfg.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=25,
        check_val_every_n_epoch=20,
        callbacks=[ckpt_call, early_call],
        fast_dev_run=10 if cfg.debug else False,
    )

    torch.set_float32_matmul_precision("medium")
    trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Loads the best model.
    trainer.test(lightning_module, dataloaders=test_dataloader)
    # Flush the logger.
    logger.finalize(trainer.state)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
