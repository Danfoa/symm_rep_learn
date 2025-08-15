import math

import torch
from escnn.nn import FieldType, GeometricTensor
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import default_collate

from symm_rep_learn.inference.ncp import NCPConditionalCDF
from symm_rep_learn.models.conditional_quantile_regression.cqr import get_coverage, get_relaxed_coverage, get_set_size
from symm_rep_learn.models.neural_conditional_probability.ncp import NCP


def get_train_logger_and_callbacks(
    run_path: str, cfg: DictConfig, val_metric: str
) -> tuple[ModelCheckpoint, EarlyStopping, int]:
    """
    Create ModelCheckpoint and EarlyStopping callbacks for training.

    Parameters:
    -----------
    run_path : str
        Directory path where checkpoints will be saved
    cfg : DictConfig
        Configuration object containing optimization settings
    val_metric : str
        Validation metric to monitor for checkpointing and early stopping

    Returns:
    --------
    tuple[ModelCheckpoint, EarlyStopping, WandbLogger]
        Configured checkpoint and early stopping callbacks, and logger
    """

    logger = WandbLogger(
        save_dir=run_path, project=cfg.proj_name, log_model=False, config=OmegaConf.to_container(cfg, resolve=True)
    )

    BEST_CKPT_NAME = "best"

    ckpt_call = ModelCheckpoint(
        dirpath=run_path,
        filename=BEST_CKPT_NAME,
        monitor=val_metric,
        save_top_k=1,
        save_last=True,
        mode="min",
        every_n_epochs=5,
    )

    assert cfg.optim.check_val_every_n_epoch < cfg.optim.max_epochs, (
        f"check_val_every_n_epoch {cfg.optim.check_val_every_n_epoch} must be less than max_epochs {cfg.optim.max_epochs}"
    )
    assert cfg.optim.patience * cfg.optim.check_val_every_n_epoch <= cfg.optim.max_epochs, (
        f"patience {cfg.optim.patience} * check_val_every_n_epoch {cfg.optim.check_val_every_n_epoch} must be less than max_epochs {cfg.optim.max_epochs}"
    )

    # Fix for all runs independent on the train_ratio chosen. This way we compare on effective number of "epochs"
    check_val_every_n_epoch = (
        cfg.optim.check_val_every_n_epoch
    )  # max(5, int(cfg.optim.max_epochs // cfg.optim.check_val_n_times))
    effective_patience = cfg.optim.patience // check_val_every_n_epoch
    early_call = EarlyStopping(val_metric, patience=effective_patience, mode="min")

    return ckpt_call, early_call, logger


def get_model(cfg: DictConfig, x_type, y_type) -> torch.nn.Module:
    embedding_dim = cfg.architecture.embedding_dim
    dim_x = x_type.size
    dim_y = y_type.size

    if cfg.model.lower() == "encp":  # Equivariant NCP
        import escnn
        from escnn.nn import FieldType
        from symm_learning.models.emlp import EMLP

        from symm_rep_learn.models.neural_conditional_probability.encp import ENCP

        G = x_type.representation.group

        reg_rep = G.regular_representation

        kwargs = dict(
            hidden_layers=cfg.architecture.hidden_layers,
            activation=cfg.architecture.activation,
            hidden_units=cfg.architecture.hidden_units,
            bias=False,
        )
        if cfg.architecture.residual_encoder:
            from symm_rep_learn.nn.equiv_layers import ResidualEncoder

            lat_rep = [reg_rep] * max(1, math.ceil(cfg.architecture.embedding_dim // reg_rep.size))
            lat_x_type = FieldType(
                gspace=escnn.gspaces.no_base_space(G), representations=list(y_type.representations) + lat_rep
            )
            lat_y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=lat_rep)
            x_embedding = EMLP(in_type=x_type, out_type=lat_x_type, **kwargs)
            y_embedding = ResidualEncoder(encoder=EMLP(in_type=y_type, out_type=lat_y_type, **kwargs), in_type=y_type)
            assert y_embedding.out_type.size == x_embedding.out_type.size
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
            orth_reg=cfg.gamma,
            centering_reg=cfg.gamma_centering,
            learnable_change_of_basis=cfg.learnable_change_basis,
        )

        return eNCPop
    elif cfg.model.lower() == "ncp":  # NCP
        from symm_rep_learn.models.neural_conditional_probability.ncp import NCP
        from symm_rep_learn.mysc.utils import class_from_name
        from symm_rep_learn.nn.layers import MLP

        activation = class_from_name("torch.nn", cfg.architecture.activation)
        kwargs = dict(
            output_shape=embedding_dim,
            n_hidden=cfg.architecture.hidden_layers,
            layer_size=cfg.architecture.hidden_units,
            activation=activation,
            bias=False,
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
            embedding_dim_x=embedding_dim,
            embedding_dim_y=embedding_dim,
            orth_reg=cfg.gamma,
            centering_reg=cfg.gamma_centering,
            momentum=cfg.momentum,
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
    x_batch, y_batch = batch

    x_mean, x_var = x_moments
    y_mean, y_var = y_moments

    # Standardize the data
    x_batch = (x_batch - x_mean) / torch.sqrt(x_var)
    y_batch = (y_batch - y_mean) / torch.sqrt(y_var)

    # Check if augmentation should be applied for this split
    should_augment = (
        (split == "train" and ds_cfg.augment_train)
        or (split == "test" and ds_cfg.augment_test)
        or (split == "val" and ds_cfg.augment_val)
    )

    if should_augment:
        G = x_type.fibergroup
        g = G.sample()  # Uniformly sample a group element.

        if g != G.identity:
            x_aug = x_type.transform_fibers(x_batch, g)
            y_aug = y_type.transform_fibers(y_batch, g)

            if split == "train":
                # For training, replace with augmented data
                x_batch, y_batch = x_aug, y_aug
            else:
                # For test/val, append augmented data to original
                x_batch = torch.cat([x_batch, x_aug], dim=0)
                y_batch = torch.cat([y_batch, y_aug], dim=0)

    if geometric_tensor:
        x_batch = GeometricTensor(x_batch, x_type)
        y_batch = GeometricTensor(y_batch, y_type)

    return x_batch, y_batch


@torch.no_grad()
def inference_metrics(
    model, x_cond, y_gt, y_train, x_type, y_type, alpha: float = 0.05, lstsq: bool = True, y_obs_dims: dict = None
):
    """Args:
        model: NCP or ENCP model.
        x_cond: (batch, x_dim) tensor of the conditioning values.
        y_gt: (batch, y_dim) tensor of ground truth target data to regress with uncertainty quantification.
        y_train: (n_train, y_dim) tensor of Y training data (standardized).
        y_type: (FieldType) output field type.
        y_obs_dims: (dict[str, slice]) dictionary with names of observables in the Y vector (e.g., "force": slice(0, 3)).)
        y_moments: Mean and variance of the Y data, used to standardize the data.

    Returns:
        metrics: dict with the following
            - coverage: Coverage of the predicted quantiles (Uncertainty quantification).
            - relaxed_coverage: Relaxed coverage of the predicted quantiles (Uncertainty quantification).
            - set_size: Size of the predicted quantile sets (Uncertainty quantification).
            - mse: Mean Squared Error of the predicted values (Regression).
            - mae: Mean Absolute Error of the predicted values (Regression).
    """

    G = x_type.fibergroup
    x_transformed = [x_type.transform_fibers(x_cond, g) for g in G.elements]
    y_transformed = [y_type.transform_fibers(y_gt, g) for g in G.elements]
    x_cond = torch.cat(x_transformed, dim=0)
    y_gt = torch.cat(y_transformed, dim=0)

    # # if isinstance(model, ENCP):
    #     encp_ccdf = ENCPConditionalCDF(
    #         model=model, y_train=y_type(y_train), support_discretization_points=100, lstsq=lstsq
    #     )
    #     q_low, q_high = encp_ccdf.conditional_quantiles(x_cond=x_type(x_cond), alpha=alpha)
    if isinstance(model, NCP):
        # Instanciate Inference model to regress next-state expectation and conditional CDF
        ncp_state_reg = NCPRegressor(model=model, y_train=y_train, zy_train=y_train, lstsq=lstsq)
        ncp_ccdf = NCPConditionalCDF(model=model, y_train=y_train, support_discretization_points=50, lstsq=lstsq)
        y_pred = ncp_state_reg(x_cond=x_cond)
        q_low, q_high = ncp_ccdf.conditional_quantiles(x_cond=x_cond, alpha=alpha)

    else:
        raise ValueError(f"Model type {type(model)} not supported.")

    y_pred = torch.tensor(y_pred).to(y_gt.device, y_gt.dtype)
    q_low = torch.tensor(q_low).to(y_gt.device, y_gt.dtype)
    q_high = torch.tensor(q_high).to(y_gt.device, y_gt.dtype)

    metrics = dict(
        coverage=get_coverage(q_low, q_high, target=y_gt),
        relaxed_coverage=get_relaxed_coverage(q_low, q_high, target=y_gt),
        set_size=get_set_size(q_low, q_high),
        mse=torch.mean((y_pred - y_gt) ** 2).item(),
        mae=torch.mean(torch.abs(y_pred - y_gt)).item(),
    )

    if y_obs_dims is not None:  # Create metrics per observable
        for obs_name, obs_dims in y_obs_dims.items():
            metrics |= {
                f"{obs_name}/coverage": get_coverage(q_low[:, obs_dims], q_high[:, obs_dims], target=y_gt[:, obs_dims]),
                f"{obs_name}/relaxed_coverage": get_relaxed_coverage(
                    q_low[:, obs_dims], q_high[:, obs_dims], target=y_gt[:, obs_dims]
                ),
                f"{obs_name}/set_size": get_set_size(q_low[:, obs_dims], q_high[:, obs_dims]),
                f"{obs_name}/mse": torch.mean((y_pred[:, obs_dims] - y_gt[:, obs_dims]) ** 2).item(),
                f"{obs_name}/mae": torch.mean(torch.abs(y_pred[:, obs_dims] - y_gt[:, obs_dims])).item(),
            }

    return metrics
