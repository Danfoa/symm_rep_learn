import torch
import math

from omegaconf import DictConfig


def get_model(cfg: DictConfig, x_type, y_type) -> torch.nn.Module:
    embedding_dim = cfg.architecture.embedding_dim
    dim_x = x_type.size
    dim_y = y_type.size

    if cfg.model.lower() == "encp":  # Equivariant NCP
        import escnn
        from escnn.nn import FieldType
        from symm_learning.models.emlp import EMLP

        from symm_rep_learn.models.equiv_ncp import ENCP

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
            gamma=cfg.gamma,
            gamma_centering=cfg.gamma_centering,
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
            gamma_centering=cfg.gamma_centering,
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
