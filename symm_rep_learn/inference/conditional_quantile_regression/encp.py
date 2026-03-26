# Created  at 29/03/25
from __future__ import annotations

import torch
from escnn.group import Representation

from symm_rep_learn.models.neural_conditional_probability.encp import ENCP

from .ncp import NCPConditionalCDF


class ENCPConditionalCDF(NCPConditionalCDF):
    def __init__(
        self,
        model: ENCP,
        y_train: torch.Tensor,
        y_rep: Representation | None = None,
        **ncp_ccdf_kwargs,
    ):
        # For now do data-agumentation for the discretization _______________________________________-
        if y_rep is None:
            y_rep = getattr(model, "y_rep", None)
        if y_rep is None and hasattr(y_train, "type") and hasattr(y_train.type, "representation"):
            y_rep = y_train.type.representation

        if not isinstance(y_train, torch.Tensor):
            if hasattr(y_train, "tensor"):
                y_train = y_train.tensor
            else:
                raise TypeError(f"Expected y_train as torch.Tensor, got {type(y_train)}")

        if y_rep is None:
            raise ValueError(
                "Could not infer y_rep. Provide `y_rep` explicitly or pass an ENCP model exposing `model.y_rep`."
            )

        self.y_rep = y_rep
        self.G = self.y_rep.group
        assert y_train.shape[-1] == self.y_rep.size, (
            f"Expected y_train shape (..., {self.y_rep.size}), got {tuple(y_train.shape)}"
        )
        y_train = y_train.cpu()
        # Do data-agumentation. TODO: We could use equivariance constraints to avoid augmentation.
        Gy_train = [y_train]
        for g in self.G.elements[1:]:
            rep_g = torch.tensor(self.y_rep(g), dtype=y_train.dtype, device=y_train.device)
            Gy_train.append(torch.einsum("ij,...j->...i", rep_g, y_train))
        Gy_train = torch.cat(Gy_train, dim=0)

        # Initialize NCPConditionalCDF with augmented data
        super().__init__(model=model, y_train=Gy_train, **ncp_ccdf_kwargs)
