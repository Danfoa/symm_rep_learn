# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 29/03/25
from __future__ import annotations

import torch
from escnn.nn import GeometricTensor

from symm_rep_learn.models.neural_conditional_probability.encp import ENCP

from .ncp import NCPConditionalCDF


class ENCPConditionalCDF(NCPConditionalCDF):
    def __init__(self, model: ENCP, y_train: GeometricTensor, **ncp_ccdf_kwargs):
        # For now do data-agumentation for the discretization _______________________________________-
        self.y_type = y_train.type
        self.G = self.y_type.fibergroup
        y_train = y_train.tensor.cpu()
        # Do data-agumentation. TODO: We could use equivariance constraints to avoid augmentation.
        Gy_train = [y_train]
        for g in self.G.elements[1:]:
            rep_g = torch.tensor(self.y_type.representation(g), dtype=y_train.dtype, device=y_train.device)
            Gy_train.append(torch.einsum("ij,...j->...i", rep_g, y_train))
        Gy_train = torch.cat(Gy_train, dim=0)

        # Initialize NCPConditionalCDF with augmented data
        super().__init__(model=model, y_train=Gy_train, **ncp_ccdf_kwargs)
