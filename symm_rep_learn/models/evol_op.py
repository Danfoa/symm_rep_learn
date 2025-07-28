# Created by danfoa at 19/12/24
from __future__ import annotations

import logging

import torch

from symm_rep_learn.models.ncp import NCP

log = logging.getLogger(__name__)

import torch.nn.utils.parametrize as P
import torch.nn.utils.parametrizations as param_zoo  # convenience alias


class Symmetric(torch.nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


# Neural Conditional Probability (NCP) modelule ========================================================================
class EvolutionOperator(NCP):
    def __init__(
        self,
        embedding_state: torch.nn.Module,
        state_embedding_dim: int,
        self_adjoint: bool = False,
        **ncp_kwargs,
    ):
        super(EvolutionOperator, self).__init__(
            embedding_x=embedding_state,
            embedding_y=embedding_state,
            embedding_dim_x=state_embedding_dim,
            embedding_dim_y=state_embedding_dim,
            **ncp_kwargs,
        )

        self.self_adjoint = self_adjoint
        # Remove/deregister unnecessary modules
        self.data_norm_y = self.data_norm_x

        if self.self_adjoint:  # Truncated operator is self-adjoint
            # Deregister spectral norm
            P.remove_parametrizations(self.Dr, "weight", leave_parametrized=True)
            # First symmetrize the operator
            P.register_parametrization(self.Dr, "weight", Symmetric())
            # Then apply spectral norm
            param_zoo.spectral_norm(self.Dr, name="weight")

    # def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
    #     """Forward pass of the Evolution Operator.

    #     Differently from NCP, the forward pass of the Evolution Operator uses the same
    #     embedding function for both state and next_state (in a single forward pass).

    #     Args:
    #         state (torch.Tensor): Input state tensor.
    #         next_state (torch.Tensor): Input next state tensor.
    #     """
    #     assert x is not None or y is not None, "At least one of x or y must be provided."

    #     if y is None:
    #         x_samples = x.shape[0]
    #         state_traj = x
    #     elif x is None:
    #         x_samples = 0
    #         state_traj = y
    #     else:
    #         x_samples = x.shape[0]
    #         state_traj = torch.cat([x, y], dim=0) if y is not None else x

    #     fstate = self._embedding_x(state_traj)
    #     fstate_c = self.data_norm_x(fstate)

    #     # Split the embeddings back to the original batch size
    #     fx_c = fstate_c[:x_samples]
    #     hy_c = fstate_c[x_samples:] if y is not None else None

    #     return fx_c, hy_c
