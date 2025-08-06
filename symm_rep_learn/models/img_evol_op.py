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
class ImgEvolutionOperator(NCP):
    def __init__(
        self,
        embedding_state: torch.nn.Module,
        state_embedding_dim: int,
        self_adjoint: bool = False,
        **ncp_kwargs,
    ):
        super(ImgEvolutionOperator, self).__init__(
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

    def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
        """Compute the embedding of the present and next images

        Args:
            state (torch.Tensor): Input state tensor.
            next_state (torch.Tensor): Input next state tensor.
        """
        assert x is not None or y is not None, "At least one of x or y must be provided."
        assert x is None or x.ndim == 4, "x must be a 4D tensor (batch_size, channels, height, width)."
        assert y is None or y.ndim == 4, "y must be a 4D tensor (batch_size, channels, height, width)."

        if y is None:
            x_samples = x.shape[0]
            state_traj = x
        elif x is None:
            x_samples = 0
            state_traj = y
        else:  # Process present and next image using the same embedding in a single forward pass
            x_samples = x.shape[0]
            state_traj = torch.cat([x, y], dim=0) if y is not None else x

        # fstate = self._embedding_x(state_traj)
        fstate_c = self._embedding_x(state_traj)
        # fstate_c = self.data_norm_x(fstate)

        # Separate past and next image embeddings
        fx_c = fstate_c[:x_samples]
        hy_c = fstate_c[x_samples:] if y is not None else None

        return fx_c, hy_c

    def fit_linear_decoder(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> torch.nn.Conv2d:
        device = next(self.parameters()).device
        # Get the training data to fit the linear decoder _____________________________________________
        hy_train = []
        zy_train = []

        for y, zy in train_dataloader:
            _, hy = self(y=y.to(device))  # shape: (n_samples, embedding_dim)
            hy_train.append(hy.detach().cpu())
            zy_train.append(zy.detach().cpu())
        hy_train = torch.cat(hy_train, dim=0)  # (B, r_y, H, W)
        zy_train = torch.cat(zy_train, dim=0)  # (B, z(y), H, W)

        # Center the target variable __________________________________________________________________
        # Compute mean over batch and spatial/temporal dimensions
        zy_mean = zy_train.mean(axis=[0] + list(range(2, zy_train.ndim)), keepdim=False)
        zy_train_c = zy_train - zy_mean  # (B, z(y), H, W)

        # Fit a linear map from h(y) to z_c(y) using least squares ____________________________________
        # TODO: Add regularization.
        # Flatten the tensors for linear regression
        hy_train_flat = hy_train.permute(0, 2, 3, 1).reshape(-1, hy_train.shape[1])  # (B*H*W, r_y)
        zy_train_c_flat = zy_train_c.permute(0, 2, 3, 1).reshape(-1, zy_train_c.shape[1])  # (B*H*W, z(y))
        out = torch.linalg.lstsq(hy_train_flat, zy_train_c_flat)
        Czyhy = out.solution.T  # (z(y), r_y)

        # Create the linear image decoder in the form of a Conv2D layer with a 1x1 kernel ____________
        linear_decoder = torch.nn.Conv2d(
            in_channels=hy_train.shape[1],
            out_channels=zy_train_c.shape[1],
            kernel_size=1,
            stride=1,
            bias=True,
        )
        linear_decoder.eval()
        linear_decoder.weight.data = Czyhy[..., None, None]  # 1x1 kernel is the linear map
        linear_decoder.bias.data = zy_mean  # Add mean as bias

        return linear_decoder.to(device)

    def conditional_expectation(self, x: torch.Tensor = None, hy2zy: torch.nn.Conv2d = None):
        """Compute the conditional expectation of the target variable z(y | x) given the present state x.
        Args:
            x (torch.Tensor): Input state tensor of shape (B, r_x, H, W).
            hy2zy (torch.nn.Conv2d): Linear map from h(y) to z(y). Computed using `fit_linear_decoder`.
        Returns:
            torch.Tensor: Conditional expectation of the target variable z(y | x) of shape (B, |z(y)|, H, W).
        """
        fx, _ = self(x=x, y=None)  # fx: (B, r_x, H, W)
        # Evolve state observations
        hy_cond_x = torch.einsum("bxhw,xy->byhw", fx, self.truncated_operator)

        if hy2zy is None:
            return hy_cond_x
        else:
            # Decode to the target variable
            zy_cond_x = hy2zy(hy_cond_x)
            return zy_cond_x  # (B, z(y | x), H, W) - conditional expectation of the target variable given x
