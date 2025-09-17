# Created by danfoa at 19/12/24
from __future__ import annotations

import logging
import math

import torch

from symm_rep_learn.models.neural_conditional_probability.ncp import NCP
from symm_rep_learn.nn.layers import ResidualEncoder
from symm_rep_learn.nn.losses import contrastive_low_rank_loss

log = logging.getLogger(__name__)

import torch.nn.utils.parametrizations as param_zoo  # convenience alias
import torch.nn.utils.parametrize as P


class Symmetric(torch.nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class EvolOp1D(NCP):
    """Evolution operator for 1D/flat state spaces."""

    def __init__(
        self,
        embedding_state: torch.nn.Module,
        state_embedding_dim: int,
        state_dim: int = None,
        self_adjoint: bool = False,
        **ncp_kwargs,
    ):
        super(EvolOp1D, self).__init__(
            embedding_x=embedding_state,
            embedding_y=embedding_state,
            embedding_dim_x=state_embedding_dim,
            embedding_dim_y=state_embedding_dim,
            **ncp_kwargs,
        )

        self.self_adjoint = self_adjoint

        if self.self_adjoint:  # Truncated operator is self-adjoint
            # Deregister spectral norm
            P.remove_parametrizations(self.Dr, "weight", leave_parametrized=True)
            # First symmetrize the operator
            P.register_parametrization(self.Dr, "weight", Symmetric())
            # Then apply spectral norm
            param_zoo.spectral_norm(self.Dr, name="weight")

        if not isinstance(self._embedding_x, ResidualEncoder):
            assert state_dim is not None, "state_dim must be provided for non-residual encoders."
            self._trainable_lin_dec = torch.nn.Linear(in_features=self.dim_fx, out_features=state_dim)

    def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
        """Compute the time-delayed state embeddings.

        Given the state x and next state y, this function computes the time-delayed state embeddings
        f(x) and h(y) using a unique non-linear encoder function, and the truncated conditional expectation operator
        such that:

        f(x) = \phi(x)

        h(y) = Dr @ \phi(y)

        """

        if self.training:
            assert x is not None and y is not None, "Both x and y must be provided during training."
            x_samples = x.shape[0]
            state_traj = torch.cat([x, y], dim=0)
            fs = self._embedding_x(state_traj)  # f(x) = [f_1(x), ..., f_r(x)]
            fx, hy = fs[:x_samples], fs[x_samples:]
            self.ema_stats(fx, hy)  # Update mean and covariance statistics
            fs_mean = (self.ema_stats.mean_x + self.ema_stats.mean_y) / 2
            fx_c = fx - fs_mean
            hy_c = torch.nn.functional.linear(hy - fs_mean, self.truncated_operator)
        else:
            fs_mean = (self.ema_stats.mean_x + self.ema_stats.mean_y) / 2

            if x is None and y is not None:
                fs = self._embedding_x(y)
                fx_c = None
                hy_c = torch.nn.functional.linear(fs - fs_mean, self.truncated_operator)
            elif y is None and x is not None:
                fs = self._embedding_x(x)
                fx_c = fs - fs_mean
                hy_c = None
            else:
                state_traj = torch.cat([x, y], dim=0)
                fs = self._embedding_x(state_traj)
                x_samples = x.shape[0]
                fx_c = fs[:x_samples] - fs_mean
                hy_c = torch.nn.functional.linear(fs[x_samples:] - fs_mean, self.truncated_operator)

        return fx_c, hy_c

    @torch.no_grad()
    def evolution_operator(self, reg=1e-5) -> torch.Tensor:
        """Returns the evolution operator defined as Cov(f(x))^-1 Cov(f(x), f(x'))"""
        regularizer = reg * torch.eye(self.dim_fx, out=torch.empty_like(self.ema_stats.cov_xx))
        reg_cov = regularizer + self.ema_stats.cov_xx
        evolution_operator = torch.linalg.solve(reg_cov, self.ema_stats.cov_xy)
        return evolution_operator

    def regression_loss(self, fx_c: torch.Tensor, hy_c: torch.Tensor, x: torch.Tensor, y: torch.Tensor = None):
        """Compute regression loss for 1D evolution operator.

        Args:
            fx_c: (torch.Tensor) of shape (batch_size, r_x) *centered* embedding functions of a subspace of L^2(X)
            hy_c: (torch.Tensor) of shape (batch_size, r_y) *centered* embedding functions of a subspace of L^2(Y)
            x: (torch.Tensor) of shape (batch_size, state_dim) input state
            y: (torch.Tensor) of shape (batch_size, state_dim) target state (optional for residual encoders)

        Returns:
            MSE loss: || hy_c - Dr @ fx_c ||_F^2
        """
        hy_c_pred = self.evolve_latent_state(fx_c)

        if isinstance(self._embedding_x, ResidualEncoder):
            # Compute the MSE loss
            residual_dims = self._embedding_x.residual_dims
            mse_loss = torch.nn.functional.mse_loss(
                input=hy_c_pred[:, residual_dims], target=hy_c[:, residual_dims], reduction="mean"
            )
        else:
            assert y is not None, "y must be provided when using a non-residual encoder."
            y_pred = self._trainable_lin_dec(hy_c_pred)
            mse_loss = torch.nn.functional.mse_loss(input=y_pred, target=y, reduction="mean")

        orthonormal_reg_x, orthonormal_reg_y, metrics = self.orthonormality_regularization(fx_c, hy_c)

        loss = mse_loss + self.gamma * (orthonormal_reg_x / self.dim_fx + orthonormal_reg_y / self.dim_hy)

        metrics["mse_loss"] = mse_loss.item()
        with torch.no_grad():
            clora_err, loss_metrics = contrastive_low_rank_loss(fx_c, hy_c, self.truncated_operator)
            metrics |= loss_metrics
            metrics["||k(x,y) - k_r(x,y)||"] = clora_err.mean().detach().item()

        return loss, metrics


class EvolOp2D(NCP):
    """Evolution operator for 2D/images state spaces."""

    def __init__(
        self,
        embedding_state: torch.nn.Module,
        state_embedding_dim: int,
        self_adjoint: bool = False,
        **ncp_kwargs,
    ):
        super(EvolOp2D, self).__init__(
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

        if not isinstance(self._embedding_x, ResidualEncoder):
            self._trainable_lin_dec = torch.nn.Conv2d(in_channels=self.dim_fx, out_channels=1, kernel_size=1)

    def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
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
        fstate = self._embedding_x(state_traj)
        fstate_flat = fstate.permute(0, 2, 3, 1).reshape(-1, self.dim_fx)  # Flatten the state embeddings
        _ = self.data_norm_x(fstate_flat)  # Center the embeddings
        mean = self.data_norm_x.mean  # Get the mean of the embeddings
        fstate_c = fstate - mean[..., None, None]  # Center the embeddings

        # Separate past and next image embeddings
        fx_c = fstate_c[:x_samples]
        hy_c = fstate_c[x_samples:] if y is not None else None

        return fx_c, hy_c

    # def loss(self, fx_c: torch.Tensor, hy_c: torch.Tensor, *args, **kwargs):
    #     fx_c_flat = fx_c.permute(0, 2, 3, 1).reshape(-1, self.dim_fx)
    #     hy_c_flat = hy_c.permute(0, 2, 3, 1).reshape(-1, self.dim_hy)
    #     return super().loss(fx_c_flat, hy_c_flat, *args, **kwargs)

    def fit_linear_decoder(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        ridge_reg: float = 1e-3,
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

        # Fit a linear map from h(y) to z_c(y) using least squares with ridge regularization _________
        hy_flat = hy_train.permute(0, 2, 3, 1).reshape(-1, self.dim_hy)  # X, (N, r_y = |h(y)|)
        zy_c_flat = zy_train_c.permute(0, 2, 3, 1).reshape(-1, zy_train_c.shape[1])  # Y, (N, |z(y)|)

        if ridge_reg > 0.0:
            dim_hy = hy_flat.shape[1]
            dim_zy = zy_c_flat.shape[1]

            eye = torch.eye(dim_hy, device=hy_flat.device, dtype=hy_flat.dtype)
            X_aug = torch.cat([hy_flat, math.sqrt(ridge_reg) * eye], dim=0)  # (N+d, d)
            Y_aug = torch.cat(
                [zy_c_flat, torch.zeros(dim_hy, dim_zy, device=hy_flat.device, dtype=hy_flat.dtype)], dim=0
            )
            Czyhy = torch.linalg.lstsq(X_aug, Y_aug).solution.T  # (|z(y)|, r_y)
        else:
            Czyhy = torch.linalg.lstsq(hy_flat, zy_c_flat).solution.T  # (|z(y)|, r_y)

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
            x (torch.Tensor): Input state tensor of shape (B, |X|, H, W).
            hy2zy (torch.nn.Conv2d): Linear map from h(y) to z(y). Computed using `fit_linear_decoder`.
        Returns:
            torch.Tensor: Conditional expectation of the target variable z(y | x) of shape (B, |z(y)|, H, W).
        """
        fx_c, _ = self(x=x, y=None)  # fx: (B, r_x, H, W)
        # Evolve state observations
        hy_cond_x = self.evolve_latent_state(fx_c)  # (B, r_y, H, W)

        if hy2zy is None:
            return hy_cond_x
        else:
            # Decode to the target variable
            zy_cond_x = hy2zy(hy_cond_x)
            return zy_cond_x  # (B, z(y | x), H, W) - conditional expectation of the target variable given x

    def evolve_latent_state(self, fx_c: torch.Tensor):
        """Evolve the latent state fx_c using the truncated operator Dr.

        Args:
            fx_c (torch.Tensor): Centered latent state of shape (B, r_x, H, W).

        Returns:
            torch.Tensor: Evolved latent state of shape (B, r_y, H, W).
        """
        Dr = self.truncated_operator
        hy_c = torch.nn.functional.conv2d(
            input=fx_c,
            weight=(Dr.T)[..., None, None],  # (r_y, r_x, 1, 1),
            bias=None,
            stride=1,
        )
        return hy_c

    def regression_loss(self, fx_c: torch.Tensor, hy_c: torch.Tensor, x: torch.Tensor, y: torch.Tensor = None):
        """TODO.

        Args:
            fx_c: (torch.Tensor) of shape (batch_size,r_x, H, W) *centered* embedding functions of a subspace of L^2(X)
            hy_c: (torch.Tensor) of shape (batch_size,r_y, H, W) *centered* embedding functions of a subspace of L^2(Y)

        Returns:
            MSE loss: || hy_c - Dr @ fx_c ||_F^2
        """
        hy_c_pred = self.evolve_latent_state(fx_c)

        if isinstance(self._embedding_x, ResidualEncoder):
            # Compute the MSE loss
            residual_dims = self._embedding_x.residual_dims
            mse_loss = torch.nn.functional.mse_loss(
                input=hy_c_pred[:, residual_dims, :, :], target=hy_c[:, residual_dims, :, :], reduction="mean"
            )
        else:
            assert y is not None, "y must be provided when using a non-residual encoder."
            y_pred = self._trainable_lin_dec(hy_c_pred)
            mse_loss = torch.nn.functional.mse_loss(input=y_pred, target=y, reduction="mean")

        fx_c_flat = fx_c.permute(0, 2, 3, 1).reshape(-1, self.dim_fx)  # Flatten the state embeddings
        hy_c_flat = hy_c.permute(0, 2, 3, 1).reshape(-1, self.dim_hy)
        orthonormal_reg_x, orthonormal_reg_y, metrics = self.orthonormality_regularization(fx_c_flat, hy_c_flat)

        loss = mse_loss + self.gamma * (orthonormal_reg_x / self.dim_fx + orthonormal_reg_y / self.dim_hy)

        metrics["mse_loss"] = mse_loss.item()
        with torch.no_grad():
            clora_err, loss_metrics = contrastive_low_rank_loss(fx_c, hy_c, self.truncated_operator)
            metrics |= loss_metrics
            metrics["||k(x,y) - k_r(x,y)||"] = clora_err.mean().detach().item()

        return loss, metrics
