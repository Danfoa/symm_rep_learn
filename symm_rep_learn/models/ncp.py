# Created by danfoa at 19/12/24
from __future__ import annotations

import logging

import torch
from symm_learning.nn import DataNorm

from symm_rep_learn.nn import contrastive_low_rank_loss, orthonormality_regularization

log = logging.getLogger(__name__)


# Neural Conditional Probability (NCP) modelule ========================================================================
class NCP(torch.nn.Module):
    def __init__(
        self,
        embedding_x: torch.nn.Module,
        embedding_y: torch.nn.Module,
        embedding_dim_x: int,
        embedding_dim_y: int,
        orth_reg=0.1,  # Will be multiplied by the embedding_dim
        centering_reg=0.01,  # Penalizes probability mass distortion
        momentum=0.9,  # 1.0 Batch stats to center the embeddings and compute covariance.
    ):
        super(NCP, self).__init__()

        assert orth_reg > 0 or centering_reg > 0, (
            "If you desire to train NCP without orthonormal regularization, set `centering_reg` to a positive value, "
            "since this is a constraint of the optimization problem penalized via a lagrangian multiplier."
        )

        self.gamma = orth_reg
        self.gamma_centering = centering_reg

        self.dim_fx = embedding_dim_x
        self.dim_hy = embedding_dim_y

        self._embedding_x = embedding_x
        self._embedding_y = embedding_y

        # Initialize the parameters of the low-rank approximation of the operator
        self.Dr = torch.nn.Linear(in_features=self.dim_hy, out_features=self.dim_fx, bias=False)
        # Initialization close to identity. Full rank operator.
        with torch.no_grad():
            self.Dr.weight.data = torch.eye(self.dim_fx, self.dim_hy) + 1e-4 * torch.randn(self.dim_fx, self.dim_hy)
        # Ensure the truncated operator has spectral norm maximum of 1
        torch.nn.utils.parametrizations.spectral_norm(module=self.Dr, name="weight")

        # Layers that center the embedding functions and keep track of mean and covariance. Variance is penalized in orthonormality
        self.data_norm_x = DataNorm(embedding_dim_x, momentum=momentum, compute_cov=True, only_centering=True)
        self.data_norm_y = DataNorm(embedding_dim_y, momentum=momentum, compute_cov=True, only_centering=True)

    def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
        """Forward pass of the NCP operator.

        Computes non-linear transformations of the input random variables x and y, and returns r-dimensional embeddings
        f(x) = [f_1(x), ..., f_r(x)] and h(y) = [h_1(y), ..., h_r(y)] representing the top r-singular functions of
        the conditional expectation operator such that E_p(y|x)[h_i(y)] = σ_i f_i(x) for i=1,...,r.

        Args:
            x: (torch.Tensor) of shape (..., d) representing the input random variable x.
            y: (torch.Tensor) of shape (..., d) representing the input random variable y.

        Returns:
            fx_c: (torch.Tensor) of shape (..., r) *centered* basis functions of subspace of L^2(X) .
            hy_c: (torch.Tensor) of shape (..., r) *centered* basis functions of subspace of L^2(Y) .
        """
        assert x is not None or y is not None, "At least one of x or y must be provided."

        if x is not None:
            fx = self._embedding_x(x)  # f(x) = [f_1(x), ..., f_r(x)]
            fx_c = self.data_norm_x(fx)  # f_c(x) = f(x) - E_p(x)[f(x)]
        else:
            fx_c = None

        if y is not None:
            hy = self._embedding_y(y)  # h(y) = [h_1(y), ..., h_r(y)]
            hy_c = self.data_norm_y(hy)  # h_c(y) = h(y) - E_p(y)[h(y)]
        else:
            hy_c = None

        return fx_c, hy_c

    def pointwise_mutual_dependency(self, x: torch.Tensor, y: torch.Tensor):
        """Return the estimated pointwise mutual dependency between the random variables x and y.

        Args:
            x: (torch.Tensor) of shape (N, dx, ...) representing the input random variable x. Where `...` denote spatial/temporal dimensions if any.
            y: (torch.Tensor) of shape (N, dy, ...) representing the input random variable y. Where `...` denote the same spatial/temporal dimensions as x.

        Returns:
            k_r:  (torch.Tensor) of shape (N, ...) representing the approximated pointwise mutual dependency between x and
             y defined as PMD = p(x,y)/p(x)p(y) ≈ k_r(x,y).
        """
        assert x.ndim >= 2, "x must be at least a 2D tensor."
        assert y.ndim >= 2, "y must be at least a 2D tensor."

        fx_c, hy_c = self(x, y)

        Dr = self.truncated_operator
        k_r = 1 + torch.einsum("bx...,xy,by...->b...", fx_c, Dr, hy_c)

        return k_r

    @property
    def truncated_operator(self):
        """Return the truncated operator"""
        return self.Dr.weight  # Triggers the spectral normalization.

    def orthonormality_regularization(self, fx_c: torch.Tensor, hy_c: torch.Tensor):
        Cfx, Chy = self.data_norm_x.cov, self.data_norm_y.cov
        fx_mean, hy_mean = self.data_norm_x.mean, self.data_norm_y.mean
        # orthonormal_reg_fx = ||Cx - I||_F^2 + 2 ||E_p(x) f(x)||_F^2
        orthonormal_reg_x, metrics_x = orthonormality_regularization(x=fx_c, Cx=Cfx, x_mean=fx_mean, var_name="x")
        # orthonormal_reg_hy = ||Cy - I||_F^2 + 2 ||E_p(y) h(y)||_F^2
        orthonormal_reg_y, metrics_y = orthonormality_regularization(x=hy_c, Cx=Chy, x_mean=hy_mean, var_name="y")

        metrics = metrics_x | metrics_y  # Combine metrics from both regularizations

        return orthonormal_reg_x, orthonormal_reg_y, metrics

    def loss(self, fx_c: torch.Tensor, hy_c: torch.Tensor):
        """TODO.

        Args:
            fx_c: (torch.Tensor) of shape (batch_size,r_x, ...) *centered* embedding functions of a subspace of L^2(X)
            hy_c: (torch.Tensor) of shape (batch_size,r_y, ...) *centered* embedding functions of a subspace of L^2(Y)

        Returns:
            loss: L = -2 ||Cxy||_F^2 + tr(Cxy Cx Cxy^T Cy) - 1
                + γ(||Cx - I||_F^2 + ||Cy - I||_F^2 + ||E_p(x) f(x)||_F^2 + ||E_p(y) h(y)||_F^2)
            metrics: Scalar valued metrics to monitor during training.
        """
        assert fx_c.shape[1] == self.dim_fx, f"Expected fx (batch_size, {self.dim_fx},...), got {fx_c.shape}"
        assert hy_c.shape[1] == self.dim_hy, f"Expected hy (batch_size, {self.dim_hy},...), got {hy_c.shape}"
        dx = self.dim_fx
        dy = self.dim_hy
        # Orthonormal regularization _________________________________________
        orthonormal_reg_x, orthonormal_reg_y, metrics = self.orthonormality_regularization(fx_c, hy_c)
        # Centering penalization __________________________________________________
        # Lagrange multiplier term for centering constraint of basis functions
        cent_reg_x = self.data_norm_x.mean.pow(2).sum()  #  ||E_x f(x)||_F^2
        cent_reg_y = self.data_norm_y.mean.pow(2).sum()  #  ||E_y h(y)||_F^2
        cent_reg = cent_reg_x / dx + cent_reg_y / dy
        # Operator truncation error = ||E - E_r||_HS^2 ____________________________________________________
        # E_r = Cxy -> ||E - E_r||_HS - ||E||_HS = -2 ||Cxy||_F^2 + tr(Cxy Cy Cxy^T Cx)
        Dr = self.truncated_operator
        clora_err, loss_metrics = contrastive_low_rank_loss(fx_c, hy_c, Dr)

        metrics |= loss_metrics
        # Total loss ____________________________________________________________________________________
        loss = (
            clora_err + self.gamma * (orthonormal_reg_x / dx + orthonormal_reg_y / dy) + self.gamma_centering * cent_reg
        )
        # Logging metrics _______________________________________________________________________________
        with torch.no_grad():
            metrics |= {
                "||k(x,y) - k_r(x,y)||": clora_err.detach().item(),
            }
        return loss, metrics


if __name__ == "__main__":
    from symm_rep_learn.nn.layers import MLP

    in_dim, out_dim, embedding_dim_x, embedding_dim_y = 10, 4, 40, 50

    fx = MLP(
        input_shape=in_dim,
        output_shape=embedding_dim_x,
        n_hidden=3,
        layer_size=64,
        activation=torch.nn.GELU,
    )
    hy = MLP(
        input_shape=out_dim,
        output_shape=embedding_dim_y,
        n_hidden=3,
        layer_size=64,
        activation=torch.nn.GELU,
    )
    ncp = NCP(fx, hy, embedding_dim_x=embedding_dim_x, embedding_dim_y=embedding_dim_y, momentum=1.0)
    ncp.train()

    for _ in range(10):
        x = torch.randn(10, in_dim)
        y = torch.randn(10, out_dim)

        f, h = ncp(x, y)

        # print(f.shape, h.shape)

        loss, metrics = ncp.loss(f, h)
        # print(loss.item())

        with torch.no_grad():
            pmd = ncp.pointwise_mutual_dependency(x, y)
