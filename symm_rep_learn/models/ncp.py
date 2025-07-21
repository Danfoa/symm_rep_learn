# Created by danfoa at 19/12/24
from __future__ import annotations

import logging

import torch

from symm_rep_learn.mysc.statistics import (
    cov_norm_squared_unbiased_estimation,
    cross_cov_norm_squared_unbiased_estimation,
)

log = logging.getLogger(__name__)


# Neural Conditional Probability (NCP) modelule ========================================================================
class NCP(torch.nn.Module):
    def __init__(
        self,
        embedding_x: torch.nn.Module,
        embedding_y: torch.nn.Module,
        embedding_dim: int,
        gamma=0.1,  # Will be multiplied by the embedding_dim
        gamma_centering=None,  # Penalizes probability mass distortion
        learnable_change_basis: bool = False,
    ):
        super(NCP, self).__init__()
        self.gamma = gamma
        self.gamma_centering = gamma_centering if gamma_centering is not None else 0.0
        self.embedding_dim = embedding_dim
        if learnable_change_basis:
            self.embedding_x = torch.nn.Sequential(embedding_x, torch.nn.Linear(embedding_dim, embedding_dim))
            self.embedding_y = torch.nn.Sequential(embedding_y, torch.nn.Linear(embedding_dim, embedding_dim))
        else:
            self.embedding_x = embedding_x
            self.embedding_y = embedding_y

        # Create parameters according to the chosen truncated operator bias
        embedding_dim = self.embedding_dim
        D_r = torch.eye(embedding_dim) + 1e-4 * torch.randn(embedding_dim, embedding_dim)
        self.Dr_params = torch.nn.Parameter(D_r, requires_grad=True)

        # Register buffers for the statistics of the embedding functions
        self._register_stats_buffers()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Forward pass of the NCP operator.

        Computes non-linear transformations of the input random variables x and y, and returns r-dimensional embeddings
        f(x) = [f_1(x), ..., f_r(x)] and h(y) = [h_1(y), ..., h_r(y)] representing the top r-singular functions of
        the conditional expectation operator such that E_p(y|x)[h_i(y)] = σ_i f_i(x) for i=1,...,r.

        Args:
            x: (torch.Tensor) of shape (..., d) representing the input random variable x.
            y: (torch.Tensor) of shape (..., d) representing the input random variable y.

        Returns:
            fx: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^2(X).
            hy: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^2(Y).
        """
        fx = self.embedding_x(x)  # f(x) = [f_1(x), ..., f_r(x)]
        hy = self.embedding_y(y)  # h(y) = [h_1(y), ..., h_r(y)]

        return fx, hy

    def pointwise_mutual_dependency(self, x: torch.Tensor, y: torch.Tensor):
        """Return the estimated pointwise mutual dependency between the random variables x and y.

        Args:
            x: (torch.Tensor) of shape (N, dx) representing the input random variable x.
            y: (torch.Tensor) of shape (N, dy) representing the input random variable y.

        Returns:
            k_r:  (torch.Tensor) of shape (N,) representing the approximated pointwise mutual dependency between x and
             y defined as PMD = p(x,y)/p(x)p(y) ≈ k_r(x,y).
        """
        fx, hy = self(x, y)
        fx_c, hy_c = fx - self.mean_fx, hy - self.mean_hy

        Dr = self.truncated_operator
        k_r = 1 + torch.einsum("...x,xy,...y->...", fx_c, Dr, hy_c)

        return k_r

    def pointwise_mutual_information(self, x: torch.Tensor, y: torch.Tensor):
        """Return the estimated pointwise mutual information between the random variables x and y.

        Args:
            x: (torch.Tensor) of shape (N, dx) representing the input random variable x.
            y: (torch.Tensor) of shape (N, dy) representing the input random variable y.

        Returns:
            pmi: (torch.Tensor) of shape (N,) representing the approximated  pointwise mutual information between x and
            y defined as PMI := ln(p(x,y) / p(x)p(y)) ≈ ln(k_r(x,y)).
        """
        k_r = self.pointwise_mutual_dependency(x, y)
        k_r_pos = torch.clamp(k_r, min=1e-6)  # Need to clamp to avoid NaNs.
        pmi = torch.log(k_r_pos)
        # Check no NaN  or Inf values
        assert torch.isfinite(pmi).all(), "NaN or Inf values found in the PMI estimation"
        return pmi

    def loss(self, fx: torch.Tensor, hy: torch.Tensor):
        """TODO.

        Args:
            fx: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^2(X)
            hy: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^2(Y)

        Returns:
            loss: L = -2 ||Cxy||_F^2 + tr(Cxy Cx Cxy^T Cy) - 1
                + γ(||Cx - I||_F^2 + ||Cy - I||_F^2 + ||E_p(x) f(x)||_F^2 + ||E_p(y) h(y)||_F^2)
            metrics: Scalar valued metrics to monitor during training.
        """
        assert fx.shape[-1] == hy.shape[-1] == self.embedding_dim, (
            f"Expected number of singular functions to be {self.embedding_dim}, got {fx.shape[-1]} and {hy.shape[-1]}."
        )

        # Center basis functions and update mean_fx, mean_hy, Cx, Cy, Cxy
        fx_c, hy_c = self.update_fns_statistics(fx, hy)

        # Orthonormal regularization and centering penalization _________________________________________
        # orthonormal_reg_fx = ||Cx - I||_F^2 + 2 ||E_p(x) f(x)||_F^2
        # orthonormal_reg_hy = ||Cy - I||_F^2 + 2 ||E_p(y) h(y)||_F^2
        orthonormal_reg_fx, orthonormal_reg_hy, metrics = self.orthonormality_penalization(fx_c, hy_c)

        # Operator truncation error = ||E - E_r||_HS^2 ____________________________________________________
        # E_r = Cxy -> ||E - E_r||_HS - ||E||_HS = -2 ||Cxy||_F^2 + tr(Cxy Cy Cxy^T Cx)
        truncation_err, loss_metrics = self.unbiased_truncation_error_matrix_form(fx_c, hy_c)

        metrics |= loss_metrics if loss_metrics is not None else {}
        # Total loss ____________________________________________________________________________________
        loss = (
            truncation_err
            + self.gamma * (orthonormal_reg_fx + orthonormal_reg_hy) / (2 * self.embedding_dim)
            +
            # Control barrier function on model constraint.
            self.gamma_centering * torch.exp((self.mean_fx**2 + self.mean_hy**2).sum())
        )
        # Logging metrics _______________________________________________________________________________
        with torch.no_grad():
            metrics |= {
                "||k(x,y) - k_r(x,y)||": truncation_err.detach().item(),
            }
        return loss, metrics

    def unbiased_truncation_error_matrix_form(self, fx_c, hy_c) -> tuple[torch.Tensor, dict]:
        """Implementation of ||E - E_r||_HS^2, while assuming E_r is a full matrix.

        Case 1: Orthogonal basis functions give:
            E_r = Cxy -> ||E - E_r||_HS - ||E||_HS = -2 ||Cxy||_F^2 + tr(Cxy Cy Cxy^T Cx) + 1
        Case 2: TODO: Trainable E_r matrix

        Args:
            fx_c: (torch.Tensor) of shape (n_samples, r) representing the centered singular functions of a subspace
            of L^2(X).
            hy_c: (torch.Tensor) of shape (n_samples, r) representing the centered singular functions of a subspace
            of L^2(Y).

        Returns:
            (torch.Tensor) representing the unbiased truncation error.
            (dict) Metrics to monitor during training.
        """
        metrics = {}
        n_samples = fx_c.shape[0]
        Dr = self.truncated_operator  # Dr = Dr.T
        # k_r(x,y) = 1 + f(x)^T Dr h(y)
        # truncated_err = -2 * E_p(x,y)[k_r(x,y)] + E_p(x)p(y)[k_r(x,y)^2]
        pmd_mat = 1 + torch.einsum("nx,xy,my->nm", fx_c, Dr, hy_c)  # (n_samples, n_samples)
        # E_p(x,y)[k_r(x,y)] = diag(pmd_mat).mean()
        E_pxy_kr = torch.diag(pmd_mat).mean()
        pmd_squared = pmd_mat**2
        # E_p(x)p(y)[k_r(x,y)^2]  # Note we remove the samples from the joint in the diagonal
        E_px_py_kr = (pmd_squared.sum() - pmd_squared.diag().sum()) / (n_samples * (n_samples - 1))
        truncation_err = (-2 * E_pxy_kr) + (E_px_py_kr) + 1

        with torch.no_grad():
            prob_mass_distortion = (1 - pmd_mat.mean()) ** 2  # Always 0 for batch due to centering.
            metrics |= {
                "E_p(x)p(y) k_r(x,y)^2": E_px_py_kr.detach() - 1,
                "E_p(x,y) k_r(x,y)": E_pxy_kr.detach() - 1,
                "Prob Mass Distortion": prob_mass_distortion,
            }

        return truncation_err, metrics

    def orthonormality_penalization(self, fx_c, hy_c, permutation=None):
        """Computes orthonormality and centering regularization penalization for a batch of feature vectors.

        Computes finite sample unbiased empirical estimates of the term:
        || Vx - I ||_F^2 = || Cx - I ||_F^2 + 2 || E_p(x) f(x) ||^2
                         = tr(Cx^2) - 2 tr(Cx) + r + 2 || E_p(x) f(x) ||^2
        Where Vx ∈ R^{r+1 x r+1} is the uncentered covariance matrix assuming the constant function is included as the
        first fn. Cx ∈ R^{r x r} is the centered covariance matrix of the feature vectors f_c(x) = f(x) - mean(f(x)).
        That is, [Vx]_ij = E_p(x) f_i(x) f_j(x) and [Cx]_ij = E_p(x) f_c,i(x) f_c,j(x), where f_c = f - E_p(x) f(x).

        The unbiased estimate requires to have two sample sets of marginal distribution p(x), such that
        D = {x_1, ..., x_n} and D' = {x'_1, ..., x'_n} are independent sampling sets from p(x). To achieve this in
        practice we shuffle D to obtain D' and to get f(x) and f(x').
        Then we have that the unbiased estimate is computed by:

        || Vx - I ||_F^2 ≈ E_(x,x')~p(x) [(f_c(x).T f_c(x'))^2] - 2 tr(Cx) + r + 2 ||f_mean||^2

        Args:
            fx_c: (n_samples, r) Centered feature vectors f_c(x) = [f_c,1(x), ..., f_c,r(x)].
            hy_c: (n_samples, r) Centered feature vectors h_c(y) = [h_c,1(y), ..., h_c,r(y)].
            return_inner_prod: (bool) If True, return intermediate inner products.
            permutation: (torch.Tensor) Permutation tensor to shuffle the samples in the batch.

        Returns:
            Regularization term as a scalar tensor.
        """
        # Compute unbiased empirical estimates ||Cx||_F^2 = E_(x,x')~p(x) [(f_c(x).T f_c(x'))^2]
        Cx_fro_2 = cov_norm_squared_unbiased_estimation(fx_c, False, permutation=permutation)
        tr_Cx = torch.trace(self.Cx)  # E[f_c(x)^T f_c(x)] =  tr(Cx)
        fx_centering_loss = (self.mean_fx**2).sum()  # ||E_p(x) (f(x_i))||^2
        embedding_dim_x = fx_c.shape[-1]

        Cy_fro_2 = cov_norm_squared_unbiased_estimation(hy_c, False, permutation=permutation)
        tr_Cy = torch.trace(self.Cy)  # E[h_c(y)^T h_c(y)] = tr(Cy)
        hy_centering_loss = (self.mean_hy**2).sum()  # ||E_p(y) (h(y_i))||^2
        embedding_dim_y = hy_c.shape[-1]

        # orthonormality_fx = tr(Cx^2) - 2 tr(Cx) # + 2 || E_p(x) f(x) ||^2 + r_x = |Fx|
        orthonormality_fx = Cx_fro_2 - 2 * tr_Cx + embedding_dim_x + 2 * fx_centering_loss
        # orthonormality_hy = tr(Cy^2) - 2 tr(Cy) + 2 || E_p(y) h(y) ||^2 + r_y = |Hy|
        orthonormality_hy = Cy_fro_2 - 2 * tr_Cy + embedding_dim_y + 2 * hy_centering_loss

        with torch.no_grad():
            # Divide by the embedding dimension to standardize metrics across experiments.
            metrics = {
                "tr(Cx)": (tr_Cx / embedding_dim_x).item(),
                "||mu_x||": torch.sqrt(fx_centering_loss).item(),
                "||Vx - I||_F^2": (orthonormality_fx / embedding_dim_x).item(),
                "tr(Cy)": (tr_Cy / embedding_dim_y).item(),
                "||mu_y||": torch.sqrt(hy_centering_loss).item(),
                "||Vy - I||_F^2": (orthonormality_hy / embedding_dim_y).item(),
            }

        return orthonormality_fx, orthonormality_hy, metrics

    def update_fns_statistics(self, fx: torch.Tensor, hy: torch.Tensor):
        """Update the statistics of the embedding functions.

        Computes the mean and covariance matrices of the embedding functions f(x) and h(y) for the batch of samples.

        Args:
            fx: (torch.Tensor) of shape (n_samples, r) representing the basis functions of a subspace of L^2(X).
            hy: (torch.Tensor) of shape (n_samples, r) representing the basis functions of a subspace of L^2(Y).

        Returns:
            fx_c: (torch.Tensor) centered basis functions
            hy_c: (torch.Tensor) centered basis functions
        """
        n_samples = fx.shape[0]

        eps = 1e-6 * torch.eye(self.embedding_dim, device=fx.device, dtype=fx.dtype)
        self.mean_fx = fx.mean(dim=0, keepdim=True)
        self.mean_hy = hy.mean(dim=0, keepdim=True)

        # Centering before (centered/un-centered) covariance estimation is key for numerical stability.
        fx_c, hy_c = fx - self.mean_fx, hy - self.mean_hy
        self.Cxy = torch.einsum("nr,nc->rc", fx_c, hy_c) / (n_samples - 1)
        self.Cx = torch.einsum("nr,nc->rc", fx_c, fx_c) / (n_samples - 1) + eps
        self.Cy = torch.einsum("nr,nc->rc", hy_c, hy_c) / (n_samples - 1) + eps

        return fx_c, hy_c

    @property
    def truncated_operator(self):
        Dr_symm = self.Dr_params @ self.Dr_params.T  # Ensure its symmetric
        eigval_max = torch.linalg.eigvalsh(Dr_symm)[-1]
        Dr = Dr_symm / eigval_max
        return Dr

    def _register_stats_buffers(self):
        # Matrix containing the cross-covariance matrix form of the operator Cyx: L^2(Y) -> L^2(X)
        self.register_buffer("Cxy", torch.zeros(self.embedding_dim, self.embedding_dim))
        # Matrix containing the covariance matrix form of the operator Cx: L^2(X) -> L^2(X)
        self.register_buffer("Cx", torch.zeros(self.embedding_dim, self.embedding_dim))
        # Matrix containing the covariance matrix form of the operator Cy: L^2(Y) -> L^2(Y)
        self.register_buffer("Cy", torch.zeros(self.embedding_dim, self.embedding_dim))
        # Expectation of the embedding functions
        self.register_buffer("mean_fx", torch.zeros((1, self.embedding_dim)))
        self.register_buffer("mean_hy", torch.zeros((1, self.embedding_dim)))


if __name__ == "__main__":
    from symm_rep_learn.nn.layers import MLP

    in_dim, out_dim, embedding_dim = 10, 4, 40

    fx = MLP(
        input_shape=in_dim,
        output_shape=embedding_dim,
        n_hidden=3,
        layer_size=64,
        activation=torch.nn.GELU,
    )
    hy = MLP(
        input_shape=out_dim,
        output_shape=embedding_dim,
        n_hidden=3,
        layer_size=64,
        activation=torch.nn.GELU,
    )
    ncp = NCP(fx, hy, embedding_dim=embedding_dim)

    x = torch.randn(10, in_dim)
    y = torch.randn(10, out_dim)

    fx, hy = ncp(x, y)

    loss, metrics = ncp.loss(fx, hy)
    print(loss)
    with torch.no_grad():
        pmd = ncp.pointwise_mutual_dependency(x, y)
    print(pmd)
