# Created by danfoa at 19/12/24
from __future__ import annotations
import torch

import logging

from torch.linalg import matrix_norm
from wandb.sdk.internal.profiler import torch_trace_handler

log = logging.getLogger(__name__)


# Neural Conditional Probability (NCP) model.
class NCP(torch.nn.Module):

    def __init__(self, x_fns: torch.nn.Module, y_fns: torch.nn.Module, embedding_dim=None, gamma=0.01):
        super(NCP, self).__init__()
        self.gamma = gamma
        self.embedding_dim = embedding_dim
        self.singular_fns_x = x_fns
        self.singular_fns_y = y_fns
        # NCP does not need to have trainable svals.
        self.svals = torch.nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)
        self._svals_estimated = False
        # Matrix containing the cross-covariance matrix form of the operator Cyx: L^2(Y) -> L^2(X)
        self.register_buffer('Cxy', torch.zeros(embedding_dim, embedding_dim))
        # Matrix containing the covariance matrix form of the operator Cx: L^2(X) -> L^2(X)
        self.register_buffer('Cx', torch.zeros(embedding_dim, embedding_dim))
        # Matrix containing the covariance matrix form of the operator Cy: L^2(Y) -> L^2(Y)
        self.register_buffer('Cy', torch.zeros(embedding_dim, embedding_dim))
        # Expectation of the embedding functions
        self.register_buffer('mean_fx', torch.zeros((1, embedding_dim)))
        self.register_buffer('mean_hy', torch.zeros((1, embedding_dim)))
        # Use multiple batch information to keep bettwe estimates of expectations
        self._running_stats = False  # TODO: Implement running mean

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
        fx = self.singular_fns_x(x)  # f(x) = [f_1(x), ..., f_r(x)]
        hy = self.singular_fns_y(y)  # h(y) = [h_1(y), ..., h_r(y)]

        # TODO: After svals are identified applied change of basis to the singular functions.
        pass

        return fx, hy

    def loss(self, fx: torch.Tensor, hy: torch.Tensor):
        """ TODO
        Args:
            fx: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^2(X)
            hy: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^2(Y)
        Returns:
            loss: L = -2 ||Cxy||_F^2 + tr(Cxy Cx Cxy^T Cy) - 1
                + γ(||Cx - I||_F^2 + ||Cy - I||_F^2 + ||E_p(x) f(x)||_F^2 + ||E_p(y) h(y)||_F^2)
            metrics: Scalar valued metrics to monitor during training.
        """
        assert fx.shape[-1] == hy.shape[-1] == self.embedding_dim, \
            f"Expected number of singular functions to be {self.embedding_dim}, got {fx.shape[-1]} and {hy.shape[-1]}."

        self.update_fns_statistics(fx, hy)  # Update mean_fx and mean_hy
        # Main loss
        # truncation_err = -2 ||Cxy||_F^2 + tr(Cxy Cy Cxy^T Cx) - 1
        Cxy_f_norm = matrix_norm(self.Cxy, ord='fro') ** 2
        A = torch.einsum('ab,bc,cd,de->ae', self.Cxy, self.Cy, self.Cxy, self.Cx)
        truncation_err = -2 * Cxy_f_norm + torch.trace(A)
        # truncation_err = self.truncation_error(fx, hy)

        # Regularization terms, encouraging orthonormality and centered basis functions
        # orth_reg = ||Cx - I||_F^2 + ||Cy - I||_F^2
        # I = torch.eye(self.embedding_dim, device=fx.device, dtype=fx.dtype)
        # orth_reg = matrix_norm(self.Cx - I, ord='fro') ** 2 + matrix_norm(self.Cy - I, ord='fro') ** 2
        # center_reg = ||mean_fx := E_p(x) f(x)||_F^2 + ||mean_hy := E_p(y) h(y)||_F^2
        mean_fx_norm, mean_hy_norm = torch.linalg.norm(self.mean_fx), torch.linalg.norm(self.mean_hy)
        # center_reg = mean_fx_norm ** 2 + mean_hy_norm ** 2
        # reg = orth_reg + 2 * center_reg  # Numerically unstability appears here
        reg = self.orthonormality_penalization(fx, self.mean_fx) + self.orthonormality_penalization(hy, self.mean_hy)
        # Total loss ___________________________________________________________
        # loss = truncation_err + self.gamma * (orth_reg + center_reg)
        loss = truncation_err + (self.embedding_dim * self.gamma) * reg
        # Logging metrics ______________________________________________________
        metrics = {
            "||E - E_r||_HS": truncation_err.detach(),
            # "orth_reg":       orth_reg.detach(),
            "||Cxy||_F^2":    Cxy_f_norm.detach() / self.embedding_dim,
            "||mu_x||":       mean_fx_norm.detach(),
            "||mu_y||":       mean_hy_norm.detach(),
            }
        metrics |= self.batch_metrics()
        return loss, metrics

    def truncation_error(self, fx, hy):
        eps = 1e-6 * torch.eye(self.embedding_dim, device=fx.device, dtype=fx.dtype)
        n_samples = fx.shape[0]
        fx_c = fx - fx.mean(dim=0, keepdim=True)
        hy_c = hy - hy.mean(dim=0, keepdim=True)

        Cxy = torch.einsum('nr,nc->rc', fx_c, hy_c) / (n_samples - 1)
        Cx = torch.einsum('nr,nc->rc', fx_c, fx_c) / (n_samples - 1) + eps
        Cy = torch.einsum('nr,nc->rc', hy_c, hy_c) / (n_samples - 1) + eps

        A = torch.einsum('ab,bc,cd,de->ae', Cxy, Cy, Cxy, Cx)

        return - 2 * torch.sum(Cxy ** 2) + torch.trace(A)

    # def reg_term(self, fx, hy):
    #     fx_p = fx[torch.randperm(fx.shape[0])]
    #     hy_p = hy[torch.randperm(hy.shape[0])]
    #     t1_u = torch.mean((torch.einsum('ik,jk->ij', fx, fx_p) + 1) ** 2)
    #     t1_v = torch.mean((torch.einsum('ik,jk->ij', hy, hy_p) + 1) ** 2)
    #     t2_u = - 2 * torch.einsum('ik,ik->', fx, fx) / fx.shape[0]
    #     t2_v = - 2 * torch.einsum('ik,ik->', hy, hy) / hy.shape[0]
    #     return t1_u + t1_v + t2_u + t2_v + (fx.shape[1] + hy.shape[1] - 2)

    @staticmethod
    def orthonormality_penalization(fx, fx_mean=None):
        """ Computes orthonormality and centering regularization penalization for a batch of feature vectors.

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

        || Vx - I ||_F^2 ≈ E_(x,x')~p(x) [(f_c(x).T f_c(x'))^2] - 2 E_p(x)[f_c(x).T f_c(x)] + r + 2 E_p(x) f(x)^T f(x)
                         ≈ 1/N^2 Σ_i,j=1^N (f_c(x_i).T f_c(x_j))^2 -
                         2/N Σ_i=1^N (f_c(x_i).T f_c(x_i)) +
                         r +
                         2 ||1/N Σ_i=1^N (f(x_i))||^2
        Args:
            fx: (n_samples, r) Feature vectors f(x) = [f_1(x), ..., f_r(x)].
            fx_mean: (r,) Mean of the feature vectors E_p(x) f(x).

        Returns:
            Regularization term as a scalar tensor.
        """
        n_samples = fx.shape[0]
        # Compute centered vectors if not provided
        fx_mean = fx_mean if fx_mean is not None else fx.mean(dim=0, keepdim=True)
        fx_c = fx - fx_mean     #  f_c = f - E_p(x) f(x) = [f_c,1(x), ..., f_c,r(x)]
        fx_c_p = fx_c[torch.randperm(n_samples)]       # = [f_c,1(x'), ..., f_c,r(x')]

        # Precompute inner products for centered and uncentered vectors
        inner_prod_x_xp = torch.mm(fx_c, fx_c_p.T)             # (n_samples, n_samples)  Symmetric matrix.
        # diag_vec = torch.diag(inner_prod_x_xp)               # 1. Get the diagonal as a flattened 1D tensor
        # # 2. Get the strictly upper-triangular elements (excluding diagonal) as a flattened 1D tensor
        # row_idx, col_idx = torch.triu_indices(n_samples, n_samples, offset=1)
        # upper_diag_flat = inner_prod_x_xp[row_idx, col_idx]
        # entries = torch.concatenate((diag_vec**2, 2*upper_diag_flat**2))
        # term1 = ((diag_vec**2).sum() + (2*upper_diag_flat**2).sum()) / (n_samples ** 2)

        # Compute unbiased empirical estimates
        term1 = (inner_prod_x_xp ** 2).mean()                           # E_(x,x')~p(x) [(f_c(x)^T f_c(x'))^2]
        term2 = -2 * torch.sum(fx_c * fx_c, dim=1).mean()              # -2 E[f_c(x)^T f_c(x)]
        cst = fx.shape[-1]                                             # Dimensionality r
        centering_loss = 2 * (fx_mean ** 2).sum()                       # 2 ||E_p(x) (f(x_i))||^2

        # Combine terms
        regularization = term1 + term2 + cst + centering_loss
        return regularization


    def mutual_information(self, x: torch.Tensor, y: torch.Tensor):
        """Compute the exponential of the mutual information between the random variables x and y.

        The conditional expectation's kernel function k(x,y) = p(x,y) / p(x)p(y), is by definition the exponential of
        the mutual information between two evaluations of the random variables x and y, that is: MI = ln(k(x,y)).

        In the chosen basis sets of the approximated function spaces L^2(X) and L^2(Y), the approximated kernel function
        is computed as: k_r(x,y) = 1 + Σ_i,j=1^r Cxy_ij f_i(x) h_j(y).

        TODO: When the singular basis are appropriatedly identified after training we can compute the kernel function by
        k_r(x,y) = 1 + Σ_i=1^r σ_i f_i(x) h_i(y).

        Args:
            x: (torch.Tensor) of shape (..., d) representing the input random variable x.
            y: (torch.Tensor) of shape (..., d) representing the input random variable y.
        Returns:
            (torch.Tensor) representing the expected mutual information between x and y.
        """
        fx, hy = self(x, y)
        fx_centered, hy_centered = fx - self.mean_fx, hy - self.mean_hy
        # Compute the kernel function
        kr = 1 + torch.einsum('nr,rc,nc->n', fx_centered, self.Cxy, hy_centered)
        return torch.log(kr)

    @torch.no_grad()
    def batch_metrics(self):
        return {
            "||Cx||_F": torch.linalg.norm(self.Cx, ord='fro') ** 2 / self.embedding_dim,  # Should converge to 1
            "||Cy||_F": torch.linalg.norm(self.Cy, ord='fro') ** 2 / self.embedding_dim,  # Should converge to 1
            }

    def update_fns_statistics(self, fx: torch.Tensor, hy: torch.Tensor):
        n_samples = fx.shape[0]

        if self._running_stats:
            raise NotImplementedError("Running mean is not implemented yet.")
        else:
            eps = 1e-6 * torch.eye(self.embedding_dim, device=fx.device, dtype=fx.dtype)
            self.mean_fx = fx.mean(dim=0, keepdim=True)
            self.mean_hy = hy.mean(dim=0, keepdim=True)

            fx_c, hy_c = fx - self.mean_fx, hy - self.mean_hy
            # Center covariances
            self.Cxy = torch.einsum('nr,nc->rc', fx_c, hy_c) / (n_samples - 1)
            self.Cx = torch.einsum('nr,nc->rc', fx_c, fx_c) / (n_samples - 1) + eps
            self.Cy = torch.einsum('nr,nc->rc', hy_c, hy_c) / (n_samples - 1) + eps
            # Center covariances without unnecessary subtraction on the batch data
            # Cxy_mean = self.mean_fx.T @ self.mean_hy * n_samples
            # self.Cxy = (torch.einsum('nr,nc->rc', fx, hy) - Cxy_mean) / (n_samples - 1)
            # Cx_mean = self.mean_fx.T @ self.mean_fx * n_samples
            # self.Cx = (torch.einsum('nr,nc->rc', fx, fx) - Cx_mean) / (n_samples - 1) + eps
            # Cy_mean = self.mean_hy.T @ self.mean_hy * n_samples
            # self.Cy = (torch.einsum('nr,nc->rc', hy, hy) - Cy_mean) / (n_samples - 1) + eps


if __name__ == "__main__":
    from NCP.nn.layers import MLP

    in_dim, out_dim, embedding_dim = 10, 4, 40

    fx = MLP(input_shape=in_dim, output_shape=embedding_dim, n_hidden=3, layer_size=64, activation=torch.nn.GELU)
    hy = MLP(input_shape=out_dim, output_shape=embedding_dim, n_hidden=3, layer_size=64, activation=torch.nn.GELU)
    ncp = NCP(fx, hy, embedding_dim=embedding_dim)

    x = torch.randn(10, in_dim)
    y = torch.randn(10, out_dim)

    fx, hy = ncp(x, y)

    loss, metrics = ncp.loss(fx, hy)
    print(loss)
    with torch.no_grad():
        mi = ncp.mutual_information(x, y)
    print(mi)
