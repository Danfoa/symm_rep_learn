# Created by danfoa at 19/12/24
from __future__ import annotations

from idlelib.pyshell import fix_x11_paste

import torch

import logging

from torch.linalg import matrix_norm
from wandb.sdk.internal.profiler import torch_trace_handler

from NCP.mysc.statistics import cov_norm_squared_unbiased_estimation, cross_cov_norm_squared_unbiased_estimation

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

    def mutual_information(self, fx: torch.Tensor, hy: torch.Tensor):
        """Compute the exponential of the mutual information between the random variables x and y.

        The conditional expectation's kernel function k(x,y) = p(x,y) / p(x)p(y), is by definition the exponential of
        the mutual information between two evaluations of the random variables x and y, that is: MI = ln(k(x,y)).

        In the chosen basis sets of the approximated function spaces L^2(X) and L^2(Y), the approximated kernel function
        is computed as: k_r(x,y) = 1 + Σ_i,j=1^r Cxy_ij f_i(x) h_j(y).

        TODO: When the singular basis are appropriatedly identified after training we can compute the kernel function by
        k_r(x,y) = 1 + Σ_i=1^r σ_i f_i(x) h_i(y).

        Args:
            fx: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^2(X).
            hy: (torch.Tensor) of shape (..., r) representing the singular functions of a subspace of L^
        Returns:
            (torch.Tensor) representing the expected mutual information between x and y.
        """
        fx_c, hy_c = fx - self.mean_fx, hy - self.mean_hy
        # Compute the kernel function
        kr = 1 + torch.einsum('nr,rc,nc->n', fx_c, self.Cxy, hy_c)
        # Check no NaN  or Inf values
        assert torch.isfinite(kr).all(), "NaN or Inf values found in the kernel function."
        return kr

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
        n_samples = fx.shape[0]

        self.update_fns_statistics(fx, hy)  # Update mean_fx and mean_hy
        fx_c, hy_c = fx - self.mean_fx, hy - self.mean_hy
        p_perm = torch.randperm(n_samples, device=fx.device)  # Permutation to obtain (x', y') ~ p(x, y)
        pp_perm = torch.randperm(n_samples, device=fx.device)  # Permutation to obtain (x'', y'') ~ p(x, y)
        fx_cp, hy_cp = fx_c[p_perm], hy_c[p_perm]  # f_c(x') and h_c(y')

        # Orthonormal regularization and centering penalization _________________________________________
        # orth_reg_fx = ||Cx - I||_F^2 + ||Cy - I||_F^2 + 2 ||E_p(x) f(x)||_F^2 + 2 ||E_p(y) h(y)||_F^2
        orthonormal_reg, metrics, inner_prod = self.orthonormality_penalization(
            fx_c, hy_c, return_inner_prod=True, permutation=p_perm
            )
        fx_fxp, hy_hyp = inner_prod  # [fx_fxp]nm = (f_c(x_n).T f_c(x'_m)), [hy_hyp]nm = (h_c(y_n).T h_c(y'_m))

        # Operator truncation error = -2 ||Cxy||_F^2 + tr(Cxy Cy Cxy^T Cx) - 1___________________________
        # Biased estimation:
        # Cxy_F_2_biased = matrix_norm(self.Cxy, ord='fro') ** 2
        Pxyx = torch.einsum('ab,bc,cd,de->ae', self.Cxy, self.Cy, self.Cxy, self.Cx)
        tr_Pxyx_biased = torch.trace(Pxyx)
        # Unbiased estimation:   [fxp_hypp]nm = (f_c(x'_n).T h_c(y''_m)), [fxpp_hyp]nm = (f_c(x''_n).T h_c(y'_m))
        Cxy_F_2, (fxp_hypp, fxpp_hyp) = cross_cov_norm_squared_unbiased_estimation(fx_cp, hy_cp,
                                                                                   return_inner_prods=True)
        # Pxyx = 1/N^4 Σ_n,m,l,k (f(x_n).T f(x'_m)) (h(y_n).T h(y'_m))(h(y'_m).T f(x''_l))(h(y''_l).T f(x'_k))
        # tr_Pxyx = torch.einsum('nm,nm,lm,kl->', fx_fxp, hy_hyp, fxpp_hyp, fxp_hypp) / (n_samples ** 4)  # not working
        truncation_err = -2 * Cxy_F_2 - 1 + tr_Pxyx_biased - 1
        # truncation_err = -2 * Cxy_F_2_biased + tr_Pxyx - 1  # TODO: Unbiased tr_Pxyx not working properly

        # Total loss ____________________________________________________________________________________
        # loss = truncation_err + self.gamma * (orth_reg + center_reg)
        loss = truncation_err + (self.embedding_dim * self.gamma) * (orthonormal_reg)
        # Logging metrics _______________________________________________________________________________
        with torch.no_grad():
            metrics |= {
                "||E - E_r||_HS":             truncation_err.detach(),
                "||E - E_r||_HS/biased":      self.biased_truncation_error(fx, hy).detach(),
                "||Cxy||_F^2":                Cxy_F_2.detach() / self.embedding_dim,
                # "tr(Cxy Cy Cxy^T Cx)": tr_Pxyx.detach() / self.embedding_dim,
                "tr(Cxy Cy Cxy^T Cx)/biased": tr_Pxyx_biased.detach() / self.embedding_dim,
                "||Cxy||_F^2/biased":         torch.linalg.matrix_norm(self.Cxy) ** 2 / self.embedding_dim,
                "||Cx||_F^2/biased":          torch.linalg.matrix_norm(self.Cx) ** 2 / self.embedding_dim,
                "||Cy||_F^2/biased":          torch.linalg.matrix_norm(self.Cy) ** 2 / self.embedding_dim,
                }
        return loss, metrics

    def biased_truncation_error(self, fx, hy):
        eps = 1e-6 * torch.eye(self.embedding_dim, device=fx.device, dtype=fx.dtype)
        n_samples = fx.shape[0]
        fx_c = fx - fx.mean(dim=0, keepdim=True)
        hy_c = hy - hy.mean(dim=0, keepdim=True)

        Cxy = torch.einsum('nr,nc->rc', fx_c, hy_c) / (n_samples - 1)
        Cx = torch.einsum('nr,nc->rc', fx_c, fx_c) / (n_samples - 1) + eps
        Cy = torch.einsum('nr,nc->rc', hy_c, hy_c) / (n_samples - 1) + eps

        A = torch.einsum('ab,bc,cd,de->ae', Cxy, Cy, Cxy, Cx)

        return - 2 * torch.sum(Cxy ** 2) + torch.trace(A) - 1

    def orthonormality_penalization(self, fx_c, hy_c, return_inner_prod=False, permutation=None):
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
        if return_inner_prod:
            Cx_fro_2, fx_fxp = cov_norm_squared_unbiased_estimation(fx_c, True, permutation=permutation)
            Cy_fro_2, hy_hyp = cov_norm_squared_unbiased_estimation(hy_c, True, permutation=permutation)
        else:
            Cx_fro_2 = cov_norm_squared_unbiased_estimation(fx_c, False, permutation=permutation)
            Cy_fro_2 = cov_norm_squared_unbiased_estimation(hy_c, False, permutation=permutation)
            fx_fxp, hy_hyp = None, None

        tr_Cx = torch.trace(self.Cx)  # E[f_c(x)^T f_c(x)] =  tr(Cx)
        fx_centering_loss = (self.mean_fx ** 2).sum()  # ||E_p(x) (f(x_i))||^2

        tr_Cy = torch.trace(self.Cy)  # E[h_c(y)^T h_c(y)] = tr(Cy)
        hy_centering_loss = (self.mean_hy ** 2).sum()  # ||E_p(y) (h(y_i))||^2

        embedding_dim = fx_c.shape[-1]  # Dimensionality r

        # orthonormality_fx = tr(Cx^2) - 2 tr(Cx) + 2 || E_p(x) f(x) ||^2 + r
        orthonormality_fx = Cx_fro_2 - 2 * tr_Cx + 2 * fx_centering_loss + embedding_dim
        # orthonormality_hy = tr(Cy^2) - 2 tr(Cy) + 2 || E_p(y) h(y) ||^2 + r
        orthonormality_hy = Cy_fro_2 - 2 * tr_Cy + 2 * hy_centering_loss + embedding_dim
        # Combine terms
        regularization = orthonormality_fx + orthonormality_hy

        with torch.no_grad():
            metrics = {
                f"||Cx||_F^2":     Cx_fro_2 / embedding_dim,
                f"||mu_x||":       torch.sqrt(fx_centering_loss),
                f"||Vx - I||_F^2": orthonormality_fx / embedding_dim,
                f"||tr(Cx)||":     tr_Cx / embedding_dim,
                f"||Cy||_F^2":     Cy_fro_2 / embedding_dim,
                f"||mu_y||":       torch.sqrt(hy_centering_loss),
                f"||Vy - I||_F^2": orthonormality_hy / embedding_dim,
                f"||tr(Cy)||":     tr_Cy / embedding_dim,
                }

        if return_inner_prod:
            return regularization, metrics, (fx_fxp, hy_hyp)
        else:
            return regularization, metrics

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
        fx, hy = ncp(x, y)
        mi = ncp.mutual_information(fx, hy)
    print(mi)
