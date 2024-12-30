# Created by danfoa at 19/12/24
from __future__ import annotations
import torch

import logging


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
        self.register_buffer('Cyx', torch.zeros(embedding_dim, embedding_dim))
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
        # approx_err = -2 ||Cxy||_F^2 + tr(Cxy Cx Cxy^T Cy) - 1
        Cxy_f_norm = torch.linalg.norm(self.Cxy, ord='fro') ** 2
        prod_C = torch.einsum('ij,jk,kl,ln->in', self.Cxy, self.Cx, self.Cxy, self.Cy)
        approx_err = -2 * Cxy_f_norm + torch.trace(prod_C) - 1

        # Regularization terms, encouraging orthonormality and centered basis functions
        # orth_reg = ||Cx - I||_F^2 + ||Cy - I||_F^2
        I = torch.eye(self.embedding_dim, device=fx.device, dtype=fx.dtype)
        orth_reg = (torch.sum((self.Cx - I) ** 2) + torch.sum((self.Cy - I) ** 2)) / 2
        # center_reg = ||mean_fx := E_p(x) f(x)||_F^2 + ||mean_hy := E_p(y) h(y)||_F^2
        mean_fx_norm, mean_hy_norm = torch.linalg.norm(self.mean_fx), torch.linalg.norm(self.mean_hy)
        center_reg = mean_fx_norm ** 2 + mean_hy_norm ** 2

        # Total loss ___________________________________________________________
        loss = approx_err + self.gamma * (orth_reg + center_reg)
        # Logging metrics ______________________________________________________
        metrics = {
            "||Cxy||_F": Cxy_f_norm.detach() / self.embedding_dim,
            "||mu_x||":  mean_fx_norm.detach(),
            "||mu_y||":  mean_hy_norm.detach(),
            }
        metrics |= self.batch_metrics()
        return loss, metrics

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
            eps = 1e-6 * torch.eye(self.embedding_dim)
            self.mean_fx = fx.mean(dim=0, keepdim=True)
            self.mean_hy = hy.mean(dim=0, keepdim=True)
            self.Cxy = (torch.einsum('nr,nc->rc', fx, hy) - self.mean_fx.T @ self.mean_fx) / (n_samples - 1)
            self.Cx = (torch.einsum('nr,nr->r', fx, fx) - self.mean_fx.T @ self.mean_fx) / (n_samples - 1) + eps
            self.Cy = (torch.einsum('nr,nr->r', hy, hy) - self.mean_hy.T @ self.mean_hy) / (n_samples - 1) + eps


if __name__ == "__main__":
    from NCP.nn.layers import MLP
    in_dim, out_dim, embedding_dim = 10,4,40
    fx = MLP(input_shape=in_dim, output_shape=embedding_dim, n_hidden=3, layer_size=64, activation=torch.nn.ReLU)
    hy = MLP(input_shape=out_dim, output_shape=embedding_dim, n_hidden=3, layer_size=64, activation=torch.nn.ReLU)
    ncp = NCP(fx, hy, embedding_dim=embedding_dim)
    x = torch.randn(10, in_dim)
    y = torch.randn(10, out_dim)
    fx, hy = ncp(x, y)
    loss, metrics = ncp.loss(fx, hy)
    print(loss)
    with torch.no_grad():
        mi = ncp.mutual_information(x, y)
    print(mi)