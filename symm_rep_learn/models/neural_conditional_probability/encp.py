# Created  at 19/12/24
from __future__ import annotations

import logging

import escnn
import torch
import torch.nn.functional as F
from escnn.group import Representation
from symm_learning.nn import eEMAStats, eLinear
from symm_learning.nn.disentangled import Change2DisentangledBasis
from symm_learning.representation_theory import direct_sum, isotypic_decomp_rep

from symm_rep_learn.models.neural_conditional_probability.ncp import NCP

log = logging.getLogger(__name__)


# Equivariant Neural Conditional Probabily (e-NCP) module ==============================================================
class ENCP(NCP):
    def __init__(
        self,
        embedding_x: torch.nn.Module,
        embedding_y: torch.nn.Module,
        **ncp_kwargs,
    ):
        self._validate_embeddings(embedding_x=embedding_x, embedding_y=embedding_y)
        self.G = embedding_x.out_rep.group
        self.x_rep, self.y_rep = embedding_x.in_rep, embedding_y.in_rep
        # Given any representations of the embeddings of x and y, we need to change basis to the isotypic basis.
        embedding_x_iso = torch.nn.Sequential(embedding_x, Change2DisentangledBasis(in_rep=embedding_x.out_rep))
        embedding_y_iso = torch.nn.Sequential(embedding_y, Change2DisentangledBasis(in_rep=embedding_y.out_rep))
        fx_rep, hy_rep = embedding_x_iso[-1].out_rep, embedding_y_iso[-1].out_rep

        self.rep_fx_iso = isotypic_decomp_rep(fx_rep)
        self.rep_hy_iso = isotypic_decomp_rep(hy_rep)

        # Intialize the NCP module
        super(ENCP, self).__init__(
            embedding_x=embedding_x_iso,
            embedding_y=embedding_y_iso,
            embedding_dim_x=fx_rep.size,
            embedding_dim_y=hy_rep.size,
            **ncp_kwargs,
        )

        # Replace truncated operator with equivariant linear layer
        # Equivariant Linear layer from lat singular basis to lat singular basis.
        self.Dr = eLinear(in_rep=fx_rep, out_rep=hy_rep, bias=False)
        # Reinitialize the (nparams,)
        with torch.no_grad():
            torch.nn.init.uniform_(self.Dr.weight_dof, a=-1, b=1)
        self.Dr.invalidate_cache()

        # Replace the EMA stats layer
        self.ema_stats = eEMAStats(
            x_rep=fx_rep,
            y_rep=hy_rep,
            momentum=self.ema_stats.momentum,
            center_with_running_mean=self.ema_stats.center_with_running_mean,
        )

        # Custom logic for spectral normalization of equiv layers ___________________________________
        # Buffers for spectral normalization power iteration (2D Dr: (out_dim=hy_size, in_dim=fx_size))
        # u in R^{out_dim} (hy_rep.size), v in R^{in_dim} (fx_rep.size)
        u = F.normalize(self.Dr.weight.new_empty(hy_rep.size).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(self.Dr.weight.new_empty(fx_rep.size).normal_(0, 1), dim=0, eps=1e-12)
        self.register_buffer("_sn_u", u, persistent=True)
        self.register_buffer("_sn_v", v, persistent=True)

    @staticmethod
    def _validate_embeddings(embedding_x: torch.nn.Module, embedding_y: torch.nn.Module) -> None:
        for emb_name, embedding in (("embedding_x", embedding_x), ("embedding_y", embedding_y)):
            assert hasattr(embedding, "in_rep") and hasattr(embedding, "out_rep"), (
                f"{emb_name} must expose in_rep/out_rep attributes. "
                f"Got {type(embedding).__name__} without one of these attributes."
            )
            assert isinstance(embedding.in_rep, Representation), (
                f"{emb_name}.in_rep must be an escnn.group.Representation, got {type(embedding.in_rep)}"
            )
            assert isinstance(embedding.out_rep, Representation), (
                f"{emb_name}.out_rep must be an escnn.group.Representation, got {type(embedding.out_rep)}"
            )

        assert embedding_x.out_rep.group == embedding_y.out_rep.group, (
            f"Embedding groups must match: {embedding_x.out_rep.group} != {embedding_y.out_rep.group}"
        )

    def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
        if self.training:
            assert x is not None and y is not None, "Both x and y must be provided during training."
            fx = self._embedding_x(x)  # f(x) = [f_1(x), ..., f_r(x)]
            hy = self._embedding_y(y)  # h(y) = [h_1(y), ..., h_r(y)]
            self.ema_stats(fx, hy)  # Update mean and covariance statistics
            fx_c = fx - self.ema_stats.mean_x  # f_c(x) = f(x) - E_p(x)[f(x)]
            hy_c = hy - self.ema_stats.mean_y  # h_c(y) = h(y) - E_p(y)[h(y)]
        else:
            fx_c = self._embedding_x(x) - self.ema_stats.mean_x if x is not None else None
            hy_c = self._embedding_y(y) - self.ema_stats.mean_y if y is not None else None

        return fx_c, hy_c

    @property
    def truncated_operator(self):
        # Expand the equivariant linear layer into its dense matrix form (out_dim, in_dim)
        Dr = self.Dr.weight

        # Spectral normalization via power iteration ------------------------------
        # Copied from SpectralNorm in PyTorch
        do_power_iteration = self.training
        eps = 1e-12
        n_power_iters = 1

        u, v = self._sn_u, self._sn_v
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(n_power_iters):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(Dr.t(), u), dim=0, eps=eps, out=v)
                    u = F.normalize(torch.mv(Dr, v), dim=0, eps=eps, out=u)
                if n_power_iters > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(Dr, v))
        Dr = Dr / sigma
        return Dr

    def fit_linear_decoder(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        ridge_reg: float = 1e-3,
        lstsq: bool = False,
        z_rep: Representation | None = None,
        z_type=None,
    ) -> torch.nn.Linear:
        if z_rep is None and z_type is not None:
            z_rep = z_type if isinstance(z_type, Representation) else getattr(z_type, "representation", None)
            if z_rep is None:
                raise TypeError(
                    f"z_type must be a Representation or expose a `.representation` attribute, got {type(z_type)}"
                )

        lin_decoder = super(ENCP, self).fit_linear_decoder(
            train_dataloader=train_dataloader,
            ridge_reg=ridge_reg,
            lstsq=lstsq,
        )

        # Project the linear decoder to the Equivariant subpsace if z_rep is provided
        dtype, device = lin_decoder.weight.dtype, lin_decoder.weight.device
        if z_rep is not None:
            with torch.no_grad():
                orbit = [lin_decoder.weight]
                G = self.G
                for g in G.elements[1:]:
                    rep_hy_g = torch.tensor(self.rep_hy_iso(g), dtype=dtype, device=device)
                    rep_z_g = torch.tensor(z_rep(g), dtype=dtype, device=device)
                    orbit.append(torch.einsum("ij,jk,kl->il", rep_z_g, lin_decoder.weight, rep_hy_g.T))

                G_weight = torch.stack(orbit, dim=0)
                lin_decoder.weight = torch.mean(G_weight, dim=0)

                from symm_learning.linalg import invariant_orthogonal_projector

                P = invariant_orthogonal_projector(z_rep).to(device=device, dtype=dtype)
                bias = P @ lin_decoder.weight.T
                lin_decoder.bias.data = bias

        return lin_decoder


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(5)

    x_rep = G.regular_representation  # ρ_Χ
    y_rep = direct_sum([G.regular_representation] * 10)  # ρ_Y
    lat_rep = direct_sum([G.regular_representation] * 12)  # ρ_Ζ
    x_rep.name, y_rep.name, lat_rep.name = "rep_X", "rep_Y", "rep_L2"

    x_embedding = eLinear(in_rep=x_rep, out_rep=lat_rep, bias=False)
    y_embedding = eLinear(in_rep=y_rep, out_rep=lat_rep, bias=False)

    model = ENCP(x_embedding, y_embedding)

    print(model)
    n_samples = 100
    X = torch.randn(n_samples, x_rep.size)
    Y = torch.randn(n_samples, y_rep.size)
    fx, hy = model(X, Y)
    print(fx.shape, hy.shape)
