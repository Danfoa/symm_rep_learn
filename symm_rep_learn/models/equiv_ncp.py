# Created by danfoa at 19/12/24
from __future__ import annotations

import logging

import escnn.nn
import lightning
import numpy as np
import symm_learning
import torch
from escnn.group import directsum
from escnn.nn import FieldType, GeometricTensor
from symm_learning.linalg import isotypic_signal2irreducible_subspaces
from symm_learning.nn.disentangled import Change2DisentangledBasis

from symm_rep_learn.models.ncp import NCP
from symm_rep_learn.mysc.statistics import (
    cov_norm_squared_unbiased_estimation,
)

log = logging.getLogger(__name__)


# Equivariant Neural Conditional Probabily (e-NCP) module ==============================================================
class ENCP(NCP):
    def __init__(
        self,
        embedding_x: escnn.nn.EquivariantModule,
        embedding_y: escnn.nn.EquivariantModule,
        gamma=1.0,
        gamma_centering=None,
        learnable_change_of_basis: bool = False,
    ):
        self.G = embedding_x.out_type.fibergroup
        # Given any Field types of the embeddings of x and y, we need to change basis to the isotypic basis.
        embedding_x_iso = escnn.nn.SequentialModule(
            embedding_x, Change2DisentangledBasis(in_type=embedding_x.out_type, learnable=learnable_change_of_basis)
        )
        embedding_y_iso = escnn.nn.SequentialModule(
            embedding_y, Change2DisentangledBasis(in_type=embedding_y.out_type, learnable=learnable_change_of_basis)
        )

        # Isotypic subspace are identified by the irrep id associated with the subspace
        self.n_iso_subspaces = len(embedding_x_iso.out_type.representations)
        self.iso_subspace_ids = [iso_rep.irreps[0] for iso_rep in embedding_x_iso.out_type.representations]
        self.iso_subspace_dims = [iso_rep.size for iso_rep in embedding_x_iso.out_type.representations]
        self.irreps_dim = {irrep_id: self.G.irrep(*irrep_id).size for irrep_id in self.iso_subspace_ids}
        self.iso_subspace_slice = [
            slice(s, e) for s, e in zip(embedding_x_iso.out_type.fields_start, embedding_x_iso.out_type.fields_end)
        ]
        self.iso_irreps_multiplicities = [
            space_dim // self.irreps_dim[id] for space_dim, id in zip(self.iso_subspace_dims, self.iso_subspace_ids)
        ]
        if self.G.trivial_representation.id in self.iso_subspace_ids:
            self.idx_inv_subspace = self.iso_subspace_ids.index(self.G.trivial_representation.id)
        else:
            self.idx_inv_subspace = None

        # Intialize the NCP module
        super().__init__(
            embedding_x=embedding_x_iso,
            embedding_y=embedding_y_iso,
            embedding_dim=embedding_x_iso.out_type.size,
            orth_reg=gamma,
            centering_reg=gamma_centering,
        )

    def forward(self, x: GeometricTensor, y: GeometricTensor):
        """Forward pass of the eNCP operator.

        Computes non-linear transformations of the input random variables x and y, and returns r-dimensional embeddings
        f(x) = [f_1(x), ..., f_r(x)] and h(y) = [h_1(y), ..., h_r(y)] representing the top r-singular functions of
        the conditional expectation operator such that E_p(y|x)[h_i(y)] = σ_i f_i(x) for i=1,...,r.

        Args:
            x: (GeometricTensor) of shape (..., d_x) representing the input x.
            y: (GeometricTensor) of shape (..., d_y) representing the input y.

        Returns:
            fx: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(X)
            hy: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(Y)
        """
        fx = self.embedding_x(x)  # f(x) = [f_1(x), ..., f_r(x)]
        hy = self.embedding_y(y)  # h(y) = [h_1(y), ..., h_r(y)]
        return fx, hy

    def pointwise_mutual_dependency(self, x: GeometricTensor, y: GeometricTensor):
        fx, hy = self(x, y)
        # k(x, y) = 1 + Σ_i=1^r σ_i f_i(x) h_i(y)
        # Einsum can do this operation faster in GPU with some internal optimizations.
        fx_c = self._orth_proj_isotypic_subspaces(fx)
        hy_c = self._orth_proj_isotypic_subspaces(hy)
        # Center if inv subspace is present.
        if self.idx_inv_subspace is not None:
            fx_c[self.idx_inv_subspace] = fx_c[self.idx_inv_subspace] - self.mean_fx
            hy_c[self.idx_inv_subspace] = hy_c[self.idx_inv_subspace] - self.mean_hy

        Dr = self.truncated_operator
        fx_c = torch.cat(fx_c, dim=-1)
        hy_c = torch.cat(hy_c, dim=-1)
        k_r = 1 + torch.einsum("...x,xy,...y->...", fx_c, Dr, hy_c)
        return k_r

    def orthonormality_penalization(
        self, fx_c: GeometricTensor, hy_c: GeometricTensor, return_inner_prod=False, permutation=None
    ):
        """Computes orthonormality and centering regularization penalization for a batch of feature vectors.

        Computes finite sample unbiased empirical estimates of the term:
        || Vx - I ||_F^2 = || Cx - I ||_F^2 + 2 || E_p(x) f(x) ||^2
                         = || ⊕_k Cx_k - I_r_k ||_F^2 + 2 || E_p(x) f^inv (x) ||^2
                         = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || Cx_k - I_r_k ||_F^2
                         = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || Cx_k ||_F^2 - 2tr(Cx_k) + r_k
                         = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || (Dx_k ⊗ I_ρk) ||_F^2 - 2tr(Dx_k ⊗ I_ρk) + r_k
                         = 2 || E_p(x) f^inv (x) ||^2 + Σ_k |ρk| (||Dx_k||_F^2 - 2tr(Dx_k)) + r_k
        || Vy - I ||_F^2 = 2 || E_p(y) h^inv (y) ||^2 + Σ_k |ρk| (||Dy_k||_F^2 - 2tr(Dy_k)) + r_k
        Args:
            fx_c: (n_samples, r) Centered feature vectors f_c(x) = [f_c,1(x), ..., f_c,r(x)].
            hy_c: (n_samples, r) Centered feature vectors h_c(y) = [h_c,1(y), ..., h_c,r(y)].
            return_inner_prod: (bool) If True, return intermediate inner products.
            permutation: (torch.Tensor) Permutation tensor to shuffle the samples in the batch.

        Returns:
            Regularization term as a scalar tensor.
        """
        assert fx_c.type == self.embedding_x.out_type and hy_c.type == self.embedding_y.out_type  # Iso basis.
        # Project embedding into the isotypic subspaces
        fx_c_iso, reps_Fx_iso = self._orth_proj_isotypic_subspaces(z=fx_c), fx_c.type.representations
        hy_c_iso, reps_Hy_iso = self._orth_proj_isotypic_subspaces(z=hy_c), hy_c.type.representations

        Cx_iso_fro_2, Cy_iso_fro_2 = [], []
        trCx_iso, trCy_iso = [], []
        for k, (fx_ck, rep_x_k, hy_ck, rep_y_k) in enumerate(zip(fx_c_iso, reps_Fx_iso, hy_c_iso, reps_Hy_iso)):
            irrep_dim = self.irreps_dim[self.iso_subspace_ids[k]]
            # Flatten the realizations along irreducible subspaces, while preserving sampling from the joint dist.
            zx = isotypic_signal2irreducible_subspaces(fx_ck, rep_x_k)  # (n_samples * |ρ_k|, r_k / |ρ_k|)
            zy = isotypic_signal2irreducible_subspaces(hy_ck, rep_y_k)  # (n_samples * |ρ_k|, r_k / |ρ_k|)
            r_xk, r_yk = fx_ck.shape[-1], hy_ck.shape[-1]  # r_k
            # Compute unbiased empirical estimates ||Dx_k||_F^2
            Dx_k_fro_2 = cov_norm_squared_unbiased_estimation(zx, False, permutation=permutation)
            Dy_k_fro_2 = cov_norm_squared_unbiased_estimation(zy, False, permutation=permutation)
            # Trace terms without need of unbiased estimation
            tr_Dx_k = torch.trace(self.__getattr__(f"Dx_{k}"))  # tr(Dx_k)
            tr_Dy_k = torch.trace(self.__getattr__(f"Dy_{k}"))  # tr(Dy_k)
            #  ||Cx_k||_F^2 := |ρk| (||Dx_k||_F^2 - 2tr(Dx_k)) + r_k
            Cx_k_fro_2 = irrep_dim * (Dx_k_fro_2 - 2 * tr_Dx_k) + r_xk
            #  ||Cy_k||_F^2 := |ρk| (||Dy_k||_F^2 - 2tr(Dy_k)) + r_k
            Cy_k_fro_2 = irrep_dim * (Dy_k_fro_2 - 2 * tr_Dy_k) + r_yk
            Cx_iso_fro_2.append(Cx_k_fro_2)
            Cy_iso_fro_2.append(Cy_k_fro_2)
            trCx_iso.append(tr_Dx_k)
            trCy_iso.append(tr_Dy_k)
            # Cx_k_fro_2_biased = torch.linalg.matrix_norm(self.Cx(k)) ** 2
            # Cy_k_fro_2_biased = torch.linalg.matrix_norm(self.Cy(k)) ** 2

        Cx_I_err_fro_2 = sum(Cx_iso_fro_2)  # ||Cx - I||_F^2 = Σ_k ||Cx_k - I_r_k||_F^2,
        Cy_I_err_fro_2 = sum(Cy_iso_fro_2)  # ||Cy - I||_F^2 = Σ_k ||Cy_k - I_r_k||_F^2
        trCx = sum(trCx_iso)  # tr(Cx) = Σ_k tr(Dx_k)
        trCy = sum(trCy_iso)  # tr(Cy) = Σ_k tr(Dy_k)
        # Cx_fro_2_biased = torch.linalg.matrix_norm(self.Cx(None)) ** 2
        # Cy_fro_2_biased = torch.linalg.matrix_norm(self.Cy(None)) ** 2

        # TODO: Unbiased estimation of mean squared
        fx_centering_loss = (self.mean_fx**2).sum()  # ||E_p(x) (f(x_i))||^2
        hy_centering_loss = (self.mean_hy**2).sum()  # ||E_p(y) (h(y_i))||^2

        # ||Vx - I||_F^2 = ||Cx - I||_F^2 + 2||E_p(x) f(x)||^2
        orthonormality_fx = Cx_I_err_fro_2 + 2 * fx_centering_loss
        # ||Vy - I||_F^2 = ||Cy - I||_F^2 + 2||E_p(y) h(y)||^2
        orthonormality_hy = Cy_I_err_fro_2 + 2 * hy_centering_loss

        with torch.no_grad():
            embedding_dim_x, embedding_dim_y = self.embedding_x.out_type.size, self.embedding_y.out_type.size
            metrics = {
                "||Cx||_F^2": Cx_I_err_fro_2 / embedding_dim_x,
                "tr(Cx)": trCx / embedding_dim_x,
                "||mu_x||": torch.sqrt(fx_centering_loss),
                "||Vx - I||_F^2": orthonormality_fx / embedding_dim_x,
                #
                "||Cy||_F^2": Cy_I_err_fro_2 / embedding_dim_y,
                "tr(Cy)": trCy / embedding_dim_y,
                "||mu_y||": torch.sqrt(hy_centering_loss),
                "||Vy - I||_F^2": orthonormality_hy / embedding_dim_y,
            }

        if return_inner_prod:
            raise NotImplementedError("Inner products not implemented yet.")
        else:
            return orthonormality_fx, orthonormality_hy, metrics

    def unbiased_truncation_error_matrix_form(self, fx_c: GeometricTensor, hy_c: GeometricTensor):
        """Implementation of ||E - E_r||_HS^2, while assuming E_r is a full matrix.

        Args:
            fx_c: (torch.Tensor) of shape (n_samples, r) representing the centered singular functions of a subspace
            of L^2(X).
            hy_c: (torch.Tensor) of shape (n_samples, r) representing the centered singular functions of a subspace
            of L^2(Y).

        Returns:
            (torch.Tensor) representing the unbiased truncation error.
        """
        assert fx_c.type == self.embedding_x.out_type and hy_c.type == self.embedding_y.out_type  # Iso basis.
        metrics = {}

        # k_r(x,y) = 1 + f(x)^T Dr h(y) = 1 + Σ_κ f_κ(x)^T Dr_κ h_κ(y)
        n_samples = fx_c.shape[0]
        Dr = self.truncated_operator  # Dr = Dr.T
        # Project embedding into the isotypic subspaces
        fx_c_iso = self._orth_proj_isotypic_subspaces(z=fx_c)
        hy_c_iso = self._orth_proj_isotypic_subspaces(z=hy_c)
        Dr_iso = [Dr[s:e, s:e] for s, e in zip(fx_c.type.fields_start, fx_c.type.fields_end)]

        # Sequential is slower but enable us to log the metrics we want.
        # pmd_mat = 1
        E_pxy_kr_iso, E_px_py_kr_iso = [], []
        truncation_err_iso = []
        for fx_ci, hy_ci, Dr_i in zip(fx_c_iso, hy_c_iso, Dr_iso):
            k_r_i = torch.einsum("nx,xy,my->nm", fx_ci, Dr_i, hy_ci)  # (n_samples, n_samples)
            # pmd_mat = pmd_mat + k_r_i
            E_pxy_kr_i = torch.diag(k_r_i).mean()
            E_pxy_kr_iso.append(E_pxy_kr_i)
            k_r_i2 = k_r_i**2
            E_px_py_kr_i = (k_r_i2.sum() - k_r_i2.diag().sum()) / (n_samples * (n_samples - 1))
            E_px_py_kr_iso.append(E_px_py_kr_i)
            truncation_err_iso.append(-2 * E_pxy_kr_i + E_px_py_kr_i)
        # truncated_err = -2 * E_p(x,y)[k_r(x,y)] + E_p(x)p(y)[k_r(x,y)^2]
        E_pxy_kr = sum(E_pxy_kr_iso)
        # E_p(x)p(y)[k_r(x,y)^2]  # Note we remove the samples from the joint in the diagonal
        E_px_py_kr = sum(E_px_py_kr_iso)
        truncation_err = (-2 * E_pxy_kr) + (E_px_py_kr)
        with torch.no_grad():
            metrics |= {
                "E_p(x)p(y) k_r(x,y)^2": E_px_py_kr.detach(),
                "E_p(x,y) k_r(x,y)": E_pxy_kr.detach(),
            }
            for k in range(self.n_iso_subspaces):
                metrics[f"||k(x,y) - k_r(x,y)||iso/{k}"] = truncation_err_iso[k]
                metrics[f"E_p(x,y) k_r(x,y)iso/{k}"] = E_pxy_kr_iso[k]
                metrics[f"E_p(x)p(y) k_r(x,y)^2iso/{k}"] = E_px_py_kr_iso[k]

        return truncation_err, metrics

    def update_fns_statistics(self, fx: GeometricTensor, hy: GeometricTensor):
        assert isinstance(fx, GeometricTensor) and isinstance(hy, GeometricTensor), (
            f"Expected Geometric Tensors got f(x): {type(fx)} and h(y): {type(hy)}"
        )
        assert fx.type == self.embedding_x.out_type and hy.type == self.embedding_y.out_type
        _device, _dtype = fx.tensor.device, fx.tensor.dtype

        # Get projections into isotypic subspaces.  fx_iso[k] = fx^(k), hy_iso[k] = hy^(k)
        fx_iso, reps_Fx_iso = self._orth_proj_isotypic_subspaces(z=fx), fx.type.representations
        hy_iso, reps_Hy_iso = self._orth_proj_isotypic_subspaces(z=hy), hy.type.representations

        if self.idx_inv_subspace is not None:
            self.mean_fx = fx_iso[self.idx_inv_subspace].mean(dim=0, keepdim=True)
            self.mean_hy = hy_iso[self.idx_inv_subspace].mean(dim=0, keepdim=True)

        # Compute the empirical covariance matrices
        nk = self.n_iso_subspaces
        for k, fx_k, hy_k, rep_x_k, rep_y_k in zip(range(nk), fx_iso, hy_iso, reps_Fx_iso, reps_Hy_iso):
            # Centered observations (non-trivial subspaces are centered by construction)
            fx_ck = fx_k if k != self.idx_inv_subspace else fx_k - self.mean_fx
            hy_ck = hy_k if k != self.idx_inv_subspace else hy_k - self.mean_hy
            Cxy_k, _ = symm_learning.stats._isotypic_cov(x=fx_ck, y=hy_ck, rep_x=rep_x_k, rep_y=rep_y_k)
            Cx_k, _ = symm_learning.stats._isotypic_cov(x=fx_ck, rep_x=rep_x_k)
            Cy_k, _ = symm_learning.stats._isotypic_cov(x=hy_ck, rep_x=rep_y_k)
            setattr(self, f"Cxy_{k}", Cxy_k)
            setattr(self, f"Cx_{k}", Cx_k)
            setattr(self, f"Cy_{k}", Cy_k)

        # Center basis functions. Centering occurs only at the G-invariant subspace.
        fx_c, hy_c = fx.tensor.clone(), hy.tensor.clone()
        if self.idx_inv_subspace is not None:
            inv_subspace_dims = self.iso_subspace_slice[self.idx_inv_subspace]
            fx_c[..., inv_subspace_dims] = fx.tensor[..., inv_subspace_dims] - self.mean_fx
            hy_c[..., inv_subspace_dims] = hy.tensor[..., inv_subspace_dims] - self.mean_hy
        fx_c = GeometricTensor(fx_c, self.embedding_x.out_type)
        hy_c = GeometricTensor(hy_c, self.embedding_y.out_type)

        return fx_c, hy_c

    @property
    def truncated_operator(self):
        # D_r is diagonal and is stable (that is has eivalues <= 1)
        D_r, _ = self._Dr.expand_parameters()  # Expand the equiv lin layer into its matrix form
        Dr_symm = (D_r @ D_r.T) / 2  # Ensure its symmetric.
        eigval_max = torch.linalg.eigvalsh(Dr_symm)[-1]
        Dr = Dr_symm / eigval_max
        return Dr

    def Cxy(self, iso_idx=None):
        """Compute the cross-covariance matrix Cxy.

        Args:
            iso_idx (int, optional): Index of the isotypic subspace. If None, compute the block diagonal matrix.

        Returns:
            torch.Tensor: The cross-covariance matrix.
        """
        if iso_idx is None:
            Cxy_iso = [self.Cxy(k) for k in range(self.n_iso_subspaces)]
            return torch.block_diag(*Cxy_iso)
        else:
            assert 0 <= iso_idx < self.n_iso_subspaces, f"{iso_idx} not in [0,{self.n_iso_subspaces}]"
            return self.__getattr__(f"Cxy_{iso_idx}")

    def Cx(self, iso_idx=None):
        """Compute the covariance matrix Cx.

        Args:
            iso_idx (int, optional): Index of the isotypic subspace. If None, compute the block diagonal matrix.

        Returns:
            torch.Tensor: The covariance matrix.
        """
        if iso_idx is None:
            Cx_iso = [self.Cx(k) for k in range(self.n_iso_subspaces)]
            return torch.block_diag(*Cx_iso)
        else:
            assert 0 <= iso_idx < self.n_iso_subspaces, f"{iso_idx} not in [0,{self.n_iso_subspaces}]"
            return self.__getattr__(f"Cx_{iso_idx}")

    def Cy(self, iso_idx=None):
        """Compute the covariance matrix Cy.

        Args:
            iso_idx (int, optional): Index of the isotypic subspace. If None, compute the block diagonal matrix.

        Returns:
            torch.Tensor: The covariance matrix.
        """
        if iso_idx is None:
            Cy_iso = [self.Cy(k) for k in range(self.n_iso_subspaces)]
            return torch.block_diag(*Cy_iso)
        else:
            assert 0 <= iso_idx < self.n_iso_subspaces, f"{iso_idx} not in [0,{self.n_iso_subspaces}]"
            return self.__getattr__(f"Cy_{iso_idx}")

    def _orth_proj_isotypic_subspaces(self, z: GeometricTensor) -> list[torch.Tensor]:
        """Compute the orthogonal projection of the input tensor into the isotypic subspaces."""
        z_iso = [z.tensor[..., s:e] for s, e in zip(z.type.fields_start, z.type.fields_end)]
        return z_iso

    def _create_op_parameters(self):
        lat_singular_type = self.embedding_x.out_type

        # Equivariant Linear layer from lat singular basis to lat singular basis.
        self._Dr = escnn.nn.Linear(in_type=lat_singular_type, out_type=lat_singular_type, bias=False)
        # Reinitialize the (nparams,)
        self._Dr.weights.data = torch.nn.init.uniform_(self._Dr.weights.data, a=-1, b=1)
        Dr, _ = self._Dr.expand_parameters()
        # sval_max = torch.linalg.matrix_norm(Dr, 2)
        Dr = (Dr @ Dr.T) / 2
        sval_max = torch.linalg.matrix_norm(Dr, 2)
        self._Dr.weights.data = self._Dr.weights.data / sval_max

    def _register_stats_buffers(self):
        """Register the buffers for the running mean, Covariance and Cross-Covariance matrix matrix."""
        if self.idx_inv_subspace is not None:
            dim_x_inv_subspace = self.embedding_x.out_type.representations[self.idx_inv_subspace].size
            dim_y_inv_subspace = self.embedding_y.out_type.representations[self.idx_inv_subspace].size
            self.register_buffer("mean_fx", torch.zeros((1, dim_x_inv_subspace)))
            self.register_buffer("mean_hy", torch.zeros((1, dim_y_inv_subspace)))

        for iso_idx, iso_id, iso_subspace_dim in zip(
            range(self.n_iso_subspaces), self.iso_subspace_ids, self.iso_subspace_dims
        ):
            irrep_dim = self.irreps_dim[iso_id]  # |ρ_k|  Dimension of the irrep
            self.register_buffer(f"Cxy_{iso_idx}", torch.zeros(iso_subspace_dim, iso_subspace_dim))
            # Matrix containing the DoF of Cx^(k) = Dx ⊗ Iρ_k: L^2(X^(k)) -> L^2(X^(k))
            self.register_buffer(f"Cx_{iso_idx}", torch.zeros(iso_subspace_dim, iso_subspace_dim))
            # Matrix containing the DoF of Cy^(k) = Dy ⊗ Iρ_k: L^2(Y^(k)) -> L^2(Y^(k))
            self.register_buffer(f"Cy_{iso_idx}", torch.zeros(iso_subspace_dim, iso_subspace_dim))


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(5)

    x_rep = G.regular_representation  # ρ_Χ
    y_rep = directsum([G.regular_representation] * 10)  # ρ_Y
    lat_rep = directsum([G.regular_representation] * 12)  # ρ_Ζ
    x_rep.name, y_rep.name, lat_rep.name = "rep_X", "rep_Y", "rep_L2"

    type_X = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[x_rep])
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[y_rep])
    lat_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[lat_rep])

    χ_embedding = escnn.nn.Linear(type_X, lat_type)
    y_embedding = escnn.nn.Linear(type_Y, lat_type)

    model = ENCP(χ_embedding, y_embedding)

    print(model)
    n_samples = 100
    X = torch.randn(n_samples, x_rep.size)
    Y = torch.randn(n_samples, y_rep.size)
    k = model(GeometricTensor(X, type_X), GeometricTensor(Y, type_Y))
    print("Done")

    # Check all training pipeline ========================================
    from torch.utils.data import DataLoader, TensorDataset, default_collate

    batch_size = 256

    # ESCNN equivariant models expect GeometricTensors.
    def geom_tensor_collate_fn(batch) -> [GeometricTensor, GeometricTensor]:
        x_batch, y_batch = default_collate(batch)
        return GeometricTensor(x_batch, type_X), GeometricTensor(y_batch, type_Y)

    n_samples = 1000
    X_train, X_val = torch.randn(n_samples, x_rep.size), torch.randn(int(n_samples * 0.15), x_rep.size)
    Y_train, Y_val = torch.randn(n_samples, y_rep.size), torch.randn(int(n_samples * 0.15), y_rep.size)

    # Dataset
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    # Dataloaders
    collate_fn = geom_tensor_collate_fn if isinstance(model, ENCP) else default_collate
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Simple train and backward loop test:

    model.train()
    for _ in range(100):
        for x, y in train_dataloader:
            with torch.autograd.set_detect_anomaly(True):
                fx, hy = model(x, y)
                loss, metrics = model.loss(fx, hy)
                loss.backward()
            break
        break

    # Train using lightning_______________________________________________
    from symm_rep_learn.models.lightning_modules import TrainingModule

    light_module = TrainingModule(
        model, optimizer_fn=torch.optim.Adam, optimizer_kwargs=dict(lr=1e-3), loss_fn=model.loss
    )
    trainer = lightning.Trainer(max_epochs=50)

    torch.set_float32_matmul_precision("medium")
    trainer.fit(light_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    #
    # log_std = torch.nn.Parameter(n,)
    # _std = torch.exp(log_std)
    #
    # std = torch.mean([rep_A(g) _std for g in G.elements])   #  std = 1/|G| Sum_g∈G rep_A(g) _std
