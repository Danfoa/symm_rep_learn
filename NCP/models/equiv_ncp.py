# Created by danfoa at 19/12/24
from __future__ import annotations
import logging

import escnn.nn
import lightning
import numpy as np
import torch
from escnn.group import directsum
from escnn.nn import FieldType, GeometricTensor

from NCP.models.ncp import NCP
from NCP.mysc.rep_theory_utils import field_type_to_isotypic_basis
from NCP.mysc.statistics import cov_norm_squared_unbiased_estimation, cross_cov_norm_squared_unbiased_estimation
from NCP.mysc.symm_algebra import (invariant_orthogonal_projector, isotypic_cross_cov,
                                   isotypic_signal2irreducible_subspaces)

log = logging.getLogger(__name__)


# Equivariant Neural Conditional Probabily (e-NCP) module ==============================================================
class ENCP(NCP):

    def __init__(self,
                 embedding_x: escnn.nn.EquivariantModule,
                 embedding_y: escnn.nn.EquivariantModule,
                 gamma=0.001,
                 truncated_op_bias: str = 'Cxy',
                 ):
        # Get field type in the singular-isotypic basis
        assert embedding_x.out_type == embedding_y.out_type, "Embeddings field types must be the same"
        lat_singular_type = field_type_to_isotypic_basis(embedding_x.out_type)
        self.G = lat_singular_type.fibergroup
        # Take any input field-type, add a G-equivariant linear layer, parameterizing a change of basis to the
        # Iso-singular basis. (singular functions clustered by isotypic subspaces)
        x2singular = escnn.nn.Linear(in_type=embedding_x.out_type, out_type=lat_singular_type, bias=False)
        y2singular = escnn.nn.Linear(in_type=embedding_y.out_type, out_type=lat_singular_type, bias=False)
        embedding_x = escnn.nn.SequentialModule(embedding_x, x2singular)
        embedding_y = escnn.nn.SequentialModule(embedding_y, y2singular)

        # Isotypic subspace are identified by the irrep id associated with the subspace
        self.n_iso_subspaces = len(lat_singular_type.representations)
        self.iso_subspace_ids = [iso_rep.irreps[0] for iso_rep in lat_singular_type.representations]
        self.iso_subspace_dims = [iso_rep.size for iso_rep in lat_singular_type.representations]
        self.irreps_dim = {irrep_id: self.G.irrep(*irrep_id).size for irrep_id in self.iso_subspace_ids}
        self.iso_subspace_slice = [
            slice(s, e) for s, e in zip(lat_singular_type.fields_start, lat_singular_type.fields_end)
            ]
        self.iso_irreps_multiplicities = [
            space_dim // self.irreps_dim[id] for space_dim, id in zip(self.iso_subspace_dims, self.iso_subspace_ids)
            ]
        if self.G.trivial_representation.id in self.iso_subspace_ids:
            self.idx_inv_subspace = self.iso_subspace_ids.index(self.G.trivial_representation.id)
        else:
            self.idx_inv_subspace = None

        # Intialize the NCP module
        super().__init__(embedding_x=embedding_x,
                         embedding_y=embedding_y,
                         embedding_dim=lat_singular_type.size,
                         gamma=gamma,
                         truncated_op_bias=truncated_op_bias)

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

        k_iso = []
        if self.truncated_op_bias == 'Cxy':
            for iso_idx, fx_ci, hy_ci in zip(range(self.n_iso_subspaces), fx_c, hy_c):
                Cxy_i = self.Cxy(iso_idx=iso_idx)
                k_iso.append(torch.einsum('nr,rc,nc->n', fx_ci, Cxy_i, hy_ci))
        elif self.truncated_op_bias == 'diag':
            for iso_idx, fx_ci, hy_ci in zip(range(self.n_iso_subspaces), fx_c, hy_c):
                Cxy_i = self.Cxy(iso_idx=iso_idx)
                k_iso.append(torch.einsum('nr,r,nc->n', fx_ci, torch.diag(Cxy_i), hy_ci))
        elif self.truncated_op_bias == 'svals':
            k_iso.append(torch.einsum('nr,r,nc->n', torch.cat(fx_c, dim=-1), self.svals, torch.cat(hy_c, dim=-1)))
        elif self.truncated_op_bias == "full_rank":
            Dr = self.truncated_operator
            fx_c = torch.cat(fx_c, dim=-1)
            hy_c = torch.cat(hy_c, dim=-1)
            k_r = 1 + torch.einsum('...x,xy,...y->...', fx_c, Dr, hy_c)
            return k_r
        else:
            raise ValueError(f"Invalid truncated operator bias: {self.truncated_op_bias}")

        k_r = 1 + sum(k_iso)
        return k_r

    def orthonormality_penalization(self,
                                    fx_c: GeometricTensor,
                                    hy_c: GeometricTensor,
                                    return_inner_prod=False,
                                    permutation=None):
        """ Computes orthonormality and centering regularization penalization for a batch of feature vectors.

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
            tr_Dx_k = torch.trace(self.__getattr__(f'Dx_{k}'))  # tr(Dx_k)
            tr_Dy_k = torch.trace(self.__getattr__(f'Dy_{k}'))  # tr(Dy_k)
            #  ||Cx_k||_F^2 := |ρk| (||Dx_k||_F^2 - 2tr(Dx_k)) + r_k
            Cx_k_fro_2 = irrep_dim * (Dx_k_fro_2 - 2 * tr_Dx_k) + r_xk
            #  ||Cy_k||_F^2 := |ρk| (||Dy_k||_F^2 - 2tr(Dy_k)) + r_k
            Cy_k_fro_2 = irrep_dim * (Dy_k_fro_2 - 2 * tr_Dy_k) + r_yk
            Cx_iso_fro_2.append(Cx_k_fro_2)
            Cy_iso_fro_2.append(Cy_k_fro_2)
            # Cx_k_fro_2_biased = torch.linalg.matrix_norm(self.Cx(k)) ** 2
            # Cy_k_fro_2_biased = torch.linalg.matrix_norm(self.Cy(k)) ** 2

        Cx_I_err_fro_2 = sum(Cx_iso_fro_2)  # ||Cx - I||_F^2 = Σ_k ||Cx_k - I_r_k||_F^2,
        Cy_I_err_fro_2 = sum(Cy_iso_fro_2)  # ||Cy - I||_F^2 = Σ_k ||Cy_k - I_r_k||_F^2
        # Cx_fro_2_biased = torch.linalg.matrix_norm(self.Cx(None)) ** 2
        # Cy_fro_2_biased = torch.linalg.matrix_norm(self.Cy(None)) ** 2

        # TODO: Unbiased estimation of mean squared
        fx_centering_loss = (self.mean_fx ** 2).sum()  # ||E_p(x) (f(x_i))||^2
        hy_centering_loss = (self.mean_hy ** 2).sum()  # ||E_p(y) (h(y_i))||^2

        # ||Vx - I||_F^2 = ||Cx - I||_F^2 + 2||E_p(x) f(x)||^2
        orthonormality_fx = Cx_I_err_fro_2 + 2 * fx_centering_loss
        # ||Vy - I||_F^2 = ||Cy - I||_F^2 + 2||E_p(y) h(y)||^2
        orthonormality_hy = Cy_I_err_fro_2 + 2 * hy_centering_loss

        with torch.no_grad():
            embedding_dim_x, embedding_dim_y = self.embedding_x.out_type.size, self.embedding_y.out_type.size
            metrics = {f"||Cx||_F^2":     Cx_I_err_fro_2 / embedding_dim_x,
                       f"||mu_x||":       torch.sqrt(fx_centering_loss),
                       f"||Vx - I||_F^2": orthonormality_fx / embedding_dim_x,
                       #
                       f"||Cy||_F^2":     Cy_I_err_fro_2 / embedding_dim_y,
                       f"||mu_y||":       torch.sqrt(hy_centering_loss),
                       f"||Vy - I||_F^2": orthonormality_hy / embedding_dim_y,
                       } | {
                          f"||Cx||_F^2iso/{k}": Cx_k_fro_2 / fx_c_iso[k].shape[-1] for k, Cx_k_fro_2 in
                          enumerate(Cx_iso_fro_2)
                          } | {
                          f"||Cy||_F^2iso/{k}": Cy_k_fro_2 / hy_c_iso[k].shape[-1] for k, Cy_k_fro_2 in
                          enumerate(Cy_iso_fro_2)
                          }

        if return_inner_prod:
            raise NotImplementedError("Inner products not implemented yet.")
        else:
            return orthonormality_fx, orthonormality_hy, metrics

    def unbiased_truncation_error_matrix_form(self, fx_c: GeometricTensor, hy_c: GeometricTensor):
        """ Implementation of ||E - E_r||_HS^2, while assuming E_r is a full matrix.

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
        """
        assert fx_c.type == self.embedding_x.out_type and hy_c.type == self.embedding_y.out_type  # Iso basis.
        metrics = {}
        if self.truncated_op_bias == "Cxy":
            # Project embedding into the isotypic subspaces
            fx_c_iso, reps_Fx_iso = self._orth_proj_isotypic_subspaces(z=fx_c), fx_c.type.representations
            hy_c_iso, reps_Hy_iso = self._orth_proj_isotypic_subspaces(z=hy_c), hy_c.type.representations

            Cxy_iso_fro_2, tr_Pxyx_iso = [], []
            for k, (fx_ci, rep_x_k, hy_ci, rep_y_k) in enumerate(zip(fx_c_iso, reps_Fx_iso, hy_c_iso, reps_Hy_iso)):
                irrep_dim = self.irreps_dim[self.iso_subspace_ids[k]]
                # Flatten the realizations along irreducible subspaces, while preserving sampling from the joint dist.
                zx = isotypic_signal2irreducible_subspaces(fx_ci, rep_x_k)  # (n_samples * |ρ_k|, r_k / |ρ_k|)
                zy = isotypic_signal2irreducible_subspaces(hy_ci, rep_y_k)  # (n_samples * |ρ_k|, r_k / |ρ_k|)
                Dxy_k_fro_2 = cross_cov_norm_squared_unbiased_estimation(x=zx, y=zy)
                #  ||Cxy_k||_F^2 := |ρk| ||Dxy_k||_F^2
                Cxy_k_fro_2 = irrep_dim * Dxy_k_fro_2
                # Cxy_k_fro_2_biased = torch.linalg.matrix_norm(self.Cxy(k)) ** 2
                Cxy_iso_fro_2.append(Cxy_k_fro_2)
                # Trace term: Pxyx_k := Cxy_k Cy_k Cxy_k Cx_k = (Dxy_k Dy_k Dxy_k.T Dx_k) ⊗ I_|ρk|
                Dxy_k, Dx_k, Dy_k = self.__getattr__(f'Dxy_{k}'), self.__getattr__(f'Dx_{k}'), self.__getattr__(f'Dy_{k}')
                Dxyx_k = torch.einsum('ab,bc,cd,de->ae', Dxy_k, Dy_k, Dxy_k, Dx_k)
                # tr(Pxyx_k) := tr(Dxy_k Dy_k Dxy_k.T Dx_k) * |ρk|
                tr_Pxyx_iso.append(torch.trace(Dxyx_k) * irrep_dim)

            Cxy_F_2 = sum(Cxy_iso_fro_2)  # ||Cxy||_F^2 = Σ_k ||Cxy_k||_F^2,
            tr_Pxyx_biased = sum(tr_Pxyx_iso)  # tr(Pxyx) = Σ_k tr(Pxyx_k)
            truncation_err = -2 * Cxy_F_2 + tr_Pxyx_biased

            with torch.no_grad():
                metrics |= {"E_p(x)p(y) k_r(x,y)^2": tr_Pxyx_biased.detach(),
                            "E_p(x,y) k_r(x,y)":     Cxy_F_2.detach(),
                            }
                for k, (Cxy_k_fro_2, tr_Pxyx_k) in enumerate(zip(Cxy_iso_fro_2, tr_Pxyx_iso)):
                    metrics[f"E_p(x,y) k_r(x,y)iso/{k}"] = Cxy_k_fro_2 / self.iso_subspace_dims[k]
                    metrics[f"E_p(x)p(y) k_r(x,y)^2iso/{k}"] = tr_Pxyx_k
                    metrics[f"||k(x,y) - k_r(x,y)||iso/{k}"] = -2 * Cxy_k_fro_2 + tr_Pxyx_k
        elif self.truncated_op_bias == "full_rank":
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
                k_r_i = torch.einsum('nx,xy,my->nm', fx_ci, Dr_i, hy_ci)  # (n_samples, n_samples)
                # pmd_mat = pmd_mat + k_r_i
                E_pxy_kr_i = torch.diag(k_r_i).mean()
                E_pxy_kr_iso.append(E_pxy_kr_i)
                k_r_i2 = k_r_i ** 2
                E_px_py_kr_i = (k_r_i2.sum() - k_r_i2.diag().sum()) / (n_samples * (n_samples - 1))
                E_px_py_kr_iso.append(E_px_py_kr_i)
                truncation_err_iso.append(-2 * E_pxy_kr_i + E_px_py_kr_i)
            # truncated_err = -2 * E_p(x,y)[k_r(x,y)] + E_p(x)p(y)[k_r(x,y)^2]
            E_pxy_kr = sum(E_pxy_kr_iso)
            # E_p(x)p(y)[k_r(x,y)^2]  # Note we remove the samples from the joint in the diagonal
            E_px_py_kr = sum(E_px_py_kr_iso)
            truncation_err = (-2 * E_pxy_kr) + (E_px_py_kr)
            with torch.no_grad():
                metrics |= {"E_p(x)p(y) k_r(x,y)^2": E_px_py_kr.detach(),
                            "E_p(x,y) k_r(x,y)":     E_pxy_kr.detach(),
                            }
                for k in range(self.n_iso_subspaces):
                    metrics[f"||k(x,y) - k_r(x,y)||iso/{k}"] = truncation_err_iso[k]
                    metrics[f"E_p(x,y) k_r(x,y)iso/{k}"] = E_pxy_kr_iso[k]
                    metrics[f"E_p(x)p(y) k_r(x,y)^2iso/{k}"] = E_px_py_kr_iso[k]
        else:
            raise ValueError(f"Invalid truncated operator bias: {self.truncated_op_bias}")

        return truncation_err, metrics

    def unbiased_truncation_error_diag_form(self, fx_c: GeometricTensor, hy_c: GeometricTensor):
        """ Implementation of ||E - E_r||_HS^2, while assuming E_r is diagonal. """
        assert fx_c.type == self.embedding_x.out_type and hy_c.type == self.embedding_y.out_type  # Iso basis.
        # Project embedding into the isotypic subspaces
        fx_c_iso, reps_Fx_iso = self._orth_proj_isotypic_subspaces(z=fx_c), fx_c.type.representations
        hy_c_iso, reps_Hy_iso = self._orth_proj_isotypic_subspaces(z=hy_c), hy_c.type.representations

        # Basis of singular functions and isotypic basis have the same symmetry group reoresentation, hence:
        use_expectations = self.truncated_op_bias == 'diag' or self.truncated_op_bias == 'Cxy'
        # Singular vectors with symmetry constrained multiplicities.
        Dr = self.svals if not use_expectations else torch.diag(self.Cxy())
        Dr_iso = [Dr[s:e] for s, e in zip(fx_c.type.fields_start, fx_c.type.fields_end)]

        tr_CxSCyS_iso, tr_CxyS_iso = [], []
        for k, (fx_ck, hy_ck, Dr_k) in enumerate(zip(fx_c_iso, hy_c_iso, Dr_iso)):
            irrep_dim = self.irreps_dim[self.iso_subspace_ids[k]]
            # Tr (Cx Σ Cy Σ) = Σ_k tr(Cx_k Dr_k Cy_k Dr_k) * |ρ_k|
            tr_CxSCyS = torch.einsum("ik, il, k, jl, jk, l->", fx_ck, fx_ck, Dr_k, hy_ck, hy_ck, Dr_k) / (
                    fx_ck.shape[0] - 1) ** 2
            # Tr(Cxy Σ) = Σ_k tr(Cxy_k Dr_k)
            tr_CxyS = torch.einsum("ik, ik, k->", fx_ck, hy_ck, Dr_k) / (fx_ck.shape[0] - 1)
            tr_CxSCyS_iso.append(tr_CxSCyS)
            tr_CxyS_iso.append(tr_CxyS)

        tr_CxyS = sum(tr_CxyS_iso)  # tr(Cxy Σ) = Σ_k tr(Cxy_k Dr_k)
        tr_CxSCyS = sum(tr_CxSCyS_iso)  # tr(Cx Σ Cy Σ) = Σ_k tr(Cx_k Dr_k Cy_k Dr_k)

        truncation_err = -2 * tr_CxyS + tr_CxSCyS
        with torch.no_grad():
            metrics = {f"||E - diag(E_r)||_HSiso/{k}": -2 * A + B for k, (A, B) in
                       enumerate(zip(tr_CxyS_iso, tr_CxSCyS_iso))}
        # L = -2 tr (Cxy Σ) + tr (Cx Σ Cy Σ)
        return truncation_err, metrics

    def update_fns_statistics(self, fx: GeometricTensor, hy: GeometricTensor):
        assert isinstance(fx, GeometricTensor) and isinstance(hy, GeometricTensor), \
            f"Expected Geometric Tensors got f(x): {type(fx)} and h(y): {type(hy)}"
        assert fx.type == self.embedding_x.out_type and hy.type == self.embedding_y.out_type
        device, dtype = fx.tensor.device, fx.tensor.dtype

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
            _, Dxy_k = isotypic_cross_cov(X=fx_ck, Y=hy_ck, rep_X=rep_x_k, rep_Y=rep_y_k, centered=True)
            _, Dx_k = isotypic_cross_cov(X=fx_ck, Y=fx_ck, rep_X=rep_x_k, rep_Y=rep_x_k, centered=True)
            _, Dy_k = isotypic_cross_cov(X=hy_ck, Y=hy_ck, rep_X=rep_y_k, rep_Y=rep_y_k, centered=True)
            setattr(self, f'Dxy_{k}', Dxy_k)
            setattr(self, f'Dx_{k}', Dx_k)
            setattr(self, f'Dy_{k}', Dy_k)

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
    def svals(self):
        """Ensures the multiplicities of singular values required to satisfy the equivariance constraint.

        Each singular space, can be thought of being associated with an instance of an irrep of the group. The
        dimensionality of the space is hence the dimensionality of the irrep, which implies that the singular values
        have multiplicities equal to the dimensionality of the irrep.

        Returns:
            The singular values in the form of a tensor.
        """
        unique_svals = torch.exp(-self.log_svals ** 2)
        return unique_svals.repeat_interleave(repeats=self.sval_multiplicities.to(unique_svals.device))

    @property
    def truncated_operator(self):
        # Compute the kernel function
        if self.truncated_op_bias == "svals":
            svals = self.svals
            return torch.diag(svals)
        elif self.truncated_op_bias == "full_rank":
            # D_r is diagonal and is stable (that is has eivalues <= 1)
            D_r, _ = self._Dr.expand_parameters()  # Expand the equiv lin layer into its matrix form
            a = D_r.detach().cpu().numpy()
            Dr_symm = (D_r @ D_r.T ) / 2 # Ensure its symmetric.
            eigval_max = torch.linalg.eigvalsh(Dr_symm)[-1]
            Dr = Dr_symm / eigval_max
            return Dr
        elif self.truncated_op_bias == "diag":
            return torch.diag(torch.diag(self.Cxy()))
        elif self.truncated_op_bias == "Cxy":
            return self.Cxy()
        else:
            raise ValueError(f"Unknown truncated operator bias: {self.truncated_op_bias}.")

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
            assert 0 <= iso_idx < self.n_iso_subspaces, \
                f"Invalid isotypic subspace index {iso_idx} !in [0,{self.n_iso_subspaces}]"
            irrep_dim = self.irreps_dim[self.iso_subspace_ids[iso_idx]]
            Dxy_k = self.__getattr__(f"Dxy_{iso_idx}")
            return torch.kron(Dxy_k, torch.eye(irrep_dim, dtype=Dxy_k.dtype, device=Dxy_k.device))

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
            assert 0 <= iso_idx < self.n_iso_subspaces, \
                f"Invalid isotypic subspace index {iso_idx} !in [0,{self.n_iso_subspaces}]"
            irrep_dim = self.irreps_dim[self.iso_subspace_ids[iso_idx]]
            Dx_k = self.__getattr__(f"Dx_{iso_idx}")
            return torch.kron(Dx_k, torch.eye(irrep_dim, dtype=Dx_k.dtype, device=Dx_k.device))

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
            assert 0 <= iso_idx < self.n_iso_subspaces, \
                f"Invalid isotypic subspace index {iso_idx} !in [0,{self.n_iso_subspaces}]"
            irrep_dim = self.irreps_dim[self.iso_subspace_ids[iso_idx]]
            Dy_k = self.__getattr__(f"Dy_{iso_idx}")
            return torch.kron(Dy_k, torch.eye(irrep_dim, dtype=Dy_k.dtype, device=Dy_k.device))

    def _orth_proj_isotypic_subspaces(self, z: GeometricTensor) -> [torch.Tensor]:
        """Compute the orthogonal projection of the input tensor into the isotypic subspaces."""

        z_iso = [z.tensor[..., s:e] for s, e in zip(z.type.fields_start, z.type.fields_end)]
        return z_iso

    def _process_truncated_op_bias(self):
        truncated_op_bias = self.truncated_op_bias
        embedding_dim = self.embedding_dim
        lat_singular_type = self.embedding_x.out_type

        self._is_truncated_op_diag = False
        # Diagonal Biases
        if truncated_op_bias == 'svals':
            # Store the sval trainable parameters / degrees of freedom (dof)
            num_sval_dof = np.sum(self.iso_irreps_multiplicities)  # There is one sval per irrep
            assert num_sval_dof == len(lat_singular_type.irreps), f"{num_sval_dof} != {len(lat_singular_type.irreps)}"
            self.log_svals = torch.nn.Parameter(
                torch.normal(mean=0., std=2. / num_sval_dof, size=(num_sval_dof,)), requires_grad=True
                )
            # vector storing the multiplicity of each singular value
            self.sval_multiplicities = torch.tensor(
                [self.irreps_dim[irrep_id] for irrep_id in lat_singular_type.irreps]
                )
            self._is_truncated_op_diag = True
        elif truncated_op_bias == "diag":
            self._is_truncated_op_diag = True
            pass  # No parameters to define
        # Full rank biases
        elif truncated_op_bias == "full_rank":
            # Equivariant Linear layer from lat singular basis to lat singular basis.
            self._Dr = escnn.nn.Linear(in_type=lat_singular_type, out_type=lat_singular_type, bias=False)
        elif truncated_op_bias == "Cxy":
            pass  # No parameters to define
        else:
            raise ValueError(f"Unknown truncated operator bias: {truncated_op_bias}.")

    def _register_stats_buffers(self):
        """Register the buffers for the running mean, Covariance and Cross-Covariance matrix matrix."""

        if self.idx_inv_subspace is not None:
            dim_x_inv_subspace = self.embedding_x.out_type.representations[self.idx_inv_subspace].size
            dim_y_inv_subspace = self.embedding_y.out_type.representations[self.idx_inv_subspace].size
            self.register_buffer('mean_fx', torch.zeros((1, dim_x_inv_subspace)))
            self.register_buffer('mean_hy', torch.zeros((1, dim_y_inv_subspace)))

        for iso_idx, iso_id, iso_subspace_dim in zip(
                range(self.n_iso_subspaces), self.iso_subspace_ids, self.iso_subspace_dims
                ):
            irrep_dim = self.irreps_dim[iso_id]  # |ρ_k|  Dimension of the irrep
            effective_dim = int(iso_subspace_dim // irrep_dim)
            # Matrix containing the DoF of Cxy^(k) = Dxy ⊗ Iρ_k: L^2(Y^(k)) -> L^2(X^(k))
            self.register_buffer(f'Dxy_{iso_idx}', torch.zeros(effective_dim, effective_dim))
            # Matrix containing the DoF of Cx^(k) = Dx ⊗ Iρ_k: L^2(X^(k)) -> L^2(X^(k))
            self.register_buffer(f'Dx_{iso_idx}', torch.zeros(effective_dim, effective_dim))
            # Matrix containing the DoF of Cy^(k) = Dy ⊗ Iρ_k: L^2(Y^(k)) -> L^2(Y^(k))
            self.register_buffer(f'Dy_{iso_idx}', torch.zeros(effective_dim, effective_dim))


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
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from NCP.models.ncp_lightning_module import TrainingModule

    light_module = TrainingModule(model, optimizer_fn=torch.optim.Adam, optimizer_kwargs=dict(lr=1e-3),
                                  loss_fn=model.loss)
    trainer = lightning.Trainer(max_epochs=50)

    torch.set_float32_matmul_precision('medium')
    trainer.fit(light_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
