# Created by danfoa at 19/12/24
import escnn.nn
import numpy as np
import torch
from escnn.group import directsum
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from NCP.mysc.rep_theory_utils import field_type_to_isotypic_basis
from NCP.mysc.symm_algebra import isotypic_cross_cov
from NCP.nn.layers import SingularLayer


class eNCPOperator(torch.nn.Module):

    def __init__(self, x_fns: EquivariantModule, y_fns: EquivariantModule, gamma=0.01):
        super(eNCPOperator, self).__init__()
        self.gamma = 0.01
        # Get field type in the singular-isotypic basis
        x_singular_type = field_type_to_isotypic_basis(x_fns.out_type)
        y_singular_type = field_type_to_isotypic_basis(y_fns.out_type)

        # Take any input field-type, add a G-equivariant linear layer, parameterizing a change of basis to the
        # Iso-singular basis. (singular functions clustered by isotypic subspaces)
        x2singular = escnn.nn.Linear(in_type=x_fns.out_type, out_type=x_singular_type)
        y2singular = escnn.nn.Linear(in_type=y_fns.out_type, out_type=y_singular_type)
        self.singular_fns_x = escnn.nn.SequentialModule(x_fns, x2singular)
        self.singular_fns_y = escnn.nn.SequentialModule(y_fns, y2singular)
        self.G = self.singular_fns_x.in_type.fibergroup

        assert x_singular_type.size == y_singular_type.size, "Fn spaces of diff dimensionality not yet supported"
        # Isotypic subspace are identified by the irrep id associated with the subspace
        self.iso_subspaces_id = [iso_rep.irreps[0] for iso_rep in y_singular_type.representations]
        self.iso_subspaces_dim = [iso_rep.size for iso_rep in y_singular_type.representations]
        self.irreps_dim = {irrep_id: self.G.irrep(*irrep_id).size for irrep_id in self.iso_subspaces_id}
        self.iso_subspace_irrep_dim = [self.irreps_dim[id] for id in self.iso_subspaces_id]  # For completeness
        self.iso_irreps_multiplicities = [
            space_dim // self.irreps_dim[id] for space_dim, id in zip(self.iso_subspaces_dim, self.iso_subspaces_id)
            ]
        if self.G.trivial_representation.id in self.iso_subspaces_id:
            self.idx_inv_subspace = self.iso_subspaces_id.index(self.G.trivial_representation.id)
        else:
            self.idx_inv_subspace = None

        # Store the sval trainable parameters / degrees of freedom (dof)
        num_sval_dof = np.sum(self.iso_irreps_multiplicities)  # There is one sval per irrep
        assert num_sval_dof == len(y_singular_type.irreps), f"{num_sval_dof} != {len(y_singular_type.irreps)}"
        # TODO: Enable different initializations for this parameter
        self.sval_dof = SingularLayer(num_sval_dof)
        # vector storing the multiplicity of each singular value
        self.sval_multiplicities = torch.tensor(
            [self.irreps_dim[irrep_id] for irrep_id in y_singular_type.irreps]
            )
        # TODO: Buffers for centering and whitening

    @property
    def svals(self):
        """Ensures the multiplicities of singular values required to satisfy the equivariance constraint.

        Each singular space, can be thought of being associated with an instance of an irrep of the group. The
        dimensionality of the space is hence the dimensionality of the irrep, which implies that the singular values
        have multiplicities equal to the dimensionality of the irrep.

        Returns:
            The singular values in the form of a tensor.
        """
        unique_svals = self.sval_dof.svals
        return unique_svals.repeat_interleave(repeats=self.sval_multiplicities.to(unique_svals.device))


    def forward(self, x: GeometricTensor, y: GeometricTensor):
        """ Forward pass of the eNCP operator.

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
        fx = self.singular_fns_x(x)  # f(x) = [f_1(x), ..., f_r(x)]
        hy = self.singular_fns_y(y)  # h(y) = [h_1(y), ..., h_r(y)]
        return fx, hy

    def exp_mutual_information(self, svals: torch.Tensor, fx: torch.Tensor, hy: torch.Tensor):
        # k(x, y) = 1 + Σ_i=1^r σ_i f_i(x) h_i(y)
        # Einsum can do this operation faster in GPU with some internal optimizations.
        k_centered = torch.einsum('i,...i,...i->...', svals, fx.tensor, hy.tensor)
        k = 1 + k_centered
        return k

    def loss(self, x: GeometricTensor, y: GeometricTensor):
        device, dtype = x.tensor.device, x.tensor.dtype
        _tensor_kwargs = dict(device=device, dtype=dtype)

        sqrt_svals = torch.sqrt(self.svals)

        fx, hy = self(x, y)
        # Center functions by computing the empirical mean only from the G-invariant subspace
        fx, mean_fx = self.center_fns(fx)
        hy, mean_hy = self.center_fns(hy)
        # Multiply by the square root of the singular values
        fx = GeometricTensor(fx.tensor * sqrt_svals, fx.type)
        hy = GeometricTensor(hy.tensor * sqrt_svals, hy.type)

        # Get projections into isotypic subspaces.  fx_iso[k] = fx^(k), hy_iso[k] = hy^(k)
        fx_iso = [fx.tensor[..., s:e] for s, e in zip(fx.type.fields_start, fx.type.fields_end)]
        reps_Fx_iso = fx.type.representations
        hy_iso = [fx.tensor[..., s:e] for s, e in zip(hy.type.fields_start, hy.type.fields_end)]
        reps_Hy_iso = hy.type.representations
        sqrt_svals_iso = [sqrt_svals[s:e] for s, e in zip(fx.type.fields_start, fx.type.fields_end)]

        loss_iso, Cx_iso, Cy_iso, Cxy_iso = [], [], [], []
        orth_iso_x, orth_iso_y, cent_iso_x, cent_iso_y = [], [], [], []
        for fx_k, hy_k, rep_x_k, rep_y_k, sqrt_sval_k in zip(fx_iso, hy_iso, reps_Fx_iso, reps_Hy_iso, sqrt_svals_iso):
            Cx_k = isotypic_cross_cov(X=fx_k, Y=fx_k, rep_X=rep_x_k, rep_Y=rep_x_k, centered=False)
            Cy_k = isotypic_cross_cov(X=hy_k, Y=hy_k, rep_X=rep_y_k, rep_Y=rep_y_k, centered=False)
            Cxy_k = isotypic_cross_cov(X=fx_k, Y=hy_k, rep_X=rep_x_k, rep_Y=rep_y_k, centered=False)
            Ix, Iy = torch.eye(Cx_k.shape[0], **_tensor_kwargs), torch.eye(Cy_k.shape[0], **_tensor_kwargs)
            orth_x_k = torch.linalg.norm(Cx_k - Ix, ord='fro') ** 2  # ||C_x - I||_F^2
            orth_y_k = torch.linalg.norm(Cy_k - Iy, ord='fro') ** 2  # ||C_y - I||_F^2
            loss_k = torch.trace((Cx_k @ Cy_k) - 2 * Cxy_k)         # tr(C_x C_y - 2 C_xy)
            Cx_iso.append(Cx_k), Cy_iso.append(Cy_k), Cxy_iso.append(Cxy_k)
            orth_iso_x.append(orth_x_k), orth_iso_y.append(orth_y_k), loss_iso.append(loss_k)

        mi_loss = sum(loss_iso)                                 # tr(block_diag(C1,...Cn)) = Σ_i tr(C_i)
        orth_reg = sum(orth_iso_x) + sum(orth_iso_y)            # ||block_diag(C1,...Cn) - I||_F^2 = Σ_i ||C_i - I||_F^2
        center_reg = mean_fx.norm() ** 2 + mean_hy.norm() ** 2    # Centering regularization
        loss = mi_loss + self.gamma * (orth_reg + 2*center_reg)
        metrics = dict(mi_loss=mi_loss.detach(), orth_reg=orth_reg.detach(), center_reg=center_reg.detach())
        return loss, metrics


    def center_fns(self, f: GeometricTensor):
        """Centers the functions by removing the mean of their G-invariant components.

        Args:
            f: (GeometricTensor) The functions to be centered of shape (..., d), field_type assumed to be in the
            isotypic basis.

        Returns:
            f: (GeometricTensor) (..., d) The centered functions.
            mean_f: (torch.Tensor) (d,) The mean of the functions.
        """
        mean_f = torch.zeros((f.shape[-1]), device=f.tensor.device, dtype=f.tensor.dtype)

        if self.idx_inv_subspace is not None:
            inv_subspace_start = f.type.fields_start[self.idx_inv_subspace]
            inv_subspace_end = f.type.fields_end[self.idx_inv_subspace]
            f_inv = f.tensor[..., inv_subspace_start:inv_subspace_end]
            # Compute the mean using the batch dimension as samples
            mean_f_inv = f_inv.mean(dim=0)
            f.tensor[..., inv_subspace_start:inv_subspace_end] = f_inv - mean_f_inv
            mean_f[inv_subspace_start:inv_subspace_end] = mean_f_inv

        return f, mean_f




if __name__ == "__main__":

    G = escnn.group.Icosahedral()

    x_rep = G.regular_representation                      # ρ_Χ
    y_rep = directsum([G.regular_representation] * 2)     # ρ_Y
    lat_rep = directsum([G.regular_representation] * 4)   # ρ_Ζ

    x_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[x_rep])
    y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[y_rep])
    lat_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[lat_rep])

    χ_embedding = escnn.nn.Linear(x_type, lat_type)
    y_embedding = escnn.nn.Linear(y_type, lat_type)

    model = eNCPOperator(χ_embedding, y_embedding)

    x = torch.randn(10, x_rep.size)
    y = torch.randn(10, y_rep.size)
    x = GeometricTensor(x, x_type)
    y = GeometricTensor(y, y_type)
    k = model(x, y)
    model.loss(x, y)
    print("Done")


