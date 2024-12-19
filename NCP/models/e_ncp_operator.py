# Created by danfoa at 19/12/24
import escnn.nn
import numpy as np
import torch
from escnn.group import directsum, Representation

from NCP.model import NCPOperator
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from NCP.mysc.rep_theory_utils import field_type_to_isotypic_basis, isotypic_decomp_rep
from NCP.nn.layers import SingularLayer


class eNCPOperator(torch.nn.Module):

    def __init__(self, x_fns: EquivariantModule, y_fns: EquivariantModule):
        super(eNCPOperator, self).__init__()

        # Get field type in the singular-isotypic basis
        x_singular_type = field_type_to_isotypic_basis(x_fns.out_type)
        y_singular_type = field_type_to_isotypic_basis(y_fns.out_type)

        # Take any input field-type, add a G-equivariant linear layer, parameterizing a change of basis to the
        # Iso-singular basis. (singular functions clustered by isotypic subspaces)
        x2singular = escnn.nn.Linear(in_type=x_fns.out_type, out_type=x_singular_type)
        y2singular = escnn.nn.Linear(in_type=y_fns.out_type, out_type=y_singular_type)
        self.x_singular_fns = escnn.nn.SequentialModule(x_fns, x2singular)
        self.y_singular_fns = escnn.nn.SequentialModule(y_fns, y2singular)
        self.G = self.x_singular_fns.in_type.fibergroup

        assert x_singular_type.size == y_singular_type.size, f"Fn spaces of diff dimensionality not yet supported"
        # Isotypic subspace are identified by the irrep id associated with the subspace
        self.iso_subspaces_id = [iso_rep.irreps[0] for iso_rep in y_singular_type.representations]
        self.iso_subspaces_dim = [iso_rep.size for iso_rep in y_singular_type.representations]
        self.irreps_dim = {irrep_id: self.G.irrep(*irrep_id).size for irrep_id in self.iso_subspaces_id}
        self.iso_subspace_irrep_dim = [self.irreps_dim[id] for id in self.iso_subspaces_id]  # For completeness
        self.iso_irreps_multiplicities = [
            space_dim // self.irreps_dim[id] for space_dim, id in zip(self.iso_subspaces_dim, self.iso_subspaces_id)
            ]
        self.has_inv_subspace = self.G.trivial_representation.id in self.iso_subspaces_id

        # Store the sval trainable parameters / degrees of freedom (dof)
        num_sval_dof = np.sum(self.iso_irreps_multiplicities)  # There is one sval per irrep
        assert num_sval_dof == len(y_singular_type.irreps), f"{num_sval_dof} != {len(y_singular_type.irreps)}"
        # TODO: Enable different initializations for this parameter
        self.sval_dof = torch.nn.Parameter(
            torch.Tensor(torch.normal(mean=0, std=2. / y_type.size, size=(num_sval_dof,))), requires_grad=True
            )
        # vector storing the multiplicity of each singular value
        self.sval_multiplicities = torch.tensor(
            [self.irreps_dim[irrep_id] for irrep_id in y_singular_type.irreps]
            )
        # TODO: Buffers for centering and whitening

    def compute_svals(self):
        """ Ensures the multiplicities of singular values required to satisfy the equivariance constraint.

        Each singular space, can be thought of being associated with an instance of an irrep of the group. The
        dimensionality of the space is hence the dimensionality of the irrep, which implies that the singular values
        have multiplicities equal to the dimensionality of the irrep.

        Returns:
            The singular values in the form of a tensor.
        """
        unique_svals = torch.exp(-self.sval_dof ** 2)
        svals = unique_svals.repeat_interleave(repeats=self.sval_multiplicities.to(unique_svals.device))
        return svals


    def forward(self, x: GeometricTensor, y: GeometricTensor, postprocess=None):
        sing_fn_x = self.x_singular_fns(x)
        sing_fn_y = self.y_singular_fns(y)
        svals = self.compute_svals().to(sing_fn_x.tensor.device)

        # k(x, y) = 1 + Σ_i=1^r σ_i f_i(x) h_i(y)
        # Einsum can do this operation faster in GPU with some internal optimizations.
        k_centered = torch.einsum('i,...i,...i->...', svals, sing_fn_x.tensor, sing_fn_y.tensor)
        k = 1 + k_centered
        return k





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

    print("Done")


