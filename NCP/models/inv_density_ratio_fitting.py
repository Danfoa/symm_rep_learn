# Created by danfoa at 16/01/25
import escnn
import torch.nn
from escnn.group import Representation
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from NCP.models.density_ratio_fitting import DRF
from NCP.nn.layers import Lambda


# Density Ratio Fitting.
class InvDRF(DRF):

    def __init__(self, embedding: EquivariantModule, gamma: float = 0.01):

        assert embedding.out_type.size == 1, "The output of the embedding must be a scalar."
        self.in_type = embedding.in_type
        self.pmd_type = embedding.out_type
        rep_pmd: Representation = self.pmd_type.representation
        assert len(rep_pmd._irreps_multiplicities) == 1
        assert rep_pmd.irreps[0] == rep_pmd.group.trivial_representation.id, \
            f"The embedding must provide a {rep_pmd.group} invariant scalar output."

        # Remove the Geometric tensor wrapper before using the methods of DRF
        torch_embedding = torch.nn.Sequential(
            Lambda(lambda x: self._tensor2geom_tensor(x)),  # Ensure input type is rep_x , rep_y
            embedding,
            Lambda(lambda x: self._geom_tensor2tensor(x))
            )

        super().__init__(embedding=torch_embedding, gamma=gamma)

    def _geom_tensor2tensor(self, x: GeometricTensor):
        assert x.type == self.pmd_type, f"Expected GeometricTensor of type {self.pmd_type}, got {x.type}"
        return x.tensor

    def _tensor2geom_tensor(self, x: torch.Tensor):
        return self.in_type(x)


if __name__ == "__main__":
    from NCP.nn.equiv_layers import IMLP

    G = escnn.group.DihedralGroup(6)
    x_rep = G.regular_representation  # ρ_Χ
    y_rep = G.regular_representation  # ρ_Y

    xy_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[x_rep, y_rep])

    imlp = IMLP(in_type=xy_type, out_dim=1, hidden_layers=5, hidden_units=128, bias=False)
    idrf = InvDRF(embedding=imlp)

    x = torch.randn(10, x_rep.size)
    y = torch.randn(10, y_rep.size)
    pmd_mat = idrf(x=x, y=y)

    assert pmd_mat.size() == (10, 10)
