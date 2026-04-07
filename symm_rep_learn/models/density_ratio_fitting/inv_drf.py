# Created  at 16/01/25
import escnn
import torch.nn
from escnn.group import Representation

from symm_rep_learn.models.density_ratio_fitting import DRF
from symm_rep_learn.nn.layers import Lambda

from symm_learning.representation_theory import direct_sum

# Density Ratio Fitting.
class InvDRF(DRF):
    def __init__(self, embedding: torch.nn.Module, gamma: float = 0.01):
        out_rep = getattr(embedding, "out_rep", None)
        out_type = getattr(embedding, "out_type", None)
        out_size = None
        if out_rep is not None:
            out_size = out_rep.size
        elif out_type is not None:
            out_size = out_type.size
        assert out_size == 1, "The output of the embedding must be a scalar."
        # TODO: Assert embedding is invariant.
        super().__init__(embedding=embedding, gamma=gamma)



if __name__ == "__main__":
    from symm_learning.models import iMLP

    G = escnn.group.DihedralGroup(6)
    x_rep = G.regular_representation  # ρ_Χ
    y_rep = G.regular_representation  # ρ_Y
    xy_rep = direct_sum([x_rep, y_rep], name="xy_rep")

    imlp = iMLP(in_rep=xy_rep, out_dim=1, hidden_layers=5, hidden_units=128, bias=False)
    idrf = InvDRF(embedding=imlp)

    x = torch.randn(10, x_rep.size)
    y = torch.randn(10, y_rep.size)
    pmd_mat = idrf(x=x, y=y)

    assert pmd_mat.size() == (10, 10)
