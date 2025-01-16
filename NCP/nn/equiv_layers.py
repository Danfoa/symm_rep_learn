# Created by danfoa at 26/12/24
from __future__ import annotations

from math import ceil
from typing import Tuple

import escnn
import torch
from escnn.nn import EquivariantModule, FieldType, FourierPointwise, GeometricTensor

from NCP.mysc.rep_theory_utils import isotypic_decomp_rep


class EMLP(EquivariantModule):

    def __init__(
            self,
            in_type: FieldType,
            out_type: FieldType,
            hidden_layers: int = 1,
            hidden_units: int = 128,
            activation: str = "ReLU",
            bias: bool = True,
            hidden_irreps: list | tuple = None
            ):
        super(EMLP, self).__init__()
        assert hidden_layers > 0, "A MLP with 0 hidden layers is equivalent to a linear layer"
        self.G = in_type.fibergroup
        self.in_type, self.out_type = in_type, out_type

        hidden_irreps = hidden_irreps or self.G.regular_representation.irreps
        hidden_irreps = set(hidden_irreps)
        signal_dim = sum(self.G.irrep(*id).size for id in hidden_irreps)
        # Number of multiplicities / signals in the hidden layers
        channels = int(ceil(hidden_units // signal_dim))

        layers = []
        layer_in_type = in_type
        for i in range(hidden_layers):
            layer = FourierBlock(
                in_type=layer_in_type, irreps=hidden_irreps, channels=channels, activation=activation, bias=bias
                )
            layer_in_type = layer.out_type
            layers.append(layer)

        # Head layer
        layers.append(escnn.nn.Linear(in_type=layer_in_type, out_type=out_type, bias=bias))
        self.net = escnn.nn.SequentialModule(*layers)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.net(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.net.evaluate_output_shape(input_shape)

    def extra_repr(self) -> str:
        return f"{self.G}-equivariant MLP: in={self.in_type}, out={self.out_type}"

class FourierBlock(EquivariantModule):

    def __init__(self,
                 in_type: FieldType,
                 irreps: tuple | list,
                 channels: int,
                 activation: str,
                 bias: bool = True,
                 grid_kwargs: dict=None):
        super(FourierBlock, self).__init__()
        G = in_type.fibergroup
        gspace = in_type.gspace
        grid_kwargs = grid_kwargs or self.get_group_kwargs(G)

        self.act = FourierPointwise(gspace,
                               channels=channels,
                               irreps=list(irreps),
                               function=f"p_{activation.lower()}",
                               inplace=True,
                               **grid_kwargs)

        self.in_type = in_type
        self.out_type = self.act.in_type
        self.linear = escnn.nn.Linear(in_type=in_type, out_type=self.act.in_type, bias=bias)
        self.model = escnn.nn.SequentialModule(self.linear, self.act)

    def forward(self, *input):
        return self.model(*input)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.linear.evaluate_output_shape(input_shape)

    @staticmethod
    def get_group_kwargs(group: escnn.group.Group):
        grid_type = 'regular' if not group.continuous else 'rand'
        N = group.order() if not group.continuous else 10
        kwargs = dict()

        if isinstance(group, escnn.group.DihedralGroup):
            N = N // 2
        elif isinstance(group, escnn.group.DirectProductGroup):
            G1_args = FourierBlock.get_group_kwargs(group.G1)
            G2_args = FourierBlock.get_group_kwargs(group.G2)
            kwargs.update({f"G1_{k}": v for k, v in G1_args.items()})
            kwargs.update({f"G2_{k}": v for k, v in G2_args.items()})

        return dict(N=N, type=grid_type, **kwargs)



if __name__ == "__main__":

    G = escnn.group.DihedralGroup(6)
    x_rep = G.representations['irrep_1,1']              # ρ_Χ
    # G = escnn.group.CyclicGroup(2)
    # x_rep = G.representations['irrep_1']  # ρ_Χ
    y_rep = isotypic_decomp_rep(G.regular_representation)     # ρ_Y
    y_iso_reps = tuple(y_rep.attributes['isotypic_reps'].values())

    type_X = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[x_rep])
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=y_iso_reps)

    emlp = EMLP(in_type=type_X, out_type=type_Y, hidden_layers=2, hidden_units=64, bias=False)
    emlp.check_equivariance()
    print(emlp)

    n_samples = 5
    x = GeometricTensor(torch.randn(n_samples, x_rep.size), type_X)
    y = emlp(x)

    y_iso = [y.tensor[..., s:e] for s, e in zip(y.type.fields_start, y.type.fields_end)]

    for irrep_id, y_p in zip(y_rep.attributes['isotypic_reps'].keys(), y_iso):
        print(f"{irrep_id}: \n{torch.linalg.norm(y_p)}")
