from typing import Tuple

import escnn.nn
from escnn.nn import FieldType, GeometricTensor
from symm_learning.models import EMLP

from .cqr import cqr_loss


class eCQR(escnn.nn.EquivariantModule):
    def __init__(self, in_type: FieldType, out_type: FieldType, gamma: float, **mlp_kwargs):
        super(eCQR, self).__init__()
        assert 0 < gamma <= 1, "gamma must be in (0, 1]"
        self.in_type = in_type
        self.out_type = out_type
        self.low_q_nn = EMLP(in_type=in_type, out_type=out_type, **mlp_kwargs)
        self.up_q_nn = EMLP(in_type=in_type, out_type=out_type, **mlp_kwargs)
        self.gamma = gamma

    def forward(self, x: GeometricTensor):
        low_q = self.low_q_nn(x)
        up_q = self.up_q_nn(x)

        return low_q, up_q

    def loss(self, loq_q, up_q, target):
        return cqr_loss(loq_q.tensor, up_q.tensor, target.tensor, self.gamma)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (self.out_type.size,)
