from typing import Tuple

import torch
import escnn.nn
from escnn.group import Representation
from symm_learning.models import eMLP

from .cqr import cqr_loss


class eCQR(torch.nn.Module):
    def __init__(self, in_rep: Representation, out_rep: Representation, gamma: float, **mlp_kwargs):
        super(eCQR, self).__init__()
        assert 0 < gamma <= 1, "gamma must be in (0, 1]"
        self.low_q_nn = eMLP(in_rep, out_rep, **mlp_kwargs)
        self.up_q_nn = eMLP(in_rep, out_rep, **mlp_kwargs)
        self.gamma = gamma

    def forward(self, x: torch.Tensor):
        low_q = self.low_q_nn(x)
        up_q = self.up_q_nn(x)

        return low_q, up_q

    def loss(self, loq_q, up_q, target):
        return cqr_loss(loq_q.tensor, up_q.tensor, target.tensor, self.gamma)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (self.out_type.size,)
