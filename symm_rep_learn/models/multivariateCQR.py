from typing import Tuple

import escnn.nn
import torch.nn
from escnn.nn import FieldType, GeometricTensor

from symm_rep_learn.nn.equiv_layers import EMLP
from symm_rep_learn.nn.layers import MLP


class MultivariateCQR(torch.nn.Module):
    def __init__(self, dim_x: int, dim_y: int, gamma: float, **mlp_kwargs):
        super(MultivariateCQR, self).__init__()
        assert 0 < gamma <= 1, "gamma must be in (0, 1]"
        self.low_q_nn = MLP(input_shape=dim_x, output_shape=dim_y, **mlp_kwargs)
        self.up_q_nn = MLP(input_shape=dim_x, output_shape=dim_y, **mlp_kwargs)
        self.gamma = gamma

    def forward(self, x: torch.Tensor):
        low_q = self.low_q_nn(x)
        up_q = self.up_q_nn(x)

        return low_q, up_q

    def loss(self, loq_q, up_q, target):
        return cqr_loss(loq_q, up_q, target, self.gamma)


class EquivMultivariateCQR(escnn.nn.EquivariantModule):
    def __init__(self, in_type: FieldType, out_type: FieldType, gamma: float, **mlp_kwargs):
        super(EquivMultivariateCQR, self).__init__()
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


def cqr_loss(loq_q, up_q, target, gamma):
    loss_lo = pinball_loss(input=loq_q, target=target, gamma=gamma / 2)
    loss_up = pinball_loss(input=up_q, target=target, gamma=1 - (gamma / 2))

    metrics = {}
    with torch.no_grad():
        metrics["coverage"] = get_coverage(loq_q, up_q, target)
        metrics["relaxed_coverage"] = get_relaxed_coverage(loq_q, up_q, target)
        metrics["set_size"] = get_set_size(loq_q, up_q)

    loss = loss_lo + loss_up
    return loss, metrics


def get_coverage(loq_q, up_q, target):
    coverage = (
        ((loq_q <= target).all(dim=1, keepdim=True) & (target <= up_q).all(dim=1, keepdim=True)).type_as(target).mean()
    )
    return coverage


def get_relaxed_coverage(loq_q, up_q, target):
    coverage = ((loq_q <= target) & (target <= up_q)).type_as(target).float().mean(dim=1).mean()
    return coverage


def get_set_size(loq_q, up_q):
    set_size = (up_q - loq_q).abs().prod(dim=1, keepdim=True).mean()
    return set_size


def pinball_loss(input: torch.Tensor, target: torch.Tensor, gamma: float):
    """Pinball loss of https://arxiv.org/pdf/2107.07511 (page 8) to learn (gamma/2)-quantiles.

    Args:
        input: Model output.
        target: Ground truth.
        gamma: Quantile to regress.

    """
    errors = target - input
    loss = torch.maximum(errors * gamma, -errors * (1 - gamma)).mean()
    return loss
