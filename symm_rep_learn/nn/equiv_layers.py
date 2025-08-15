# Created by danfoa at 26/12/24
from __future__ import annotations

from typing import Tuple

import torch
from escnn.nn import EquivariantModule, FieldType, GeometricTensor


class ResidualEncoder(EquivariantModule):
    """Residual encoder for symm_rep_learn. This encoder processes batches of shape (batch_size, dim_y) and
    returns (batch_size, embedding_dim + dim_y).
    """

    def __init__(self, encoder: EquivariantModule, in_type: FieldType):
        super(ResidualEncoder, self).__init__()
        self.encoder = encoder
        self.in_type = in_type
        self.out_type = FieldType(
            gspace=encoder.out_type.gspace, representations=in_type.representations + encoder.out_type.representations
        )

    def forward(self, input: GeometricTensor):
        embedding = self.encoder(input)
        out = torch.cat([input.tensor, embedding.tensor], dim=-1)
        return self.out_type(out)

    def decode(self, encoded_x: torch.Tensor):
        x = encoded_x[..., self.residual_dims]
        return x

    @property
    def residual_dims(self):
        return slice(0, self.in_type.size)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (len(self.out_type.size),)
