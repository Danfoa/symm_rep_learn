from __future__ import annotations

import torch


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

    def extra_repr(self):
        return "function={}".format(self.func)


class ResidualEncoder(torch.nn.Module):
    """Residual encoder for symm_rep_learn. This encoder processes batches of shape (batch_size, dim_y) and
    returns (batch_size, embedding_dim + dim_y).
    """

    def __init__(self, encoder: torch.nn.Module, in_dim: int):
        super(ResidualEncoder, self).__init__()
        self.encoder = encoder
        self.in_dim = in_dim

    def forward(self, input: torch.Tensor):
        embedding = self.encoder(input)
        out = torch.cat([input, embedding], dim=1)
        return out

    def decode(self, encoded_x: torch.Tensor):
        x = encoded_x[:, self.residual_dims, ...]
        return x

    @property
    def residual_dims(self):
        return slice(0, self.in_dim)
