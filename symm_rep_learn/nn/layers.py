from __future__ import annotations

import torch
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential


class MLPBlock(Module):
    def __init__(self, input_size, output_size, dropout=0.0, activation=ReLU, bias=True):
        super(MLPBlock, self).__init__()
        self.linear = Linear(input_size, output_size, bias=bias)
        self.dropout = Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        out = self.linear(x)
        out = self.dropout(out)
        out = self.activation(out)
        return out


class MLP(Module):
    def __init__(
        self,
        input_shape,
        n_hidden,
        layer_size,
        output_shape,
        dropout=0.0,
        activation=ReLU,
        iterative_whitening=False,
        bias=False,
    ):
        super(MLP, self).__init__()
        if isinstance(layer_size, int):
            layer_size = [layer_size] * n_hidden
        if n_hidden == 0:
            layers = [Linear(input_shape, output_shape, bias=False)]
        else:
            layers = []
            for layer in range(n_hidden):
                if layer == 0:
                    layers.append(MLPBlock(input_shape, layer_size[layer], dropout, activation, bias=bias))
                else:
                    layers.append(MLPBlock(layer_size[layer - 1], layer_size[layer], dropout, activation, bias=bias))

            layers.append(Linear(layer_size[-1], output_shape, bias=False))
            if iterative_whitening:
                layers.append(IterNorm(output_shape))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvMLP(Module):
    # convolutional network for 28 by 28 images (TODO: debug needed for non rgb)
    def __init__(
        self, n_hidden, layer_size, output_shape, dropout=0.0, rgb=False, activation=ReLU, iterative_whitening=False
    ):
        super(ConvMLP, self).__init__()
        if rgb:
            self.conv1 = Conv2d(3, 6, 5)
        else:
            self.conv1 = Conv2d(1, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)

        self.act = ReLU()

        if rgb:
            input_shape = 6 * 5 * 5
        else:
            input_shape = 16 * 4 * 4

        self.mlp = MLP(input_shape, n_hidden, layer_size, output_shape, dropout, activation, iterative_whitening)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.mlp(x)


class IterNorm(Module):
    """Applies IterNorm (Algorithm 1 of https://arxiv.org/pdf/1904.03441.pdf)."""

    def __init__(
        self, num_features, eps: float = 1e-5, momentum: float = 0.1, newton_iterations: int = 5, affine: bool = False
    ):
        super(IterNorm, self).__init__()
        self.num_features = num_features
        self.newton_iterations = newton_iterations
        self.eps = eps
        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros(1, self.num_features))
        self.register_buffer("running_whitening_mat", torch.eye(self.num_features, self.num_features))

    def _compute_whitening_matrix(self, X: torch.Tensor):
        assert X.dim() == 2, "Only supporting 2D Tensors"
        if X.shape[1] != self.num_features:
            return ValueError(
                f"The feature dimension of the input tensor ({X.shape[1]}) does not match the num_features attribute ("
                f"{self.num_features})"
            )
        covX = torch.cov(X.T, correction=0) + self.eps * torch.eye(self.num_features, dtype=X.dtype, device=X.device)
        # Normalize the maximum eigenvalue of the cov matrix to be one such that Newton's method converges (eq.4)
        norm_covX = covX / torch.trace(covX)
        P = torch.eye(self.num_features, dtype=X.dtype, device=X.device)
        for k in range(self.newton_iterations):
            P = 0.5 * (3 * P - torch.matrix_power(P, 3) @ norm_covX)
        whitening_mat = P / torch.trace(covX)
        X_mean = X.mean(0, keepdim=True)
        return X_mean, whitening_mat

    def _update_running_stats(self, mean, whitening_mat):
        with torch.no_grad():
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_whitening_mat = (
                self.momentum * whitening_mat + (1 - self.momentum) * self.running_whitening_mat
            )

    def forward(self, X: torch.Tensor):
        self._update_running_stats(*self._compute_whitening_matrix(X))
        return (X - self.running_mean) @ self.running_whitening_mat


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
