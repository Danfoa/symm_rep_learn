import torch
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential


# TODO: Unless we define custom initialization schemes, this class seems rather unnecesary.
class SingularLayer(Module):
    def __init__(self, d):
        super(SingularLayer, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.Tensor(torch.normal(mean=0.0, std=2.0 / d, size=(d,))), requires_grad=True
        )
        # high = np.sqrt(np.log(4)- np.log(3))
        # low = np.sqrt(np.log(4))
        # self.weights = torch.nn.Parameter(torch.Tensor(low+(high-low)*torch.rand(d,)), requires_grad=True)

    @property
    def svals(self):
        return torch.exp(-(self.weights**2))

    def forward(self, x):
        return x * self.svals


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


# class iterative_normalization_py(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, *args, **kwargs):
#         X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
#         # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
#         ctx.g = X.size(1) // nc
#         x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
#         _, d, m = x.size()
#         saved = []
#         if training:
#             # calculate centered activation by subtracted mini-batch mean
#             mean = x.mean(-1, keepdim=True)
#             xc = x - mean
#             saved.append(xc)
#             # calculate covariance matrix
#             P = [None] * (ctx.T + 1)
#             P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
#             Sigma = torch.baddbmm(eps, P[0], 1.0 / m, xc, xc.transpose(1, 2))
#             # reciprocal of trace of Sigma: shape [g, 1, 1]
#             rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
#             saved.append(rTr)
#             Sigma_N = Sigma * rTr
#             saved.append(Sigma_N)
#             for k in range(ctx.T):
#                 P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N)
#             saved.extend(P)
#             wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
#             running_mean.copy_(momentum * mean + (1.0 - momentum) * running_mean)
#             running_wmat.copy_(momentum * wm + (1.0 - momentum) * running_wmat)
#         else:
#             xc = x - running_mean
#             wm = running_wmat
#         xn = wm.matmul(xc)
#         Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
#         ctx.save_for_backward(*saved)
#         return Xn

#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         (grad,) = grad_outputs
#         saved = ctx.saved_variables
#         xc = saved[0]  # centered input
#         rTr = saved[1]  # trace of Sigma
#         sn = saved[2].transpose(-2, -1)  # normalized Sigma
#         P = saved[3:]  # middle result matrix,
#         g, d, m = xc.size()

#         g_ = grad.transpose(0, 1).contiguous().view_as(xc)
#         g_wm = g_.matmul(xc.transpose(-2, -1))
#         g_P = g_wm * rTr.sqrt()
#         wm = P[ctx.T]
#         g_sn = 0
#         for k in range(ctx.T, 1, -1):
#             P[k - 1].transpose_(-2, -1)
#             P2 = P[k - 1].matmul(P[k - 1])
#             g_sn += P2.matmul(P[k - 1]).matmul(g_P)
#             g_tmp = g_P.matmul(sn)
#             g_P.baddbmm_(1.5, -0.5, g_tmp, P2)
#             g_P.baddbmm_(1, -0.5, P2, g_tmp)
#             g_P.baddbmm_(1, -0.5, P[k - 1].matmul(g_tmp), P[k - 1])
#         g_sn += g_P
#         # g_sn = g_sn * rTr.sqrt()
#         g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
#         g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2.0 * g_tr) * (-0.5 / m * rTr)
#         # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
#         g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
#         grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
#         return grad_input, None, None, None, None, None, None, None


# class IterativeWhitening(torch.nn.Module):
#     def __init__(
#         self,
#         num_features,
#         num_groups=1,
#         num_channels=None,
#         T=3,
#         dim=2,
#         eps=1e-5,
#         momentum=0.1,
#         affine=False,
#         *args,
#         **kwargs,
#     ):
#         super(IterativeWhitening, self).__init__()
#         # assert dim == 4, 'IterNorm does not support 2D'
#         self.T = T
#         self.eps = eps
#         self.momentum = momentum
#         self.num_features = num_features
#         self.affine = affine
#         self.dim = dim
#         if num_channels is None:
#             num_channels = (num_features - 1) // num_groups + 1
#         num_groups = num_features // num_channels
#         while num_features % num_channels != 0:
#             num_channels //= 2
#             num_groups = num_features // num_channels
#         assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(
#             num_features, num_groups
#         )
#         self.num_groups = num_groups
#         self.num_channels = num_channels
#         shape = [1] * dim
#         shape[1] = self.num_features
#         if self.affine:
#             self.weight = torch.nn.Parameter(torch.Tensor(*shape))
#             self.bias = torch.nn.Parameter(torch.Tensor(*shape))
#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)

#         self.register_buffer("running_mean", torch.zeros(num_groups, num_channels, 1))
#         # running whiten matrix
#         self.register_buffer("running_wm", torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         # self.reset_running_stats()
#         if self.affine:
#             torch.nn.init.ones_(self.weight)
#             torch.nn.init.zeros_(self.bias)

#     def forward(self, X: torch.Tensor):
#         X_hat = iterative_normalization_py.apply(
#             X, self.running_mean, self.running_wm, self.num_channels, self.T, self.eps, self.momentum, self.training
#         )
#         # affine
#         if self.affine:
#             return X_hat * self.weight + self.bias
#         else:
#             return X_hat

#     def extra_repr(self):
#         return (
#             "{num_features}, num_channels={num_channels}, T={T}, eps={eps}, "
#             "momentum={momentum}, affine={affine}".format(**self.__dict__)
#         )
