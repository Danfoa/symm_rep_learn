import torch
from torch.nn import Module, Linear, Dropout, ReLU, Sequential
import numpy as np

class SingularLayer(Module):
    def __init__(self, input_shape):
        super(SingularLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(np.random.normal(0, 2/input_shape, input_shape)), requires_grad=True)

    def forward(self, x):
        return x * torch.exp(-self.weights**2)

class MLPOperator(Module):
    def __init__(self, input_shape, n_hidden, layer_size, output_shape, dropout=0., iterative_whitening=False):
        super(MLPOperator, self).__init__()
        if isinstance(layer_size, int):
            layer_size = [layer_size]*n_hidden
        if n_hidden == 0:
            layers = [Linear(input_shape, output_shape, bias=False)]
        else:
            layers = []
            for layer in range(n_hidden):
                if layer == 0:
                    layers = [Linear(input_shape, layer_size[layer])]
                    layers.append(Dropout(p=dropout))
                    layers.append(ReLU())
                else:
                    layers.append(Linear(layer_size[layer-1], layer_size[layer]))
                    layers.append(Dropout(p=dropout))
                    layers.append(ReLU())
            layers.append(Linear(layer_size[-1], output_shape, bias=False))
            if iterative_whitening:
                layers.append(IterativeWhitening(output_shape))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class IterativeWhitening(Module):
    # Algorithm 1 of https://arxiv.org/pdf/1904.03441.pdf
    def __init__(self, input_size, newton_iterations: int = 5, eps:float = 1e-5, momentum=0.1):
        self.input_size = input_size
        self.newton_iterations = newton_iterations
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(1, self.input_size))
        self.register_buffer('running_whitening_mat', torch.zeros(self.input_size, self.input_size))
    
    def _compute_whitening_matrix(self, X: torch.Tensor):
        assert X.dim == 2, "Only supporting 2D Tensors"
        if X.shape[1] != self.input_size:
            return ValueError(f"The feature dimension of the input tensor ({X.shape[1]}) does not match the input_size attribute ({self.input_size})")
        covX = torch.cov(X.T, correction=0) + self.eps*torch.eye(self.input_size, dtype=X.dtype, device=X.device)
        norm_covX = covX / torch.trace(covX)
        P = torch.eye(self.input_size, dtype=X.dtype, device=X.device)
        for k in range(self.newton_iterations):
            P = 0.5*(3*P - torch.matrix_power(P, 3)@norm_covX)
        whitening_mat = P / torch.trace(covX)
        X_mean = X.mean(0, keepdim=True)
        return X_mean, whitening_mat
    
    def _update_running_stats(self, mean, whitening_mat):
        self.running_mean = self.momentum*mean + (1 - self.momentum)*self.running_mean
        self.running_whitening_mat = self.momentum*whitening_mat + (1 - self.momentum)*self.running_whitening_mat
    
    def forward(self, X: torch.Tensor):
        self._update_running_stats(self._compute_whitening_matrix(X))
        return (X - self.running_mean)@self.running_whitening_mat