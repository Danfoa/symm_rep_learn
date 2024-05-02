import torch
from torch.nn import Module, Linear, Dropout, ReLU, Sequential, Conv2d, MaxPool2d

class SingularLayer(Module):
    def __init__(self, d):
        super(SingularLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(torch.normal(mean=0.,std=2./d,size=(d,))), requires_grad=True)

    def forward(self, x):
        return x * torch.exp(-self.weights**2)

class MLPBlock(Module):
    def __init__(self, input_size, output_size, dropout=0., activation=ReLU):
        super(MLPBlock, self).__init__()
        self.linear = Linear(input_size, output_size)
        self.dropout = Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        out = self.linear(x)
        out = self.dropout(out)
        out = self.activation(out)
        return out

class MLP(Module):
    def __init__(self, input_shape, n_hidden, layer_size, output_shape, dropout=0., activation=ReLU, iterative_whitening=False):
        super(MLP, self).__init__()
        if isinstance(layer_size, int):
            layer_size = [layer_size]*n_hidden
        if n_hidden == 0:
            layers = [Linear(input_shape, output_shape, bias=False)]
        else:
            layers = []
            for layer in range(n_hidden):
                if layer == 0:
                    layers.append(MLPBlock(input_shape, layer_size[layer], dropout, activation))
                else:
                    layers.append(MLPBlock(layer_size[layer-1], layer_size[layer], dropout, activation))

            layers.append(Linear(layer_size[-1], output_shape, bias=False))
            if iterative_whitening:
                layers.append(IterativeWhitening(output_shape))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ConvMLP(Module):
    # convolutional network for 28 by 28 images (TODO: debug needed for non rgb)
    def __init__(self, n_hidden, layer_size, output_shape, dropout=0., rgb=False, activation=ReLU, iterative_whitening=False):
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

        mlp = MLP(input_shape,
                  n_hidden,
                  layer_size,
                  output_shape,
                  dropout,
                  activation,
                  iterative_whitening)
        
    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.mlp(x)

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