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
    def __init__(self, input_shape, n_hidden, layer_size, output_shape, dropout=0.5):
        super(MLPOperator, self).__init__()
        if isinstance(layer_size, int):
            layer_size = [layer_size]*n_hidden
        if n_hidden == 0:
            layers = [Linear(input_shape, output_shape)]
        else:
            layers = []
            for layer in range(n_hidden):
                if layer == 0:
                    layers = [Linear(input_shape, layer_size[layer])]
                    # layers.append(Dropout(p=dropout))
                    layers.append(ReLU())
                else:
                    layers.append(Linear(layer_size[layer-1], layer_size[layer]))
                    # layers.append(Dropout(p=dropout))
                    layers.append(ReLU())
            layers.append(Linear(layer_size[-1], output_shape))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)