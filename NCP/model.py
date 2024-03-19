from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from NCP.layers import SingularLayer
from NCP.nn.losses import CMELoss

class DeepSVD:
    # ideally, entries should be int
    def __init__(self, U_operator:Module, V_operator:Module, layer_size, output_shape, gamma=0.):

        self.models = ModuleDict({
            'U':U_operator, 
            'S':SingularLayer(output_shape), 
            'V':V_operator
        })
        self.losses = []
        self.val_losses = []
        self.gamma = gamma

    def save_after_training(self, X, Y):
        self.training_X = X
        self.training_Y = Y

    def fit(self, X, Y, X_val, Y_val, optimizer:Optimizer, optimizer_kwargs:dict, epochs=1000,lr=1e-3, gamma=None):
        if gamma is not None:
            self.gamma = gamma
        optimizer = Optimizer(self.models.parameters(), **optimizer_kwargs)

        pbar = tqdm(range(epochs))

        # random split of X and Y
        X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.5)
        X1, X2, Y1, Y2 = torch.Tensor(X1), torch.Tensor(X2), torch.Tensor(Y1), torch.Tensor(Y2)

        X1_val, X2_val, Y1_val, Y2_val = train_test_split(X, Y, test_size=0.5)
        X1_val, X2_val, Y1_val, Y2_val = torch.Tensor(X1_val), torch.Tensor(X2_val), torch.Tensor(Y1_val), torch.Tensor(Y2_val)

        last_val_loss = torch.inf

        self.save_after_training(X, Y)

        for i in pbar:

            optimizer.zero_grad()
            self.models.train()

            z1 = self.models['U'](X1)
            z2 = self.models['U'](X2)
            z3 = self.models['V'](Y1)
            z4 = self.models['V'](Y2)

            loss = CMELoss(gamma=gamma)
            l = loss(z1, z2, z3, z4, self.models['S'])
            l.backward()
            optimizer.step()
            self.losses.append(l.detach().numpy())
            pbar.set_description(f'epoch = {i}, loss = {l}')

            # validation step:
            val_l = loss(X1_val, X2_val, Y1_val, Y2_val, self.models['S'])
            self.val_losses.append(val_l.detach().numpy())

            #if i%1000 == 0:
            #    print(list(self.models['U'].parameters()), list(self.models['V'].parameters()), list(self.models['S'].parameters()))

    def get_losses(self):
        return self.losses

    def get_val_losses(self):
        return self.val_losses

    def predict(self, X, observable = lambda x :x ):
        self.models.eval()

        if not torch.is_tensor(X):
            X = torch.Tensor(X)

        Y = torch.Tensor(self.training_Y)

        Ux = self.models['U'](torch.Tensor(X)).detach().numpy()
        Vy = self.models['V'](Y).detach().numpy()

        print('U(x)', Ux)
        print('V(y)', Vy)

        fY = np.outer(np.squeeze(observable(self.training_Y)), np.ones(Vy.shape[-1]))
        bias = np.mean(fY)
        print('fY', fY)

        Vy_fY = np.mean(Vy * fY, axis=0)
        print('VyfY', Vy_fY)
        sigma_U_fY_VY = self.models['S'].weights.detach().numpy() * Ux * Vy_fY
        val = np.sum(sigma_U_fY_VY, axis=-1)
        # print(bias, val)
        return bias + val