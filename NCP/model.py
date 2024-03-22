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
    def __init__(self, U_operator:Module, V_operator:Module, output_shape, gamma=0.):

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

    def fit(self, X, Y, X_val, Y_val, optimizer:Optimizer, optimizer_kwargs:dict, epochs=1000,lr=1e-3, gamma=None, seed=None):
        if gamma is not None:
            self.gamma = gamma
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 0

        torch.manual_seed(self.seed)
        optimizer = optimizer(self.models.parameters(), **optimizer_kwargs)
        pbar = tqdm(range(epochs))

        # random split of X and Y
        X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.5, random_state=self.seed)
        X1, X2, Y1, Y2 = torch.Tensor(X1), torch.Tensor(X2), torch.Tensor(Y1), torch.Tensor(Y2)

        X1_val, X2_val, Y1_val, Y2_val = train_test_split(X_val, Y_val, test_size=0.5, random_state=self.seed)
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

            loss = CMELoss(gamma=self.gamma)
            l = loss(z1, z2, z3, z4, self.models['S'])
            l.backward()
            optimizer.step()
            self.losses.append(l.detach().numpy())
            pbar.set_description(f'epoch = {i}, loss = {l}')

            # validation step:
            with torch.no_grad():
                self.models.eval()
                z1_val = self.models['U'](X1_val)
                z2_val = self.models['U'](X2_val)
                z3_val = self.models['V'](Y1_val)
                z4_val = self.models['V'](Y2_val)
                val_l = loss(z1_val, z2_val, z3_val, z4_val, self.models['S'])
                self.val_losses.append(val_l.detach().numpy())

            #if i%1000 == 0:
            #    print(list(self.models['U'].parameters()), list(self.models['V'].parameters()), list(self.models['S'].parameters()))

    def get_losses(self):
        return self.losses

    def get_val_losses(self):
        return self.val_losses

    def predict(self, X, observable = lambda x :x ):
        self.models.eval()
        n = self.training_X.shape[0]
        if not torch.is_tensor(X):
            X = torch.Tensor(X)

        Y = torch.Tensor(self.training_Y)

        # Ux = self.models['U'](X).detach().numpy()
        # Vy = self.models['V'](torch.Tensor(self.training_Y)).detach().numpy()

        # whitening of Ux and Vy
        Ux = self.models['U'](torch.Tensor(self.training_X))
        Vy = self.models['V'](torch.Tensor(self.training_Y))

        cov_X = Ux.T @ Ux * n**-1
        cov_Y = Vy.T @ Vy * n**-1

        sqrt_cov_X = sqrtmh(cov_X)
        sqrt_cov_Y = sqrtmh(cov_Y)

        Ux = torch.linalg.lstsq(sqrt_cov_X,Ux.T).solution.T.detach().numpy()
        Vy = torch.linalg.lstsq(sqrt_cov_Y,Vy.T).solution.T.detach().numpy()

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

def sqrtmh(A: torch.Tensor):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices. Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228 <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH