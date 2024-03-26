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
    def __init__(self, U_operator:Module, V_operator:Module, output_shape, gamma=0., device='cpu'):

        self.models = ModuleDict({
            'U':U_operator.to(device),
            'S':SingularLayer(output_shape).to(device),
            'V':V_operator.to(device)
        })
        self.losses = []
        self.val_losses = []
        self.gamma = gamma
        self.device = device

    def save_after_training(self, X, Y):
        self.training_X = X
        self.training_Y = Y

    def fit(self, X, Y, X_val, Y_val, optimizer:Optimizer, optimizer_kwargs:dict, epochs=1000,lr=1e-3, gamma=None, seed=None, wandb=None):
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
        X1, X2, Y1, Y2 = (torch.Tensor(X1).to(self.device), torch.Tensor(X2).to(self.device),
                          torch.Tensor(Y1).to(self.device), torch.Tensor(Y2).to(self.device))

        X1_val, X2_val, Y1_val, Y2_val = train_test_split(X_val, Y_val, test_size=0.5, random_state=self.seed)
        X1_val, X2_val, Y1_val, Y2_val = (torch.Tensor(X1_val).to(self.device), torch.Tensor(X2_val).to(self.device),
                                          torch.Tensor(Y1_val).to(self.device), torch.Tensor(Y2_val).to(self.device))

        last_val_loss = torch.inf

        self.save_after_training(X, Y)

        if wandb:
            wandb.watch(self.models, log="all", log_freq=10)
            # for _, module in self.models.items():
            #     wandb.watch(module, log="all", log_freq=10)

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
            self.losses.append(l.detach().cpu().numpy())
            pbar.set_description(f'epoch = {i}, loss = {l}')

            # validation step:
            with torch.no_grad():
                self.models.eval()
                z1_val = self.models['U'](X1_val)
                z2_val = self.models['U'](X2_val)
                z3_val = self.models['V'](Y1_val)
                z4_val = self.models['V'](Y2_val)
                val_l = loss(z1_val, z2_val, z3_val, z4_val, self.models['S'])
                self.val_losses.append(val_l.detach().cpu().numpy())
                if wandb and i%10 == 0:
                    for module_name, module in self.models.items():
                        for name, param in module.named_parameters():
                            wandb.log({module_name+'_'+name: param.detach().cpu().numpy()})

            if wandb:
                wandb.log({"train_loss": l, "val_loss": val_l})
            #if i%1000 == 0:
            #    print(list(self.models['U'].parameters()), list(self.models['V'].parameters()), list(self.models['S'].parameters()))

    def get_losses(self):
        return self.losses

    def get_val_losses(self):
        return self.val_losses

    def predict(self, X, observable = lambda x :x ):
        self.models.eval()
        n = self.training_X.shape[0]
        eps = 2 * torch.finfo(torch.get_default_dtype()).eps
        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)

        Y = torch.Tensor(self.training_Y).to(self.device)

        # whitening of Ux and Vy
        sigma = torch.sqrt(torch.exp(-self.models['S'].weights**2))
        print(sigma)

        Ux = self.models['U'](X)
        Vy = self.models['V'](Y)

        if Ux.shape[-1] > 1:
            Ux = Ux - torch.outer(torch.mean(Ux, axis=-1), torch.ones(Ux.shape[-1], device=self.device))
            Vy = Vy - torch.outer(torch.mean(Vy, axis=-1), torch.ones(Vy.shape[-1], device=self.device))

        Ux = Ux @ torch.diag(sigma)
        Vy = Vy @ torch.diag(sigma)

        cov_X = Ux.T @ Ux * n**-1
        cov_Y = Vy.T @ Vy * n**-1
        cov_XY = Ux.T @ Vy * n**-1

        # write in a stable way
        sqrt_cov_X_inv = torch.linalg.pinv(sqrtmh(cov_X))
        sqrt_cov_Y_inv = torch.linalg.pinv(sqrtmh(cov_Y))

        M = sqrt_cov_X_inv @ cov_XY @ sqrt_cov_Y_inv
        # sing_vec_l, sing_val, sing_vec_r = torch.linalg.svd(M)
        e_val, sing_vec_l = torch.linalg.eigh(M @ M.T)
        sing_vec_l = sing_vec_l[:, e_val >= eps]
        e_val= e_val[e_val >= eps]
        print('e_val', e_val)
        #sing_vec_l, sing_val, sing_vec_r = torch.svd(Ux.T @ Vy * n**-1)
        sing_val = torch.sqrt(e_val)
        sing_vec_r = (M.T @ sing_vec_l) / sing_val
        print('sing val', sing_val)
        print('sing_vec_l', sing_vec_l)
        print('sing_vec_r', sing_vec_r)

        # Ux = Ux - torch.outer(torch.mean(Ux, axis=-1), torch.ones(Ux.shape[-1], device=self.device))
        # Vy = Vy - torch.outer(torch.mean(Vy, axis=-1), torch.ones(Vy.shape[-1], device=self.device))

        Ux = (Ux @ sqrt_cov_X_inv @ sing_vec_l).detach().cpu().numpy()
        Vy = (Vy @ sqrt_cov_Y_inv @ sing_vec_r).detach().cpu().numpy()

        # Vy = Vy - np.outer(np.mean(Vy, axis=-1), np.ones(Ux.shape[-1]))
        # Ux = Ux - np.outer(np.mean(Ux, axis=-1), np.ones(Ux.shape[-1]))

        # print(Ux @ Ux.T)
        # print(Vy @ Vy.T)

        # centering on Ux and Vx maybe before?

        print('U(x)', Ux)
        print('V(y)', Vy)

        fY = np.outer(np.squeeze(observable(self.training_Y)), np.ones(Vy.shape[-1]))
        bias = np.mean(fY)
        print('fY', fY)

        Vy_fY = np.mean(Vy * fY, axis=0)
        print('VyfY', Vy_fY)
        sigma_U_fY_VY = tonp(sing_val) * Ux * Vy_fY
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

def tonp(x):
    return x.detach().cpu().numpy()