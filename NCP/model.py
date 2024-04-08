from torch.nn import Module, ModuleDict
from torch.optim import Optimizer
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from NCP.layers import SingularLayer
from NCP.nn.losses import CMELoss
from NCP.utils import tonp, frnp

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
        self.cond_number_cov_X = []
        self.cond_number_cov_Y = []
        self.cond_number_cov_XY = []
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

            if wandb and i % 10 == 0:
                wandb.log({'z1': tonp(z1)})
                wandb.log({'z2': tonp(z2)})
                wandb.log({'z3': tonp(z3)})
                wandb.log({'z4': tonp(z4)})
                wandb.log({'cov_z1': tonp(z1.T @ z1)})
                centered_z1 = z1 - z1.mean(axis=-1).reshape(-1, 1)
                wandb.log({'centered_cov_z1': tonp(centered_z1.T @ centered_z1)})

            loss = CMELoss(gamma=self.gamma)
            l = loss(z1, z2, z3, z4, self.models['S'])
            l.backward()
            optimizer.step()
            self.losses.append(tonp(l))
            pbar.set_description(f'epoch = {i}, loss = {l}')

            # validation step:
            with torch.no_grad():
                self.models.eval()
                z1_val = self.models['U'](X1_val)
                z2_val = self.models['U'](X2_val)
                z3_val = self.models['V'](Y1_val)
                z4_val = self.models['V'](Y2_val)
                val_l = loss(z1_val, z2_val, z3_val, z4_val, self.models['S'])
                self.val_losses.append(tonp(val_l))
                if wandb and i%10 == 0:
                    for module_name, module in self.models.items():
                        for name, param in module.named_parameters():
                            if module_name == 'S':
                                wandb.log({module_name+'_'+name: tonp(module.weights).squeeze()})
                            else:
                                wandb.log({module_name+'_'+name: tonp(param).squeeze()})

            if wandb:
                wandb.log({"train_loss": l, "val_loss": val_l})
            #if i%1000 == 0:
            #    print(list(self.models['U'].parameters()), list(self.models['V'].parameters()), list(self.models['S'].parameters()))
    def get_losses(self):
        return self.losses

    def get_val_losses(self):
        return self.val_losses

    def predict(self, X_test, observable = lambda x :x, postprocess = None, Y=None):
        if postprocess: # postprocessing can be 'centering' or 'whitening'
            Ux, sigma, Vy = self.postprocess_UV(X_test, postprocess)
        else:
            sigma = torch.sqrt(torch.exp(-self.models['S'].weights ** 2))
            Ux = self.models['U'](frnp(X_test, self.device))
            Vy = self.models['V'](frnp(self.training_Y, self.device))
            Ux, sigma, Vy = tonp(Ux), tonp(sigma), tonp(Vy)

        fY = np.outer(np.squeeze(observable(self.training_Y)), np.ones(Vy.shape[-1]))
        bias = np.mean(fY)

        Vy_fY = np.mean(Vy * fY, axis=0)
        sigma_U_fY_VY = sigma * Ux * Vy_fY
        val = np.sum(sigma_U_fY_VY, axis=-1)

        return bias + val

    # adapt functions to work with function from R^d to {0,1}

    def conditional_probability(self, interval_A, interval_B, postprocessing = None):
        Y_train = self.training_Y
        X_train = self.training_X

        Ux, sigma, Vy = self.postprocess_UV(X_train, postprocessing)

        n = Y_train.shape[0]
        x_A = indicator_fn(X_train, interval_A)
        y_B = indicator_fn(Y_train, interval_B)
        Ux_A = Ux[x_A, :].sum(axis=0)*n**-1
        Vy_B = Vy[y_B, :].sum(axis=0)*n**-1

        conditional_prob = y_B.mean() + (sigma * Ux_A * (x_A.mean() ** -1) * Vy_B).sum(axis=-1)
        return conditional_prob

    def joint_probability(self, interval_A, interval_B, postprocessing = None):
        Y_train = self.training_Y
        X_train = self.training_X

        Ux, sigma, Vy = self.postprocess_UV(X_train, postprocessing)

        n = Y_train.shape[0]
        x_A = indicator_fn(X_train, interval_A)
        y_B = indicator_fn(Y_train, interval_B)

        Ux_A = Ux[x_A, :].sum(axis=0) * n ** -1
        Vy_B = Vy[y_B, :].sum(axis=0) * n ** -1
        joint_prob = 1 + (sigma * Ux_A * (x_A.mean() ** -1) * Vy_B * (y_B.mean() ** -1)).sum(axis=-1)
        return joint_prob

    def postprocess_UV(self, X_test, postprocess, Y_test=None):
        if postprocess == 'centering':
            sigma = torch.sqrt(torch.exp(-self.models['S'].weights ** 2))
            Ux, Vy = self.centering(X_test)
        elif postprocess == 'whitening':
            Ux, sigma, Vy = self.whitening(X=X_test)

        return tonp(Ux), tonp(sigma), tonp(Vy)

    def centering(self, X=None, Y=None):
        self.models.eval()

        Ux_train = self.models['U'](frnp(self.training_X, self.device))
        Vy_train = self.models['V'](frnp(self.training_Y, self.device))
        self.mean_Ux = torch.mean(Ux_train, axis=0)
        self.mean_Vy = torch.mean(Vy_train, axis=0)

        if X is not None:
            if not torch.is_tensor(X):
                X = frnp(X, self.device)
            Ux = self.models['U'](X)
        else:
            Ux = Ux_train

        if Ux_train.shape[-1] > 1:
            Ux_centered = Ux - torch.outer(torch.ones(Ux.shape[0], device=self.device), self.mean_Ux)
            Vy_centered = Vy_train - torch.outer(torch.ones(Vy_train.shape[0], device=self.device), self.mean_Vy)

        return Ux_centered, Vy_centered

    def whitening(self, X=None, Y=None):
        n = self.training_X.shape[0]
        sigma = torch.sqrt(torch.exp(-self.models['S'].weights ** 2))

        Ux_centered, Vy_centered = self.centering()

        Ux_centered = Ux_centered @ torch.diag(sigma)
        Vy_centered = Vy_centered @ torch.diag(sigma)

        cov_X = Ux_centered.T @ Ux_centered * n ** -1
        cov_Y = Vy_centered.T @ Vy_centered * n ** -1
        cov_XY = Ux_centered.T @ Vy_centered * n ** -1

        # print(torch.linalg.cond(cov_X))
        # print(torch.linalg.cond(cov_Y))

        # write in a stable way
        sqrt_cov_X_inv = torch.linalg.pinv(sqrtmh(cov_X))
        sqrt_cov_Y_inv = torch.linalg.pinv(sqrtmh(cov_Y))

        M = sqrt_cov_X_inv @ cov_XY @ sqrt_cov_Y_inv
        e_val, sing_vec_l = torch.linalg.eigh(M @ M.T)
        # print(torch.sqrt(e_val))
        e_val, sing_vec_l = self._filter_reduced_rank_svals(e_val, sing_vec_l)
        sing_val = torch.sqrt(e_val)
        sing_vec_r = (M.T @ sing_vec_l) / sing_val

        if X is not None:
            if not torch.is_tensor(X):
                X = frnp(X, self.device)
            Ux = self.models['U'](X)

            Ux = Ux - torch.outer(torch.ones(Ux.shape[0], device=self.device), self.mean_Ux)
            Ux = Ux @ torch.diag(sigma)
        else:
            Ux = Ux_centered

        Ux = Ux @ sqrt_cov_X_inv @ sing_vec_l
        Vy = Vy_centered @ sqrt_cov_Y_inv @ sing_vec_r

        return Ux, sing_val, Vy

    def _filter_reduced_rank_svals(self, values, vectors):
        eps = 2 * torch.finfo(torch.get_default_dtype()).eps
        # Filtering procedure.
        # Create a mask which is True when the real part of the eigenvalue is negative or the imaginary part is nonzero
        is_invalid = torch.logical_or(torch.abs(torch.real(values)) <= eps,
                                      torch.imag(vectors) != 0 if torch.is_complex(values) else torch.zeros(len(values), device=self.device))
        # Check if any is invalid take the first occurrence of a True value in the mask and filter everything after that
        if torch.any(is_invalid):
            values = values[~is_invalid].real
            vectors = vectors[:, ~is_invalid]

        # sort_perm = topk(values, len(values)).indices
        # values = values[sort_perm]
        # vectors = vectors[:, sort_perm]

        # # Assert that the eigenvectors do not have any imaginary part
        # assert torch.all(
        #     torch.imag(vectors) == 0 if torch.is_complex(values) else torch.ones(len(values))
        # ), "The eigenvectors should be real. Decrease the rank or increase the regularization strength."

        # Take the real part of the eigenvectors
        vectors = torch.real(vectors)
        values = torch.real(values)
        return values, vectors

def sqrtmh(A: torch.Tensor):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices. Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228 <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

def indicator_fn(x, interval):
    return ((interval[0] <= x) & (x <= interval[1])).squeeze()