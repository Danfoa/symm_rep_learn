from torch.nn import Module
import torch

from NCP.nn.layers import SingularLayer
from NCP.utils import tonp, frnp, sqrtmh, cross_cov, filter_reduced_rank_svals
import lightning as L
from copy import deepcopy

def ensure_torch(x):
    if torch.is_tensor(x):
        return x
    else:
        return frnp(x)

class NCPOperator(Module):
    def __init__(self, U_operator:Module, V_operator:Module, U_operator_kwargs:dict, V_operator_kwargs:dict):

        super(NCPOperator, self).__init__()
        if U_operator_kwargs['output_shape'] == V_operator_kwargs['output_shape']:
            d = U_operator_kwargs['output_shape'] # number of latent dimensions
        else:
            raise ValueError('Number of latent dimensions must be the same for U_operator and V_operator.')

        self.U = U_operator(**U_operator_kwargs)
        self.V = V_operator(**V_operator_kwargs)
        self.S = SingularLayer(d)

        # buffers for centering
        self.register_buffer('_mean_Ux', torch.zeros(d))
        self.register_buffer('_mean_Vy', torch.zeros(d))

        #buffers for whitening
        self.register_buffer('_sqrt_cov_X_inv', torch.eye(d))
        self.register_buffer('_sqrt_cov_Y_inv', torch.eye(d))
        self.register_buffer('_sing_val', torch.ones(d))
        self.register_buffer('_sing_vec_l', torch.eye(d))
        self.register_buffer('_sing_vec_r', torch.eye(d))

    def forward(self, x, y, postprocess=None):
        Ux, sigma, Vy = self.postprocess_UV(x, y, postprocess)

        # return F.relu(torch.sum(sigma * Ux * Vy, axis=-1))
        return torch.sum(sigma * Ux * Vy, axis=-1)

    def postprocess_UV(self, X, Y, postprocess=None):
        sigma = torch.sqrt(torch.exp(-self.S.weights ** 2))
        if postprocess is None:
            Ux = self.U(X)
            Vy = self.V(Y)
        else:
            if postprocess == 'centering':
                Ux, Vy = self.centering(X, Y)
            elif postprocess == 'whitening':
                Ux, sigma, Vy = self.whitening(X, Y)
        return Ux, sigma, Vy

    def centering(self, X, Y):
        Ux = self.U(X)
        Vy = self.V(Y)

        Ux_centered = Ux - torch.outer(torch.ones(Ux.shape[0]), self._mean_Ux)
        Vy_centered = Vy - torch.outer(torch.ones(Vy.shape[0]), self._mean_Vy)

        return Ux_centered, Vy_centered

    def whitening(self, X, Y):
        sigma = torch.sqrt(torch.exp(-self.S.weights ** 2))

        Ux = self.U(X)
        Ux = Ux - torch.outer(torch.ones(Ux.shape[0]), self._mean_Ux)
        Ux = Ux @ torch.diag(sigma)
        Ux = Ux @ self._sqrt_cov_X_inv @ self._sing_vec_l

        Vy = self.V(Y)
        Vy = Vy - torch.outer(torch.ones(Vy.shape[0]), self._mean_Vy)
        Vy = Vy @ torch.diag(sigma)
        Vy = Vy @ self._sqrt_cov_Y_inv @ self._sing_vec_r

        return Ux, self._sing_val, Vy

    def conditional_expectation(self, X, Y_sampling, observable=lambda x: x, postprocess=None):
        X = ensure_torch(X)
        Y_sampling = ensure_torch(Y_sampling)

        Ux, sigma, Vy = self.postprocess_UV(X, Y_sampling, postprocess)
        fY = observable(Y_sampling)
        bias = torch.mean(fY, axis=0)

        fY = fY.unsqueeze(-1).repeat((1,1,Vy.shape[-1]))
        Vy = Vy.unsqueeze(1).repeat((1, fY.shape[1], 1))
        Ux = Ux.repeat((1, fY.shape[1], 1))

        Vy_fY = torch.mean(Vy * fY, axis=0)
        sigma_U_fY_VY = sigma * Ux * Vy_fY
        val = torch.sum(sigma_U_fY_VY, axis=-1)

        return tonp(bias + val)

    def cdf(self, X, Y_sampling, observable=lambda x: x, postprocess=None):
        X = ensure_torch(X)
        Y_sampling = ensure_torch(Y_sampling)

        # observable is a vector to scalar function
        fY = torch.stack([observable(y_i) for y_i in torch.unbind(Y_sampling, dim=-1)], dim=-1).flatten() # Pytorch equivalent of numpy.apply_along_axis
        candidates = torch.argsort(fY)
        probas = torch.cumsum(torch.ones(fY.shape[0]), -1) / fY.shape[0]  # vector of [k/n], k \in [n]

        Ux, sigma, Vy = self.postprocess_UV(X, Y_sampling, postprocess)
        Ux = Ux.flatten()

        # estimating the cdf of the function f on X_t
        cdf = torch.zeros(candidates.shape[0])
        for i, val in enumerate(fY[candidates]):
            Ify = torch.outer((fY <= val), torch.ones(Vy.shape[1]))  # indicator function of fY < fY[i], put into shape (n_sample, latent_dim)
            EVyFy = torch.mean(Vy * Ify, axis=0)  # for all latent dim, compute E (Vy * fY)
            cdf[i] = probas[i] + torch.sum(sigma * Ux * EVyFy)

        return tonp(fY[candidates].flatten()), tonp(cdf)

    def pdf(self, X, Y_sampling, p_y=lambda x:x, observable=lambda x: x, postprocess=None):
        # observable is a vector to scalar function
        fY = torch.stack([observable(x_i) for x_i in torch.unbind(Y_sampling, dim=-1)], dim=-1) # Pytorch equivalent of numpy.apply_along_axis
        fY_sorted, _ = torch.sort(fY, dim=0)

        pdf = (1 + self.forward(X, Y_sampling, postprocess)) * p_y(fY_sorted)
        return tonp(fY_sorted), tonp(pdf)

    def _compute_data_statistics(self, X, Y):
        sigma = torch.sqrt(torch.exp(-self.S.weights ** 2))
        Ux = self.U(X.type_as(sigma))
        Vy = self.V(Y.type_as(sigma))
        self._mean_Ux = torch.mean(Ux, axis=0)
        self._mean_Vy = torch.mean(Vy, axis=0)

        Ux_centered = Ux - torch.outer(torch.ones(Ux.shape[0]).type_as(Ux), self._mean_Ux)
        Vy_centered = Vy - torch.outer(torch.ones(Vy.shape[0]).type_as(Vy), self._mean_Vy)

        Ux_centered = Ux_centered @ torch.diag(sigma)
        Vy_centered = Vy_centered @ torch.diag(sigma)

        cov_X = torch.cov(Ux_centered.T)
        cov_Y = torch.cov(Vy_centered.T)
        cov_XY = cross_cov(Ux_centered.T, Vy_centered.T)

        # write in a stable way
        self._sqrt_cov_X_inv = torch.linalg.pinv(sqrtmh(cov_X))
        self._sqrt_cov_Y_inv = torch.linalg.pinv(sqrtmh(cov_Y))

        M = self._sqrt_cov_X_inv @ cov_XY @ self._sqrt_cov_Y_inv
        e_val, sing_vec_l = torch.linalg.eigh(M @ M.T)
        e_val, self._sing_vec_l = filter_reduced_rank_svals(e_val, sing_vec_l)
        self._sing_val = torch.sqrt(e_val)
        self._sing_vec_r = (M.T @ self._sing_vec_l) / self._sing_val

class NCPModule(L.LightningModule):
    def __init__(
            self,
            model: NCPOperator,
            optimizer_fn: torch.optim.Optimizer,
            optimizer_kwargs: dict,
            loss_fn: torch.nn.Module,
            loss_kwargs: dict,
    ):
        super(NCPModule, self).__init__()
        self.model = model
        self._optimizer = optimizer_fn
        _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
        if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
            self.lr = _tmp_opt_kwargs.pop("lr")
            self.opt_kwargs = _tmp_opt_kwargs
        else:
            self.lr = 1e-3
            raise Warning(
                "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing it to the optimizer_kwargs argument."
            )
        self.loss_fn = loss_fn(**loss_kwargs)
        self.train_loss = []
        self.val_loss = []

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self._optimizer(self.parameters(), **kw)

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.batch = batch

        X, Y = batch
        loss = self.loss_fn
        l = loss(X, Y, self.model)
        self.log('train_loss', l, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss.append(tonp(l))

        return l

    def validation_step(self, batch, batch_idx):
        X, Y = batch

        loss = self.loss_fn
        l = loss(X, Y, self.model)
        self.log('val_loss', l, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.append(tonp(l))
        return l

    def on_fit_end(self):
        X, Y = self.batch
        self.model._compute_data_statistics(X, Y)
        del self.batch