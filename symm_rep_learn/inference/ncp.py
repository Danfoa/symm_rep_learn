# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 29/03/25
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from symm_rep_learn.models.ncp import NCP
from symm_rep_learn.nn.layers import ResidualEncoder


class NCPRegressor(torch.nn.Module):
    def __init__(self, model: NCP, y_train, zy_train, lstsq=False, analytic_residual=False):
        super(NCPRegressor, self).__init__()
        self.model = model
        self.lstsq = lstsq
        self.analytic_residual = analytic_residual
        self.device = next(model.parameters()).device
        self.out_dim = zy_train.shape[-1]

        assert zy_train.shape[0] == y_train.shape[0], "Y train and Z(Y) train must have the same number of samples"
        assert y_train.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.ndim}"
        assert zy_train.ndim == 2, f"Z(Y) train must have shape (n_train, z(y)_dim) got {zy_train.ndim}"

        zy_train = zy_train.to(self.device)
        y_train = y_train.to(self.device)

        n_train = zy_train.shape[0]

        # Compute the expectation of the r.v `z(y)` from the training dataset.
        self.mean_zy = zy_train.mean(axis=0, keepdim=True)
        zy_train_c = zy_train - self.mean_zy

        # Compute the embeddings of the entire y training dataset. And the linear regression between z(y) and h(y)
        self.Czyhy = torch.zeros((zy_train.shape[-1], model.embedding_dim), device=self.device)

        hy_train = model.embedding_y(y_train)  # shape: (n_train, embedding_dim)

        if analytic_residual and isinstance(model.embedding_y, ResidualEncoder):
            y_dims_in_hy = model.embedding_y.residual_dims
            for dim in range(y_dims_in_hy.start, y_dims_in_hy.stop):
                self.Czyhy[dim, dim] = 1
        else:
            if lstsq:
                out = torch.linalg.lstsq(hy_train, zy_train_c)
                self.Czyhy = out.solution.T
                assert self.Czyhy.shape == (zy_train.shape[-1], hy_train.shape[-1]), f"Invalid shape {self.Czyhy.shape}"
            else:  # Compute empirical expectation
                self.Czyhy = (1 / n_train) * torch.einsum("by,bh->yh", zy_train_c, hy_train)

    def forward(self, x_cond):
        x_cond = x_cond.to(self.device)
        fx_cond = self.model.embedding_x(x_cond)  # shape: (n_test, embedding_dim)

        # Check formula 12 from https://arxiv.org/pdf/2407.01171
        Dr = self.model.truncated_operator
        zy_deflated_basis_expansion = torch.einsum("bf,fh,yh->by", fx_cond, Dr, self.Czyhy)
        zy_pred = self.mean_zy + zy_deflated_basis_expansion

        return zy_pred


class NCPConditionalCDF(torch.nn.Module):
    def __init__(self, model: NCP, y_train, support_discretization_points=500, **ncp_regressor_kwargs):
        super(NCPConditionalCDF, self).__init__()

        assert y_train.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.ndim}"

        y_min, y_max = torch.min(y_train, dim=0).values, torch.max(y_train, dim=0).values
        self.discretization_points = support_discretization_points
        # self.support_obs -> (discret_points, y_dim)
        self.support_obs = np.linspace(y_min.numpy(), y_max.numpy(), self.discretization_points)
        # Compute the indicator function of (y_i <= y_i') for each y_i' in the discretized support of y_i
        # cdf_obs_ind -> (n_samples, discretization_points, y_dim)
        cdf_obs_ind = (y_train.detach().cpu().numpy()[:, None, :] <= self.support_obs[None, :, :]).astype(int)

        # Estimate the CDF
        self.marginal_CDF = cdf_obs_ind.mean(axis=0)  # (discretization_points,) -> F(y') -> P(Y <= y')
        cdf_obs_ind_c = torch.tensor(cdf_obs_ind - self.marginal_CDF, dtype=torch.float32)

        self.n_obs_dims = y_train.shape[1]
        self.NCP_regressors = []

        for dim in range(self.n_obs_dims):
            # Compute the conditional indicator sets per each y' in the support given X=x.
            dim_ncp_regressor = NCPRegressor(
                model=model, y_train=y_train, zy_train=cdf_obs_ind_c[..., dim], **ncp_regressor_kwargs
            )
            self.NCP_regressors.append(dim_ncp_regressor)

    def forward(self, x_cond: torch.Tensor):
        """Predicts the Conditional Cumulative Distribution Function of the random variable Y given X=x.

        Args:
            x_cond:

        Returns:

        """
        ccdf = []
        for dim in range(self.n_obs_dims):
            dim_ncp_regressor = self.NCP_regressors[dim]
            ccdf_obs_ind_pred = (
                self.marginal_CDF[..., dim] + dim_ncp_regressor(x_cond=x_cond).detach().cpu().numpy().squeeze()
            )
            ccdf_obs_ind_pred = np.max([ccdf_obs_ind_pred, np.zeros_like(ccdf_obs_ind_pred)], axis=0)
            # Smooth the predicted Conditional Cumulative Distribution Function
            # ccdf_obs_ind_smooth = self.smooth_cdf(self.support_obs[..., dim], np.squeeze(ccdf_obs_ind_pred))
            # Filter out points below 0 from approximation of the NCP.
            ccdf.append(ccdf_obs_ind_pred)
        return ccdf

    def conditional_quantiles(self, x_cond: torch.Tensor, alpha=0.05):
        """Predicts the Conditional Quantiles of the random variable Y given X=x."""
        assert alpha is None or 0 < alpha <= 1, f"Alpha must be in the range (0, 1] got {alpha}"
        ccdf = self.forward(x_cond)
        q_low, q_high = [], []
        for dim in range(self.n_obs_dims):
            dim_ccdf = ccdf[dim]
            if dim_ccdf.ndim == 2:  # Multiple conditioning points:
                q_low_per_x, q_high_per_ = [], []
                for x_cond_idx in range(dim_ccdf.shape[0]):
                    low_qx, high_qx = self.find_best_quantile(self.support_obs[..., dim], ccdf[dim][x_cond_idx], alpha)
                    q_low_per_x.append(low_qx)
                    q_high_per_.append(high_qx)
                q_low.append(np.asarray(q_low_per_x))
                q_high.append(np.asarray(q_high_per_))
            elif dim_ccdf.ndim == 1:
                low_qx, high_qx = self.find_best_quantile(self.support_obs[..., dim], ccdf[dim], alpha)
                q_low.append(low_qx)
                q_high.append(high_qx)
            else:
                raise ValueError(f"Invalid shape {dim_ccdf.shape}")
        q_low = np.asarray(q_low).T
        q_high = np.asarray(q_high).T
        # q_low = torch.tensor([q[0, ...] for q in quantiles])
        # q_high = torch.tensor([q[1, ...] for q in quantiles])
        return q_low, q_high

    @staticmethod
    def smooth_cdf(values, cdf):  # Moved smooth_cdf here from NCP/utils.py
        scdf = IsotonicRegression(y_min=0.0, y_max=cdf.max()).fit_transform(values, cdf)
        if scdf.max() <= 0:
            return np.zeros(values.shape)
        scdf = scdf / scdf.max()
        return scdf

    @staticmethod
    def find_best_quantile(x_support, cdf_x, alpha):
        """Find the best quantile interval of the random variable Y given X=x at confidence level alpha.

        Args:
            x_support (numpy.ndarray): The values of the random variable Y of shape (n_samples,)
            cdf_x  (numpy.ndarray): The cumulative distribution function values corresponding to Y of shape (n_samples,)
            alpha (float): The confidence level, where 0 < alpha <= 1.

        Returns:
            tuple: A tuple (q_low, q_high) where q_low is the lower quantile and q_high is the upper quantile of the random variable Y given X=x.
                These quantiles estimate the region of the support within which the random variable Y lies with confidence level (1-alpha).
        """
        x_support = x_support.flatten()
        t0 = 0
        t1 = 1
        best_t0 = 0
        best_t1 = -1
        best_size = np.inf
        while t0 < len(cdf_x):
            # stop if left border reaches right end of discretisation
            if cdf_x[t1] - cdf_x[t0] >= 1 - alpha:
                # if x[t0], x[t1] is a confidence interval at level alpha, compute length and compare to best
                size = x_support[t1] - x_support[t0]
                if size < best_size:
                    best_t0 = t0
                    best_t1 = t1
                    best_size = size
                # moving t1 to the right will only increase the size of the interval, so we can safely move t0 to the right
                t0 += 1
            elif t1 == len(cdf_x) - 1:
                # if x[t0], x[t1] is not a confidence interval with confidence at least level alpha,
                # and t1 is already at the right limit of the discretisation, then there remains no more pertinent intervals
                break
            else:
                # if moving x[t0] to the right reduces the level, we need to increase t1
                t1 += 1
        return x_support[best_t0], x_support[best_t1]
