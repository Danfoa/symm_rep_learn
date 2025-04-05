# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 29/03/25
from typing import Tuple

import escnn
import numpy as np
import torch
from escnn.gspaces import no_base_space
from escnn.nn import FieldType, GeometricTensor
from symm_learning.stats import invariant_orthogonal_projector

from symm_rep_learn.inference.ncp import NCPConditionalCDF
from symm_rep_learn.models.equiv_ncp import ENCP
from symm_rep_learn.nn.layers import ResidualEncoder


class ENCPConditionalCDF(escnn.nn.EquivariantModule):
    def __init__(
        self, model: ENCP, y_train: GeometricTensor, support_discretization_points=500, **ncp_regressor_kwargs
    ):
        super(ENCPConditionalCDF, self).__init__()
        assert isinstance(y_train, GeometricTensor), f"Y train must be a GeometricTensor got {type(y_train)}"
        assert model.embedding_y.in_type == y_train.type, "`y_train` and `model.embedding_y` must have the same type"
        assert y_train.tensor.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.tensor.ndim}"
        self.in_type: FieldType = model.embedding_x.in_type
        self.out_type: FieldType = model.embedding_y.in_type
        self.y_type = self.out_type

        self.discretization_points = support_discretization_points

        # Compute G-invariant bounds for the support of the random variable Y
        y_min = torch.min(y_train.tensor, dim=0, keepdim=True).values
        y_max = torch.max(y_train.tensor, dim=0, keepdim=True).values
        self.G = self.in_type.fibergroup
        G_y_min = [self.out_type.transform_fibers(y_min, g) for g in self.G.elements]
        G_y_max = [self.out_type.transform_fibers(y_max, g) for g in self.G.elements]
        y_min = torch.min(torch.stack(G_y_min), dim=0).values.reshape(self.out_type.size)
        y_max = torch.max(torch.stack(G_y_max), dim=0).values.reshape(self.out_type.size)
        # print(y_min.shape)
        # self.support_obs -> (discret_points, y_dim)
        self.support_obs = np.linspace(y_min.numpy(), y_max.numpy(), self.discretization_points)
        # print(self.support_obs.shape)
        # Compute the indicator function of (y_i <= y_i') for each y_i' in the discretized support of y_i
        # cdf_obs_ind -> (n_samples, discretization_points, y_dim)
        G_y_train = [self.out_type.transform_fibers(y_train.tensor, g) for g in self.G.elements]
        # print([t.shape for t in G_y_train])
        G_y_train = torch.cat(G_y_train, dim=0)  # (G * n_samples, y_dim)
        # print(G_y_train.shape)
        cdf_obs_ind = (G_y_train.detach().cpu().numpy()[:, None, ...] <= self.support_obs[None, ...]).astype(int)

        # Estimate the CDF
        self.marginal_CDF = cdf_obs_ind.mean(axis=0)  # (discretization_points,) -> F(y') -> P(Y <= y')
        cdf_obs_ind_c = torch.tensor(cdf_obs_ind - self.marginal_CDF, dtype=torch.float32)

        zy_type = FieldType(
            gspace=no_base_space(self.G), representations=[self.G.trivial_representation] * self.discretization_points
        )

        self.n_obs_dims = y_train.shape[1]
        self.NCP_regressors = []

        for dim in range(self.n_obs_dims):
            # Compute the conditional indicator sets per each y' in the support given X=x.
            dim_ncp_regressor = ENCPRegressor(
                model=model,
                y_train=self.out_type(G_y_train),
                zy_train=zy_type(cdf_obs_ind_c[..., dim]),
                **ncp_regressor_kwargs,
            )
            self.NCP_regressors.append(dim_ncp_regressor)

    def forward(self, x_cond: GeometricTensor):
        """Predicts the Conditional Cumulative Distribution Function of the random variable Y given X=x.

        Args:
            x_cond:

        Returns:

        """
        assert isinstance(x_cond, GeometricTensor), f"X condition must be a GeometricTensor got {type(x_cond)}"
        ccdf = []
        for dim in range(self.n_obs_dims):
            dim_ncp_regressor = self.NCP_regressors[dim]
            ccdf_obs_ind_pred = (
                self.marginal_CDF[..., dim] + dim_ncp_regressor(x_cond=x_cond).tensor.detach().cpu().numpy().squeeze()
            )
            ccdf_obs_ind_pred = np.max([ccdf_obs_ind_pred, np.zeros_like(ccdf_obs_ind_pred)], axis=0)
            # Smooth the predicted Conditional Cumulative Distribution Function
            # ccdf_obs_ind_smooth = self.smooth_cdf(self.support_obs[..., dim], np.squeeze(ccdf_obs_ind_pred))
            # Filter out points below 0 from approximation of the NCP.
            ccdf.append(ccdf_obs_ind_pred)

        ccdf = np.asarray(ccdf)
        # bound predictions to [0, 1]
        ccdf = np.clip(ccdf, 0, 1)
        return ccdf

    @torch.no_grad()
    def conditional_quantiles(self, x_cond: GeometricTensor, alpha=0.05):
        """Predicts the Conditional Quantiles of the random variable Y given X=x."""
        assert alpha is None or 0 < alpha <= 1, f"Alpha must be in the range (0, 1] got {alpha}"
        ccdf = self.forward(x_cond)
        q_low, q_high = [], []
        for dim in range(self.n_obs_dims):
            dim_ccdf = ccdf[dim]
            if dim_ccdf.ndim == 2:  # Multiple conditioning points:
                q_low_per_x, q_high_per_ = [], []
                for x_cond_idx in range(dim_ccdf.shape[0]):
                    low_qx, high_qx = NCPConditionalCDF.find_best_quantile(
                        self.support_obs[..., dim], ccdf[dim][x_cond_idx], alpha
                    )
                    q_low_per_x.append(low_qx)
                    q_high_per_.append(high_qx)
                q_low.append(np.asarray(q_low_per_x))
                q_high.append(np.asarray(q_high_per_))
            elif dim_ccdf.ndim == 1:
                low_qx, high_qx = NCPConditionalCDF.find_best_quantile(self.support_obs[..., dim], ccdf[dim], alpha)
                q_low.append(low_qx)
                q_high.append(high_qx)
            else:
                raise ValueError(f"Invalid shape {dim_ccdf.shape}")
        q_low = np.asarray(q_low).T
        q_high = np.asarray(q_high).T
        q_low = torch.tensor(q_low)
        q_high = torch.tensor(q_high)
        # G_q_low = [self.y_type.transform_fibers(q_low[None], g) for g in self.G.elements]
        # G_q_high = [self.y_type.transform_fibers(q_high[None], g) for g in self.G.elements]
        # q_low = torch.squeeze(torch.mean(torch.stack(G_q_low), dim=0), 0)
        # q_high = torch.squeeze(torch.mean(torch.stack(G_q_high), dim=0), 0)
        # q_high = torch.squeeze(torch.mean(torch.stack(G_q_high), dim=0), 0)

        # print(q_low.shape)
        # print(q_high.shape)

        return q_low.cpu().numpy(), q_high.cpu().numpy()

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.out_type.size, self.n_obs_dims


class ENCPRegressor(escnn.nn.EquivariantModule):
    def __init__(
        self, model: ENCP, y_train: GeometricTensor, zy_train: GeometricTensor, lstsq=False, analytic_residual=False
    ):
        super(ENCPRegressor, self).__init__()
        assert isinstance(y_train, GeometricTensor), f"Y train must be a GeometricTensor got {type(y_train)}"
        assert isinstance(zy_train, GeometricTensor), f"z(y) train must be a GeometricTensor got {type(y_train)}"
        assert model.embedding_y.in_type == y_train.type, "`y_train` and `model.embedding_y` must have the same type"
        assert y_train.tensor.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.tensor.ndim}"

        self.model = model
        self.in_type = self.model.embedding_x.in_type
        self.out_type = zy_train.type

        self.lstsq = lstsq
        self.analytic_residual = analytic_residual
        self.device = next(model.parameters()).device

        assert zy_train.shape[0] == y_train.shape[0], "Y train and Z(Y) train must have the same number of samples"
        assert y_train.tensor.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.tensor.ndim}"
        assert zy_train.tensor.ndim == 2, f"Z(Y) train must have shape (n_train, z(y)_dim) got {zy_train.tensor.ndim}"

        zy_train = zy_train.to(self.device)
        y_train = y_train.to(self.device)

        # Compute the expectation of the r.v `z(y)` from the training dataset.
        mean_zy = zy_train.tensor.mean(axis=0)
        inv_projector = invariant_orthogonal_projector(self.out_type.representation).to(self.device)
        self.mean_zy = inv_projector @ mean_zy
        # Centered observables
        zy_train_c = zy_train.tensor - self.mean_zy

        # Compute the embeddings of the entire y training dataset. And the linear regression between z(y) and h(y)
        self.Czyhy = torch.zeros((zy_train.shape[-1], model.embedding_dim), device=self.device)
        hy_train = model.embedding_y(y_train).tensor  # shape: (n_train, embedding_dim)

        if analytic_residual and isinstance(model.embedding_y[0], ResidualEncoder):
            # Y is embedded in the encoded vector h(y), we can get the prediction using indexing.
            res_encoder = model.embedding_y[0]
            change2iso_module = model.embedding_y[-1]
            Qiso2y = change2iso_module.Qin2iso.T
            self.Czyhy = Qiso2y[res_encoder.residual_dims, :]
        else:  # Compute the symmetry aware linear regression from h(y) to y
            rep_Hy = model.embedding_y.out_type.representation
            rep_Zy = self.out_type.representation
            if lstsq:  # TODO: symmetry aware lstsq
                import linear_operator_learning as lol

                self.Czyhy = lol.nn.symmetric.linalg.lstsq(X=hy_train, Y=zy_train_c, rep_X=rep_Hy, rep_Y=rep_Zy)
            else:  # Symmetry aware basis expansion coefficients.
                import linear_operator_learning as lol

                self.Czyhy = lol.nn.symmetric.stats.covariance(X=hy_train, Y=zy_train_c, rep_X=rep_Hy, rep_Y=rep_Zy)

    def forward(self, x_cond: GeometricTensor):
        x_cond = x_cond.to(self.device)
        fx_cond = self.model.embedding_x(x_cond)  # shape: (n_test, embedding_dim)

        # Check formula 12 from https://arxiv.org/pdf/2407.01171
        Dr = self.model.truncated_operator
        zy_deflated_basis_expansion = torch.einsum("bf,fh,yh->by", fx_cond.tensor, Dr, self.Czyhy)
        zy_pred = self.mean_zy + zy_deflated_basis_expansion

        return self.out_type(zy_pred)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (self.out_type.size)
