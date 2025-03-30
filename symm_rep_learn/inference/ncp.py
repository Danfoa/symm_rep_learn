# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 29/03/25
import torch

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
