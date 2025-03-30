# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 29/03/25
import torch

from symm_rep_learn.models.ncp import NCP


@torch.no_grad()
def ncp_regression(model: NCP, x_cond, y_train, zy_train, lstsq=False, analytic_residual=False):
    """Use the NCP model to regress an observation of the random variable `y`.

    Given the observable z(y) this function computes the conditional expectation of the observable given the
    conditioning variable `x`, that is: E_y|x [z(y) | x]

    """
    assert zy_train.shape[0] == y_train.shape[0], "Y train and Z(Y) train must have the same number of samples"
    assert y_train.ndim == 2, f"Y train must have shape (n_train, y_dim) {y_train.ndim}"
    assert zy_train.ndim == 2, f"Z(Y) train must have shape (n_train, z(y)_dim) got {zy_train.ndim}"
    device = next(model.parameters()).device

    x_cond = x_cond.to(device)
    zy_train = zy_train.to(device)
    y_train = y_train.to(device)

    n_train = zy_train.shape[0]

    # Compute the expectation of the r.v `z(y)` from the training dataset.
    mean_zy = zy_train.mean(axis=0)
    zy_train_c = zy_train - mean_zy

    # Compute the embeddings of the entire y training dataset. And the linear regression between z(y) and h(y)
    Czyhy = torch.zeros((zy_train.shape[-1], model.embedding_dim), device=device)

    hy_train = model.embedding_y(y_train)  # shape: (n_train, embedding_dim)
    from symm_rep_learn.nn.layers import ResidualEncoder

    if analytic_residual and isinstance(model.embedding_y, ResidualEncoder):
        y_dims_in_hy = model.embedding_y.residual_dims
        for dim in range(y_dims_in_hy.start, y_dims_in_hy.stop):
            Czyhy[dim, dim] = 1
    else:
        if lstsq:
            out = torch.linalg.lstsq(hy_train, zy_train_c)
            Czyhy = out.solution.T
            assert Czyhy.shape == (zy_train.shape[-1], hy_train.shape[-1]), f"Invalid shape {Czyhy.shape}"
        else:  # Compute empirical expectation
            Czyhy = (1 / n_train) * torch.einsum("by,bh->yh", zy_train_c, hy_train)

    fx_cond = model.embedding_x(x_cond)  # shape: (n_test, embedding_dim)

    # Check formula 12 from https://arxiv.org/pdf/2407.01171
    Dr = model.truncated_operator
    zy_deflated_basis_expansion = torch.einsum("bf,fh,yh->by", fx_cond, Dr, Czyhy)
    zy_pred = mean_zy + zy_deflated_basis_expansion

    return zy_pred
