import torch


def cross_cov_norm_squared_unbiased_estimation(
    x: torch.Tensor, y: torch.Tensor, return_inner_prods=False, permutation=None
):
    """Compute the unbiased estimation of ||Cxy||_F^2 from a batch of samples.

    Given the Covariance matrix Cxy = E_p(x,y) [x.T y], this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from two independent sampling sets.

    ||Cxy||_F^2 = tr(Cxy^T Cxy) = Σ_i Σ_j (E_(x,y)~p(x,y) [x_i y_j]) (E_(x',y')~p(x,y) [x_j y_i'])
                 = E_((x,y),(x',y'))~p(x,y) [(x.T y') (x'.T y)]
                 = 1/N^2 Σ_n Σ_m [(x_n.T y'_m) (x'_m.T y_n)]

    Args:
        x: (n_samples, r) Centered realizations of a random variable x = [x_1, ..., x_r].
        y: (n_samples, r) Centered realizations of a random variable y = [y_1, ..., y_r].
        return_inner_prods: (bool) If True, return intermediate inner products.
        permutation: (torch.Tensor) (Optional) Permutation of the samples.

    Returns:
        cov_fro_norm: (torch.Tensor) Unbiased estimation of ||Cxy||_F^2.
        x_yp: (torch.Tensor) (Optional) Inner product E_(x,y')~p(x,y) (x.T y') of shape (n_samples, n_samples).
        xp_y: (torch.Tensor) (Optional) Inner product E_(x',y)~p(x,y) (x'.T y) of shape (n_samples, n_samples).
    """
    n_samples = x.shape[0]

    # Permute the rows independently to simulate independent sampling
    perm = permutation if permutation is not None else torch.randperm(n_samples)
    xp = x[perm]  # Independent sampling of x'
    yp = y[perm]  # Independent sampling of y'

    if return_inner_prods:
        # Compute the inner products
        x_yp = torch.einsum("ij,kj->ik", x, yp)  # (n_samples, n_samples) x_yp_n,m = (x_n.T y'_m)
        xp_y = torch.einsum("ik,jk->ij", xp, y)  # (n_samples, n_samples) xp_y_n,m = (x'_n.T y_m)
        # Compute 1/N^2 Σ_n Σ_m [(x_n.T y'_m) (x'_m.T y_n)]
        cov_fro_norm = torch.trace(torch.mm(x_yp, xp_y)) / (n_samples**2)
        return cov_fro_norm, (x_yp, xp_y)
    else:  # Equivalent einsum implementation, without intermediate storage
        # Compute 1/N^2 Σ_n Σ_m [(x_n.T y'_m) (x'_m.T y_n)]
        val = torch.einsum("nj,mj,mk,nk->", x, yp, xp, y)
        cov_fro_norm = val / (n_samples**2)
        return cov_fro_norm


def cov_norm_squared_unbiased_estimation(x: torch.Tensor, return_inner_prod=False, permutation=None):
    """Compute the unbiased estimation of ||Cx||_F^2 from a batch of samples.

    Given the Covariance matrix Cx = E_p(x) [x.T x], this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from a single sampling set.

    ||Cx||_F^2 = tr(Cx^TCx) = Σ_i Σ_j (E_(x) [x_i x_j]) (E_(x') [x_j x_i'])
                 = E_(x,x')~p(x) [(x.T x')^2]
                 = 1/N^2 Σ_n Σ_m [(x_n.T x'_m)^2]

    Args:
        x: (n_samples, r) Centered realizations of a random variable x = [x_1, ..., x_r].
        return_inner_prod: (bool) If True, return intermediate inner products.

    Returns:
        cov_fro_norm: (torch.Tensor) Unbiased estimation of ||Cx||_F^2.
        xxp: (torch.Tensor) (Optional) Inner product E_(x,x')~p(x) (x.T x') of shape (n_samples, n_samples).
    """
    n_samples = x.shape[0]

    perm = permutation if permutation is not None else torch.randperm(n_samples)
    xp = x[perm]  # Independent sampling of x'

    if return_inner_prod:
        x_xp = torch.einsum("ij,kj->ik", x, xp)  # (n_samples, n_samples) x_xp_n,m = (x_n.T x'_m)
        # Compute 1/N^2 Σ_n Σ_m [(x_n.T x'_m)^2]
        cov_fro_norm = (x_xp**2).sum() / (n_samples**2)
        return cov_fro_norm, x_xp
    else:  # Equivalent einsum implementation, without intermediate storage
        return cross_cov_norm_squared_unbiased_estimation(x=x, y=x, return_inner_prods=False)


def test_cross_cov_and_cov():
    # 1. Generate random data
    torch.manual_seed(42)
    N, r = 100, 5  # e.g., 100 samples, dimension 5
    x = torch.randn(N, r)
    y = torch.randn(N, r)

    # 2. Center the data
    x -= x.mean(dim=0, keepdim=True)
    y -= y.mean(dim=0, keepdim=True)

    # 3. Test cross-covariance norm squared
    val_single_einsum = cross_cov_norm_squared_unbiased_estimation(x, y, return_inner_prods=False)
    val_two_step, (x_yp, xp_y) = cross_cov_norm_squared_unbiased_estimation(x, y, return_inner_prods=True)

    diff_cross = (val_single_einsum - val_two_step).abs().item()
    print("cross-cov single-einsum:", val_single_einsum.item())
    print("cross-cov two-step     :", val_two_step.item())
    print("Difference (cross-cov) :", diff_cross)

    # 4. Test covariance norm squared
    val_single_einsum_cov = cov_norm_squared_unbiased_estimation(x, return_inner_prod=False)
    val_two_step_cov, x_xp = cov_norm_squared_unbiased_estimation(x, return_inner_prod=True)

    diff_cov = (val_single_einsum_cov - val_two_step_cov).abs().item()
    print("cov single-einsum  :", val_single_einsum_cov.item())
    print("cov two-step       :", val_two_step_cov.item())
    print("Difference (cov)   :", diff_cov)


if __name__ == "__main__":
    test_cross_cov_and_cov()
