import torch


def cross_cov_norm_squared_unbiased_estimation(x: torch.Tensor, y: torch.Tensor):
    """Compute the unbiased estimation of ||Cxy||_F^2 from a batch of samples.

    Given the Covariance matrix Cxy = E_p(x,y) [x.T y], this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from a single batch of paired samples.

    ||Cxy||_F^2 = tr(Cxy^T Cxy) = Σ_i Σ_j (E_(x,y)~p(x,y) [x_i y_j]) (E_(x',y')~p(x,y) [x_j y_i'])
                 = E_((x,y),(x',y'))~p(x,y) [(x.T y') (x'.T y)]
                 = 1/(N(N-1)) Σ_{n!=m} [(x_n.T y_m) (x_m.T y_n)]

    The diagonal terms n = m would contribute (x_n.T y_n)^2, which reuse the same joint sample twice.
    Those self-pairs are not distributed as two independent draws from p(x, y), so they must be removed.

    Args:
        x: (n_samples, r) Centered realizations of a random variable x = [x_1, ..., x_r].
        y: (n_samples, r) Centered realizations of a random variable y = [y_1, ..., y_r].

    Returns:
        cov_fro_norm: (torch.Tensor) Unbiased estimation of ||Cxy||_F^2.
    """
    n_samples = x.shape[0]

    # Sum over all ordered pairs (n, m): (x_n^T y_m) (x_m^T y_n).
    total_sum = torch.einsum("nj,mj,mk,nk->", x, y, x, y)
    # Subtract the diagonal self-pairs Σ_n (x_n^T y_n)^2.  The non-linear bias of the estimaiton.
    diag_sum = torch.einsum("nj,nj,nk,nk->", x, y, x, y)
    cov_fro_norm = (total_sum - diag_sum) / (n_samples * (n_samples - 1))
    return cov_fro_norm


def cov_norm_squared_unbiased_estimation(x: torch.Tensor):
    """Compute the unbiased estimation of ||Cx||_F^2 from a batch of samples.

    Given the Covariance matrix Cx = E_p(x) [x.T x], this function computes an unbiased estimation
    of the Frobenius norm of the covariance matrix from a single sampling set.

    ||Cx||_F^2 = tr(Cx^TCx) = Σ_i Σ_j (E_(x) [x_i x_j]) (E_(x') [x_j x_i'])
                 = E_(x,x')~p(x) [(x.T x')^2]
                 = 1/(N(N-1)) Σ_{n!=m} [(x_n.T x_m)^2]

    Args:
        x: (n_samples, r) Centered realizations of a random variable x = [x_1, ..., x_r].

    Returns:
        cov_fro_norm: (torch.Tensor) Unbiased estimation of ||Cx||_F^2.
    """
    return cross_cov_norm_squared_unbiased_estimation(x=x, y=x)


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
    val_cross = cross_cov_norm_squared_unbiased_estimation(x, y)
    print("cross-cov:", val_cross.item())

    # 4. Test covariance norm squared
    val_cov = cov_norm_squared_unbiased_estimation(x)
    print("cov      :", val_cov.item())


if __name__ == "__main__":
    test_cross_cov_and_cov()
