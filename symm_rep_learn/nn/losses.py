import numpy as np
import torch
from torch import Tensor

from symm_rep_learn.mysc.statistics import cov_norm_squared_unbiased_estimation


def contrastive_low_rank_loss_memory_heavy(fx_c, hy_c, Dr) -> tuple[Tensor, dict]:
    """Implementation of ||E - E_r||_HS^2, while assuming E_r is a full matrix.

    Args:
        fx_c: (Tensor) of shape (n_samples, r_x) centered embedding functions spanning a subspace of L^2(X).
        hy_c: (Tensor) of shape (n_samples, r_y) centered embedding functions spanning a subspace of L^2(Y).
        Dr: (Tensor) of shape (r_x, r_y) representing the truncated operator.

    Returns:
        (Tensor) Low-rank approximation error loss.
        (dict) Metrics to monitor during training.
    """
    metrics = {}
    n_samples = fx_c.shape[0]
    # k_r(x,y) = 1 + f(x)^T Dr h(y)
    # truncated_err = -2 * E_p(x,y)[k_r(x,y)] + E_p(x)p(y)[k_r(x,y)^2]
    # TODO: This incurs in O(n^2) memory complextity, which wont scale for large n_samples.
    pmd_mat = 1 + torch.einsum("nx,xy,my->nm", fx_c, Dr, hy_c)  # (n_samples, n_samples)
    # E_p(x,y)[k_r(x,y)] = diag(pmd_mat).mean()
    E_pxy_kr = torch.diag(pmd_mat).mean()
    pmd_squared = pmd_mat**2
    # E_p(x)p(y)[k_r(x,y)^2]  # Note we remove the samples from the joint in the diagonal
    E_px_py_kr = (pmd_squared.sum() - pmd_squared.diag().sum()) / (n_samples * (n_samples - 1))
    truncation_err = (-2 * E_pxy_kr) + (E_px_py_kr) + 1

    with torch.no_grad():
        metrics |= {
            "E_p(x)p(y) k_r(x,y)^2": E_px_py_kr.detach() - 1,
            "E_p(x,y) k_r(x,y)": E_pxy_kr.detach() - 1,
        }

    return truncation_err, metrics


def contrastive_low_rank_loss_old(fx_c, hy_c, Dr) -> tuple[torch.Tensor, dict]:
    """Memory-efficient implementation of ||E - E_r||_HS^2, avoiding n^2 memory.

    This implementation uses the Gram matrix trick to compute E_p(x)p(y)[k_r(x,y)^2] without
    creating the full n x n kernel matrix pmd_mat. Instead of computing all pairwise kernel
    evaluations k_r(x_i, y_j) = 1 + f(x_i)^T Dr h(y_j), we reformulate the computation using:

    1. Gram matrices: Gx = f(x)^T @ f(x) (r_x x r_x) and Gy = h(y)^T @ h(y) (r_y x r_y)
    2. Matrix operations on smaller matrices to recover the required sums
    3. Diagonal vs off-diagonal separation to compute the correct expectation terms

    Args:
        fx_c: (Tensor) shape (n, r_x), centered embedding functions f(x) for X.
        hy_c: (Tensor) shape (n, r_y), centered embedding functions h(y) for Y.
        Dr: (Tensor) shape (r_x, r_y), representing the truncated operator.

    Returns:
        (Tensor) Low-rank approximation error loss.
        (dict) Metrics to monitor during training.
    """
    n_samples = fx_c.shape[0]
    U = fx_c @ Dr  # (n, r_y)
    s_diag = torch.sum(U * hy_c, dim=1)  # (n,)

    # E_p(x,y)[k_r(x,y)]
    E_pxy_kr = 1.0 + s_diag.mean()

    # For E_p(x)p(y)[k_r(x,y)^2], compute using Gram matrix trick
    Fx_sum = fx_c.sum(dim=0)  # (r_x,)
    Hy_sum = hy_c.sum(dim=0)  # (r_y,)
    sum_s_all = Fx_sum @ Dr @ Hy_sum  # scalar

    Gx = fx_c.T @ fx_c  # (r_x, r_x)
    Gy = hy_c.T @ hy_c  # (r_y, r_y)
    DrT_Dr = Dr.T @ Gx @ Dr  # (r_y, r_y)
    sum_s2_all = torch.sum(DrT_Dr * Gy)  # tr(Gy @ Dr^T Gx Dr)

    sum_s_diag = s_diag.sum()
    sum_s2_diag = torch.sum(s_diag**2)

    sum_k2_all = n_samples**2 + 2.0 * sum_s_all + sum_s2_all
    sum_k2_diag = n_samples + 2.0 * sum_s_diag + sum_s2_diag
    sum_k2_off = sum_k2_all - sum_k2_diag

    E_pxpy_k2 = sum_k2_off / (n_samples * (n_samples - 1))
    truncation_err = -2.0 * E_pxy_kr + E_pxpy_k2 + 1.0

    with torch.no_grad():
        metrics = {
            "E_p(x)p(y) k_r(x,y)^2": E_pxpy_k2 - 1,
            "E_p(x,y) k_r(x,y)": E_pxy_kr - 1,
        }

    return truncation_err, metrics


def contrastive_low_rank_loss(
    fx_c: Tensor,  # (B, r_x, *spatial)
    hy_c: Tensor,  # (B, r_y, *spatial)
    Dr: Tensor,  # (r_x, r_y)
) -> tuple[Tensor, dict]:
    """Memory-efficient implementation of ||E - E_r||_HS^2, avoiding n^2 memory.

    This implementation uses the Gram matrix trick to compute E_p(x)p(y)[k_r(x,y)^2] without
    creating the full n x n kernel matrix pmd_mat. Instead of computing all pairwise kernel
    evaluations k_r(x_i, y_j) = 1 + f(x_i)^T Dr h(y_j), we reformulate the computation using:

    1. Gram matrices: Gx = f(x)^T @ f(x) (r_x x r_x) and Gy = h(y)^T @ h(y) (r_y x r_y)
    2. Matrix operations on smaller matrices to recover the required sums
    3. Diagonal vs off-diagonal separation to compute the correct expectation terms

    Args:
        fx_c: (Tensor) shape (n, r_x, ...), centered embedding functions f(x) for X. Where `...` denote spatial/temporal dimensions if any.
        hy_c: (Tensor) shape (n, r_y, ...), centered embedding functions h(y) for Y. Where `...` denote the same spatial/temporal dimensions as x.
        Dr: (Tensor) shape (r_x, r_y), representing the truncated operator.

    Returns:
        (Tensor) Low-rank approximation error loss. The loss is averaged over the spatial dimensions (...).
    """

    # ---------- shapes & convenience ----------
    B, r_x, *spatial = fx_c.shape  # spatial may be []
    r_y = hy_c.size(1)
    S = fx_c[0, 0].numel()  # Product of spatial dimensions

    # reshape: batch dim first, latent dim second, spatial flattened last
    fx_flat = fx_c.reshape(B, r_x, S)  # (B, r_x, S)
    hy_flat = hy_c.reshape(B, r_y, S)  # (B, r_y, S)

    # ---------- core algebra (vectorised over S) ----------
    # U = f · D  -> (B, r_y, S)
    U_flat = torch.einsum("brs,rc->bcs", fx_flat, Dr)
    s_diag = (U_flat * hy_flat).sum(dim=1)  # (B, S)

    E_pxy_kr = 1.0 + s_diag.mean(dim=0)  # (S,)

    # batch sums (for ∑_i,j  )
    Fx_sum = fx_flat.sum(dim=0)  # (r_x, S)
    Hy_sum = hy_flat.sum(dim=0)  # (r_y, S)
    sum_s_all = (Fx_sum.t() @ Dr * Hy_sum.t()).sum(dim=1)  # (S,)

    # Gram blocks, computed per-location in parallel
    H_b = hy_flat.permute(2, 0, 1)  # (S, B, r_y)
    U_b = U_flat.permute(2, 0, 1)  # (S, B, r_y)
    G_H = torch.bmm(H_b.transpose(1, 2), H_b)  # (S, r_y, r_y)
    G_U = torch.bmm(U_b.transpose(1, 2), U_b)  # (S, r_y, r_y)
    sum_s2_all = (G_H * G_U).sum(dim=(1, 2))  # (S,)

    sum_s_diag = s_diag.sum(dim=0)  # (S,)
    sum_s2_diag = (s_diag**2).sum(dim=0)  # (S,)

    n = B
    sum_k2_all = n * n + 2.0 * sum_s_all + sum_s2_all
    sum_k2_diag = n + 2.0 * sum_s_diag + sum_s2_diag
    E_pxpy_k2 = (sum_k2_all - sum_k2_diag) / (n * (n - 1))

    trunc_err_vec = -2.0 * E_pxy_kr + E_pxpy_k2 + 1.0  # (S,)

    # ---------- reshape to spatial dimensions ----------
    if spatial:  # e.g. (H,W,…)
        loss = trunc_err_vec.reshape(*spatial)  # Returns tensor of shape spatial
    else:  # no spatial dims
        loss = trunc_err_vec.squeeze()  # scalar tensor

    # ---------- metrics ----------
    with torch.no_grad():
        metrics = {
            "E_p(x,y) k_r(x,y)": E_pxy_kr.mean() - 1.0,
            "E_p(x)p(y) k_r(x,y)^2": E_pxpy_k2.mean() - 1.0,
        }

    return loss, metrics


def orthonormality_regularization(x, Cx: Tensor = None, x_mean: Tensor = None, var_name="x"):
    """Computes orthonormality and centering regularization penalization for a batch of feature vectors.

    Computes finite sample unbiased empirical estimates of the term:
    || Vx - I ||_F^2 = || Cx - I ||_F^2 + 2 || E_p(x) f(x) ||^2
                     = tr(Cx^2) - 2 tr(Cx) + r + 2 || E_p(x) f(x) ||^2
    Where Vx ∈ R^{r+1 x r+1} is the uncentered covariance matrix assuming the constant function is included as the
    first fn. Cx ∈ R^{r x r} is the centered covariance matrix of the feature vectors f_c(x) = f(x) - mean(f(x)).
    That is, [Vx]_ij = E_p(x) f_i(x) f_j(x) and [Cx]_ij = E_p(x) f_c,i(x) f_c,j(x), where f_c = f - E_p(x) f(x).

    The unbiased estimate requires to have two sample sets of marginal distribution p(x), such that
    D = {x_1, ..., x_n} and D' = {x'_1, ..., x'_n} are independent sampling sets from p(x). To achieve this in
    practice we shuffle D to obtain D' and to get f(x) and f(x').
    Then we have that the unbiased estimate is computed by:

    || Vx - I ||_F^2 ≈ E_(x,x')~p(x) [(f_c(x).T f_c(x'))^2] - 2 tr(Cx) + r + 2 ||f_mean||^2

    Args:
        x: (Tensor) of shape (n_samples, r) feature vectors. Assumed to be centered if x_mean is provided.
        Cx: (Tensor) of shape (r, r) covariance matrix of the feature vectors. If None, it is computed from x.
        x_mean: (Tensor) of shape (1, r) mean of the feature vectors. If None, it is computed from x.
        var_name: (str) Name of the variable for metric names (e.g., 'x' or 'y').

    Returns:
        Regularization term as a scalar tensor.
        metrics: (dict) Dictionary containing the computed metrics for monitoring during training.
    """
    if x_mean is None:
        x_mean = x.mean(dim=0, keepdim=True).to(x.dtype, x.device)  # Compute mean if not provided
        x_c = x - x_mean  # Centered feature vectors
    else:
        x_c = x  # Assume x is already centered

    dim_x = x.shape[-1]  # Embedding dimension

    if Cx is None:
        Cx = torch.cov(x_c.T, correction=0).to(x.dtype, x.device)  # Covariance matrix if not provided

    # Compute unbiased empirical estimates ||Cx||_F^2 = E_(x,x')~p(x) [(f_c(x).T f_c(x'))^2]
    # Cx_fro_2 = cov_norm_squared_unbiased_estimation(x_c, False)
    Cx_fro_2 = torch.linalg.matrix_norm(Cx, ord="fro") ** 2
    tr_Cx = torch.trace(Cx)  # E[f_c(x)^T f_c(x)] =  tr(Cx)
    fx_centering_loss = (x_mean**2).sum()  # ||E_p(x) (f(x_i))||^2

    # orthonormality_x = tr(Cx^2) - 2 tr(Cx)  + 2 || E_p(x) f(x) ||^2 + dim_x
    orthonormality_x = Cx_fro_2 - 2 * tr_Cx + dim_x + 2 * fx_centering_loss

    with torch.no_grad():
        # Divide by the embedding dimension to standardize metrics across experiments.
        metrics = {
            f"||mu_{var_name}||": (torch.sqrt(fx_centering_loss) / dim_x).item(),
            f"||V{var_name} - I||_F^2": (orthonormality_x / dim_x).item(),
        }

    return orthonormality_x, metrics


import escnn
import symm_learning


def symm_orthonormality_regularization(
    x: torch.Tensor, rep_x: escnn.group.Representation, Cx: Tensor = None, x_mean: Tensor = None, var_name="x"
):
    """Computes orthonormality and centering regularization penalization for a batch of feature vectors.

    Computes finite sample unbiased empirical estimates of the term:
    || Vx - I ||_F^2 = || Cx - I ||_F^2 + 2 || E_p(x) f(x) ||^2
                        = || ⊕_k Cx_k - I_r_k ||_F^2 + 2 || E_p(x) f^inv (x) ||^2
                        = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || Cx_k - I_r_k ||_F^2
                        = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || Cx_k ||_F^2 - 2tr(Cx_k) + r_k
                        = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || (Dx_k ⊗ I_ρk) ||_F^2 - 2tr(Dx_k ⊗ I_ρk) + r_k
                        = 2 || E_p(x) f^inv (x) ||^2 + Σ_k |ρk| (||Dx_k||_F^2 - 2tr(Dx_k)) + r_k

    Args:
        x: (GeometricTensor) of shape (n_samples, r) feature vectors. Assumed to be centered if x_mean is provided.
        Cx: (Tensor) of shape (r, r) covariance matrix of the feature vectors. If None, it is computed from x.
        x_mean: (Tensor) of shape (1, r) mean of the feature vectors. If None, it is computed from x.
        var_name: (str) Name of the variable for metric names (e.g., 'x' or 'y').

    Returns:
        Regularization term as a scalar tensor.
    """
    assert rep_x.attributes["in_isotypic_basis"], "Representation must have isotypic basis."

    if x_mean is None:
        if "invariant_orthogonal_projector" not in rep_x.attributes:
            symm_learning.stats.var_mean(x, rep_x)  #  Compute the inv projector
        P_inv = rep_x.attributes["invariant_orthogonal_projector"].to(x.dtype, x.device)
        x_mean = x.mean(dim=0, keepdim=True).to(x.dtype, x.device)  # Compute mean if not provided
        x_c = x - torch.einsum("ij,...j->...i", P_inv, x_mean)  # Centered feature vectors
    else:
        x_c = x  # Assume x is already centered

    dim_x = x.shape[-1]  # Embedding dimension

    if Cx is None:
        Cx = symm_learning.stats.cov(x=x_c, y=x_c, rep_x=rep_x, rep_y=rep_x)

    # Project embedding into the isotypic subspaces
    #   def _orth_proj_isotypic_subspaces(self, z: GeometricTensor) -> list[torch.Tensor]:
    # """Compute the orthogonal projection of the input tensor into the isotypic subspaces."""
    # z_iso = [z.tensor[..., s:e] for s, e in zip(z.type.fields_start, z.type.fields_end)]
    # return z_iso
    # x_c_iso = [x_c[..., idx] for idx in rep_x.attributes["isotypic_subspace_dims"].values()]
    # reps_x_iso = rep_x.attributes["isotypic_reps"].values()
    iso_subspaces_dims = rep_x.attributes["isotypic_subspace_dims"]
    Cx_iso_fro_2 = []
    trCx_iso = []
    for k, (irrep_id, rep_x_k) in enumerate(rep_x.attributes["isotypic_reps"].items()):
        irrep_dim = rep_x_k.size // len(rep_x_k.irreps)
        x_c_k = x_c[..., iso_subspaces_dims[irrep_id]]
        r_xk = x_c_k.shape[-1]
        # Flatten the realizations along irreducible subspaces, while preserving sampling from the joint dist.
        zx = symm_learning.linalg.isotypic_signal2irreducible_subspaces(x_c_k, rep_x_k)
        # Compute unbiased empirical estimates ||Dx_k||_F^2
        Dx_k_fro_2 = cov_norm_squared_unbiased_estimation(zx, False)
        # Trace terms without need of unbiased estimation
        Dx_k = Cx[iso_subspaces_dims[irrep_id], iso_subspaces_dims[irrep_id]]
        tr_Dx_k = torch.trace(Dx_k)  # tr(Dx_k)
        #  ||Cx_k||_F^2 := |ρk| (||Dx_k||_F^2 - 2tr(Dx_k)) + r_k
        Cx_k_fro_2 = irrep_dim * (Dx_k_fro_2 - 2 * tr_Dx_k) + r_xk
        Cx_iso_fro_2.append(Cx_k_fro_2)
        trCx_iso.append(tr_Dx_k)
        # Cx_k_fro_2_biased = torch.linalg.matrix_norm(self.Cx(k)) ** 2

    Cx_I_err_fro_2 = sum(Cx_iso_fro_2)  # ||Cx - I||_F^2 = Σ_k ||Cx_k - I_r_k||_F^2,
    trCx = sum(trCx_iso)  # tr(Cx) = Σ_k tr(Dx_k)
    # Cx_fro_2_biased = torch.linalg.matrix_norm(self.Cx(None)) ** 2

    x_centering_loss = (x_mean**2).sum()  # ||E_p(x) (f(x_i))||^2

    # ||Vx - I||_F^2 = ||Cx - I||_F^2 + 2||E_p(x) f(x)||^2
    orthonormality_x = Cx_I_err_fro_2 + 2 * x_centering_loss

    with torch.no_grad():
        metrics = {
            f"||mu_{var_name}||": torch.sqrt(x_centering_loss),
            f"||V{var_name} - I||_F^2": orthonormality_x / dim_x,
        }

    return orthonormality_x, metrics


if __name__ == "__main__":
    print("Testing spatial loss function: each location should match non-spatial loss")
    print("=" * 65)

    # Fixed parameters
    B, r_x, r_y = 128, 8, 6
    torch.manual_seed(42)

    # Test 1: Compare spatial (3x4) with individual non-spatial computations
    spatial_shape = (3, 4)
    total_spatial = spatial_shape[0] * spatial_shape[1]

    # Generate data
    base_data_x = torch.randn(B, r_x, total_spatial, dtype=torch.float32)
    base_data_y = torch.randn(B, r_y, total_spatial, dtype=torch.float32)
    Dr = torch.randn(r_x, r_y, dtype=torch.float32)

    # Reshape to spatial
    fx_spatial = base_data_x.reshape(B, r_x, *spatial_shape)  # (B, r_x, 3, 4)
    hy_spatial = base_data_y.reshape(B, r_y, *spatial_shape)  # (B, r_y, 3, 4)

    # Get spatial loss (should return tensor of shape (3, 4))
    spatial_losses, _ = contrastive_low_rank_loss(fx_spatial, hy_spatial, Dr)

    print(f"Spatial losses shape: {spatial_losses.shape}")
    print(f"Expected shape: {spatial_shape}")

    # Compute individual losses for each spatial location
    individual_losses = torch.zeros(*spatial_shape)
    tolerance = 1e-6
    all_match = True

    for i in range(spatial_shape[0]):
        for j in range(spatial_shape[1]):
            # Extract data for this spatial location
            idx = i * spatial_shape[1] + j
            fx_single = base_data_x[:, :, idx]  # (B, r_x)
            hy_single = base_data_y[:, :, idx]  # (B, r_y)

            # Compute loss using non-spatial function
            single_loss, _ = contrastive_low_rank_loss_memory_heavy(fx_single, hy_single, Dr)
            individual_losses[i, j] = single_loss.item()

            # Compare with spatial loss at this location
            spatial_loss_val = spatial_losses[i, j].item()
            assert np.isclose(single_loss, spatial_loss_val, atol=1e-5, rtol=1e-5)
