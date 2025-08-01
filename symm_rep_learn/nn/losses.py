import torch
from torch import Tensor

from symm_rep_learn.mysc.statistics import cov_norm_squared_unbiased_estimation

# class CLoRaLoss(torch.nn.Module):
#     """Contrastive Low-Rank Approximation Loss.

#     Computes an estimate of the Hilbert-Schmidt norm of the difference between a conditional expectation operator
#     :math:`E` and its low-rank (*matrix*) approximation :math:`E_r`. The loss is computed via the kernel function
#     that defines the linear-integral operator.

#     The loss is defined as:
#     .. math::
#         || E - \mathbf{E}_r ||_{HS}^2 \leq -2 \mathbb{E}_p(x,y)[k_r(x,y)] + \mathbb{E}_p(x)\mathbb{E}_p(y)[k_r(x,y)^2]

#     TODO.
#     """

#     def __init__(self, orthogonality_reg: float, centering_reg: float):
#         super().__init__()
#         self.gamma = orthogonality_reg
#         self.beta = centering_reg

#     def forward(
#         self,
#         Dr: Tensor,
#         fx_c: Tensor,
#         hy_c: Tensor,
#         fx_mean: Tensor,
#         hy_mean: Tensor,
#         Cfx: Tensor,
#         Chy: Tensor,
#     ):
#         """TODO.

#         Args:
#             fx_c: (Tensor) of shape (..., r) *centered* embedding functions of a subspace of L^2(X)
#             hy_c: (Tensor) of shape (..., r) *centered* embedding functions of a subspace of L^2(Y)

#         Returns:
#             loss: ||E - E_r||_HS^2 <= -2 E_p(x,y)[k_r(x,y)] + E_p(x)p(y)[k_r(x,y)^2]
#             metrics: Scalar valued metrics to monitor during training.
#         """
#         assert Dr.shape == (fx_c.shape[-1], hy_c.shape[-1])
#         assert Cfx.shape == (fx_c.shape[-1], fx_c.shape[-1])
#         assert Chy.shape == (hy_c.shape[-1], hy_c.shape[-1])

#         # Orthonormal regularization and centering penalization _________________________________________
#         # orthonormal_reg_fx = ||Cx - I||_F^2 + 2 ||E_p(x) f(x)||_F^2
#         orthonormal_reg_x, metrics_x = orthonormality_regularization(x=fx_c, Cx=Cfx, x_mean=fx_mean, var_name="x")
#         # orthonormal_reg_hy = ||Cy - I||_F^2 + 2 ||E_p(y) h(y)||_F^2
#         orthonormal_reg_y, metrics_y = orthonormality_regularization(x=hy_c, Cx=Chy, x_mean=hy_mean, var_name="y")

#         metrics = metrics_x | metrics_y  # Combine metrics from both regularizations

#         # Operator truncation error = ||E - E_r||_HS^2 ____________________________________________________
#         # E_r = Cxy -> ||E - E_r||_HS - ||E||_HS = -2 ||Cxy||_F^2 + tr(Cxy Cy Cxy^T Cx)
#         clora_err, loss_metrics = contrastive_low_rank_loss(fx_c, hy_c, Dr)

#         metrics |= loss_metrics if loss_metrics is not None else {}
#         # Total loss ____________________________________________________________________________________
#         dx = self.embedding_dim_x
#         dy = self.embedding_dim_y
#         loss = clora_err + self.gamma * (orthonormal_reg_x / dx + orthonormal_reg_y / dy)
#         # Logging metrics _______________________________________________________________________________
#         with torch.no_grad():
#             metrics |= {
#                 "||E - E_r(x,y)||_HS": clora_err.detach().item(),
#             }
#         return loss, metrics


def contrastive_low_rank_loss(fx_c, hy_c, Dr) -> tuple[Tensor, dict]:
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
    Cx_fro_2 = cov_norm_squared_unbiased_estimation(x_c, False)
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
