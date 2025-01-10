# Created by danfoa at 19/12/24
from __future__ import annotations
import escnn
import numpy as np
import torch
from escnn.group import change_basis, directsum, IrreducibleRepresentation, Representation


def symmetric_moments(x: torch.Tensor | np.numpy, rep_X: Representation) -> [torch.Tensor, torch.Tensor]:
    """Compute the mean and standard deviation of observations with known symmetry representations """
    assert len(x.shape) == 2, f"Expected x to have shape (n_samples, n_features), got {x.shape}"
    x = torch.Tensor(x) if isinstance(x, np.ndarray) else x
    G = rep_X.group
    # Allocate the mean and variance arrays.
    mean, var = torch.zeros(rep_X.size), torch.ones(rep_X.size)
    # Change basis of the observation to expose the irrep G-stable subspaces
    Qx_T, Qx = torch.Tensor(rep_X.change_of_basis_inv), torch.Tensor(rep_X.change_of_basis)

    # Get the dimensions of each irrep.
    S = torch.zeros((rep_X.size, rep_X.size))
    irreps_dimension = []
    cum_dim = 0
    for irrep_id in rep_X.irreps:
        irrep = G.irrep(*irrep_id)
        # Get dimensions of the irrep in the original basis
        irrep_dims = range(cum_dim, cum_dim + irrep.size)
        irreps_dimension.append(irrep_dims)
        if irrep_id == G.trivial_representation.id:
            S[irrep_dims, irrep_dims] = 1
        cum_dim += irrep.size

    # Compute the mean of the observation.
    # The mean of a symmetric random variable (rv) lives in the subspaces associated with the trivial/inv irreps.
    has_trivial_irreps = G.trivial_representation.id in rep_X.irreps
    if has_trivial_irreps:
        avg_projector = Qx @ S @ Qx_T
        # Compute the mean in a single vectorized operation
        mean_empirical = torch.mean(x, dim=0)
        # Project to the inv-subspace and map back to the original basis
        mean = torch.einsum('...ij,...j->...i', avg_projector, mean_empirical)

    # Compute the variance of the observable by computing a single variance per irrep G-stable subspace.
    # To do this, we project the observations to the basis exposing the irreps, compute the variance per
    # G-stable subspace, and map the variance back to the original basis.
    x_iso_centered = torch.einsum('...ij,...j->...i', Qx_T, x - mean)
    var_irrep_basis = torch.ones_like(var)
    for irrep_id, irrep_dims in zip(rep_X.irreps, irreps_dimension):
        irrep = G.irrep(*irrep_id)
        x_irrep_centered = x_iso_centered[..., irrep_dims]
        assert x_irrep_centered.shape[-1] == irrep.size, \
            f"Obs irrep shape {x_irrep_centered.shape} != {irrep.size}"
        # Since the irreps are unitary/orthogonal transformations, we are constrained compute a unique variance
        # for all dimensions of the irrep G-stable subspace, as scaling the dimensions independently would break
        # the symmetry of the rv. As a centered rv the variance is the expectation of the squared rv.
        var_irrep = torch.mean(x_irrep_centered ** 2)  # Single scalar variance per G-stable subspace
        # Ensure the multipliticy of the variance is equal to the dimension of the irrep.
        var_irrep_basis[irrep_dims] = var_irrep
    # Convert the variance from the irrep/spectral basis to the original basis
    Cov = Qx @ torch.diag(var_irrep_basis) @ Qx_T
    var = torch.diagonal(Cov)

    # TODO: Move this check to Unit test as it is computationally demanding to check this at runtime.
    # Ensure the mean is equivalent to computing the mean of the orbit of the recording under the group action
    # aug_obs = []
    # for g in G.elements:
    #     g_obs = np.einsum('...ij,...j->...i', rep_obs(g), obs_original_basis)
    #     aug_obs.append(g_obs)
    #
    # aug_obs = np.concatenate(aug_obs, axis=0)   # Append over the trajectory dimension
    # mean_emp = np.mean(aug_obs, axis=(0, 1))
    # assert np.allclose(mean, mean_emp, rtol=1e-3, atol=1e-3), f"Mean {mean} != {mean_emp}"

    # var_emp = np.var(aug_obs, axis=(0, 1))
    # assert np.allclose(var, var_emp, rtol=1e-2, atol=1e-2), f"Var {var} != {var_emp}"
    return mean, var

def isotypic_cross_cov(
        X: torch.Tensor, 
        Y: torch.Tensor, 
        rep_X: Representation, 
        rep_Y: Representation, 
        centered=True,
        ):
    """Cross covariance of signals between isotypic subspaces of the same type.

    This function exploits the fact that the cross-covariance of signals between isotypic subspaces of the same type
    is constrained to be of the block form:

    Cxy = Cov(X, Y) = Dxy ⊗ I_d, where d = dim(irrep) and Dxy ∈ R^{mχ x my}  and Cyx ∈ R^{(mx * d) x (my * d)}

    Being mx and my the multiplicities of the irrep in X and Y respectively. This implies that the matrix Dxy
    represents the free parameters of the cross-covariance we are required to estimate. To do so we reshape
    the signals X ∈ R^{n x (mx x d)} and Y ∈ R^{n x (my x d)} to X_sing ∈ R^{(d x n) x mχ} and Y_sing ∈ R^{(d * n) x my}
    respectively. Ensuring all dimensions of the irreducible subspaces associated to each multiplicity of the irrep are
    considered as a single dimension for estimating Dxy = 1/(n*d) X_sing^T Y_sing.

    Args:
        X: torch.Tensor, shape (..., n, mx * d) where n is the number of samples and mx the multiplicity of the irrep in X.
        Y: torch.Tensor, shape (..., n, my * d) where n is the number of samples and my the multiplicity of the irrep in Y.
        rep_X: escnn.nn.Representation, composed of mx copies of an irrep of type k. rep_X = ⊕_i^mx ρ_k
        rep_Y: escnn.nn.Representation, composed of my copies of an irrep of type k. rep_Y = ⊕_i^my ρ_k
        centered: bool, whether to center the signals before computing the cross-covariance.
    Returns:
        Cxy: torch.Tensor, (mx * d, my * d) the cross-covariance matrix between the isotypic subspaces of X and Y.
        Dxy: torch.Tensor, (mx, my) free parameters of the cross-covariance matrix in the isotypic basis.
    """
    assert len(rep_X._irreps_multiplicities) == len(rep_Y._irreps_multiplicities) == 1, \
        f"Expected group representation of an isotypic subspace.I.e., with only one type of irrep. \nFound: " \
        f"{list(rep_X._irreps_multiplicities.keys())} in rep_X, {list(rep_Y._irreps_multiplicities.keys())} in rep_Y."
    assert rep_X.group == rep_Y.group, f"{rep_X.group} != {rep_Y.group}"
    irrep_id  = rep_X.irreps[0]  # Irrep id of the isotypic subspace
    assert irrep_id == rep_Y.irreps[0], \
        f"Irreps {irrep_id} != {rep_Y.irreps[0]}. Hence signals are orthogonal and Cxy=0."
    assert rep_X.size == X.shape[-1], f"Expected signal shape to be (..., {rep_X.size}) got {X.shape}"
    assert rep_Y.size == Y.shape[-1], f"Expected signal shape to be (..., {rep_Y.size}) got {Y.shape}"

    # Get information about the irreducible representation present in the isotypic subspace
    irrep_dim = rep_X.group.irrep(*irrep_id).size
    mk_X = rep_X._irreps_multiplicities[irrep_id]   # Multiplicity of the irrep in X
    mk_Y = rep_Y._irreps_multiplicities[irrep_id]   # Multiplicity of the irrep in Y

    # If required we must change bases to the isotypic bases.
    Qx_T, Qx = rep_X.change_of_basis_inv, rep_X.change_of_basis
    Qy_T, Qy = rep_Y.change_of_basis_inv, rep_Y.change_of_basis
    x_in_iso_basis = np.allclose(Qx_T, np.eye(Qx_T.shape[0]), atol=1e-6, rtol=1e-4)
    y_in_iso_basis = np.allclose(Qy_T, np.eye(Qy_T.shape[0]), atol=1e-6, rtol=1e-4)
    if x_in_iso_basis:
        X_iso = X
    else:
        Qx_T = torch.Tensor(Qx_T).to(device=X.device, dtype=X.dtype)
        Qx = torch.Tensor(Qx).to(device=X.device, dtype=X.dtype)
        X_iso = torch.einsum('...ij,...j->...i', Qx_T, X)   # x_iso = Q_x2iso @ x
    if np.allclose(Qy_T, np.eye(Qy_T.shape[0]), atol=1e-6, rtol=1e-4):
        Y_iso = Y
    else:
        Qy_T = torch.Tensor(Qy_T).to(device=Y.device, dtype=Y.dtype)
        # Qy = torch.Tensor(Qy).to(device=Y.device, dtype=Y.dtype)
        Y_iso = torch.einsum('...ij,...j->...i', Qy_T, Y)   # y_iso = Q_y2iso @ y

    if irrep_dim > 1:
        # Since Cxy = Dxy ⊗ I_d  , d = dim(irrep) and D_χy ∈ R^{mχ x my}
        # We compute the constrained cross-covariance, by estimating the matrix D_χy
        # This requires reshape X_iso ∈ R^{n x p} to X_sing ∈ R^{nd x mχ} and Y_iso ∈ R^{n x q} to Y_sing ∈ R^{nd x my}
        # Ensuring all samples from dimensions of a single irrep are flattened into a row of X_sing and Y_sing
        X_sing = X_iso.view(-1, mk_X, irrep_dim).permute(0, 2, 1).reshape(-1, mk_X)
        Y_sing = Y_iso.view(-1, mk_Y, irrep_dim).permute(0, 2, 1).reshape(-1, mk_Y)
    else: # For one dimensional (real) irreps, this defaults to the standard cross-covariance
        X_sing, Y_sing = X_iso, Y_iso

    if centered and irrep_id == rep_X.group.trivial_representation.id:  # Non-trivial isotypic subspace are centered
        X_sing = X_sing - torch.mean(X_sing, dim=0, keepdim=True)
        Y_sing = Y_sing - torch.mean(Y_sing, dim=0, keepdim=True)

    n_samples = X_sing.shape[0]
    assert n_samples == X.shape[0] * irrep_dim
    
    Dxy = (1 / (n_samples - 1)) * torch.einsum('...i,...j->ij', X_sing, Y_sing)
    if irrep_dim > 1:  # Broadcast the estimates according to Cxy = Dxy ⊗ I_d.
        I_d = torch.eye(irrep_dim, device=Dxy.device, dtype=Dxy.dtype)
        Cxy_iso = torch.kron(Dxy, I_d)          
    else:
        Cxy_iso = Dxy

    # Change back to original basis if needed _______________________
    if not x_in_iso_basis:
        Cxy = Qx @ Cxy_iso
    else:
        Cxy = Cxy_iso

    if not y_in_iso_basis:
        Cxy = Cxy @ Qy_T

    return Cxy, Dxy



# TODO: Make this appropriate tests.
if __name__ == "__main__":

    G = escnn.group.DihedralGroup(6)

    for irrep in G.representations.values():
        if not isinstance(irrep, IrreducibleRepresentation):
            continue
        if irrep.id == G.trivial_representation.id:
            continue
        if irrep.size == 1:
            continue
        x_rep_iso = directsum([irrep] * 2)                   # ρ_Χ
        y_rep_iso = directsum([irrep] * 4)  # ρ_Y

        batch_size = 100
        X_iso = torch.randn(batch_size, x_rep_iso.size)
        Y_iso = torch.randn(batch_size, y_rep_iso.size)
        Cxy_iso, Dxy = isotypic_cross_cov(X_iso, Y_iso, x_rep_iso, y_rep_iso)
        Cxy_iso = Cxy_iso.numpy()

        # Testy change of basis is handled appropriately, using random change of basis.
        Qx, _ = np.linalg.qr(np.random.randn(x_rep_iso.size, x_rep_iso.size))
        Qy, _ = np.linalg.qr(np.random.randn(y_rep_iso.size, y_rep_iso.size))
        x_rep = change_basis(x_rep_iso, Qx, name=f"{x_rep_iso.name}_p")      # ρ_Χ_p = Q_Χ ρ_Χ Q_Χ^T
        y_rep = change_basis(y_rep_iso, Qy, name=f"{y_rep_iso.name}_p")      # ρ_Y_p = Q_Y ρ_Y Q_Y^T
        Qx_T, Qy_T = x_rep.change_of_basis_inv, y_rep.change_of_basis_inv
        X = torch.Tensor(np.einsum('...ij,...j->...i', Qx, X_iso.numpy()))                   # X_p = Q_x X
        Y = torch.Tensor(np.einsum('...ij,...j->...i', Qy, Y_iso.numpy()))                   # Y_p = Q_y Y
        Cxy_p, Dxy = isotypic_cross_cov(X, Y, x_rep, y_rep)
        Cxy_p = Cxy_p.numpy()

        assert np.allclose(Cxy_p, Qx @ Cxy_iso @ Qy.T, atol=1e-6, rtol=1e-4), \
            f"Expected Cxy_p - Q_x Cxy_iso Q_y^T = 0. Got \n {Cxy_p - Qx @ Cxy_iso @ Qy.T}"

        # Test that computing Cxy_iso is equivalent to computing the cross covariance on the G orbit of the data
        # using data augmentation
        GX_iso, GY_iso = [X_iso], [Y_iso]
        for g in G.elements[1:]:
            X_g = torch.Tensor(np.einsum('...ij,...j->...i', x_rep(g), X_iso.numpy()))
            Y_g = torch.Tensor(np.einsum('...ij,...j->...i', y_rep(g), Y_iso.numpy()))
            GX_iso.append(X_g)
            GY_iso.append(Y_g)
        GX_iso = torch.cat(GX_iso, dim=0)
        GY_iso = torch.cat(GY_iso, dim=0)

        Cx_iso, _ = isotypic_cross_cov(X=GX_iso, Y=GX_iso, rep_X=x_rep_iso, rep_Y=x_rep_iso)
        Cx_iso = Cx_iso.numpy()
        Cx_iso_orbit = (GX_iso.T @ GX_iso / (GX_iso.shape[0])).numpy()
        Cx_iso_orbit = np.mean([np.einsum('ij,jk,kl->il', x_rep_iso(g), Cx_iso_orbit, x_rep_iso(~g)) for g in G.elements], axis=0)
        assert np.allclose(Cx_iso, Cx_iso_orbit, atol=1e-3, rtol=1e-3) # Numerical error occurs for small sample sizes

        Cy_iso, _ = isotypic_cross_cov(X=GY_iso, Y=GY_iso, rep_X=y_rep_iso, rep_Y=y_rep_iso)
        Cy_iso = Cy_iso.numpy()
        Cy_iso_orbit = (GY_iso.T @ GY_iso / (GY_iso.shape[0])).numpy()
        Cy_iso_orbit = np.mean([np.einsum('ij,jk,kl->il', y_rep_iso(g), Cy_iso_orbit, y_rep_iso(~g)) for g in G.elements], axis=0)
        assert np.allclose(Cy_iso, Cy_iso_orbit, atol=1e-3, rtol=1e-3) # Numerical error occurs for small sample sizes

