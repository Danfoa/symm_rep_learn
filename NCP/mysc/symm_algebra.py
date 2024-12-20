# Created by danfoa at 19/12/24
import escnn
import numpy as np
import torch
from escnn.group import change_basis, directsum, IrreducibleRepresentation, Representation


def isotypic_cross_cov(
        X: torch.Tensor, Y: torch.Tensor, rep_X: Representation, rep_Y: Representation, centered=True
        ):
    """Cross covariance of signals between isotypic subspaces of the same type.

    This function exploits the fact that the cross-covariance of signals between isotypic subspaces of the same type
    is constrained to be of the block form:

    Cov(X, Y) = Θ_χυ ⊗ I_d, where d = dim(irrep) and Θ_χυ ∈ R^{mχ x mυ}

    Being mx and my the multiplicities of the irrep in X and Y respectively. This implies that the matrix Θ_χυ
    represents the free parameters of the cross-covariance we are required to estimate. To do so we reshape the
    the signals X ∈ R^{n x p} and Y ∈ R^{n x q} to X_sing ∈ R^{n x mχ} and Y_sing ∈ R^{n x mυ} respectively. Ensuring
    all dimensions of the irreducible subspaces associated to each multiplicity of the irrep are considered as a single
    dimension for estimating Θ_χυ = 1/(n*d) X_sing^T Y_sing.

    Args:
        Y (torch.Tensor): Matrix of size (n, q). n is the number of samples.
        X (torch.Tensor): Matrix of size (n, p).

    Returns:
        np.ndarray or torch.Tensor: Matrix of size (p, q) containing the cross covariance of A and B.
    """
    assert len(rep_X._irreps_multiplicities) == len(rep_Y._irreps_multiplicities) == 1, \
        f"Expected group representation of an isotypic subspace.I.e., with only one type of irrep. \nFound: " \
        f"{list(rep_X._irreps_multiplicities.keys())} in rep_X, {list(rep_Y._irreps_multiplicities.keys())} in rep_Y."
    assert rep_X.group == rep_Y.group, f"{rep_X.group} != {rep_Y.group}"
    irrep_id  = rep_X.irreps[0]  # Irrep id of the isotypic subspace
    assert irrep_id == rep_Y.irreps[0], \
        f"Irreps {irrep_id} != {rep_Y.irreps[0]}. Hence signals are orthogonal and CovXY=0."
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
        Qy = torch.Tensor(Qy).to(device=Y.device, dtype=Y.dtype)
        Y_iso = torch.einsum('...ij,...j->...i', Qy_T, Y)   # y_iso = Q_y2iso @ y

    if irrep_dim > 1:
        # Since CovXY = Θ_χy ⊗ I_d  , d = dim(irrep) and Θ_χυ ∈ R^{mχ x mυ}
        # We compute the constrained cross-covariance, by estimating the matrix Θ_χυ
        # This requires us to reshape X_iso ∈ R^{n x p} to X_sing ∈ R^{n x mχ} and Y_iso ∈ R^{n x q} to Y_sing ∈ R^{n x mυ}
        # Ensuring that the dimensions of a single irrep are flattened into the rows of X_sing and Y_sing
        X_sing = X_iso.view(-1, mk_X, irrep_dim).permute(0, 2, 1).reshape(-1, mk_X)
        Y_sing = Y_iso.view(-1, mk_Y, irrep_dim).permute(0, 2, 1).reshape(-1, mk_Y)
    else: # For one dimensional (real) irreps, this defaults to the standard cross-covariance
        X_sing, Y_sing = X_iso, Y_iso

    n_samples = X_sing.shape[0]
    assert n_samples == X.shape[0] * irrep_dim

    if centered and irrep_id == G.trivial_representation.id:  # Non-trivial isotypic subspace are centered
        X_sing = X_sing - torch.mean(X_sing, dim=0, keepdim=True)
        Y_sing = Y_sing - torch.mean(Y_sing, dim=0, keepdim=True)

    theta_xy = (1 / (n_samples - 1)) * torch.einsum('...i,...j->ij', X_sing, Y_sing)
    if irrep_dim > 1:  # Broadcast the estimates according to CovXY = Θ_χy ⊗ I_d.
        I_d = torch.eye(irrep_dim, device=theta_xy.device, dtype=theta_xy.dtype)
        CovXY_iso = torch.kron(theta_xy, I_d)          #    Efficient Kronecker product
    else:
        CovXY_iso = theta_xy

    if not x_in_iso_basis: # Change basis back to the original basis
        CovXY = Qx @ CovXY_iso
    else:
        CovXY = CovXY_iso

    if not y_in_iso_basis:
        CovXY = CovXY @ Qy_T

    return CovXY



# TODO: Make this appropriate tests.
if __name__ == "__main__":

    G = escnn.group.DihedralGroup(6)

    for irrep in G.representations.values():
        if not isinstance(irrep, IrreducibleRepresentation):
            continue
        x_rep_iso = directsum([irrep] * 2)                   # ρ_Χ
        y_rep_iso = directsum([irrep] * 4)  # ρ_Y

        batch_size = 10
        X_iso = torch.randn(batch_size, x_rep_iso.size)
        Y_iso = torch.randn(batch_size, y_rep_iso.size)
        CovXY_iso = isotypic_cross_cov(X_iso, Y_iso, x_rep_iso, y_rep_iso).numpy()

        # Testy change of basis is handled appropriately, using random change of basis.
        Qx, _ = np.linalg.qr(np.random.randn(x_rep_iso.size, x_rep_iso.size))
        Qy, _ = np.linalg.qr(np.random.randn(y_rep_iso.size, y_rep_iso.size))
        x_rep = change_basis(x_rep_iso, Qx, name=f"{x_rep_iso.name}_p")      # ρ_Χ_p = Q_Χ ρ_Χ Q_Χ^T
        y_rep = change_basis(y_rep_iso, Qy, name=f"{y_rep_iso.name}_p")      # ρ_Y_p = Q_Y ρ_Y Q_Y^T
        Qx_T, Qy_T = x_rep.change_of_basis_inv, y_rep.change_of_basis_inv
        X = torch.Tensor(np.einsum('...ij,...j->...i', Qx, X_iso.numpy()))                   # X_p = Q_x X
        Y = torch.Tensor(np.einsum('...ij,...j->...i', Qy, Y_iso.numpy()))                   # Y_p = Q_y Y
        CovXY_p = isotypic_cross_cov(X, Y, x_rep, y_rep).numpy()

        assert np.allclose(CovXY_p, Qx @ CovXY_iso @ Qy.T, atol=1e-6, rtol=1e-4), \
            f"Expected CovXY_p - Q_x CovXY_iso Q_y^T = 0. Got \n {CovXY_p - Qx @ CovXY_iso @ Qy.T}"


