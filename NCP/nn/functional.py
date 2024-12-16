import torch

from NCP.model import NCPOperator
from NCP.utils import cross_cov, random_split


def robust_cov(X, tol=1e-5):
    C = torch.cov(X)
    Cp = 0.5*(C + C.T)
    return Cp #+ tol

# def cme_score(x1:torch.Tensor, x2:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, S:SingularLayer, gamma:float):
#     loss = 0.5 * ( torch.sum(S(x1 * y2)**2, dim=1) + torch.sum(S(x2 * y1)**2, dim=1) )
#     loss -= torch.sum(S((x1 - x2) * (y1 - y2)), dim=1)
#     if gamma > 0:
#         # gamma = gamma/(2*x1.shape[0]) # 2*x1.shape[0] = n
#         x1_x2 = torch.sum(x1 * x2, dim=1)
#         y1_y2 = torch.sum(y1 * y2, dim=1)
#         loss -= gamma * x1_x2 * (1 + x1_x2)
#         loss -= gamma * y1_y2 * (1 + y1_y2)
#         loss += gamma * (torch.norm(x1)**2 + torch.norm(x2)**2 + torch.norm(y1)**2 + torch.norm(y2)**2)
#         loss += 2*gamma*x1.shape[0]
#     return torch.mean(loss)

def cme_score_cov(X:torch.Tensor, Y:torch.Tensor, NCP:NCPOperator, gamma:float):
    X1, X2, Y1, Y2 = random_split(X, Y, 2)
    U1 = NCP.S(NCP.U(X1))
    U2 = NCP.S(NCP.U(X2))
    V1 = NCP.S(NCP.V(Y1))
    V2 = NCP.S(NCP.V(Y2))

    # centered covariance matrices
    cov_U1 = robust_cov(U1.T)
    cov_U2 = robust_cov(U2.T)
    cov_V1 = robust_cov(V1.T)
    cov_V2 = robust_cov(V2.T)

    cov_U1V1 = cross_cov(U1.T, V1.T, centered=True)
    cov_U2V2 = cross_cov(U2.T, V2.T, centered=True)

    loss = (0.5 * (torch.sum(cov_U1*cov_V2) + torch.sum(cov_U2*cov_V1))
            - torch.trace(cov_U1V1) - torch.trace(cov_U2V2))

    if gamma > 0:
        d = X1.shape[-1]
        U1_mean = U1.mean(axis=0, keepdims=True)
        U2_mean = U2.mean(axis=0, keepdims=True)
        V1_mean = V1.mean(axis=0, keepdims=True)
        V2_mean = V2.mean(axis=0, keepdims=True)

        # uncentered covariance matrices
        uc_cov_U1 = cov_U1 + U1_mean @ U1_mean.T
        uc_cov_U2 = cov_U2 + U2_mean @ U2_mean.T
        uc_cov_V1 = cov_V1 + V1_mean @ V1_mean.T
        uc_cov_V2 = cov_V2 + V2_mean @ V2_mean.T

        loss_on = 0.5 * (
                torch.sum(uc_cov_U1*uc_cov_U2) - torch.trace(uc_cov_U1) - torch.trace(uc_cov_U2)
                + torch.sum(uc_cov_V1*uc_cov_V2) - torch.trace(uc_cov_V1) - torch.trace(uc_cov_V2)
        ) + d
        return loss + gamma * loss_on
    else:
        return loss


def cme_score_opti(X:torch.Tensor, Y:torch.Tensor, NCP:NCPOperator, gamma:float):
    X1, X2, Y1, Y2 = random_split(X, Y, 2)
    U1 = NCP.U(X1)
    U2 = NCP.U(X2)
    V1 = NCP.V(Y1)
    V2 = NCP.V(Y2)

    l1 = 0.5 * (U1 * NCP.S(V2)).sum(dim=1) ** 2
    l2 = 0.5 * (U2 * NCP.S(V1)).sum(dim=1) ** 2
    l3 = ((U1 - U2) * NCP.S(V1 - V2)).sum(dim=1)

    L = torch.mean(l1 + l2 - l3)

    if gamma > 0 :
        r1 = (U1 * U2).sum(dim=1)**2
        r2 = ((U1 - U2)**2).sum(dim=1)
        r3 = (V1 * V2).sum(dim=1)**2
        r4 = ((V1 - V2)**2).sum(dim=1)
        r5 = 2*U1.shape[1]

        R = torch.mean(r1 - r2 + r3 - r4 + r5)

        return L + gamma * R

    return L

def cme_score_Ustat(
        X: torch.Tensor,
        Y: torch.Tensor,
        NCP:NCPOperator,
        metric_deformation: float = 1.0,
        center: bool = True,
):
    if center:
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)

    Ux = NCP.S(NCP.U(X))
    Vy = NCP.S(NCP.V(Y))

    joint_measure_score = 2 * ((Ux * Vy).sum(dim=-1)).mean()
    product_measure_score = torch.square_(Ux @ Vy.T)
    product_measure_score.diagonal().zero_()  # Zeroing out the diagonal in place
    b = X.shape[0]
    product_measure_score = b * product_measure_score.mean() / (b - 1)
    score = joint_measure_score - product_measure_score
    if metric_deformation > 0:
        cov_X, cov_Y, cov_XY = (
            torch.cov(Ux),
            torch.cov(Vy),
            cross_cov(Ux, Vy),
        )
        R_X = log_fro_metric_deformation_loss(cov_X)
        R_Y = log_fro_metric_deformation_loss(cov_Y)
        return score - 0.5 * metric_deformation * (R_X + R_Y)
    else:
        return score

def log_fro_metric_deformation_loss(cov: torch.tensor):
    """Logarithmic + Frobenious metric deformation loss as used in :footcite:t:`Kostic2023DPNets`, defined as :math:`{{\\rm Tr}}(C^{2} - C -\\ln(C))` .

    Args:
        cov (torch.tensor): A symmetric positive-definite matrix.

    Returns:
        torch.tensor: Loss function
    """
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    loss = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return loss
