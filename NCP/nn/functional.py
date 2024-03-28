import torch
from NCP.layers import SingularLayer

def cme_score(x1:torch.Tensor, x2:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, S:SingularLayer, gamma:float):
    loss = 0.5 * ( torch.sum(S(x1 * y2)**2, dim=1) + torch.sum(S(x2 * y1)**2, dim=1) )
    loss -= torch.sum(S((x1 - x2) * (y1 - y2)), dim=1)
    if gamma > 0:
        # gamma = gamma/(2*x1.shape[0]) # 2*x1.shape[0] = n
        x1_x2 = torch.sum(x1 * x2, dim=1)
        y1_y2 = torch.sum(y1 * y2, dim=1)
        loss -= gamma * x1_x2 * (1 + x1_x2)
        loss -= gamma * y1_y2 * (1 + y1_y2)
        loss += gamma * (torch.norm(x1)**2 + torch.norm(x2)**2 + torch.norm(y1)**2 + torch.norm(y2)**2)
        loss += 2*gamma*x1.shape[0]
    return torch.mean(loss)

def cme_score_cov(x1:torch.Tensor, x2:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, S:SingularLayer, gamma:float):
    U1 = S(x1)
    U2 = S(x2)
    V1 = S(y1)
    V2 = S(y2)

    # centered covariance matrices
    cov_U1 = torch.cov(U1.T)
    cov_U2 = torch.cov(U2.T)
    cov_V1 = torch.cov(V1.T)
    cov_V2 = torch.cov(V2.T)

    U1_mean = U1.mean(axis=0, keepdims=True)
    U2_mean = U2.mean(axis=0, keepdims=True)
    V1_mean = V1.mean(axis=0, keepdims=True)
    V2_mean = V2.mean(axis=0, keepdims=True)

    cov_U1V1 = cross_cov(U1.T, V1.T, centered=True)
    cov_U2V2 = cross_cov(U2.T, V2.T, centered=True)

    loss = 0.5 * (torch.trace(cov_U1@cov_V2) + torch.trace(cov_U2@cov_V1)) - torch.trace(cov_U1V1) - torch.trace(cov_U2V2)

    if gamma > 0:
        d = x1.shape[-1]

        # uncentered covariance matrices
        uc_cov_U1 = cov_U1 + U1_mean @ U1_mean.T
        uc_cov_U2 = cov_U2 + U2_mean @ U2_mean.T
        uc_cov_V1 = cov_V1 + V1_mean @ V1_mean.T
        uc_cov_V2 = cov_V2 + V2_mean @ V2_mean.T

        loss_on = 0.5 * (torch.trace(uc_cov_U1@uc_cov_U2) - torch.trace(uc_cov_U1) - torch.trace(uc_cov_U2) +
        torch.trace(uc_cov_V1@uc_cov_V2) - torch.trace(uc_cov_V1) - torch.trace(uc_cov_V2)) + d
        return loss + gamma * loss_on
    else:
        return loss

def cross_cov(A, B, rowvar=True, bias=False, centered=True):
    """Cross covariance of two matrices.

    Args:
        A (np.ndarray or torch.Tensor): Matrix of size (n, p).
        B (np.ndarray or torch.Tensor): Matrix of size (n, q).
        rowvar (bool, optional): Whether to calculate the covariance along the rows. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor: Matrix of size (p, q) containing the cross covariance of A and B.
    """
    if rowvar is False:
        A = A.T
        B = B.T

    if centered:
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)

    C = A @ B.T

    if bias:
        return C / A.shape[1]
    else:
        return C / (A.shape[1] - 1)