import torch
from NCP.layers import SingularLayer

def cme_score(x1:torch.Tensor, x2:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, S:SingularLayer, gamma:float):
    loss = 0.5 * ( torch.sum(S(x1 * y2)**2, dim=1) + torch.sum(S(x2 * y1)**2, dim=1) )
    loss -= torch.sum(S((x1 - x2) * (y1 - y2)), dim=1)
    if gamma > 0:
        gamma = gamma/(2*x1.shape[0]) # 2*x1.shape[0] = n
        x1_x2 = torch.sum(x1 * x2, dim=1)
        y1_y2 = torch.sum(y1 * y2, dim=1)
        loss -= gamma * x1_x2 * (1 + x1_x2)
        loss -= gamma * y1_y2 * (1 + y1_y2)
        loss += gamma * (torch.norm(x1)**2 + torch.norm(x2)**2 + torch.norm(y1)**2 + torch.norm(y2)**2)
        loss += 2*gamma*x1.shape[0]
    return torch.mean(loss)

def cme_score_cov(x1:torch.Tensor, x2:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, S:SingularLayer, gamma:float):
    cov_U1 = torch.cov(S(x1).T)
    cov_U2 = torch.cov(S(x2).T)
    cov_V1 = torch.cov(S(y1).T)
    cov_V2 = torch.cov(S(y2).T)

    cov_U1V2 = cross_cov(S(x1).T, S(y1).T)
    cov_U2V1 = cross_cov(S(x2).T, S(y2).T)

    loss = 0.5*(torch.trace(cov_U1@cov_V2) + torch.trace(cov_U2@cov_V1)) - torch.trace(cov_U1V2) - torch.trace(cov_U2V1)
    return loss

def cross_cov(A, B, rowvar=True, bias=False):
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
    A = A - A.mean(axis=1, keepdims=True)

    B = B - B.mean(axis=1, keepdims=True)

    C = A @ B.T

    if bias:
        return C / A.shape[1]
    else:
        return C / (A.shape[1] - 1)