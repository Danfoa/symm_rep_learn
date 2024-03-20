import torch
from NCP.layers import SingularLayer

def cme_score(x1:torch.Tensor, x2:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, S:SingularLayer, gamma:float):
    loss = 0.5 * ( torch.sum(S(x1 * y2)**2, dim=1) + torch.sum(S(x2 * y1)**2, dim=1) )
    loss -= torch.sum(S((x1 - x2) * (y1 - y2)), dim=1)
    loss -= gamma * torch.sum(x1 * x2, dim=1) * (1 + torch.sum(x1 * x2, dim=1))
    loss -= gamma * torch.sum(y1 * y2, dim=1) * (1 + torch.sum(y1 * y2, dim=1))
    loss += gamma * (torch.norm(x1)**2 + torch.norm(x2)**2 + torch.norm(y1)**2 + torch.norm(y2)**2)
    # loss -= 2*gamma*z1.size()[0]
    return torch.mean(loss)
