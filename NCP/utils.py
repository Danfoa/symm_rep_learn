import torch
def tonp(x):
    return x.detach().cpu().numpy()

def frnp(x, device='cpu'):
    return torch.Tensor(x).to(device)