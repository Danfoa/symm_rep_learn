import torch
from sklearn.isotonic import IsotonicRegression

def tonp(x):
    return x.detach().cpu().numpy()

def frnp(x, device='cpu'):
    return torch.Tensor(x).to(device)

def smooth_cdf(values, cdf):
    scdf = IsotonicRegression(y_min=0., y_max=cdf.max()).fit_transform(values, cdf)
    scdf = scdf/scdf.max()   
    return scdf