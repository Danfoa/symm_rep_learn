import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from typing import Optional, Callable, Union
from NCP.utils import tonp, frnp

def compute_quantile_robust(values:np.ndarray, cdf:np.ndarray, alpha:Union[str, float]='all', isotonic:bool=True, rescaling:bool=True):
    # TODO: correct this code
    # correction of the cdf using isotonic regression
    if isotonic:
        for i in range(cdf.shape[0]):
            cdf[i] = IsotonicRegression(y_min=0., y_max=cdf[i].max()).fit_transform(range(cdf.shape[1]), cdf[i])
    if rescaling:
        max_cdf = np.outer(cdf.max(axis=-1), np.ones(cdf.shape[1]))
        max_cdf[max_cdf == 0] = 1.    # security to avoid errors
        cdf = cdf/max_cdf

    # if alpha = all, return the entire cdf
    if alpha=='all':
        return values, cdf

    # otherwise, search for the quantile at level alpha
    quantiles = np.zeros(cdf.shape[0])
    for j in range(cdf.shape[0]):
        for i, level in enumerate(cdf[j]):
            if level >= alpha:
                if i == 0:
                    quantiles[j] = -np.inf
                quantiles[j] = values[i-1]
                break
            
        # special case where we exceeded the maximum observed value
        if i == cdf.shape[0] - 1:
            quantiles[j] = np.inf

    return quantiles

def get_cdf(model, X, Y=None, observable = lambda x : x, postprocess = None):
    # observable is a vector to scalar function
    if Y is None: # if no Y is given, use the training data
        Y = model.training_Y
    fY = np.apply_along_axis(observable, -1, Y).flatten()
    candidates = np.argsort(fY)
    probas = np.cumsum(np.ones(fY.shape[0]))/fY.shape[0] # vector of [k/n], k \in [n]

    if postprocess:  # postprocessing can be 'centering' or 'whitening'
        Ux, sigma, Vy = model.postprocess_UV(X, postprocess)
    else:
        sigma = torch.sqrt(torch.exp(-model.models['S'].weights ** 2))
        Ux = model.models['U'](frnp(X, model.device))
        Vy = model.models['V'](frnp(Y, model.device))
        Ux, sigma, Vy = tonp(Ux), tonp(sigma), tonp(Vy)

    Ux = Ux.flatten()

    # estimating the cdf of the function f on X_t
    cdf = np.zeros(candidates.shape[0])
    for i, val in enumerate(fY[candidates]):
        Ify = np.outer((fY <= val), np.ones(Vy.shape[1]))         # indicator function of fY < fY[i], put into shape (n_sample, latent_dim)
        EVyFy = np.mean(Vy * Ify, axis=0)                         # for all latent dim, compute E (Vy * fY)
        cdf[i] = probas[i] + np.sum(sigma * Ux * EVyFy)

    return fY[candidates].flatten(), cdf

def quantile_regression(model, X, observable = lambda x : np.mean(x, axis=-1), alpha=0.01, t=1, isotonic=True, rescaling=True):
    x, cdfX = get_cdf(model, X, observable)
    return compute_quantile_robust(x, cdfX, alpha=alpha, isotonic=isotonic, rescaling=rescaling)