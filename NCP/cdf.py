import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from typing import Optional, Callable, Union

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

def get_cdf(model, X, observable = lambda x : np.mean(x, axis=-1)):
    # observable is a vector to scalar function
    fY = model.training_Y.numpy().apply_map(observable)
    candidates = np.argsort(fY)
    probas = np.cumsum(np.ones(fY.shape[0]))/fY.shape[0]

    hatUx = model.models['U'](torch.Tensor(X)).numpy() - np.sum(model.models['U'](model.training_X))
    hatVy = model.models['V'](model.training_Y).numpy() - np.sum(model.models['V'](model.training_Y))
    Sigma = model.models['S'].weights.detach().numpy()

    # estimating the cdf of the function f on X_t
    cdf = np.array([probas[i] + np.sum(Sigma * hatUx * np.sum(hatVy * (fY <= fY[candidates[i]]) , axis=-1), axis=-1) for i in range(candidates.shape[0])]).T
    return fY[candidates], cdf

def quantile_regression(model, X, observable = lambda x : np.mean(x, axis=-1), alpha=0.01, t=1, isotonic=True, rescaling=True):
    x, cdfX = model.get_cdf(X, observable)
    return compute_quantile_robust(X, cdfX, alpha=alpha, isotonic=isotonic, rescaling=rescaling)