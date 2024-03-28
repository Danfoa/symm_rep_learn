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

def get_cdf(model, X, observable = lambda x : x):
    # observable is a vector to scalar function

    fY = np.apply_along_axis(observable, 0, model.training_Y).flatten()
    candidates = np.argsort(fY)
    probas = np.cumsum(np.ones(fY.shape[0]))/fY.shape[0]

    Ux, sing_val, Vy = model.postprocess_UV(X)

    # estimating the cdf of the function f on X_t
    cdf = np.zeros((candidates.shape[0], Ux.shape[0]))
    for i, val in enumerate(candidates):
        Ify = np.outer((fY <= fY[candidates[i]]), np.ones(Vy.shape[1]))
        EVyFy = np.mean(Vy * Ify, axis=0)
        EVyFy = np.outer(np.ones(Ux.shape[0]), EVyFy)
        cdf[i] = probas[i] + np.sum(sing_val * Ux * EVyFy, axis=-1)

    return fY.flatten()[candidates], cdf

def quantile_regression(model, X, observable = lambda x : np.mean(x, axis=-1), alpha=0.01, t=1, isotonic=True, rescaling=True):
    x, cdfX = get_cdf(model, X, observable)
    return compute_quantile_robust(x, cdfX, alpha=alpha, isotonic=isotonic, rescaling=rescaling)