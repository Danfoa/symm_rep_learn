import numpy as np
import torch
from typing import Union
from NCP.utils import tonp, frnp
from NCP.metrics import smooth_cdf
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KernelDensity
import normflows as nf

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
        Ux, sigma, Vy = model.postprocess_UV(X, postprocess=postprocess, Y=Y)
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

def get_pdf(model, X, Y=None, observable = lambda x : x, postprocess = None, p_y = lambda x : 1):
    # observable is a vector to scalar function
    if Y is None: # if no Y is given, use the training data
        Y = model.training_Y
    fY = np.apply_along_axis(observable, -1, Y).flatten()
    candidates = np.argsort(fY)
    n = fY.shape[0]
    probas = np.ones(n) # vector of [1], k \in [n]

    if postprocess:  # postprocessing can be 'centering' or 'whitening'
        Ux, sigma, Vy = model.postprocess_UV(X, postprocess, Y)
    else:
        sigma = torch.sqrt(torch.exp(-model.models['S'].weights ** 2))
        Ux = model.models['U'](frnp(X, model.device))
        Vy = model.models['V'](frnp(Y, model.device))
        Ux, sigma, Vy = tonp(Ux), tonp(sigma), tonp(Vy)

    Ux = Ux.flatten()
    EVyFy = Vy[candidates]
    pdf = (probas + np.sum(sigma * Ux * EVyFy, axis=-1)) * p_y(fY[candidates])

    return fY[candidates].flatten(), pdf

def compute_moments(model, X, order=2, Y=None, observable = lambda x : x, postprocess=None):
    obs_moment = lambda x : observable(x)**order
    return model.predict(X, Y=Y, observable=obs_moment, postprocess=postprocess)

def compute_variance(model, X, Y=None, observable = lambda x : x, postprocess=None):
    e2 = compute_moments(model, X, Y=Y, order=2, observable=observable, postprocess=postprocess)
    e1 = model.predict(X, Y=Y, observable=observable, postprocess=postprocess)
    return e2 - e1**2

def quantile_regression_naive(model, X, observable = lambda x : np.mean(x, axis=-1), alpha=0.01, t=1, isotonic=True, rescaling=True):
    x, cdfX = get_cdf(model, X, observable)
    return compute_quantile_robust(x, cdfX, alpha=alpha, isotonic=isotonic, rescaling=rescaling)

class compute_marginal(KernelDensity):
    def __call__(self, x):
        if torch.is_tensor(x):
            x = tonp(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        log_probability = self.score_samples(np.array(x))
        probability = np.exp(log_probability)
        return frnp(probability)

from scipy.integrate import cumulative_trapezoid
def integrate_pdf(pdf, values):
    return cumulative_trapezoid(pdf.squeeze(), x=values.squeeze(), initial=0)

def find_best_quantile(x, cdf, alpha):
    t0 = 0
    t1 = 1
    best_t0 = 0
    best_t1 = -1
    best_size = np.inf

    while t0 < len(cdf):
        # stop if left border reaches right end of discretisation
        if cdf[t1] - cdf[t0] >= 1-alpha:
            # if x[t0], x[t1] is a confidence interval at level alpha, compute length and compare to best
            size = x[t1] - x[t0]
            if size < best_size:
                best_t0 = t0
                best_t1 = t1
                best_size = size
            # moving t1 to the right will only increase the size of the interval, so we can safely move t0 to the right
            t0 += 1
        
        elif t1 == len(cdf)-1:
            # if x[t0], x[t1] is not a confidence interval with confidence at least level alpha, 
            #and t1 is already at the right limit of the discretisation, then there remains no more pertinent intervals
            break
        else:
            # if moving x[t0] to the right reduces the level, we need to increase t1
            t1 += 1
    return x[best_t0], x[best_t1]

def quantile_regression(model, X, y_discr, alpha=0.01, postprocess='centering', marginal=None, model_type='NCP'):
    if model_type=='NCP':
        x, pdf = model.pdf(torch.Tensor(X), y_discr, p_y=marginal, postprocess=postprocess)
        ys_bis = y_discr.numpy().flatten()
        step = (ys_bis[1]-ys_bis[0])
        cdf = np.cumsum(pdf * step, -1)
        cdf = smooth_cdf(x, cdf)

    elif model_type == 'NF':
        xs = torch.Tensor(X).repeat(y_discr.size()[0], 1)
        log_prob = model.log_prob(y_discr, xs).detach().cpu().numpy()
        prob = np.exp(log_prob)
        ys_bis = y_discr.numpy().flatten()
        cdf = np.cumsum(prob)*(ys_bis[1] - ys_bis[0])
        x = y_discr.numpy().flatten()
    else:
        print('no such method', model_type)

    return find_best_quantile(x, cdf, alpha=alpha)

def quantile_regression_from_cdf(x, cdf, alpha):
    return find_best_quantile(x, cdf, alpha=alpha/2)

def compute_coverage(quantiles, values):
    cntr = 0
    for i, val in enumerate(values):
        if (val >= quantiles[i][0]) and (val <= quantiles[i][1]):
            cntr += 1
    return cntr/len(values)

def compute_coverage_length(quantiles):
    lengths = quantiles[:,1] - quantiles[:,0]
    return lengths.mean(), lengths.std()

def get_mean_from_nf(model, x, N=1000):
    #compute mean after samples N from conditional distribution according to x
    samples = model.sample(num_samples=N, context=x.repeat(N, 1))[0].detach().cpu().numpy()
    return np.mean(samples)

def get_pdf_from_nf(model:nf.ConditionalNormalizingFlow, x, ys):
    # since only one x, duplicate
    xs = x.repeat(ys.size()[0], 1)
    log_prob = model.log_prob(ys, xs).detach().cpu().numpy()
    return np.exp(log_prob)

def get_cdf_from_nf(model:nf.ConditionalNormalizingFlow, x, ys):
    # supposing that ys is computed from linspace
    prob = get_pdf_from_nf(model, x, torch.Tensor(ys))
    ys_bis = ys.numpy().flatten()
    return np.cumsum(prob)*(ys_bis[1] - ys_bis[0])