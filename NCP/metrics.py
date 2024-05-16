# distances for one dimensional distributions
import numpy as np
import warnings
from sklearn.isotonic import IsotonicRegression
from scipy.special import rel_entr

# all measures take empirical cdfs computed for the same values as input
def cdf_to_hist(x, fx=None, smooth=True, bin_length=1):
    x_copy = x.copy()
    if smooth:
        assert fx is not None, 'isotonic regression requires the values for series x'
        x_copy = smooth_cdf(fx, x_copy)
    # x_hist = np.diff(x.flatten(), prepend=0)
    x_hist = np.diff(x_copy.flatten()) # Removed the prepend argument because it introduces a spurious bin
    # x_hist = np.interp(np.linspace(0, 1, len(x_copy.flatten())), np.linspace(0, 1, len(x_hist)), x_hist) # Perfomed interpolation to match the length of x_copy
    x_swindow = np.lib.stride_tricks.sliding_window_view(x_hist, bin_length)[::bin_length]
    return x_swindow

def smooth_cdf(values, cdf): # Moved smooth_cdf here from NCP/utils.py
    scdf = IsotonicRegression(y_min=0., y_max=cdf.max()).fit_transform(values, cdf)
    if scdf.max() <= 0:
        return np.zeros(values.shape)
    scdf = scdf/scdf.max()
    return scdf

def hellinger(x,y, values=None, bin_length=1, smooth=True):
    x_hist = cdf_to_hist(x, values, smooth, bin_length)
    y_hist = cdf_to_hist(y, values, smooth, bin_length)
    # return np.sum((np.sqrt(x_hist) - np.sqrt(y_hist))**2)
    # Changed the formula to match the definition of the Hellinger distance in [https://www.tcs.tifr.res.in/~prahladh/teaching/2011-12/comm/lectures/l12.pdf; https://en.wikipedia.org/wiki/Hellinger_distance]]
    return np.linalg.norm(np.sqrt(x_hist) - np.sqrt(y_hist), ord=2) / np.sqrt(2)
def kullback_leibler(x,y, values=None, bin_length=1, smooth=True):
    eps = 1e-8
    x_hist = cdf_to_hist(x, values, smooth, bin_length)
    y_hist = cdf_to_hist(y, values, smooth, bin_length)
    # overriding the zero values to eps to avoid np.log(0) = -inf, not sure if this is the best way to handle this
    x_hist = np.where(x_hist == 0, eps, x_hist)
    y_hist = np.where(y_hist == 0, eps, y_hist)
    # normalizing the histograms to avoid negative KL divergence
    x_hist = x_hist / np.sum(x_hist)
    y_hist = y_hist / np.sum(y_hist)
    return np.sum(x_hist * (np.log(x_hist)-np.log(y_hist)))

# Added Jensen-Shannon divergence as defined in [https://en.wikipedia.org/wiki/Jensen–Shannon_divergence]
def jensen_shannon(x,y, values=None, bin_length=1, smooth=True):
    eps = 1e-8
    x_hist = cdf_to_hist(x, values, smooth, bin_length)
    y_hist = cdf_to_hist(y, values, smooth, bin_length)
    # overriding the zero values to eps to avoid np.log(0) = -inf, not sure if this is the best way to handle this
    x_hist = np.where(x_hist == 0, eps, x_hist)
    y_hist = np.where(y_hist == 0, eps, y_hist)
    # normalizing the histograms to avoid negative KL divergence
    x_hist = x_hist / np.sum(x_hist)
    y_hist = y_hist / np.sum(y_hist)
    m = (x_hist + y_hist) / 2
    return (np.sum(x_hist * (np.log(x_hist)-np.log(m))) + np.sum(y_hist * (np.log(y_hist)-np.log(m)))) * 0.5
    return (np.sum(rel_entr(x_hist, m)) + np.sum(rel_entr(y_hist, m))) * 0.5

# Added total variation distance as defined in [https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures]
def total_variation(x,y, values=None, bin_length=1, smooth=True):
    x_hist = cdf_to_hist(x, values, smooth, bin_length)
    y_hist = cdf_to_hist(y, values, smooth, bin_length)
    return np.sum(np.abs(x_hist-y_hist)) * 0.5

def wasserstein1(x,y, values=None, smooth=True):
    if smooth:
        x_treated = smooth_cdf(values, x)
        y_treated = smooth_cdf(values, y)
    else:
        x_treated = x.copy()
        y_treated = y.copy()

    # good_vals = (x_treated > 0) & (y_treated > 0) & (x_treated >= 1) & (y_treated <= 1)
    good_vals = (x_treated > 0) & (y_treated > 0) & (x_treated <= 1) & (y_treated <= 1) # fixed bug

    x_treated = x_treated[good_vals]
    y_treated = y_treated[good_vals]
    # return np.linalg.norm((x_treated.flatten()**-1)-(y_treated.flatten()**-1), ord=1)
    # in the case of p=1, the wasserstein distance can be computed as follows [https://en.wikipedia.org/wiki/Wasserstein_metric]
    return np.mean(np.abs(x_treated - y_treated))

# Added Kolmogorov-Smirnov distance as defined in [https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test]
def kolmogorov_smirnov(x, y, values=None, smooth=True):
    if smooth:
        x = smooth_cdf(values, x)
        y = smooth_cdf(values, y)
    return np.max(np.abs(x-y))

def compute_metrics(x, y, metrics='all', smooth=True, values=None):
    """
    Compute specified metrics for two probability distributions.

    Args:
    p, q: Probability distributions to compare.
    metrics: List of metric names to compute. Each metric name should be a string.
    smooth: Boolean indicating whether to smooth the cdf.
    values: Values used for smoothing the cdf.

    Returns:
    A dictionary where keys are metric names and values are computed metric values.
    """

    if smooth and values is None:
        raise ValueError('Values must be provided when smoothing the cdf')

    results = {}
    all_metrics = {
        'hellinger': hellinger,
        'kullback_leibler': kullback_leibler,
        'wasserstein1': wasserstein1,
        'kolmogorov_smirnov': kolmogorov_smirnov,
        'total_variation': total_variation,
        'jensen_shannon': jensen_shannon
    }

    if 'all' in metrics:
        metrics = all_metrics.keys()

    for metric in metrics:
        if metric in all_metrics:
            results[metric] = all_metrics[metric](x, y, smooth=smooth, values=values)
        else:
            warnings.warn(f'Metric {metric} is not implemented', UserWarning)

    return results