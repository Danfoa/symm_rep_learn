# distances for one dimensional distributions

import numpy as np
from NCP.utils import smooth_cdf

# all measures take empirical cdfs computed for the same values as input

def cdf_to_hist(x, fx=None, smooth=True, bin_length=1):
    x_copy = x.copy()
    if smooth:
        assert fx is not None, 'isotonic regression requires the values for series x'
        x_copy = smooth_cdf(fx, x_copy)
    x_hist = np.diff(x.flatten(), prepend=0)
    x_swindow = np.lib.stride_tricks.sliding_window_view(x_hist, bin_length)[::bin_length]
    return x_swindow

def hellinger(x,y, values=None, bin_length=1, smooth=True):
    x_hist = cdf_to_hist(x, values, smooth, bin_length)
    y_hist = cdf_to_hist(y, values, smooth, bin_length)
    return np.sum((np.sqrt(x_hist) - np.sqrt(y_hist))**2)

def kullback_leibler(x,y, values=None, bin_length=1, smooth=True):
    x_hist = cdf_to_hist(x, values, smooth, bin_length)
    y_hist = cdf_to_hist(y, values, smooth, bin_length)
    return np.sum(x_hist * (np.log(x_hist)-np.log(y_hist)))

def wasserstein1(x,y, values=None, smooth=True):
    if smooth:
        x_treated = smooth_cdf(values, x)
        y_treated = smooth_cdf(values, y)
    else:
        x_treated = x.copy()
        y_treated = y.copy()

    good_vals = (x_treated > 0) & (y_treated > 0) & (x_treated >= 1) & (y_treated <= 1)

    x_treated = x_treated[good_vals]
    y_treated = y_treated[good_vals]
    return np.linalg.norm((x.flatten()**-1)-(y.flatten()**-1), ord=1)