# distances for one dimensional distributions

import numpy as np

# all measures take empirical cdfs computed for the same values as input

def hellinger(x,y, fy=None, interv=None, bin_length=1):
    x_hist = np.diff(x.flatten(), prepend=0)
    y_hist = np.diff(y.flatten(), prepend=0) 

    # sliding window in case of too fine discretisation (disabled by default)
    x_swindow = np.lib.stride_tricks.sliding_window_view(x_hist, bin_length)[::bin_length]
    y_swindow = np.lib.stride_tricks.sliding_window_view(y_hist, bin_length)[::bin_length]
    return np.sum((np.sqrt(x_swindow) - np.sqrt(y_swindow))**2)

def kullback_leibler(x,y, fy=None, interv=None, bin_length=1):
    x_hist = np.diff(x.flatten(), prepend=0)
    y_hist = np.diff(y.flatten(), prepend=0)     

    good_vals = (x_hist > 0) # security in case of constant section in cdf (may happen for extreme values)

    # sliding window in case of too fine discretisation (disabled by default)
    x_swindow = np.lib.stride_tricks.sliding_window_view(x_hist[good_vals], bin_length)[::bin_length]
    y_swindow = np.lib.stride_tricks.sliding_window_view(y_hist[good_vals], bin_length)[::bin_length]
    return np.sum(x_swindow * (np.log(x_swindow)-np.log(y_swindow)))

def wasserstein1(x,y, fy=None, interv=None):
    x_copy = x.copy()
    y_copy = y.copy()

    good_vals = (x_copy > 0) & (y_copy > 0) & (x_copy >= 1) & (y_copy <= 1)

    x_copy = x_copy[good_vals]
    y_copy = y_copy[good_vals]
    return np.linalg.norm((x.flatten()**-1)-(y.flatten()**-1), ord=1)