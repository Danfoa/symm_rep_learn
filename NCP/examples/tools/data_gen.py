import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

def standardise_and_cut(X, Y, N_train, N_val, N_test):

    xscaler = StandardScaler()
    yscaler = StandardScaler()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=N_test, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=N_val, random_state=0)

    X_train = xscaler.fit_transform(X_train)
    Y_train = yscaler.fit_transform(Y_train)

    X_val = xscaler.transform(X_val)
    Y_val = yscaler.transform(Y_val)
    X_test = xscaler.transform(X_test)
    Y_test = yscaler.transform(Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, xscaler, yscaler   

def gen_additive_noise_data(noise, f, N_train=1000, N_val=1000, N_test=1000, x_min=0, x_max=5):
    '''
    creates a dataset of (X,Y) such that X is uniformly distributed between x_min and x_max and
    Y = f(X) + noise
    '''
    X = np.random.uniform(x_min, x_max, N_train+N_val+N_test)
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(X):
        Y[i] = f(xi) + noise(xi)

    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))

    return standardise_and_cut(X, Y, N_train, N_val, N_test)

def gen_switching(p, offset, scale=1, x_switch_min=2.5, N_train=1000, N_val=1000, N_test=1000, x_min=0, x_max=5):
    x_switch_max = x_switch_min + p*(x_max-x_min)
    assert x_switch_max <= x_max, 'interval too large'

    X = np.random.uniform(x_min, x_max, N_train+N_val+N_test)
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(X):
        if (xi >= x_switch_min) and (xi <= x_switch_max):
            Y[i] = xi + offset + np.random.normal(0,scale*(1+xi))
        else:
            Y[i] = xi**2 + np.random.normal(0, 1+xi)

    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))

    return standardise_and_cut(X, Y, N_train, N_val, N_test)

def gen_bimodal(main_dist=norm, alpha=0.5, mu1=-1, mu2=1, s1=1, s2=1, N_train=1000, N_val=1000, N_test=1000):
    # same experiment as mixture model of LinCDE paper
    X = np.random.uniform(-1, 1, (N_train+N_val+N_test, 20))
    Y = np.zeros(X.shape[0])
    for i, xi in enumerate(X):
        if xi[1] > 0.2:
            Y[i] = np.random.normal(0.25*xi[0], 0.3)
        else:
            a = np.random.binomial(1, 0.5)
            y1 = np.random.normal(0.25*xi[0] - 0.5, 0.25 * (0.25*xi[2] + 0.5)**2)
            y2 = np.random.normal(0.25*xi[0] + 0.5, 0.25 * (0.25*xi[2] - 0.5)**2)

            Y[i] = a * y1 + (1-a) * y2

    Y = Y.reshape((-1, 1))

    return standardise_and_cut(X, Y, N_train, N_val, N_test)   

def get_conditional_bimodal_cdf(x, y_vals):
    if x[1]>0.2:
        return norm.cdf(y_vals, loc=0.25*x[0], scale=0.3)
    
    else:
        mode1 = norm.cdf(y_vals, loc=0.25*x[0] - 0.5, scale=0.25 * (0.25*x[2] + 0.5)**2)
        mode2 = norm.cdf(y_vals, loc=0.25*x[0] + 0.5, scale=0.25 * (0.25*x[2] - 0.5)**2)

        return 0.5*mode1 + 0.5*mode2
    
    
def get_conditional_bimodal_pdf(x, y_vals):
    if x[1]>0.2:
        return norm.pdf(y_vals, loc=0.25*x[0], scale=0.3)
    
    else:
        mode1 = norm.pdf(y_vals, loc=0.25*x[0] - 0.5, scale=0.25 * (0.25*x[2] + 0.5)**2)
        mode2 = norm.pdf(y_vals, loc=0.25*x[0] + 0.5, scale=0.25 * (0.25*x[2] - 0.5)**2)

        return 0.5*mode1 + 0.5*mode2