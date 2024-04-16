import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    xscaler = StandardScaler()
    yscaler = StandardScaler()

    Xtransformed = xscaler.fit_transform(X)
    Ytransformed = yscaler.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(Xtransformed, Ytransformed, test_size=N_test, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=N_val, random_state=0)

    return X_train, X_test, X_val, Y_val, Y_train, Y_test, xscaler, yscaler

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

    xscaler = StandardScaler()
    yscaler = StandardScaler()

    Xtransformed = xscaler.fit_transform(X)
    Ytransformed = yscaler.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(Xtransformed, Ytransformed, test_size=N_test, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=N_val, random_state=0)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, xscaler, yscaler

