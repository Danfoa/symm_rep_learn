import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

def setup_plots():
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8 
    }

    plt.rcParams.update(tex_fonts)  

def plot_expectation(reg, X_train, X_test, Y_train, Y_test, xscaler, yscaler, postprocess='centering'):
    '''
    expectation plots for in sample and out of sample data
    '''

    pred = reg.conditional_expectation(X_train, Y_train, postprocess=postprocess).reshape(-1, 1)
    pred_test = reg.conditional_expectation(X_test, Y_train, postprocess=postprocess).reshape(-1, 1)

    X_out_of_sample = xscaler.transform(np.random.uniform(5,10, size=100).reshape(-1,1))
    pred_oos = reg.conditional_expectation(X_out_of_sample, Y_train, postprocess=postprocess).reshape(-1, 1)

    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    axes = axs.flatten()

    Xs = xscaler.inverse_transform(X_train)
    sorted = np.argsort(Xs.flatten())
    axes[0].scatter(Xs, 
                    yscaler.inverse_transform(Y_train), 
                    color='r', alpha=0.01)
    axes[0].plot(Xs.flatten()[sorted], 
                yscaler.inverse_transform(pred).flatten()[sorted], 'b', alpha=0.5)
    axes[0].plot(Xs.flatten()[sorted], Xs.flatten()[sorted]**2, 'red')
    axes[0].legend(['data points', 'predicted expectation', 'true expectation'])
    axes[0].set_title(f'Training data (mse = {round(mean_squared_error(Xs**2, yscaler.inverse_transform(pred)), 4)})')

    Xs = xscaler.inverse_transform(X_test)
    sorted = np.argsort(Xs.flatten())
    axes[1].scatter(Xs, 
                    yscaler.inverse_transform(Y_test), 
                    color='r', alpha=0.1)
    axes[1].plot(Xs.flatten()[sorted], 
                yscaler.inverse_transform(pred_test).flatten()[sorted], 'b', alpha=0.5)
    axes[1].plot(Xs.flatten()[sorted], Xs.flatten()[sorted]**2, 'red')
    axes[1].legend(['data points', 'predicted expectation', 'true expectation'])
    axes[1].set_title(f'Test data (mse = {round(mean_squared_error(Xs**2, yscaler.inverse_transform(pred_test)), 4)})')
    plt.show()