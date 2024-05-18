#%% Importing libraries
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import argparse
from tqdm import tqdm
from NCP.metrics import compute_metrics
from NCP.cde_fork.density_simulation import LinearGaussian, LinearStudentT, ArmaJump, SkewNormal, EconDensity, \
                                             GaussianMixture
from NCP.examples.tools.data_gen import LGGMD
from NCP.cde_fork.density_estimator import KernelMixtureNetwork, NormalizingFlowEstimator, MixtureDensityNetwork, \
    LSConditionalDensityEstimation, ConditionalKernelDensityEstimation, NeighborKernelDensityEstimation

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

from time import perf_counter

# function to convert pdf into cdf
from scipy.integrate import cumtrapz
def integrate_pdf(pdf, values):
    return cumtrapz(pdf.squeeze(), x=values.squeeze(), initial=0)
# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    # print(gc.collect()) # if it does something you should see a number as output

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
def run_experiment(density_estimator, density_estimator_kwargs, density_simulator, density_simulator_kwargs):
    filename = density_simulator().__class__.__name__ + '_' + density_estimator.__name__ + '_results.pkl'
    n_training_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    n_val = 1000
    if os.path.isfile(filename):
        results_df = pd.read_pickle(filename)
    else:
        results_df = pd.DataFrame()

    for n in tqdm(n_training_samples, desc='Training samples', total=len(n_training_samples)):
        density = density_simulator(**density_simulator_kwargs)
        X, Y = density.simulate(n_samples=n_training_samples[-1] + n_val)
        if density_simulator().__class__.__name__ == "ArmaJump":
            np.random.seed(density_simulator_kwargs['random_seed'])
            idx = np.random.permutation(len(X))
            X, Y = X[idx], Y[idx]
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if Y.ndim == 1:
            Y = Y.reshape((-1, 1))
        X_train, X_val, Y_train, Y_val = X[:n], X[-n_val:], Y[:n], Y[-n_val:]
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        X_train = xscaler.fit_transform(X_train)
        Y_train = yscaler.fit_transform(Y_train)
        X_val = xscaler.transform(X_val)
        Y_val = yscaler.transform(Y_val)

        random_seeds = np.arange(10)
        if density_estimator.__name__ in ['NeighborKernelDensityEstimation', 'ConditionalKernelDensityEstimation']:
            random_seeds = [0]
        for seed in tqdm(random_seeds, desc='Seed', total=len(random_seeds)):
            if len(results_df) > 0:
                if len(results_df[(results_df['n_samples'] == n) & (results_df['seed'] == seed)]) > 0:
                    continue

            # print(f'Running with {n} samples - seed {seed}')

            density_estimator_kwargs['random_seed'] = seed
            density_estimator_kwargs['ndim_x'] = density.ndim_x
            density_estimator_kwargs['ndim_y'] = density.ndim_y

            model = density_estimator(**density_estimator_kwargs)
            model.fit(X_train, Y_train, eval_set=(X_val, Y_val), verbose=True)

            # Computing results
            n_sampling = 19
            if density.__class__.__name__ == 'LGGMD':
                x_grid = np.zeros(
                    (n_sampling * 3, density.ndim_x))  # 3 is the number of features on which I want to condition on

                for i in range(density.ndim_x):
                    x_grid[:, i] = np.repeat(np.percentile(X_train[:, i], 50), x_grid.shape[0], axis=0)

                for i in range(3):
                    x_grid[i * n_sampling:(i + 1) * n_sampling, i] = np.percentile(X_train[:, i],
                                                                                   np.linspace(5, 95, num=n_sampling))
            else:
                x_grid = np.percentile(X_train, np.linspace(5, 95, num=n_sampling))
            ys, step = np.linspace(Y_train.min(), Y_train.max(), num=1000, retstep=True)
            ys = ys.reshape(-1, 1)

            scores = []
            for xi in x_grid:
                start = perf_counter()
                xi = xi.reshape(1, -1)
                if density.__class__.__name__ == 'LGGMD':
                    true_cdf = density.cdf(xscaler.inverse_transform(xi), yscaler.inverse_transform(ys)).squeeze()
                else:
                    true_cdf = density.cdf(np.repeat(xscaler.inverse_transform(xi), len(ys), axis=0),
                                           yscaler.inverse_transform(ys)).squeeze()

                if model.has_cdf:
                    try:
                        pred_cdf = model.cdf(np.repeat(xi, len(ys), axis=0), ys).squeeze()
                        computed_cdf = True
                    except:
                        computed_cdf = False
                else:
                    pred_pdf = model.pdf(np.repeat(xi, len(ys), axis=0), ys).squeeze()
                    pred_cdf = integrate_pdf(pred_pdf, ys)
                    computed_cdf = True

                if computed_cdf:
                    computed_metrics = compute_metrics(true_cdf.squeeze(), pred_cdf.squeeze(), metrics='all', smooth=True,
                                                   values=ys.squeeze())
                    computed_metrics['x'] = xi

                    scores.append(computed_metrics)
                    print(f'Elapsed time: {perf_counter() - start:.2f}s')

            if computed_cdf:
                result = {
                    'seed': seed,
                    'n_samples': n,
                }

                scores = pd.DataFrame(scores)
                for key in scores:
                    result[key] = [scores[key].values]

                result = pd.DataFrame(result)
                results_df = pd.concat([results_df, result], ignore_index=True)
                results_df.to_pickle(filename)

            # resetting keras
            reset_keras()


if __name__ == '__main__':
    random_seed = 42
    parser = argparse.ArgumentParser(description='Benchmarks evaluation')
    parser.add_argument('--dataset', default='econ',
                        help='dataset for which to run empirical evaluation evaluation')
    parser.add_argument('--model', default='NFlow',
                        help='model for which to run empirical evaluation evaluation')
    args = parser.parse_args()

    if args.dataset == 'econ':
        density_simulator = EconDensity
        density_simulator_kwargs = {'std': 1, 'heteroscedastic': True, 'random_seed': random_seed}
    elif args.dataset == 'gaussian_mixture':
        density_simulator = GaussianMixture
        density_simulator_kwargs = {'ndim_x': 1, 'ndim_y': 1, 'means_std': 3, 'random_seed': random_seed}
    elif args.dataset == 'linear_gaussian':
        density_simulator = LinearGaussian
        density_simulator_kwargs = {'ndim_x': 1, 'std': 0.1, 'random_seed': random_seed}
    elif args.dataset == 'arma_jump':
        density_simulator = ArmaJump
        density_simulator_kwargs = {'c': 0.1, 'arma_a1': 0.9, 'std': 0.05, 'jump_prob': 0.05, 'random_seed': random_seed}
    elif args.dataset == 'skew_normal':
        density_simulator = SkewNormal
        density_simulator_kwargs = {'random_seed': random_seed}
    elif args.dataset == 'linear_student_t':
        density_simulator = LinearStudentT
        density_simulator_kwargs = {'ndim_x': 1, 'mu': 0.0, 'mu_slope': 0.005, 'std': 0.01, 'std_slope': 0.002, 'dof_low': 2,
                                    'dof_high': 10, 'random_seed': random_seed}
    elif args.dataset == 'LGGMD':
        density_simulator = LGGMD
        density_simulator_kwargs = {'random_seed': random_seed}
    else:
        raise ValueError('Unknown dataset')

    if args.model == 'KMN':
        density_estimator = KernelMixtureNetwork
        # density_estimator_kwargs = {'name': 'kmn', 'x_noise_std': 0.2, 'y_noise_std': 0.1}
        density_estimator_kwargs = {'name': 'kmn', 'x_noise_std': 0.2, 'y_noise_std': 0.1, 'hidden_sizes': (64, 64),
                                    'hidden_nonlinearity': tf.nn.relu, 'n_training_epochs': int(1e5)}
    elif args.model == 'NFlow':
        density_estimator = NormalizingFlowEstimator
        # density_estimator_kwargs = {'name': 'nflow', 'hidden_sizes': (32, 32), 'hidden_nonlinearity': tf.nn.tanh,
        #                             'n_training_epochs': int(5e3)}
        density_estimator_kwargs = {'name': 'nflow', 'hidden_sizes': (64, 64), 'hidden_nonlinearity': tf.nn.relu,
                                    'n_training_epochs': int(1e5)}
        # density_estimator_kwargs = {'name': 'nflow', 'hidden_sizes': (32, 32), 'hidden_nonlinearity': tf.tanh}
        # density_estimator_kwargs = {'name': 'nflow', 'hidden_sizes': (16, 16), 'hidden_nonlinearity': tf.tanh}
    elif args.model == 'MDN':
        density_estimator = MixtureDensityNetwork
        # density_estimator_kwargs = {'name': 'mdn', 'x_noise_std': 0.2, 'y_noise_std': 0.1}
        density_estimator_kwargs = {'name': 'mdn', 'x_noise_std': 0.2, 'y_noise_std': 0.1, 'hidden_sizes': (64, 64),
                                    'hidden_nonlinearity': tf.nn.relu, 'n_training_epochs': int(1e5)}
    elif args.model == 'LSCDE':
        density_estimator = LSConditionalDensityEstimation
        density_estimator_kwargs = {'name': 'lscde'}
    elif args.model == 'CKDE':
        density_estimator = ConditionalKernelDensityEstimation
        density_estimator_kwargs = {'name': 'ckde'}
    elif args.model == 'NKDE':
        density_estimator = NeighborKernelDensityEstimation
        if args.dataset == 'LGGMD':
            density_estimator_kwargs = {'name': 'nkde', 'epsilon': 1.}
        else:
            density_estimator_kwargs = {'name': 'nkde'}
    else:
        raise ValueError('Unknown model')

    run_experiment(density_estimator, density_estimator_kwargs, density_simulator, density_simulator_kwargs)