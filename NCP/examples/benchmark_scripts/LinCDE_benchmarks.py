#%% Importing libraries
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from NCP.metrics import compute_metrics
from NCP.cdf import integrate_pdf
from tqdm import tqdm
from NCP.cde_fork.density_simulation import LinearGaussian, LinearStudentT, ArmaJump, SkewNormal, EconDensity, GaussianMixture
from NCP.examples.tools.lincde import lincde
from NCP.examples.tools.data_gen import LGGMD
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

def run_experiment(density_simulator, density_simulator_kwargs):
    filename = density_simulator().__class__.__name__ + '_LinCDE_results.pkl'
    n_training_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    n_val = int(1e3)

    if os.path.isfile(filename):
        results_df = pd.read_pickle(filename)
    else:
        results_df = pd.DataFrame()

    for n in tqdm(n_training_samples, desc='Training samples', total=len(n_training_samples)):

        random_seeds = np.arange(10)
        for seed in tqdm(random_seeds, desc='Seed', total=len(random_seeds)):
            # print(f'Running with {n} samples - seed {seed}')
            if len(results_df) > 0:
                if len(results_df[(results_df['n_samples'] == n) & (results_df['seed'] == seed)]) > 0:
                    continue

            density_simulator_kwargs['random_seed'] = seed
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

            ys, pred_pdf = lincde(X_train, Y_train, x_grid, ys, dataset=density.__class__.__name__,
                                  verbose='F', random_seed=seed)

            scores = []
            for i, xi in enumerate(x_grid):
                xi = xi.reshape(1, -1)
                pred_cdf = integrate_pdf(pred_pdf[i], ys)
                if density.__class__.__name__ == 'LGGMD':
                    true_cdf = density.cdf(xscaler.inverse_transform(xi), yscaler.inverse_transform(ys)).squeeze()
                else:
                    true_cdf = density.cdf(np.repeat(xscaler.inverse_transform(xi), len(ys), axis=0),
                                           yscaler.inverse_transform(ys)).squeeze()
                computed_metrics = compute_metrics(true_cdf, pred_cdf, smooth=True, values=ys)
                computed_metrics['x'] = xi

                scores.append(computed_metrics)

            result = {
                'seed': seed,
                'n_samples': n,
            }

            scores = pd.DataFrame(scores)
            for key in scores:
                result[key] = [scores[key].values]

            result = pd.DataFrame(result)
            results_df = pd.concat([results_df, result], ignore_index=True)
            results_df.to_pickle('results/' + filename)

if __name__ == '__main__':
    random_seed = 42
    parser = argparse.ArgumentParser(description='Benchmarks evaluation')
    parser.add_argument('--dataset', default=None,
                        help='dataset for which to run empirical evaluation evaluation')
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

    run_experiment(density_simulator, density_simulator_kwargs)