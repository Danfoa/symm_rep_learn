#%% Importing libraries
import argparse
import os
import time
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch.nn import GELU
from torch.optim import Adam
from tqdm import tqdm

from NCP.cde_fork.density_simulation import (
    ArmaJump,
    EconDensity,
    GaussianMixture,
    LinearGaussian,
    LinearStudentT,
    SkewNormal,
)
from NCP.cdf import compute_marginal, integrate_pdf
from NCP.examples.tools.data_gen import LGGMD
from NCP.metrics import compute_metrics
from NCP.model import NCPModule, NCPOperator
from NCP.nn.layers import MLP
from NCP.nn.losses import CMELoss
from NCP.mysc.utils import FastTensorDataLoader, from_np

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        X, Y = trainer.model.batch
        trainer.model.model._compute_data_statistics(X, Y)
        torch.save(trainer.model.model, trainer.checkpoint_callback.dirpath + '/best_model.pt')

def run_experiment(density_simulator, density_simulator_kwargs):
    filename = density_simulator().__class__.__name__ + '_NCP_results.pkl'
    n_training_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    gamma = 1e-3
    epochs = int(1e5)
    n_val = int(1e3)
    lr = 1e-3

    if os.path.isfile(filename):
        results_df = pd.read_pickle(filename)
    else:
        results_df = pd.DataFrame()

    for n in tqdm(n_training_samples, desc='Training samples', total=len(n_training_samples)):

        for seed in tqdm(range(10), desc='Seed', total=10):
            print(f'Running with {n} samples - seed {seed}')
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

            X_train_torch = from_np(X_train)
            Y_train_torch = from_np(Y_train)
            X_val_torch = from_np(X_val)
            Y_val_torch = from_np(Y_val)

            U_operator_kwargs = {
                'input_shape': X_train.shape[-1],
                'output_shape': 100,
                'n_hidden': 2,
                'layer_size': 64,
                'dropout': 0,
                'iterative_whitening': False,
                'activation': GELU
            }

            V_operator_kwargs = {
                'input_shape': Y_train.shape[-1],
                'output_shape': 100,
                'n_hidden': 2,
                'layer_size': 64,
                'dropout': 0,
                'iterative_whitening': False,
                'activation': GELU
            }

            optimizer_kwargs = {
                'lr': lr
            }

            loss_kwargs = {
                'mode': 'cov',
                'gamma': gamma
            }


            L.seed_everything(seed)

            model = NCPOperator(U_operator=MLP,
                                V_operator=MLP,
                                U_operator_kwargs=U_operator_kwargs,
                                V_operator_kwargs=V_operator_kwargs)

            NCP_module = NCPModule(
                model,
                Adam,
                optimizer_kwargs,
                CMELoss,
                loss_kwargs
            )

            train_dl = FastTensorDataLoader(X_train_torch, Y_train_torch, batch_size=len(X_train_torch), shuffle=False)
            val_dl = FastTensorDataLoader(X_val_torch, Y_val_torch, batch_size=len(X_val_torch), shuffle=False)

            early_stop = EarlyStopping(monitor="val_loss", patience=200, mode="min")
            ckpt_path = "checkpoints/" + density.__class__.__name__ + "_NCP_" + str(n) + "_" + str(seed)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            checkpoint_callback = CustomModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", dirpath=ckpt_path)

            logger_path = "lightning_logs/" + density.__class__.__name__ + "_DDPM_" + str(n) + "_" + str(seed)
            if not os.path.exists(logger_path):
                os.makedirs(logger_path)

            trainer = L.Trainer(**{
                'accelerator': 'cuda',
                'max_epochs': epochs,
                'log_every_n_steps': 0,
                'enable_progress_bar': False,
                'devices': 1,
                'enable_checkpointing': True,
                'num_sanity_val_steps': 0,
                'enable_model_summary': False,
                'check_val_every_n_epoch': 10,
                'logger': False,
            # }, callbacks=[early_stop])
            }, callbacks=[early_stop, checkpoint_callback])

            # timing the training
            start = time.perf_counter()
            trainer.fit(NCP_module, train_dataloaders=train_dl, val_dataloaders=val_dl)
            fit_time = time.perf_counter() - start
            best_model = torch.load(checkpoint_callback.dirpath + '/best_model.pt').to('cpu')

            print('N epochs: {0}'.format(NCP_module.current_epoch - early_stop.patience*trainer.check_val_every_n_epoch-1))
            print('Training loss: {0}'.format(NCP_module.train_loss[-early_stop.patience*trainer.check_val_every_n_epoch-1]))
            print('Validation loss: {0}'.format(NCP_module.val_loss[-early_stop.patience-1]))
            print('Fit time: {0}'.format(fit_time))

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

            p_y = compute_marginal(bandwidth='scott').fit(Y_train)

            results = []
            for postprocess in [None, 'centering', 'whitening']:
                scores = []
                for xi in x_grid:
                    xi = xi.reshape(1, -1)
                    fys, pred_pdf = best_model.pdf(from_np(xi), from_np(ys), postprocess=postprocess, p_y=p_y)
                    pred_pdf = np.where(pred_pdf == np.nan, 0, pred_pdf)
                    pred_cdf = integrate_pdf(pred_pdf, ys)

                    if density.__class__.__name__ == 'LGGMD':
                        true_cdf = density.cdf(xscaler.inverse_transform(xi), yscaler.inverse_transform(ys)).squeeze()
                    else:
                        true_cdf = density.cdf(np.repeat(xscaler.inverse_transform(xi), len(ys), axis=0),
                                               yscaler.inverse_transform(ys)).squeeze()
                    computed_metrics = compute_metrics(true_cdf, pred_cdf, smooth=True, values=fys)
                    computed_metrics['x'] = xi

                    scores.append(computed_metrics)

                result = {
                    'seed': seed,
                    'n_samples': n,
                    'postprocess': str(postprocess),
                }

                scores = pd.DataFrame(scores)
                for key in scores:
                    result[key] = [scores[key].values]

                results.append(result)

            results = pd.DataFrame(results)
            results_df = pd.concat([results_df, results], ignore_index=True)
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
        density_simulator_kwargs = {'random_seed': random_seed}
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
