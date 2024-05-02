#%% Importing libraries
import sys
import os
import time
from typing import Dict, Any

# sys.path.append('/media/gturri/deploy/NCP/')
import numpy as np
import pandas as pd
import argparse
import torch
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.nn import Tanh, ReLU
from NCP.nn.layers import MLP
from NCP.utils import frnp, RegressionDataset
from NCP.nn.losses import CMELoss
from NCP.model import NCPOperator, NCPModule
from NCP.metrics import compute_metrics
from NCP.cdf import compute_marginal
from tqdm import tqdm
from NCP.cde_fork.density_simulation import LinearGaussian, LinearStudentT, ArmaJump, SkewNormal, EconDensity, GaussianMixture
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# function to convert pdf into cdf
def pdf2cdf(pdf, step):
    return np.cumsum(pdf * step, -1)

class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        X, Y = trainer.model.batch
        trainer.model.model._compute_data_statistics(X, Y)

def restore_buffers_shape(model, state_dict):
    model._sing_val = torch.zeros_like(state_dict['model._sing_val']).to('cpu')
    model._sing_vec_l = torch.zeros_like(state_dict['model._sing_vec_l']).to('cpu')
    model._sing_vec_r = torch.zeros_like(state_dict['model._sing_vec_r']).to('cpu')

def run_experiment(density_simulator, density_simulator_kwargs):
    filename = density_simulator().__class__.__name__ + '_NCP_results.pkl'
    n_training_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    gamma = 1e-3
    epochs = int(1e4)

    if os.path.isfile(filename):
        results_df = pd.read_pickle(filename)
    else:
        results_df = pd.DataFrame()

    for n in tqdm(n_training_samples, desc='Training samples', total=len(n_training_samples)):

        # if n <= 1000:
        #     lr = 1e-5
        # else:
        #     lr = 1e-4

        lr = 1e-3

        density = density_simulator(**density_simulator_kwargs)
        X, Y = density.simulate(n_samples=n + 10000)
        X = X.reshape((-1, 1))
        Y = Y.reshape((-1, 1))
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=10000, random_state=0)
        X_train = xscaler.fit_transform(X_train)
        Y_train = yscaler.fit_transform(Y_train)
        X_val = xscaler.transform(X_val)
        Y_val = yscaler.transform(Y_val)

        X_train_torch = frnp(X_train)
        Y_train_torch = frnp(Y_train)
        X_val_torch = frnp(X_val)
        Y_val_torch = frnp(Y_val)

        MLP_kwargs = {
            'input_shape': X_train.shape[-1],
            'output_shape': 100,
            'n_hidden': 2,
            'layer_size': 32,
            'dropout': 0,
            'iterative_whitening': False,
            'activation': Tanh
        }

        optimizer_kwargs = {
            'lr': lr
        }

        loss_kwargs = {
            'mode': 'split',
            'gamma': gamma
        }

        for seed in tqdm(range(5), desc='Seed', total=5):
            # print(f'Running with {n} samples - seed {seed}')
            if len(results_df) > 0:
                if len(results_df[(results_df['n_samples'] == n) & (results_df['seed'] == seed)]) > 0:
                    continue

            L.seed_everything(seed)
            model = NCPOperator(U_operator=MLP, V_operator=MLP, U_operator_kwargs=MLP_kwargs, V_operator_kwargs=MLP_kwargs)
            NCP_module = NCPModule(
                model,
                Adam,
                optimizer_kwargs,
                CMELoss,
                loss_kwargs
            )

            train_dl = DataLoader(RegressionDataset(X_train_torch, Y_train_torch), batch_size=len(X_train_torch), shuffle=False,
                                  num_workers=0)
            val_dl = DataLoader(RegressionDataset(X_val_torch, Y_val_torch), batch_size=len(X_val_torch), shuffle=False, num_workers=0)

            early_stop = EarlyStopping(monitor="val_loss", patience=100, mode="min")
            checkpoint_callback = CustomModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
            trainer = L.Trainer(**{
                'accelerator': 'cuda',
                'max_epochs': epochs,
                'log_every_n_steps': 0,
                'enable_progress_bar': True,
                'devices': 1,
                'enable_checkpointing': True,
                'num_sanity_val_steps': 0,
                'enable_model_summary': False,
            }, callbacks=[early_stop, checkpoint_callback])
            # }, callbacks=[early_stop])


            # timing the training
            start = time.perf_counter()
            trainer.fit(NCP_module, train_dataloaders=train_dl, val_dataloaders=val_dl)
            print(f'Training time: {time.perf_counter() - start}')
            # print last training and validation loss
            best_model_dict = torch.load(checkpoint_callback.best_model_path)
            restore_buffers_shape(model, best_model_dict['state_dict'])
            NCP_module.load_state_dict(best_model_dict['state_dict'])
            best_model = NCP_module.model

            print('N epochs: {0}'.format(best_model_dict['epoch']))
            print('Training loss: {0}'.format(NCP_module.train_loss[best_model_dict['epoch']]))
            print('Validation loss: {0}'.format(NCP_module.val_loss[best_model_dict['epoch']]))

            # Computing results
            x_grid = np.percentile(X_train, np.linspace(10, 90, num=10))
            p1, p99 = np.percentile(Y_train, [1, 99])
            ys, step = np.linspace(p1, p99, num=1000, retstep=True)
            ys = frnp(ys.reshape(-1, 1))

            p_y = compute_marginal(bandwidth='scott').fit(Y_train)

            results = []
            for postprocess in [None, 'centering', 'whitening']:
                scores = []
                for xi in x_grid:
                    fys, pred_pdf = best_model.pdf(frnp([[xi]]), frnp(ys), postprocess=postprocess, p_y=p_y)
                    pred_cdf = pdf2cdf(pred_pdf, step)

                    true_cdf = density.cdf(xscaler.inverse_transform(np.ones_like(ys) * xi),
                                           yscaler.inverse_transform(ys)).squeeze()
                    computed_metrics = compute_metrics(true_cdf.squeeze(), pred_cdf.squeeze(), metrics='all',
                                                       smooth=True, values=fys.squeeze())
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
            results_df.to_pickle(filename)

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
        density_simulator_kwargs = {'n_kernels': 5, 'ndim_x': 1, 'ndim_y': 1, 'means_std': 1.5, 'random_seed': random_seed}
    elif args.dataset == 'linear_gaussian':
        density_simulator = LinearGaussian
        density_simulator_kwargs = {'ndim_x': 1, 'mu': 0.0, 'mu_slope': 0.005, 'std': 0.01, 'std_slope': 0.002, 'random_seed': random_seed}
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
    else:
        raise ValueError('Unknown dataset')

    run_experiment(density_simulator, density_simulator_kwargs)