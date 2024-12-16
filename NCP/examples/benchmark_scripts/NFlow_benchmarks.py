#%% Importing libraries
import argparse
import os
import time
import warnings

import lightning as L
import normflows as nf
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from normflows import ConditionalNormalizingFlow
from sklearn.preprocessing import StandardScaler
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
from NCP.cdf import integrate_pdf
from NCP.examples.tools.data_gen import LGGMD
from NCP.metrics import compute_metrics
from NCP.nn.nf_module import NFModule
from NCP.utils import FastTensorDataLoader, from_np, to_np

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        torch.save(trainer.model.model, trainer.checkpoint_callback.dirpath + '/best_model.pt')

def run_experiment(density_simulator, density_simulator_kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filename = density_simulator().__class__.__name__ + '_NFlow_results.pkl'
    n_training_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    epochs = int(1e5)
    n_val = int(1e3)
    lr = 1e-4

    if os.path.isfile(filename):
        results_df = pd.read_pickle(filename)
    else:
        results_df = pd.DataFrame()

    for n in tqdm(n_training_samples, desc='Training samples', total=len(n_training_samples)):

        random_seeds = np.arange(10)
        for seed in tqdm(random_seeds, desc='Seed', total=len(random_seeds)):
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

            optimizer_kwargs = {
                'lr': lr
            }

            L.seed_everything(seed)

            # Define flows
            context_size = X_train.shape[1]
            K = 2

            latent_size = 1
            hidden_units = 64
            hidden_layers = 2

            flows = []
            for i in range(K):
                flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units,
                                                              context_features=context_size,
                                                              num_blocks=hidden_layers)]

                flows += [nf.flows.LULinearPermute(latent_size)]

            # Set base distribution
            q0 = nf.distributions.DiagGaussian(1, trainable=True)

            # Construct flow model
            nf_model = ConditionalNormalizingFlow(q0, flows)

            nf_module = NFModule(nf_model,
                                 Adam,
                                 optimizer_kwargs)

            train_dl = FastTensorDataLoader(X_train_torch, Y_train_torch, batch_size=len(X_train_torch), shuffle=False)
            val_dl = FastTensorDataLoader(X_val_torch, Y_val_torch, batch_size=len(X_val_torch), shuffle=False)

            early_stop = EarlyStopping(monitor="val_loss", patience=200, mode="min")
            ckpt_path = "checkpoints/" + density.__class__.__name__ + "_NFlow_" + str(n) + "_" + str(seed)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            checkpoint_callback = CustomModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", dirpath=ckpt_path)


            trainer = L.Trainer(**{
                'accelerator': device,
                'max_epochs': epochs,
                'log_every_n_steps': 0,
                'enable_progress_bar': False,
                'devices': 1,
                'enable_checkpointing': True,
                'num_sanity_val_steps': 0,
                'enable_model_summary': True,
                'check_val_every_n_epoch': 5,
                'logger': False,
                # 'default_root_dir': logger_path,
            }, callbacks=[early_stop, checkpoint_callback])


            # timing the training
            start = time.perf_counter()
            trainer.fit(nf_module, train_dataloaders=train_dl, val_dataloaders=val_dl)
            fit_time = time.perf_counter() - start

            best_model = torch.load(checkpoint_callback.dirpath + '/best_model.pt').to('cpu')
            best_model.eval()

            print('N epochs: {0}'.format(nf_module.current_epoch - early_stop.patience*trainer.check_val_every_n_epoch-1))
            print('Training loss: {0}'.format(nf_module.train_loss[-early_stop.patience*trainer.check_val_every_n_epoch-1]))
            print('Validation loss: {0}'.format(nf_module.val_loss[-early_stop.patience-1]))
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

            scores = []
            for xi in x_grid:
                xi = xi.reshape(1, -1)
                pred_pdf = to_np(torch.exp(best_model.log_prob(from_np(ys), from_np(np.repeat(xi, len(ys), axis=0)))))
                pred_cdf = integrate_pdf(pred_pdf, ys)

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
    parser.add_argument('--dataset', default='econ',
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
