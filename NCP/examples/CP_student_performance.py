import os
import pickle

import lightning as L
import normflows as nf
import numpy as np
import pandas as pd
import torch
from deel.puncc.regression import SplitCP
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from tqdm import tqdm

from NCP.cdf import (
    compute_coverage,
    compute_coverage_length,
    compute_marginal,
    integrate_pdf,
    quantile_regression,
    quantile_regression_from_cdf,
)
from NCP.metrics import smooth_cdf
from NCP.models.ncp import NCPModule, NCPOperator
from NCP.nn.layers import MLP
from NCP.nn.losses import CMELoss
from NCP.nn.nf_module import NFModule
from NCP.utils import FastTensorDataLoader, from_np

datasets = ['students']
NEXP = 10
alphas = [0.1]
ntest = 1000
nval = 1000
class NCPModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        X, Y = trainer.model.batch
        trainer.model.model._compute_data_statistics(X, Y)
        torch.save(trainer.model.model, trainer.checkpoint_callback.dirpath + '/best_model.pt')

class NFModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        torch.save(trainer.model.model, trainer.checkpoint_callback.dirpath + '/best_model.pt')

def init_dico():
    return {
        'ncp_c':np.zeros((NEXP, len(alphas))),
        'ncp_w':np.zeros((NEXP, len(alphas))),
        'nf':np.zeros((NEXP, len(alphas))),
        'rfcc':np.zeros((NEXP, len(alphas)))
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Common optimizer
lr = 1e-3

optimizer = Adam
optimizer_kwargs = {
        'lr': lr
        }

gamma = 1e-2
epochs = int(1e5)
output_shape = 50

for d in datasets:
    print(d)
    if os.path.isfile('NCP/examples/results/{}_quantiles_larger.pkl'.format(d)):
        with open('NCP/examples/results/{}_coverage_larger.pkl'.format(d), 'rb') as file:
            coverage = np.load(file, allow_pickle=True)
        with open('NCP/examples/results/{}_covlength_larger.pkl'.format(d), 'rb') as file:
            size = np.load(file, allow_pickle=True)
        with open('NCP/examples/results/{}_covlengthstd_larger.pkl'.format(d), 'rb') as file:
            size_std = np.load(file, allow_pickle=True)
        with open('NCP/examples/results/{}_quantiles_larger.pkl'.format(d), 'rb') as file:
            quantiles = np.load(file, allow_pickle=True)

    else:
        coverage = init_dico()
        size = init_dico()
        size_std = init_dico()
        quantiles = {
            'ncp_c':{},
            'ncp_w':{},
            'nf':{},
            'rfcc':{},
        }

    for m in ['ncp_c', 'ncp_w', 'nf', 'rfcc']:
        if m not in quantiles.keys():
            quantiles[m] = {}
            coverage[m] = np.zeros((NEXP, len(alphas)))
            size[m] = np.zeros((NEXP, len(alphas)))
            size_std[m] = np.zeros((NEXP, len(alphas)))
        if m in ['ncp_c', 'ncp_w']:
            quantiles[m] = np.zeros((NEXP, len(alphas), ntest, 3))
        else:
            quantiles[m] = np.zeros((NEXP, len(alphas), ntest, 3))

    for exp in range(NEXP):

        df = pd.read_csv('NCP/examples/data/Student_Performance.csv')
        df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
        X = df.iloc[:, 0:-1].to_numpy()
        Y = df.iloc[:, -1].to_numpy().reshape(-1, 1)

        ntrain = len(X) - ntest - nval

        np.random.seed(exp)
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        X_train, Y_train = X[:ntrain], Y[:ntrain]
        X_val, Y_val = X[ntrain:ntrain + nval], Y[ntrain:ntrain + nval]
        X_test, Y_test = X[ntrain + nval:], Y[ntrain + nval:]

        xscaler = StandardScaler()
        yscaler = StandardScaler()

        X_train = xscaler.fit_transform(X_train)
        Y_train = yscaler.fit_transform(Y_train)

        X_val = xscaler.transform(X_val)
        Y_val = yscaler.transform(Y_val)
        X_test = xscaler.transform(X_test)
        Y_test = yscaler.transform(Y_test)

        X_train_torch = from_np(X_train)
        Y_train_torch = from_np(Y_train)
        X_val_torch = from_np(X_val)
        Y_val_torch = from_np(Y_val)
        X_test_torch = from_np(X_test)
        Y_test_torch = from_np(Y_test)

        # y discretisation for computing cdf
        spread = np.max(Y_train) - np.min(Y_train)
        p1, p99 = np.min(Y_train), np.max(Y_train)
        y_discr, step = np.linspace(p1 - 0.1 * spread, p99 + 0.1 * spread, num=1000, retstep=True)
        y_discr_torch = torch.Tensor(y_discr.reshape((-1, 1)))

        k_pdf = compute_marginal(bandwidth='scott').fit(Y_train)
        marginal = lambda x: torch.Tensor(np.exp(k_pdf.score_samples(x.reshape(-1, 1))))

        train_dl = FastTensorDataLoader(X_train_torch, Y_train_torch, batch_size=len(X_train_torch), shuffle=False)
        val_dl = FastTensorDataLoader(X_val_torch, Y_val_torch, batch_size=len(X_val_torch), shuffle=False)

        L.seed_everything(exp)

        #### # TRAINING NCP Network #####
        print('Training NCP')
        if not coverage['ncp_c'][exp, 0].any():

            MLP_kwargs_U = {
                'input_shape': X_train.shape[-1],
                'output_shape': output_shape,
                'n_hidden': 2,
                'layer_size': 32,
                'dropout': 0.,
                'iterative_whitening': False,
                'activation': torch.nn.GELU
            }

            MLP_kwargs_V = {
                'input_shape': Y_train.shape[-1],
                'output_shape': output_shape,
                'n_hidden': 2,
                'layer_size': 32,
                'dropout': 0,
                'iterative_whitening': False,
                'activation': torch.nn.GELU
            }

            loss_fn = CMELoss
            loss_kwargs = {
                'mode': 'split',
                'gamma': gamma
            }

            reg = NCPOperator(U_operator=MLP, V_operator=MLP, U_operator_kwargs=MLP_kwargs_U, V_operator_kwargs=MLP_kwargs_V)

            NCP_module = NCPModule(
                reg,
                optimizer,
                optimizer_kwargs,
                CMELoss,
                loss_kwargs
            )

            early_stop = EarlyStopping(monitor="val_loss", patience=5000, mode="min")
            ckpt_path = "checkpoints/NCP"
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            checkpoint_callback = NCPModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min",
                                                     dirpath=ckpt_path)

            logger_path = "lightning_logs/NCP"
            if not os.path.exists(logger_path):
                os.makedirs(logger_path)

            trainer = L.Trainer(**{
                'accelerator': 'auto',
                'max_epochs': epochs,
                'log_every_n_steps': 1,
                'enable_progress_bar': True,
                'devices': 1,
                'enable_checkpointing': True,
                'num_sanity_val_steps': 0,
                'check_val_every_n_epoch': 1,
                'enable_model_summary': True,
            }, callbacks=[early_stop, checkpoint_callback])

            trainer.fit(NCP_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

            # recover best model during training
            best_model = torch.load(ckpt_path + '/best_model.pt').to('cpu')
            print(checkpoint_callback.best_model_path)

            # Test coverage on test set
            cdfs = np.zeros((len(X_test), len(y_discr)))
            pdfs = np.zeros((len(X_test), len(y_discr)))
            for i, xi in tqdm(enumerate(X_test), total=len(X_test)):
                xi = xi.reshape(1, -1)
                fys, pred_pdf = best_model.pdf(from_np([[xi]]), y_discr_torch, postprocess='centering', p_y=marginal)
                pred_cdf = integrate_pdf(pred_pdf, y_discr)
                cdfs[i] = pred_cdf
                pdfs[i] = pred_pdf

            iso_cdfs = np.zeros((len(cdfs), len(y_discr)))
            for i, cdf in tqdm(enumerate(cdfs), total=len(cdfs)):
                iso_cdfs[i] = smooth_cdf(y_discr, cdf)


            for a, alpha in enumerate(alphas):
                for a, alpha in enumerate(alphas):
                    quants = []
                    for cdf in tqdm(iso_cdfs):
                        q = quantile_regression_from_cdf(y_discr, cdf, alpha)
                        quants.append(q)
                    quants = np.array(quants)
                coverage['ncp_c'][exp, a] = compute_coverage(quants, Y_test)
                size['ncp_c'][exp, a], size_std['ncp_c'][exp, a] = compute_coverage_length(quants)
                quantiles['ncp_c'][exp, a, :, 1:] = quants.squeeze()
                quantiles['ncp_c'][exp, a, :, 0] = Y_test.flatten()

            print(coverage['ncp_c'][exp, a])
            print(size['ncp_c'][exp, a], size_std['ncp_c'][exp, a])

            cdfs = np.zeros((len(X_test), len(y_discr)))
            pdfs = np.zeros((len(X_test), len(y_discr)))
            for i, xi in tqdm(enumerate(X_test), total=len(X_test)):
                xi = xi.reshape(1, -1)
                fys, pred_pdf = best_model.pdf(from_np([[xi]]), y_discr_torch, postprocess='whitening', p_y=marginal)
                pred_cdf = integrate_pdf(pred_pdf, y_discr)
                cdfs[i] = pred_cdf
                pdfs[i] = pred_pdf

            iso_cdfs = np.zeros((len(cdfs), len(y_discr)))
            for i, cdf in tqdm(enumerate(cdfs), total=len(cdfs)):
                iso_cdfs[i] = smooth_cdf(y_discr, cdf)

            for a, alpha in enumerate(alphas):
                for a, alpha in enumerate(alphas):
                    quants = []
                    for cdf in tqdm(iso_cdfs):
                        q = quantile_regression_from_cdf(y_discr, cdf, alpha)
                        quants.append(q)
                    quants = np.array(quants)
                coverage['ncp_w'][exp, a] = compute_coverage(quants, Y_test)
                size['ncp_w'][exp, a], size_std['ncp_w'][exp, a] = compute_coverage_length(
                    quants)
                quantiles['ncp_w'][exp, a, :, 1:] = quants.squeeze()
                quantiles['ncp_w'][exp, a, :, 0] = Y_test.flatten()

            print(coverage['ncp_w'][exp, a])
            print(size['ncp_w'][exp, a], size_std['ncp_w'][exp, a])

            del NCP_module
            del best_model
            del trainer
            del reg

        ### training nf
        print('Training NF')
        if not coverage['nf'][exp, 0].any():

            base = nf.distributions.base.DiagGaussian(1)

            # Define list of flows (2 flows to emulate our two MLP approach, each with more capacity than our MLPs)
            num_flows = 2
            latent_size = 1
            hidden_units = 32
            num_blocks = 2
            flows = []
            for i in range(num_flows):
                flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, num_blocks, hidden_units,
                                                            num_context_channels=X_train.shape[-1])]
                flows += [nf.flows.LULinearPermute(latent_size)]

            # If the target density is not given
            model = nf.ConditionalNormalizingFlow(base, flows)

            nf_module = NFModule(model,
                optimizer,
                optimizer_kwargs)

            early_stop = EarlyStopping(monitor="val_loss", patience=5000, mode="min")
            ckpt_path = "checkpoints/NF"
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            checkpoint_callback = NFModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", dirpath=ckpt_path)

            logger_path = "lightning_logs/NF"
            if not os.path.exists(logger_path):
                os.makedirs(logger_path)

            trainer = L.Trainer(**{
                'accelerator': 'auto',
                'max_epochs': epochs,
                'log_every_n_steps': 1,
                'enable_progress_bar': True,
                'devices': 1,
                'enable_checkpointing': True,
                'num_sanity_val_steps': 0,
                'check_val_every_n_epoch': 1,
                'enable_model_summary': True,
            }, callbacks=[early_stop, checkpoint_callback])

            trainer.fit(nf_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

            # recover best model during training
            best_model = torch.load(ckpt_path + '/best_model.pt').to('cpu')
            print(checkpoint_callback.best_model_path)

            # Test coverage on test set
            for a, alpha in enumerate(alphas):
                quants = []
                for i, xi in enumerate(tqdm(X_test)):
                    q = quantile_regression(best_model, np.array([xi]), y_discr_torch, alpha=alpha, postprocess='centering', marginal=marginal, model_type='NF')
                    quants.append(q)
                quants = np.array(quants)
                coverage['nf'][exp, a] = compute_coverage(quants, Y_test)
                size['nf'][exp, a], size_std['nf'][exp, a] = compute_coverage_length(quants)
                quantiles['nf'][exp, a, :, 1:] = quants
                quantiles['nf'][exp, a, :, 0] = Y_test.flatten()

            print(coverage['nf'][exp, a])
            print(size['nf'][exp, a], size_std['nf'][exp, a])

            del nf_module
            del best_model
            del trainer
            del model

        #####Training split conformal with random forests

        print('Training rfcc')
        if not coverage['rfcc'][exp, 0].any():
             # split conformal with random forests
            trained_linear_model = RandomForestRegressor().fit(X_train, Y_train.flatten())

            # Instanciate the split conformal wrapper for the linear model.
            # Train argument is set to False because we do not want to retrain the model
            split_cp = SplitCP(trained_linear_model, train=False)

            # With a calibration dataset, compute (and store) nonconformity scores
            split_cp.fit(X_fit=X_train, y_fit=Y_train.flatten(), X_calib=X_val, y_calib=Y_val.flatten())

            # Obtain the model's point prediction y_pred and prediction interval
            # PI = [y_pred_lower, y_pred_upper] for a target coverage of 90% (1-alpha).
            for a, alpha in enumerate(alphas):

                y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
                quants = np.array([y_pred_lower, y_pred_upper]).T
                coverage['rfcc'][exp, a] = compute_coverage(quants, Y_test)
                size['rfcc'][exp, a], size_std['rfcc'][exp, a] = compute_coverage_length(quants)
                quantiles['rfcc'][exp, a, :, 1:] = quants
                quantiles['rfcc'][exp, a, :, 0] = Y_test.flatten()

            print(coverage['rfcc'][exp, a])
            print(size['rfcc'][exp, a], size_std['rfcc'][exp, a])


        with open('NCP/examples/results/{}_coverage_larger.pkl'.format(d), 'wb+') as file:
            pickle.dump(coverage, file)
        with open('NCP/examples/results/{}_covlength_larger.pkl'.format(d), 'wb+') as file:
            pickle.dump(size, file)
        with open('NCP/examples/results/{}_covlengthstd_larger.pkl'.format(d), 'wb+') as file:
            pickle.dump(size_std, file)
        with open('NCP/examples/results/{}_quantiles_larger.pkl'.format(d), 'wb+') as file:
            pickle.dump(quantiles, file)
