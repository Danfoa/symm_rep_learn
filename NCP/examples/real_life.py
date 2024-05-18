from uci_datasets import Dataset #https://github.com/treforevans/uci_datasets
# https://github.com/deel-ai/puncc
import numpy as np
from NCP.utils import frnp

import torch
from torch.optim import Adam
from NCP.model import NCPOperator, NCPModule
from NCP.nn.layers import MLP
from NCP.nn.losses import CMELoss
from NCP.nn.nf_module import NFModule
from NCP.utils import frnp, FastTensorDataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from NCP.metrics import smooth_cdf
from NCP.cdf import compute_marginal, quantile_regression, quantile_regression_from_cdf
from NCP.examples.tools.lincde import lincde

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import normflows as nf

from deel.puncc.regression import SplitCP
from uci_datasets import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


from scipy.stats import ecdf
import pickle

import os
#print(os.getcwd())

datasets = ['bike', 
           # 'houseelectric',
            'parkinsons', 
            'slice', 
            'buzz',
            'gas',
            'protein',]
#datasets = ['autompg']
NEXP = 5
NSPLITS = 1
test_prop = 0.9
val_prop = 0.2
alphas = [0.1]

class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        X, Y = trainer.model.batch
        trainer.model.model._compute_data_statistics(X, Y)

def restore_buffers_shape(model, state_dict):
    model._sing_val = torch.zeros_like(state_dict['model._sing_val']).to('cpu')
    model._sing_vec_l = torch.zeros_like(state_dict['model._sing_vec_l']).to('cpu')
    model._sing_vec_r = torch.zeros_like(state_dict['model._sing_vec_r']).to('cpu')


def init_dico():
    return {
        'ncp':np.zeros((NSPLITS, NEXP, len(alphas))),
        'nf':np.zeros((NSPLITS, NEXP, len(alphas))),
        'mlpcc':np.zeros((NSPLITS, NEXP, len(alphas))),
        'rfcc':np.zeros((NSPLITS, NEXP, len(alphas)))
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Common optimizer
lr = 1e-3

optimizer = Adam
optimizer_kwargs = {
        'lr': lr
        }

gamma = 2e-1
epochs = int(5e3)
output_shape = 200

for d in datasets:

    if os.path.isfile('NCP/examples/figures/{}_quantiles_larger.pkl'.format(d)):
        with open('NCP/examples/figures/{}_coverage_larger.pkl'.format(d), 'rb') as file:
            coverage = np.load(file, allow_pickle=True)
        with open('NCP/examples/figures/{}_covlength_larger.pkl'.format(d), 'rb') as file:
            size = np.load(file, allow_pickle=True)
        with open('NCP/examples/figures/{}_covlengthstd_larger.pkl'.format(d), 'rb') as file:
            size_std = np.load(file, allow_pickle=True)
        with open('NCP/examples/figures/{}_quantiles_larger.pkl'.format(d), 'rb') as file:
            quantiles = np.load(file, allow_pickle=True)

    else:

        coverage = init_dico()
        size = init_dico()
        size_std = init_dico()
        quantiles = {
            'ncp':{},
            'nf':{},
            'mlpcc':{},
            'rfcc':{},
        }

    for split in range(NSPLITS):

        data = Dataset(d)
        X_train, Y_train, X_test, Y_test = data.get_split(split=split)

        #cut test set for computation time:
        if X_test.shape[0] > 200:
            X_test, Y_test = X_test[:200], Y_test[:200]

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_prop, random_state=0)
        for m in ['ncp', 'nf','mlpcc','rfcc']:
            if not m in quantiles.keys():
                quantiles[m] = {}
                coverage[m] = np.zeros((NSPLITS, NEXP, len(alphas)))
                size[m] = np.zeros((NSPLITS, NEXP, len(alphas)))
                size_std[m] = np.zeros((NSPLITS, NEXP, len(alphas)))
            quantiles[m][split] = np.zeros((NEXP, len(alphas), len(Y_test), 3))

        xscaler = StandardScaler()
        yscaler = StandardScaler()

        X_train = xscaler.fit_transform(X_train)
        Y_train = yscaler.fit_transform(Y_train)

        X_val = xscaler.transform(X_val)
        Y_val = yscaler.transform(Y_val)
        X_test = xscaler.transform(X_test)
        Y_test = yscaler.transform(Y_test)

        X_train_torch = frnp(X_train)
        Y_train_torch = frnp(Y_train)
        X_val_torch = frnp(X_val)
        Y_val_torch = frnp(Y_val)
        X_test_torch = frnp(X_test)
        Y_tes_torch = frnp(Y_test)

        #y discretisation for computing cdf
        spread = np.max(Y_train) - np.min(Y_train)
        p1, p99 = np.min(Y_train), np.max(Y_train)
        y_discr, step = np.linspace(p1-0.1*spread, p99+0.1*spread, num=5000, retstep=True)
        y_discr_torch = torch.Tensor(y_discr.reshape((-1, 1)))

        k_pdf = compute_marginal(bandwidth='scott').fit(Y_train)
        marginal = lambda x : torch.Tensor(np.exp(k_pdf.score_samples(x.reshape(-1, 1))))

        train_dl = FastTensorDataLoader(X_train_torch, Y_train_torch, batch_size=len(X_train_torch), shuffle=False)
        val_dl = FastTensorDataLoader(X_val_torch, Y_val_torch, batch_size=len(X_val_torch), shuffle=False)

        for exp in range(NEXP):

            L.seed_everything(exp)

            #### # TRAINING NCP Network #####

            if not coverage['ncp'][split, exp, 0].any():

                MLP_kwargs_U = {
                    'input_shape': X_train.shape[-1],
                    'output_shape': output_shape,
                    'n_hidden': 3,
                    'layer_size': 128,
                    'dropout': 0.,
                    'iterative_whitening': False,
                    'activation': torch.nn.ReLU
                }

                MLP_kwargs_V = {
                    'input_shape': Y_train.shape[-1],
                    'output_shape': output_shape,
                    'n_hidden': 3,
                    'layer_size':128,
                    'dropout': 0,
                    'iterative_whitening': False,
                    'activation': torch.nn.ReLU
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

                early_stop = EarlyStopping(monitor="val_loss", patience=500, mode="min")
                checkpoint_callback = CustomModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

                trainer = L.Trainer(**{
                    'accelerator': device,
                    'max_epochs': epochs,
                    'log_every_n_steps': 1,
                    'enable_progress_bar': True,
                    'devices': 1,
                    'enable_checkpointing': True,
                    'num_sanity_val_steps': 0,
                    'enable_model_summary': False,
                    }, callbacks=[early_stop, checkpoint_callback])

                trainer.fit(NCP_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

                # recover best model during training
                best_model_dict = torch.load(checkpoint_callback.best_model_path)
                restore_buffers_shape(reg, best_model_dict['state_dict'])
                NCP_module.load_state_dict(best_model_dict['state_dict'])
                best_model = NCP_module.model

                # Test coverage on test set
                for a, alpha in enumerate(alphas):
                    quants = []
                    for i, xi in enumerate(tqdm(X_test)):
                        q = quantile_regression(best_model, np.array([xi]), y_discr_torch, alpha=alpha, postprocess='centering', marginal=marginal)
                        quants.append(q)
                    quants = np.array(quants)
                    coverage['ncp'][split, exp, a] = compute_coverage(quants, Y_test)
                    size['ncp'][split, exp, a], size_std['ncp'][split, exp, a] = compute_coverage_length(quants)
                    quantiles['ncp'][split][exp, a, :, 1:] = quants
                    quantiles['ncp'][split][exp, a, :, 0] = Y_test.flatten()

                del NCP_module
                del best_model
                del trainer
                del reg

            ### training nf
            if not coverage['nf'][split, exp, 0].any():

                base = nf.distributions.base.DiagGaussian(1)

                # Define list of flows (2 flows to emulate our two MLP approach, each with more capacity than our MLPs)
                num_flows = 2
                latent_size = 1
                hidden_units = 128
                num_blocks = 3
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

                checkpoint_callback_vanilla = ModelCheckpoint(monitor="val_loss", mode="min")

                trainer = L.Trainer(**{
                    'accelerator': device,
                    'max_epochs': int(5e3),
                    'log_every_n_steps': 1,
                    'enable_progress_bar': True,
                    'devices': 1,
                    'enable_checkpointing': True,
                    'num_sanity_val_steps': 0,
                    'enable_model_summary': False,
                    }, callbacks=[checkpoint_callback_vanilla])

                trainer.fit(nf_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

                checkpoint_callback_vanilla.best_model_path
                best_model_dict = torch.load(checkpoint_callback_vanilla.best_model_path)
                nf_module.load_state_dict(best_model_dict['state_dict'])

                best_model = nf_module.model

                # Test coverage on test set            
                for a, alpha in enumerate(alphas):
                    quants = []
                    for i, xi in enumerate(tqdm(X_test)):
                        q = quantile_regression(best_model, np.array([xi]), y_discr_torch, alpha=alpha, postprocess='centering', marginal=marginal, model_type='NF')
                        quants.append(q)
                    quants = np.array(quants)
                    coverage['nf'][split, exp, a] = compute_coverage(quants, Y_test)
                    size['nf'][split, exp, a], size_std['nf'][split, exp, a] = compute_coverage_length(quants)
                    quantiles['nf'][split][exp, a, :, 1:] = quants
                    quantiles['nf'][split][exp, a, :, 0] = Y_test.flatten()

                del nf_module
                del best_model
                del trainer
                del model

            ##### Training lincde

            # ys, cdf = lincde(X_train, Y_train, X_test, y_discr, folder_location='NCP/examples/')
            # for a, alpha in enumerate(tqdm(alphas)):
            #     quants = quantile_regression_from_cdf(ys, cdf, alpha)
            #     coverage['lincde'][split, exp, a] = compute_coverage(quants, Y_test)
            #     size['lincde'][split, exp, a], size_std['lincde'][split, exp, a] = compute_coverage_length(quants)
            #     quantiles['lincde'][split][exp, a, :, 1:] = quants
            #     quantiles['lincde'][split][exp, a, :, 0] = Y_test.flatten()

            #####Training condconform

            #reg = RandomForestRegressor().fit(X_train, Y_train)

            if not coverage['rfcc'][split, exp, 0].any():

                try:

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
                        coverage['rfcc'][split, exp, a] = compute_coverage(quants, Y_test)
                        size['rfcc'][split, exp, a], size_std['nf'][split, exp, a] = compute_coverage_length(quants)
                        quantiles['rfcc'][split][exp, a, :, 1:] = quants
                        quantiles['rfcc'][split][exp, a, :, 0] = Y_test.flatten()
                except:
                    pass

            # split conformal with MLP
            if not coverage['mlpcc'][split, exp, 0].any():

                try:
                    trained_linear_model = MLPRegressor(hidden_layer_sizes=[32, 64, 128, 32, 16, 8]
                                                        ,max_iter=epochs).fit(X_train, Y_train.flatten())
                    split_cp = SplitCP(trained_linear_model, train=False)
                    split_cp.fit(X_fit=X_train, y_fit=Y_train.flatten(), X_calib=X_val, y_calib=Y_val.flatten())
                    for a, alpha in enumerate(alphas):

                        y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
                        quants = np.array([y_pred_lower, y_pred_upper]).T
                        coverage['mlpcc'][split, exp, a] = compute_coverage(quants, Y_test)
                        size['mlpcc'][split, exp, a], size_std['nf'][split, exp, a] = compute_coverage_length(quants)
                        quantiles['mlpcc'][split][exp, a, :, 1:] = quants
                        quantiles['mlpcc'][split][exp, a, :, 0] = Y_test.flatten()
                except:
                    pass


    with open('NCP/examples/figures/{}_coverage_larger.pkl'.format(d), 'wb+') as file:
        pickle.dump(coverage, file)
    with open('NCP/examples/figures/{}_covlength_larger.pkl'.format(d), 'wb+') as file:
        pickle.dump(size, file)
    with open('NCP/examples/figures/{}_covlengthstd_larger.pkl'.format(d), 'wb+') as file:
        pickle.dump(size_std, file)
    with open('NCP/examples/figures/{}_quantiles_larger.pkl'.format(d), 'wb+') as file:
        pickle.dump(quantiles, file)