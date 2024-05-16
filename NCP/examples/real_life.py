from uci_datasets import Dataset #https://github.com/treforevans/uci_datasets
import numpy as np
from NCP.utils import frnp

import torch
from torch.optim import Adam, SGD
from NCP.model import NCPOperator, NCPModule
from NCP.nn.layers import MLP
from NCP.nn.losses import CMELoss
from NCP.utils import frnp, FastTensorDataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from NCP.metrics import smooth_cdf
from NCP.cdf import compute_marginal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
#print(os.getcwd())

datasets = ['autompg','bike', 'concreteslump', 'energy','keggdirected','parkinsons']
#datasets = ['autompg']
NEXP = 5
NSPLITS = 1
test_prop = 0.9
val_prop = 0.1
alphas = [0.2, 0.1, 0.05]

def compute_coverage(quantiles, values):
    cntr = 0
    for i, val in enumerate(values):
        if (val >= quantiles[i][0]) and (val <= quantiles[i][1]):
            cntr += 1
    return cntr/len(values)

def compute_coverage_length(quantiles):
    lengths = quantiles[:,1] - quantiles[:,0]
    return lengths.mean(), lengths.std()

class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        X, Y = trainer.model.batch
        trainer.model.model._compute_data_statistics(X, Y)

def restore_buffers_shape(model, state_dict):
    model._sing_val = torch.zeros_like(state_dict['model._sing_val']).to('cpu')
    model._sing_vec_l = torch.zeros_like(state_dict['model._sing_vec_l']).to('cpu')
    model._sing_vec_r = torch.zeros_like(state_dict['model._sing_vec_r']).to('cpu')

def compute_quantile_robust(values:np.ndarray, cdf:np.ndarray, alpha, isotonic:bool=True, rescaling:bool=True):
    # TODO: correct this code
    # correction of the cdf using isotonic regression
    if isotonic:
        cdf = smooth_cdf(values, cdf)

    # otherwise, search for the quantile at level alpha
    for i, level in enumerate(cdf):
        if level >= alpha:
            if i == 0:
                quantile = -np.inf
            quantile = values[i-1]
            break
            
        # special case where we exceeded the maximum observed value
    if i == cdf.shape[0] - 1:
        quantile = np.inf

    return quantile

def find_best_quantile(x,cdf, alpha):
    t0 = 0
    t1 = 1
    best_t0 = 0
    best_t1 = 0
    best_size = np.inf

    while t0 < len(cdf):
        # stop if left border reaches right end of discretisation
        if cdf[t1] - cdf[t0] >= 1-alpha:
            # if x[t0], x[t1] is a confidence interval at level alpha, compute length and compare to best
            size = x[t1] - x[t0]
            if size < best_size:
                best_t0 = t0
                best_t1 = t1
                best_size = size
            # moving t1 to the right will only increase the size of the interval, so we can safely move t0 to the right
            t0 += 1
        
        elif t1 == len(cdf)-1:
            # if x[t0], x[t1] is not a confidence interval with confidence at least level alpha, 
            #and t1 is already at the right limit of the discretisation, then there remains no more pertinent intervals
            break
        else:
            # if moving x[t0] to the right reduces the level, we need to increase t1
            t1 += 1
    return x[best_t0], x[best_t1]

def quantile_regression(model, X, y_discr, alpha=0.01, t=1, isotonic=True, rescaling=True, postprocess='centering', marginal=None):
    if len(y_discr) <= 5000:
        x, cdfX = model.cdf(X, y_discr, postprocess=postprocess)
    else:
        x, pdf = model.pdf(torch.Tensor(X), y_discr, p_y=marginal, postprocess='whitening')
        step = (y_discr[1]-y_discr[0]).numpy()
        cdfX = np.cumsum(pdf * step, -1)
    return [compute_quantile_robust(x, cdfX, alpha=alpha/2, isotonic=isotonic, rescaling=rescaling), compute_quantile_robust(x, cdfX, alpha=1-alpha/2, isotonic=isotonic, rescaling=rescaling)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Common optimizer
lr = 1e-3

optimizer = Adam
optimizer_kwargs = {
        'lr': lr
        }

gamma = 2e-2
epochs = int(5e3)
output_shape = 100

for d in datasets:

    if os.path.isfile('NCP/examples/figures/{}_quantiles_larger.npy'.format(d)):
        continue

    coverage = np.zeros((NSPLITS, NEXP, len(alphas)))
    size = np.zeros((NSPLITS, NEXP, len(alphas)))
    size_std = np.zeros((NSPLITS, NEXP, len(alphas)))
    quantiles = {}

    for split in range(NSPLITS):

        data = Dataset(d)
        X_train, Y_train, X_test, Y_test = data.get_split(split=split)

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_prop, random_state=0)

        quantiles[split] = np.zeros((NEXP, len(alphas), len(Y_test), 3))

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

        p1, p99 = np.min(Y_train), np.max(Y_train)
        y_discr, step = np.linspace(p1, p99, num=10000, retstep=True)
        y_discr_torch = torch.Tensor(y_discr.reshape((-1, 1)))

        if len(y_discr) > 5000:
            k_pdf = compute_marginal(bandwidth='scott').fit(Y_train)
            marginal = lambda x : torch.Tensor(np.exp(k_pdf.score_samples(x.reshape(-1, 1))))
        else:
            k_pdf=None
            marginal=None

        train_dl = FastTensorDataLoader(X_train_torch, Y_train_torch, batch_size=len(X_train_torch), shuffle=False)
        val_dl = FastTensorDataLoader(X_val_torch, Y_val_torch, batch_size=len(X_val_torch), shuffle=False)

        for exp in range(NEXP):

            L.seed_everything(exp)

            # TRAINING NCP Network

            MLP_kwargs_U = {
                'input_shape': X_train.shape[-1],
                'output_shape': output_shape,
                'n_hidden': 3,
                'layer_size': [32, 64, 128],
                'dropout': 0.,
                'iterative_whitening': False,
                'activation': torch.nn.ReLU
            }

            MLP_kwargs_V = {
                'input_shape': Y_train.shape[-1],
                'output_shape': output_shape,
                'n_hidden': 3,
                'layer_size':[8, 16, 32],
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

            early_stop = EarlyStopping(monitor="val_loss", patience=100, mode="min")
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
            best_model_dict = torch.load(checkpoint_callbackç.best_model_path)
            restore_buffers_shape(reg, best_model_dict['state_dict'])
            NCP_module.load_state_dict(best_model_dict['state_dict'])
            best_model = NCP_module.model

            # Test coverage on test set
            for a, alpha in enumerate(alphas):
                quants = []
                for i, xi in enumerate(X_test):
                    q = quantile_regression(best_model, np.array([xi]), y_discr_torch, alpha=alpha, postprocess='centering', marginal=marginal)
                    quants.append(q)
                quants = np.array(quants)
                coverage[split, exp, a] = compute_coverage(quants, Y_test)
                size[split, exp, a], size_std[split, exp, a] = compute_coverage_length(quants)
                quantiles[split][exp, a, :, 1:] = quants
                quantiles[split][exp, a, :, 0] = Y_test.flatten()

            del NCP_module
            del best_model
            del trainer
            del reg

    with open('NCP/examples/figures/{}_coverage_larger.npy'.format(d), 'wb+') as file:
        np.save(file, coverage)
    with open('NCP/examples/figures/{}_covlength_larger.npy'.format(d), 'wb+') as file:
        np.save(file, size)
    with open('NCP/examples/figures/{}_covlengthstd_larger.npy'.format(d), 'wb+') as file:
        np.save(file, size_std)
    with open('NCP/examples/figures/{}_quantiles_larger.npy'.format(d), 'wb+') as file:
        np.save(file, quantiles)