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

from NCP.cdf import compute_quantile_robust

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#datasets = ['autompg','bike', 'concreteslump', 'energy','houseelectric','keggdirected','parkinsons']
datasets = ['autompg']
NEXP = 1
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

def quantile_regression(model, X, y_discr, observable = lambda x :x, alpha=0.01, t=1, isotonic=True, rescaling=True, postprocess='centering'):
    x, cdfX = model.cdf(X, y_discr, observable, postprocess=postprocess)
    return compute_quantile_robust(x, cdfX, alpha=alpha, isotonic=isotonic, rescaling=rescaling)

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
    data = Dataset(d)
    X_train, Y_train, X_test, Y_test = data.get_split(split=0)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_prop, random_state=0)

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

    p1, p99 = np.percentile(Y_train, [1, 99])
    y_discr, step = np.linspace(p1, p99, num=1000, retstep=True)
    y_discr_torch = torch.Tensor(y_discr.reshape((-1, 1)))

    train_dl = FastTensorDataLoader(X_train_torch, Y_train_torch, batch_size=len(X_train_torch), shuffle=False)
    val_dl = FastTensorDataLoader(X_val_torch, Y_val_torch, batch_size=len(X_val_torch), shuffle=False)
    print(X_train.shape)
    print(X_val.shape)
    print(Y_test.shape)

    coverage = np.zeros((NEXP, len(alphas)))
    size = np.zeros((NEXP, len(alphas)))
    size_std = np.zeros((NEXP, len(alphas)))

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
        best_model_dict = torch.load(checkpoint_callback.best_model_path)
        restore_buffers_shape(reg, best_model_dict['state_dict'])
        NCP_module.load_state_dict(best_model_dict['state_dict'])
        best_model = NCP_module.model

        # Test coverage on test set
        for a, alpha in enumerate(alphas):
            quantiles = quantile_regression(best_model, X_test, y_discr_torch, postprocess='centering')

            coverage[exp, a] = compute_coverage(quantiles, Y_test)
            size[exp, a], size_std[exp, a] = compute_coverage_length(quantiles)

    with open('figures/{}_coverage.npy'.format(d), 'wb+') as file:
        np.save(file, coverage)
    with open('figures/{}_covlength.npy'.format(d), 'wb+') as file:
        np.save(file, size)
    with open('figures/{}_covlengthstd.npy'.format(d), 'wb+') as file:
        np.save(file, size_std)