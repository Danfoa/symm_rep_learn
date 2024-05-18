import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import matplotlib.pyplot as plt
from scipy.stats import norm
from NCP.model import NCPOperator, NCPModule
from NCP.nn.layers import MLP
from NCP.cdf import get_cdf
from NCP.nn.losses import CMELoss
from NCP.utils import smooth_cdf
from NCP.metrics import hellinger, kullback_leibler, wasserstein1
from NCP.utils import frnp
import lightning as L
from NCP.nn.callbacks import LitProgressBar
from scipy.stats import laplace, cauchy, bernoulli, pareto

from NCP.examples.tools.plot_utils import setup_plots, plot_expectation
from NCP.examples.tools.data_gen import gen_additive_noise_data
setup_plots()

np.random.seed(0)
torch.manual_seed(0)
Ntrain = 50000
Nval = 1000         # val dataset will be used to sample the marginal distribution of Y
Ntest = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'