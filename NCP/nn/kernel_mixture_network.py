import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering, KMeans
from torch.autograd import Variable

from NCP.mysc.utils import from_np, to_np


def sample_center_points(y, method='all', k=100, keep_edges=False):
    """Function to define kernel centers with various downsampling alternatives
    """
    # make sure y is 1D
    y = to_np(y)
    y = y.ravel()

    # keep all points as kernel centers
    if method == 'all':
        return y

    # retain outer points to ensure expressiveness at the target borders
    if keep_edges:
        y = np.sort(y)
        centers = np.array([y[0], y[-1]])
        y = y[1:-1]
        # adjust k such that the final output has size k
        k -= 2
    else:
        centers = np.empty(0)

    if method == 'random':
        cluster_centers = np.random.choice(y, k, replace=False)

    # iteratively remove part of pairs that are closest together until everything is at least 'd' apart
    elif method == 'distance':
        raise NotImplementedError

    # use 1-D k-means clustering
    elif method == 'k_means':
        model = KMeans(n_clusters=k)
        model.fit(y.reshape(-1, 1))
        cluster_centers = model.cluster_centers_

    # use agglomerative clustering
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k, linkage='complete')
        model.fit(y.reshape(-1, 1))
        labels = pd.Series(model.labels_, name='label')
        y_s = pd.Series(y, name='y')
        df = pd.concat([y_s, labels], axis=1)
        cluster_centers = df.groupby('label')['y'].mean().values

    else:
        raise ValueError("unknown method '{}'".format(method))

    return np.append(centers, cluster_centers)

class estimator_infer_sigma(nn.Module):
    def __init__(self, ndim_x, n_centers, device):
        super(estimator_infer_sigma, self).__init__()
        self.n_centers = n_centers
        self.linear_1 = nn.Linear(ndim_x, 16).to(device)
        self.linear_2 = nn.Linear(16, 32).to(device)
        self.linear_3 = nn.Linear(32, 64).to(device)
        self.linear_4 = nn.Linear(64, 128).to(device)
        self.linear_5 = nn.Linear(128, 128).to(device)
        self.linear_6 = nn.Linear(128, self.n_centers).to(device)
        self.linear_7 = nn.Linear(32, 1).to(device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.relu(out)
        out = self.linear_3(out)
        out = self.relu(out)
        out = self.linear_4(out)
        out = self.relu(out)
        out = self.linear_5(out)
        out = self.relu(out)
        out1 = self.linear_6(out)
        out1 = self.softmax(out1)
        out2 = self.linear_7(out)
        out2 = self.softplus(out2)
        return torch.cat([out1, out2], dim=1)

class kmn_torch_infer_sigma(object):
    def __init__(self, x_train, y_train, center_sampling_method='k_means', n_centers=20, keep_edges=True,
                 estimator=None, validation_set=None, device='cpu'):
        self.device = device
        self.center_sampling_method = center_sampling_method
        self.n_centers = n_centers
        self.estimator = estimator(x_train.shape[-1], self.n_centers, self.device)
        self.keep_edges = keep_edges
        self.sigma = Variable(0.5 * torch.ones(self.n_centers), requires_grad=False)
        self.center_locs = Variable(from_np(sample_center_points(y_train, method=self.center_sampling_method,
                                                                 k=self.n_centers,
                                                                 keep_edges=self.keep_edges), self.device), requires_grad=False)

        self.n_data, self.n_features = x_train.shape

        self.x = Variable(x_train.reshape(self.n_data, self.n_features),
                                  requires_grad=False)
        self.y = Variable(y_train.reshape(self.n_data, 1), requires_grad=False)

        self.validation_present = False
        if (validation_set != None):
            self.validation_present = True
            x_val = validation_set['x']
            y_val = validation_set['y']
            self.n_data_val, _ = x_val.shape
            self.x_val = Variable(x_val.reshape(self.n_data_val, self.n_features),
                                  requires_grad=False)
            self.y_val = Variable(y_val.reshape(self.n_data_val, 1),
                                  requires_grad=False)

        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0 * np.pi)  # normalisation factor for gaussian.

    def gaussian_distribution(self, y, mu, sigma):
        result = (y - mu) * torch.reciprocal(sigma)
        result = - 0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * self.oneDivSqrtTwoPI

    def mdn_loss_function(self, weights, sigma, y):
        result = self.gaussian_distribution(y, self.center_locs, sigma) * weights
        result = torch.sum(result, dim=1)
        result = - torch.log(result)
        return torch.mean(result)

    def fit(self, learning_rate=0.01, n_iterations=300, dataset_name=''):
        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=learning_rate)
        self.loss = []
        self.loss_val = []
        min_val_loss = np.inf
        patience = 0
        for i in range(n_iterations):
            self.estimator.train()
            out = self.estimator(self.x)
            weights = out[:, 0:-1]
            sigma = torch.unsqueeze(out[:, -1], dim=1)
            l = self.mdn_loss_function(weights, sigma, self.y)
            self.loss.append(to_np(l))

            if (self.validation_present):
                self.estimator.eval()
                out = self.estimator(self.x_val)
                weights_val = out[:, 0:-1]
                sigma_val = torch.unsqueeze(out[:, -1], dim=1)
                l_val = self.mdn_loss_function(weights_val, sigma_val, self.y_val)
                self.loss_val.append(to_np(l_val))

            if (i % 100 == 0):
                if (self.validation_present):
                    print('Iter: {0} - loss: {1} - loss_val: {2} - sigma: {3}'.format(i, self.loss[-1],
                                                                                      self.loss_val[-1],
                                                                                      to_np(out[0, -1])))

                    if self.loss_val[-1] < min_val_loss:
                        min_val_loss = self.loss_val[-1]
                        torch.save(self.estimator, dataset_name + '_best_model.pt')
                        patience = 0
                    else:
                        patience += 1
                        if patience > 10:
                            print('\nEarly stopping due to no improvement in validation loss')
                            break

                    # if len(self.loss_val) > 2 and (self.loss_val[-1] > self.loss_val[-2]):
                    #     print('\nEarly stopping due to increasing validation loss')
                    #     break
                else:
                    print('Iter: {0} - loss: {1}'.format(i, self.loss[-1]))

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

    def pdf(self, x_test, y):
        out = self.estimator(x_test)
        weights = out[:, 0:-1]
        sigma = torch.unsqueeze(out[:, -1], dim=1)
        result = self.gaussian_distribution(torch.unsqueeze(y, 1), self.center_locs, sigma) * weights
        result = torch.sum(result, dim=1)
        return to_np(y), to_np(result)
