from __future__ import annotations

import logging

import escnn
import numpy as np
import scipy.stats as stats
import torch
from escnn.group import Representation, directsum
from symm_learning.stats import var_mean

from symm_rep_learn.mysc.utils import project_to_pos_semi_def

from .GMM import GaussianMixture

log = logging.getLogger(__name__)


class SymmGaussianMixture(GaussianMixture):
    """TODO: Add docstring.

    Args:
      n_kernels: number of mixture components
      ndim_x: dimensionality of X / number of random variables in X
      ndim_y: dimensionality of Y / number of random variables in Y
      mean_max_norm: std. dev. when sampling the kernel means
      sampling_seed: seed for the random_number generator
    """

    def __init__(
        self,
        n_kernels: int,
        rep_X: Representation,
        rep_Y: Representation,
        mean_max_norm=0.1,  # Dont put high values of this.
        x_subgroup_id=None,
        y_subgroup_id=None,
        sampling_seed=10,
        gmm_seed=100,
    ):
        # print(f"GMM n_kernels={n_kernels}, mean_max_norm={mean_max_norm}, rep_X={rep_X}, rep_Y={rep_Y}")
        self.random_state = np.random.RandomState(seed=sampling_seed)  # random state for sampling data
        self.random_state_params = np.random.RandomState(seed=gmm_seed)  # fixed random state for sampling GMM params
        self.random_seed = sampling_seed

        self.has_pdf = True
        self.has_cdf = True
        self.can_sample = True

        # Symmetry group
        self.G = rep_X.group
        # If a subgroup is specified, restrict the GMM to that subgroup
        if x_subgroup_id is not None:
            x_subgroup_id = self.G.subgroup_trivial_id if x_subgroup_id == "trivial" else x_subgroup_id
            self.Hx, _, G2Hx = self.G.subgroup(x_subgroup_id)
            self.G2Hx = lambda g: G2Hx(g) if G2Hx(g) is not None else self.Hx.identity
            rep_X = rep_X.restrict(x_subgroup_id)
            log.info(f"Restricting the X component to subgroup {self.Hx} of order {self.Hx.order()}")
        else:
            self.Hx, _, self.G2Hx = self.G, lambda x: x, lambda x: x
        if y_subgroup_id is not None:
            y_subgroup_id = self.G.subgroup_trivial_id if y_subgroup_id == "trivial" else y_subgroup_id
            self.Hy, _, G2Hy = self.G.subgroup(y_subgroup_id)
            self.G2Hy = lambda g: G2Hy(g) if G2Hy(g) is not None else self.Hy.identity
            rep_Y = rep_Y.restrict(y_subgroup_id)
            log.info(f"Restricting the Y component to subgroup {self.Hy} of order {self.Hy.order()}")
        else:
            self.Hy, _, self.G2Hy = self.G, lambda x: x, lambda x: x

        """  set parameters, calculate weights, means and covariances """
        self.n_kernels = n_kernels
        self.ndim = rep_X.size + rep_Y.size

        self.rep_X, self.rep_Y = rep_X, rep_Y

        self.ndim_x = rep_X.size
        self.ndim_y = rep_Y.size
        self.means_max_norm = mean_max_norm
        self.weights = self._sample_weights(n_kernels)  # shape(n_kernels,), sums to one
        # self.means = self.random_state_params.normal(
        #     loc=np.zeros([self.ndim]), scale=self.means_max_norm, size=[n_kernels, self.ndim]
        #     )  # shape(n_kernels, n_dims)
        norms_x = self.random_state_params.uniform(
            low=self.means_max_norm * 0.05, high=self.means_max_norm, size=(n_kernels, 1)
        )
        norms_y = self.random_state_params.uniform(
            low=self.means_max_norm * 0.05, high=self.means_max_norm, size=(n_kernels, 1)
        )
        unit_vects = self.random_state_params.normal(size=(n_kernels, self.ndim_x))
        unit_vects /= np.linalg.norm(unit_vects, axis=-1, keepdims=True)
        means_x = norms_x * unit_vects
        unit_vects = self.random_state_params.normal(size=(n_kernels, self.ndim_y))
        unit_vects /= np.linalg.norm(unit_vects, axis=-1, keepdims=True)
        means_y = norms_y * unit_vects
        self.means = np.concatenate([means_x, means_y], axis=-1)
        """ Sample cov matrices and assure that cov matrix is pos definite"""
        self.covariances_x = self.sample_covariances(dim=self.ndim_x, means=self.means[:, : self.ndim_x])
        self.covariances_y = self.sample_covariances(dim=self.ndim_y, means=self.means[:, self.ndim_x :])

        if not self.G.continuous:
            # To make these distributions invariant under the group action, we need to average over the group
            self.n_kernels = n_kernels * self.G.order()
            means = [self.means]
            Cxs, Cys = [self.covariances_x], [self.covariances_y]
            for g in self.G.elements:
                if g == self.G.identity:
                    continue
                if self.G2Hx(g) is not None:  # Element in the Subgroup Hx
                    g_mean_x = np.einsum("ij,kj->ki", self.rep_X(self.G2Hx(g)), self.means[:, : self.ndim_x])
                    g_Cx = np.einsum(
                        "ij,kjm,mn->kin", self.rep_X(self.G2Hx(g)), self.covariances_x, self.rep_X(~self.G2Hx(g))
                    )
                else:
                    g_mean_x, g_Cx = self.means[:, : self.ndim_x], self.covariances_x
                if self.G2Hy(g) is not None:  # Element in the Subgroup Hy
                    g_mean_y = np.einsum("ij,kj->ki", self.rep_Y(self.G2Hy(g)), self.means[:, self.ndim_x :])
                    g_Cy = np.einsum(
                        "ij,kjm,mn->kin", self.rep_Y(self.G2Hy(g)), self.covariances_y, self.rep_Y(~self.G2Hy(g))
                    )
                else:
                    g_mean_y, g_Cy = self.means[:, self.ndim_x :], self.covariances_y
                Cxs.append(g_Cx)
                Cys.append(g_Cy)
                g_means = np.concatenate([g_mean_x, g_mean_y], axis=-1)
                means.append(g_means)
            self.means = np.concatenate(means, axis=0)
            self.covariances_x = np.concatenate(Cxs, axis=0)
            self.covariances_y = np.concatenate(Cys, axis=0)
            # Prune identical kernels.
            unique_idx = np.unique(self.means, axis=0, return_index=True)[1]
            self.n_kernels = len(unique_idx)
            self.means = self.means[unique_idx]
            self.covariances_x = self.covariances_x[unique_idx]
            self.covariances_y = self.covariances_y[unique_idx]
            self.weights = np.concatenate([self.weights] * self.G.order())[unique_idx]
            self.weights /= np.sum(self.weights)
        else:
            raise NotImplementedError("Only finite groups are supported at the moment")
            # This requires us to use uniform on angle and gaussian on radius. Can be done, need to work out the details

        # some eigenvalues of the sampled covariance matrices can be exactly zero -> map to pos semi-definite subspace
        self.covariances = np.zeros(shape=(self.n_kernels, self.ndim, self.ndim))
        self.covariances[:, : self.ndim_x, : self.ndim_x] = self.covariances_x
        self.covariances[:, self.ndim_x :, self.ndim_x :] = self.covariances_y

        """ after mapping, define the remaining variables and collect frozen multivariate variables
        (x,y), x and y for later conditional draws """
        self.means_x = self.means[:, : self.ndim_x]
        self.means_y = self.means[:, self.ndim_x :]

        self.gaussians, self.gaussians_x, self.gaussians_y = [], [], []
        for i in range(self.n_kernels):
            self.gaussians.append(stats.multivariate_normal(mean=self.means[i,], cov=self.covariances[i]))
            self.gaussians_x.append(stats.multivariate_normal(mean=self.means_x[i,], cov=self.covariances_x[i]))
            self.gaussians_y.append(stats.multivariate_normal(mean=self.means_y[i,], cov=self.covariances_y[i]))

        # approximate data statistics
        self.y_mean, self.y_std = self._compute_data_statistics()

        self._running_samples = 0
        self._mi_estimate = 0

    def sample_covariances(self, dim, means):
        """Sample covariance matrices for the GMM
        Args:
          dim: dimensionality of the covariance matrices
          scale: std of the eigenvalues of the covariance matrices
          means_std: std of the means of the centers of the kernes
          num: number of covariance matrices to sample
        Returns:
          ndarray of covariance matrices with shape (number, dim, dim).
        """
        # Sample diagonal entries / eigenvalues of the covariance matrix
        num = means.shape[0]

        norms = np.linalg.norm(means, axis=-1)
        _min_norm, max_diff = np.min(norms), np.max(norms) - np.min(norms)
        max_dis = max_diff  # + min_norm
        eigenvalues = []
        for i, center in enumerate(means):
            # norm = np.linalg.norm(center)
            # Center should be 3 standard deviations away from the mean. Eigval = variance
            max_var = max_dis / 3
            # Support of the gaussian should not collapse to less than 10% of the norm.
            min_var = max_dis / 2 / 3
            min_eigval, max_eigval = min_var, max_var
            mean_eigval = (max_eigval - min_eigval) / 2
            std_eigval = max_eigval - min_eigval
            var = np.abs(self.random_state_params.normal(loc=mean_eigval, scale=std_eigval, size=(1, dim)))
            var = np.clip(var, min_eigval, max_eigval)
            eigenvalues.append(var)
        eigenvalues = np.concatenate(eigenvalues, axis=0)
        # print(f"Mean Var = {mean_eigval}, min Var = {min_var}, max Var = {max_var}")
        # Sample num orthogonal matrices usign QR decomposition
        Q, _ = np.linalg.qr(self.random_state_params.normal(size=(num, dim, dim)))
        var = np.asarray([np.diag(e) for e in eigenvalues])
        cov = np.einsum("...ij,...jk,...lk->...il", Q, var, Q)
        return project_to_pos_semi_def(cov)

    def _sample_weights(self, n_weights):
        """Samples density weights -> sum up to one
        Args:
          n_weights: number of weights
        Returns:
          ndarray of weights with shape (n_weights,).
        """
        weights = self.random_state_params.random(n_weights)
        weights /= np.sum(weights)
        return weights

    def _compute_data_statistics(self):
        """Return mean and std of the y component of the data."""
        X, Y = self.simulate(n_samples=10**4)
        Y_var, Y_mean = var_mean(torch.tensor(Y), self.rep_Y)
        return Y_mean.detach().numpy(), np.sqrt(Y_var.detach().numpy())

    def pdf_y(self, Y):
        """Marginal probability density function P(Y).

        Args:
            Y: the variable Y for the distribution P(Y), array_like, shape:(n_samples, ndim_y)

        Returns:
            the marginal distribution of Y with shape:(n_samples,)
        """
        Y = self._handle_input_dimensionality(Y)
        assert Y.shape[-1] == self.ndim_y, f"Y has the wrong dimensionality: {Y.shape[-1]} != {self.ndim_y}"
        p_y = np.sum([self.weights[i] * self.gaussians_y[i].pdf(Y) for i in range(self.n_kernels)], axis=0)
        return p_y

    def pdf_x(self, X):
        """Marginal probability density function P(X).

        Args:
            X: the variable X for the distribution P(X), array_like, shape:(n_samples, ndim_x)

        Returns:
            the marginal distribution of X with shape:(n_samples,)
        """
        X = self._handle_input_dimensionality(X)
        assert X.shape[-1] == self.ndim_x, f"X has the wrong dimensionality: {X.shape[-1]} != {self.ndim_x}"
        p_x = np.sum([self.weights[i] * self.gaussians_x[i].pdf(X) for i in range(self.n_kernels)], axis=0)
        return p_x

    def pointwise_mutual_dependency(self, X, Y):
        """Compute the pointwise mutual dependency between X and Y.

        Args:
            X: (..., ndim_x) array of samples from X
            Y: (..., ndim_y) array of samples from Y

        Returns:
            pmd:  (..., 1) pointwise mutual dependency between x and y defined as PMD = p(x,y)/p(x)p(y).
        """
        X, Y = self._handle_input_dimensionality(X), self._handle_input_dimensionality(Y)
        p_xy = self.joint_pdf(X, Y)
        p_x = self.pdf_x(X)
        p_y = self.pdf_y(Y)
        return p_xy / (p_x * p_y)

    def pointwise_mutual_information(self, X, Y):
        """Compute the mutual information between X and Y.

        Defined by MI = ln(p(x,y) / p(x)p(y))

        Args:
            X: (..., ndim_x) array of samples from X
            Y: (..., ndim_y) array of samples from Y
        Returns:
            (..., 1) Mutual information between X and Y for each sample
        """
        X, Y = self._handle_input_dimensionality(X), self._handle_input_dimensionality(Y)
        p_xy = self.joint_pdf(X, Y)
        p_x = self.pdf_x(X)
        p_y = self.pdf_y(Y)
        pmd = p_xy / (p_x * p_y)
        pmi = np.log(pmd)
        return pmi

    def normalized_pointwise_mutual_information(self, X, Y):
        X, Y = self._handle_input_dimensionality(X), self._handle_input_dimensionality(Y)
        p_xy = self.joint_pdf(X, Y)
        p_x = self.pdf_x(X)
        p_y = self.pdf_y(Y)
        return np.log(p_xy / (p_x * p_y)) / (-np.log(p_xy))

    def MI(self, X, Y):
        X, Y = self._handle_input_dimensionality(X), self._handle_input_dimensionality(Y)
        p_xy = self.joint_pdf(X, Y)
        p_x = self.pdf_x(X)
        p_y = self.pdf_y(Y)
        pmd = p_xy / (p_x * p_y)
        pmi = np.log(pmd)
        return (pmi * (p_xy / p_xy.sum())).sum()  # Weighted average of the expectation of the PMI


def test_inv(gmm: SymmGaussianMixture):
    # for n in range(10):
    x_test, y_test = gmm.simulate(n_samples=10)
    x_rep = gmm.rep_X
    y_rep = gmm.rep_Y

    G_px, G_py = [np.squeeze(gmm.pdf_x(x_test))], [np.squeeze(gmm.pdf_y(y_test))]
    G_pxy = [np.squeeze(gmm.joint_pdf(x_test, y_test))]

    H_px, H_py = [np.squeeze(gmm.pdf_x(x_test))], [np.squeeze(gmm.pdf_y(y_test))]
    H_pxy = [np.squeeze(gmm.joint_pdf(x_test, y_test))]
    for g in G.elements:
        g_x = np.einsum("ij,nj->ni", x_rep(g), x_test)
        G_px.append(np.squeeze(gmm.pdf_x(g_x)))
        g_y = np.einsum("ij,nj->ni", y_rep(g), y_test)
        G_py.append(np.squeeze(gmm.pdf_y(g_y)))
        G_pxy.append(np.squeeze(gmm.joint_pdf(g_x, g_y)))
        # Compute using the subgroups of the GMM using the data symmetry group
        g_Hx = gmm.G2Hx(g) if gmm.G2Hx(g) else gmm.Hx.identity
        g_Hy = gmm.G2Hy(g) if gmm.G2Hy(g) else gmm.Hy.identity
        g_Hx_x = np.einsum("ij,nj->ni", gmm.rep_X(g_Hx), x_test)
        H_px.append(np.squeeze(gmm.pdf_x(g_Hx_x)))
        g_Hy_y = np.einsum("ij,nj->ni", gmm.rep_Y(g_Hy), y_test)
        H_py.append(np.squeeze(gmm.pdf_y(g_Hy_y)))
        H_pxy.append(np.squeeze(gmm.joint_pdf(g_Hx_x, g_Hy_y)))

        if G == gmm.Hx and G == gmm.Hy:
            assert np.allclose(
                G_px, G_px[0], atol=1e-5, rtol=1e-5
            ), f"The marginal distribution of X is not invariant under the group action: {G_px}"
            assert np.allclose(
                G_py, G_py[0], atol=1e-5, rtol=1e-5
            ), f"The marginal distribution of Y is not invariant under the group action: {G_py}"
            assert np.allclose(
                G_pxy, G_pxy[0], atol=1e-5, rtol=1e-5
            ), f"The joint distribution of X and Y is not invariant under the group action: {G_pxy}"
        else:
            assert np.allclose(
                H_px, H_px[0], atol=1e-5, rtol=1e-5
            ), f"The marginal distribution of X is not invariant under the group action: {H_px}"
            assert np.allclose(
                H_py, H_py[0], atol=1e-5, rtol=1e-5
            ), f"The marginal distribution of Y is not invariant under the group action: {H_py}"
            assert np.allclose(
                H_pxy, H_pxy[0], atol=1e-5, rtol=1e-5
            ), f"The joint distribution of X and Y is not invariant under the group action: {H_pxy}"


if __name__ == "__main__":
    from lightning import seed_everything
    from matplotlib import pyplot as plt

    G = escnn.group.DihedralGroup(6)
    x_rep = G.regular_representation  # ρ_Χ
    y_rep = G.regular_representation
    # G = escnn.group.CyclicGroup(2)
    # x_rep = G.representations['irrep_1'] + G.representations['irrep_1']  # ρ_Χ
    # y_rep = G.representations['irrep_1']

    gmm = SymmGaussianMixture(
        10,
        rep_X=directsum([x_rep] * 2),
        rep_Y=directsum([y_rep] * 2),
        mean_max_norm=0.5,
    )
    test_inv(gmm)

    x_rep.name, y_rep.name = "rep_X", "rep_Y"
    seed_everything(np.random.randint(0, 1000))
    n_kernels_list = [1, 2, 5]
    n_dim_list = [1, 3]
    var_pmd, mean_pmd, max_pmd, min_pmd = [], [], [], []
    var_pmi, mean_pmi, max_pmi, min_pmi = [], [], [], []
    mi = []
    for n_kernels in n_kernels_list:
        print(f"n_kernels={n_kernels}")
        for n_dim in n_dim_list:
            print(f"n_dim={n_dim}")
            rep_X = directsum([x_rep] * n_dim)
            rep_Y = directsum([y_rep] * n_dim)
            gmm = SymmGaussianMixture(
                n_kernels,
                rep_X=directsum([x_rep] * n_dim),
                rep_Y=directsum([y_rep] * n_dim),
                mean_max_norm=0.5,
                sampling_seed=np.random.randint(0, 1000),
            )
            X, Y = gmm.simulate(n_samples=min(10000 * n_dim, 100000))
            pmd = gmm.pointwise_mutual_dependency(X, Y)
            pmi = np.log(pmd)
            pxy = gmm.joint_pdf(X, Y)
            mi_run = (pmi * pxy).sum() / pxy.sum()
            mi.append(mi_run)
            var_pmd.append(np.var(pmd))
            mean_pmd.append(np.mean(pmd))
            max_pmd.append(np.max(pmd))
            min_pmd.append(np.min(pmd))
            var_pmi.append(np.var(pmi))
            mean_pmi.append(np.mean(pmi))
            max_pmi.append(np.max(pmi))
            min_pmi.append(np.min(pmi))

    fig, axs = plt.subplots(1, len(n_dim_list), figsize=(20, 5), sharey=True)
    # fig.suptitle(f'mean')
    # Determine the common limits for the MI axes
    mi_min = min(mi)
    mi_max = max(mi)

    for j, n_dim in enumerate(n_dim_list):
        mean_values = [mean_pmd[i * len(n_dim_list) + j] for i in range(len(n_kernels_list))]
        max_values = [max_pmd[i * len(n_dim_list) + j] for i in range(len(n_kernels_list))]
        std_values = [3 * np.sqrt(var_pmd[i * len(n_dim_list) + j]) for i in range(len(n_kernels_list))]
        mi_values = [mi[i * len(n_dim_list) + j] for i in range(len(n_kernels_list))]
        mean_pmi_values = [mean_pmi[i * len(n_dim_list) + j] for i in range(len(n_kernels_list))]

        ax = axs[j]
        ax.plot(n_kernels_list, mean_values, label="Mean PMD", marker="o")
        ax.fill_between(
            n_kernels_list, np.asarray(mean_values) - std_values, np.asarray(mean_values) + std_values, alpha=0.2
        )
        ax.plot(n_kernels_list, max_values, label="Max PMD", marker="o")
        ax.set_title(f"n_dim={n_dim * G.order()}")
        ax.set_xlabel("n_kernels")
        ax.set_yscale("log")
        ax.legend(loc="upper left")

        ax2 = ax.twinx()
        ax2.plot(n_kernels_list, mi_values, label="Mutual Information", color="red", marker="x")
        ax2.plot(n_kernels_list, mean_pmi_values, label="Mean PMI", color="green", marker="s")
        ax2.set_ylabel("Mutual Information / Mean PMI")
        ax2.set_ylim([mi_min, mi_max])  # Set the same limits for all MI axes
        ax2.legend(loc="upper right")

    axs[0].set_ylabel("PMD Value")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
