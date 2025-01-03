import escnn
import numpy as np
import scipy.stats as stats
from escnn.group import Representation, directsum

from NCP.cde_fork.density_simulation import GaussianMixture
from NCP.cde_fork.utils.misc import project_to_pos_semi_def


class SymmGaussianMixture(GaussianMixture):
    """TODO: Add docstring

    Args:
      n_kernels: number of mixture components
      ndim_x: dimensionality of X / number of random variables in X
      ndim_y: dimensionality of Y / number of random variables in Y
      means_std: std. dev. when sampling the kernel means
      random_seed: seed for the random_number generator
    """

    def __init__(
        self, n_kernels: int, rep_X: Representation, rep_Y: Representation, means_std=1.5, random_seed=None
    ):

        self.random_state = np.random.RandomState(seed=random_seed)  # random state for sampling data
        self.random_state_params = np.random.RandomState(seed=20)  # fixed random state for sampling GMM params
        self.random_seed = random_seed

        self.has_pdf = True
        self.has_cdf = True
        self.can_sample = True

        self.G = rep_X.group
        """  set parameters, calculate weights, means and covariances """
        self.n_kernels = n_kernels
        self.ndim = rep_X.size + rep_Y.size
        self.rep = directsum([rep_X, rep_Y])
        self.rep_X, self.rep_Y = rep_X, rep_Y
        self.rep.name = "rep_Ω"
        self.ndim_x = rep_X.size
        self.ndim_y = rep_Y.size
        self.means_std = means_std
        self.weights = self._sample_weights(n_kernels)  # shape(n_kernels,), sums to one
        self.means = self.random_state_params.normal(
            loc=np.zeros([self.ndim]), scale=self.means_std, size=[n_kernels, self.ndim]
        )  # shape(n_kernels, n_dims)
        print(self.means)
        """ Sample cov matrices and assure that cov matrix is pos definite"""
        self.covariances_x = self.sample_covariances(dim=self.ndim_x, scale=0.5, means_std=self.means_std, num=n_kernels)
        self.covariances_y = self.sample_covariances(dim=self.ndim_y, scale=0.5, means_std=self.means_std, num=n_kernels)

        if not self.G.continuous:
            # To make these distributions invariant under the group action, we need to average over the group
            self.n_kernels = n_kernels * self.G.order()
            means = [self.means]
            Cxs, Cys = [self.covariances_x], [self.covariances_y]
            for g in self.G.elements:
                if g == self.G.identity: continue
                means.append(np.einsum('ij,...j->...i', self.rep(g), self.means))
                Cxs.append(np.einsum('ij,kjm,mn->kin', self.rep_X(g), self.covariances_x, self.rep_X(~g)))
                Cys.append(np.einsum('ij,kjm,mn->kin', self.rep_Y(g), self.covariances_y, self.rep_Y(~g)))
            self.means = np.concatenate(means, axis=0)
            self.covariances_x = np.concatenate(Cxs, axis=0)
            self.covariances_y = np.concatenate(Cys, axis=0)
            self.weights = np.concatenate([self.weights] * self.G.order())
            self.weights /= np.sum(self.weights)
        else:
            raise NotImplementedError("Only finite groups are supported at the moment")
            # This requires us to use uniform on angle and gaussian on radius. Can be done, need to work out the details

        # some eigenvalues of the sampled covariance matrices can be exactly zero -> map to pos semi-definite subspace
        self.covariances = np.zeros(shape=(self.n_kernels, self.ndim, self.ndim))
        self.covariances[:, :self.ndim_x, :self.ndim_x] = self.covariances_x
        self.covariances[:, self.ndim_x:, self.ndim_x:] = self.covariances_y

        """ after mapping, define the remaining variables and collect frozen multivariate variables
        (x,y), x and y for later conditional draws """
        self.means_x = self.means[:, :self.ndim_x]
        self.means_y = self.means[:, self.ndim_x:]

        self.gaussians, self.gaussians_x, self.gaussians_y = [], [], []
        for i in range(self.n_kernels):
            self.gaussians.append(stats.multivariate_normal(mean=self.means[i,], cov=self.covariances[i]))
            self.gaussians_x.append(stats.multivariate_normal(mean=self.means_x[i,], cov=self.covariances_x[i]))
            self.gaussians_y.append(stats.multivariate_normal(mean=self.means_y[i,], cov=self.covariances_y[i]))

        # approximate data statistics
        self.y_mean, self.y_std = self._compute_data_statistics()

    def sample_covariances(self, dim, scale, means_std, num=1):
        """ Sample covariance matrices for the GMM
        Args:
          dim: dimensionality of the covariance matrices
          scale: std of the eigenvalues of the covariance matrices
          means_std: std of the means of the centers of the kernes
          num: number of covariance matrices to sample
        Returns:
          ndarray of covariance matrices with shape (number, dim, dim)
        """

        # Sample diagonal entries / eigenvalues of the covariance matrix
        eigenvalues = np.abs(self.random_state_params.normal(loc=1, scale=scale, size=(num, dim)))
        # Ensure the minimum eigenvalue is 10% of the means_std
        min_eig = 0.15 * means_std
        # This controls that no kernel appears to be supported on a single line/point. / stochasticity
        eigenvalues = np.maximum(eigenvalues, min_eig)

        # eigenvalues /= eigenvalues.max()
        # Sample num orthogonal matrices usign QR decomposition
        Q, _ = np.linalg.qr(self.random_state_params.normal(size=(num, dim, dim)))
        var = np.asarray([np.diag(e) for e in eigenvalues])
        cov = np.einsum('...ij,...jk,...lk->...il', Q, var, Q)
        return project_to_pos_semi_def(cov)

    def _sample_weights(self, n_weights):
        """Samples density weights -> sum up to one
        Args:
          n_weights: number of weights
        Returns:
          ndarray of weights with shape (n_weights,)
        """
        weights = self.random_state_params.random(n_weights)
        weights /= np.sum(weights)
        return weights

    def _compute_data_statistics(self):
        """ Return mean and std of the y component of the data """
        from NCP.mysc.symm_algebra import symmetric_moments

        X, Y = self.simulate(n_samples=10**4)
        Y_mean, Y_var = symmetric_moments(Y, self.rep_Y)
        return Y_mean.detach().numpy(), np.sqrt(Y_var.detach().numpy())

    def pdf_y(self, Y):
        """ Marginal probability density function P(Y)
        Args:
            Y: the variable Y for the distribution P(Y), array_like, shape:(n_samples, ndim_y)
        Returns:
            the marginal distribution of Y with shape:(n_samples,)
        """
        Y = self._handle_input_dimensionality(Y)
        p_y = np.sum([self.weights[i] * self.gaussians_y[i].pdf(Y) for i in range(self.n_kernels)], axis=0)
        return p_y

    def pdf_x(self, X):
        """ Marginal probability density function P(X)
        Args:
            X: the variable X for the distribution P(X), array_like, shape:(n_samples, ndim_x)
        Returns:
            the marginal distribution of X with shape:(n_samples,)
        """
        X = self._handle_input_dimensionality(X)
        p_x = np.sum([self.weights[i] * self.gaussians_x[i].pdf(X) for i in range(self.n_kernels)], axis=0)
        return p_x

    def mutual_information(self, X, Y):
        """ Compute the mutual information between X and Y

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
        return p_xy / (p_x * p_y)

if __name__ == "__main__":

    G = escnn.group.DihedralGroup(5)

    x_rep = G.representations['irrep_1,2']                       # ρ_Χ
    y_rep = G.representations['irrep_1,0']
    x_rep.name, y_rep.name = "rep_X", "rep_Y"

    n_kernels = 3
    gmm = SymmGaussianMixture(n_kernels, x_rep, y_rep, means_std=5, random_seed=np.random.randint(0, 1000))

    X, Y = gmm.simulate(n_samples=10 ** 4)

    # Plot the distribution of the data X = [x,y] and Y = [z] in a plotly 3D scatter plot
    import plotly.express as px

    # Plot the centers of the kernels
    fig = px.scatter_3d(
        x=gmm.means_x[:, 0],
        y=gmm.means_x[:, 1],
        z=gmm.means_y[:, 0],
        labels={'x': 'x', 'y': 'y', 'z': 'z'},
        title="3D Scatter Plot"
        )
    fig.update_traces(
        marker=dict(color='red', size=4, opacity=1),  # Centers: red, solid
        selector=dict(mode='markers')  # Apply only to markers
        )

    # Add the samples
    fig.add_scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=Y[:, 0],
        mode='markers',
        marker=dict(
            color=np.abs(Y[:, 0]),  # Color by the absolute value of the z coordinate
            colorscale='Viridis',
            size=3,
            opacity=0.15
        )
    )

    fig.show()
    # Plot marginal distributions on the XY plane and the ZY plane
    import plotly.graph_objects as go

    fig = go.Figure(go.Histogram2dContour(
        x=X[:, 0],
        y=X[:, 1],
        colorscale='Viridis',
        xaxis='x',
        yaxis='y'
        ))
    fig.update_layout(
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),
        title='KDE Marginal distribution of X = [x,y]'
        )
    fig.show()





