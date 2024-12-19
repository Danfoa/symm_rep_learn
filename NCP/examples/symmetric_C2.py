# Created by danfoa at 18/12/24

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from torch.backends.cudnn import deterministic


def marginal_x(means, covariances, weights):
    """Creates a Gaussian mixture distribution in R. With reflectional symmetry.
    That is p(x) = p(-x).

    Args:
        means (list of float): Means of the Gaussian components.
        covariances (list of float): Variances of the Gaussian components.
        weights (list of float): Weights of the Gaussian components.

    Returns:
        function: A function that computes the PDF and samples from the mixture.
    """
    weights /= np.sum(weights)
    # TODO: symmetrize.
    components = [norm(loc=mean, scale=np.sqrt(cov)) for mean, cov in zip(means, covariances)]

    def pdf(x):  # Marginal PDF
        return sum(weight * component.pdf(x) for weight, component in zip(weights, components))

    def sample(size):
        component_choices = np.random.choice(len(weights), size=size, p=weights)
        samples = [components[i].rvs() for i in component_choices]
        return np.asarray(samples)

    assert np.isclose(pdf(-1), pdf(1)), f"p(1)={pdf(-1)}!=p(-1)={pdf(1)}"

    return pdf, sample

def marginal_y(std_transport=0.1):
    """Creates a conditional distribution p(y|x) = p(x) / |dy/dx|.
    Where y = 0.5 * x^3. And if p(x) posses C2 symmetry, then p(y) will also possess C2 symmetry.

    Args:
        pdf_x (function): The marginal PDF p(x).

    Returns:
        pdf_y, y_sampler: The conditional PDF p(y|x) and a sampler for y.
    """

    def transport_fn(x, deterministic=False):
        # Add white noise to the transport function (avoiding to break the symmetry). For every scalar/vector x
        std = std_transport if not deterministic else 0
        epsilon = np.random.normal(0, std, size=x.shape if hasattr(x, '__len__') else None)
        return  0.5 * x**3 + epsilon

    def cond_pdf(x):
        """Return the conditional pdf p(y|x) as a callable accepting x"""
        return norm(loc=0.5 * x**3, scale=std_transport)

    return cond_pdf, transport_fn

if __name__ == "__main__":
    # Example setup: Gaussian mixture with reflectional symmetry
    means = [-1.2, 0, 1.2]
    covariances = [0.08, 0.1, 0.08]
    weights = [0.4, 0.2, 0.4]


    x_pdf, x_sampler = marginal_x(means, covariances, weights)
    x_samples = x_sampler(5000)
    cpd, transport_fn = marginal_y(std_transport=0.25)
    y_samples = transport_fn(x_samples)

    # Get conditional density at specific symmetric points
    x_test = 1.2  # x
    gx_test = -x_test  # g ▻ x
    y, gy_test = transport_fn(x_test, deterministic=True), transport_fn(gx_test, deterministic=True)
    cpd_x = cpd(x_test)  # p(y| x)
    cpd_gx = cpd(gx_test)  # p(y| g ▻ x)



    # Create a figure with joint 2D and marginal plots
    # fig = plt.figure(figsize=(10, 8))
    grid = sns.JointGrid()
    # Joint KDE plot
    sns.kdeplot(x=x_samples, y=y_samples, fill=True, ax=grid.ax_joint, cmap="Blues")
    # Marginal of p(x)
    x_vals = np.linspace(min(x_samples), max(y_samples), 500)
    x_pdf_vals = x_pdf(x_vals)
    # Marginal of p(y) -- KDE for Y with symmetry adjustment
    sns.kdeplot(x=x_samples, ax=grid.ax_marg_x, fill=True)
    sns.kdeplot(y=y_samples, ax=grid.ax_marg_y, fill=True)

    grid.ax_joint.set_xticks([x_test, gx_test])
    grid.ax_joint.set_xticklabels([r"$x$", r"$g \;\triangleright_{\mathcal{X}}\; x$"])
    # Set y ticks only to the test points with labels
    grid.ax_joint.set_yticks([y, gy_test])
    grid.ax_joint.set_yticklabels([r"$y$", r"$g \;\triangleright_{\mathcal{Y}}\; y$"])
    # # Draw slightly transparent vertical lines at x_test and gx_test
    grid.ax_joint.plot([x_test, gx_test], [y, gy_test], 'ro')
    grid.ax_joint.axvline(x_test, color='r', alpha=0.3)
    grid.ax_joint.axvline(gx_test, color='r', alpha=0.3)
    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")
    plt.show()

    # Plot the conditional probability densities of the two points x_test and gx_test. In two rows
    y_max = np.max(np.abs(y_samples))
    y_range = (-y_max, y_max)
    y_vals = np.linspace(y_range[0], y_range[1], 500)
    y_probs = cpd_x.pdf(y_vals)
    y_probs_gx = cpd_gx.pdf(y_vals)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(y_vals, y_probs, label=r"$p(y|x)$")
    axs[0].fill_between(y_vals, y_probs, alpha=0.3)
    # Plot the test points
    axs[0].plot(y, cpd_x.pdf(y), 'ro')
    axs[0].set_title(r"$p(y|x)$")
    axs[1].plot(y_vals, y_probs_gx, label=r"$p(y| g \triangleright x)$")
    axs[1].fill_between(y_vals, y_probs_gx, alpha=0.3)
    axs[1].plot(gy_test, cpd_gx.pdf(gy_test), 'ro')
    axs[1].set_title(r"$p(y| g \triangleright x)$")

    plt.xlabel(r"$\mathcal{Y}$")
    plt.show()
