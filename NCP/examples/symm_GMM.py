# Created by danfoa at 18/12/24
import pathlib
from math import ceil

import escnn
import hydra
import lightning
import numpy as np
import seaborn as sns
import torch
from escnn.group import directsum, Group, Representation
from escnn.nn import FieldType, GeometricTensor
from hydra.core.hydra_config import HydraConfig
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.io as pio

from omegaconf import DictConfig, OmegaConf
from sympy.combinatorics import CyclicGroup
from torch.utils.data import DataLoader, default_collate, TensorDataset

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture
from NCP.models.ncp_lightning_module import TrainingModule

import logging

log = logging.getLogger(__name__)


def plot_analytic_joint_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                           y_samples):
    grid = sns.JointGrid(marginal_ticks=True, space=0.05)
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 200)
    y_range = np.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    # Flatten the grid to evaluate joint_pdf
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # p(x,y) -- Joint PDF
    # Compute the joint PDF for each point on the grid
    Z_flat = gmm.joint_pdf(X=X_input, Y=Y_input)
    Z = Z_flat.reshape(X_grid.shape)
    joint_contour = grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap="Blues", levels=15)

    # Select a random sample to test the conditional expectation
    x_t, y_t = x_samples[0], y_samples[0]
    g = G.elements[-1]
    gx_t, gy_t = (rep_X(g) @ [x_t]).squeeze(), (rep_Y(g) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=0.7)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.7)
    grid.ax_joint.axvline(gx_t, color='g', alpha=0.7)
    grid.ax_joint.axhline(gy_t, color='g', alpha=0.7)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=5, alpha=0.7)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=5, alpha=0.7)
    # Set limits
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    # Do plot of conditional probability density on the test conditions x and gx
    n_samples_cpd = len(y_range)
    cpd_y_vals = gmm.pdf(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    cpd_gy_vals = gmm.pdf(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    # Plot filled distributions with lines
    style = {
        "cpd_x":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$p(y | x)$",
            },
        "cpd_gx": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$p(y | g \;\triangleright_{\mathcal{X}}\; x)$",
            },
        "pdf":    {
            "color":  "blue",
            "fill":   "lightblue",
            "legend": r"$p(y)$",
            }
        }

    for key, cpd_vals in zip(["cpd_x", "cpd_gx"], [cpd_y_vals, cpd_gy_vals]):
        grid.ax_marg_y.plot(cpd_vals, y_range, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=0.6,
                            label=style[key]["legend"])

    # Plot marginal x
    pdf_x = gmm.pdf_x(X=x_range)
    grid.ax_marg_x.fill_between(x_range, 0, pdf_x, color=style["pdf"]["fill"], alpha=0.4)
    grid.ax_marg_x.plot(x_range, pdf_x, color=style["pdf"]["color"], linestyle="-", alpha=1.0, linewidth=0.6,
                        label=style["pdf"]["legend"])
    grid.ax_marg_x.set_ylim([0, None])
    # Plot marginal y
    pdf_y = gmm.pdf_y(Y=y_range)
    grid.ax_marg_y.fill_betweenx(y_range, 0, pdf_y, color=style["pdf"]["fill"], alpha=0.4)
    grid.ax_marg_y.plot(pdf_y, y_range, color=style["pdf"]["color"], linestyle="-", alpha=1.0, linewidth=0.6,
                        label=style["pdf"]["legend"])
    grid.ax_marg_y.set_xlim([0, None])

    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=8)
    grid.fig.suptitle(r"Analytic $p(x,y)$, and $p(y | x)$")
    grid.fig.tight_layout()
    return grid


def plot_analytic_prod_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                          y_samples):
    grid = sns.JointGrid(marginal_ticks=True, space=0.05)
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 200)
    y_range = np.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    # Flatten the grid to evaluate joint_pdf
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # p(x,y) -- Joint PDF
    # Compute the joint PDF for each point on the grid
    Z_flat = gmm.pdf_x(X=X_input) * gmm.pdf_y(Y=Y_input)
    Z = Z_flat.reshape(X_grid.shape)
    joint_contour = grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap="Blues", levels=15)

    # Select a random sample to test the conditional expectation
    x_t, y_t = x_samples[0], y_samples[0]
    g = G.elements[-1]
    gx_t, gy_t = (rep_X(g) @ [x_t]).squeeze(), (rep_Y(g) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=0.7)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.7)
    grid.ax_joint.axvline(gx_t, color='g', alpha=0.7)
    grid.ax_joint.axhline(gy_t, color='g', alpha=0.7)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=5, alpha=0.7)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=5, alpha=0.7)
    # Set limits
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    # Do plot of conditional probability density on the test conditions x and gx
    n_samples_cpd = len(y_range)
    cpd_y_vals = gmm.pdf(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    cpd_gy_vals = gmm.pdf(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    # Plot filled distributions with lines
    style = {
        "cpd_x":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$p(y | x)$",
            },
        "cpd_gx": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$p(y | g \;\triangleright_{\mathcal{X}}\; x)$",
            },
        "pdf":    {
            "color":  "blue",
            "fill":   "lightblue",
            "legend": r"$p(y)$",
            }
        }

    # for key, cpd_vals in zip(["cpd_x", "cpd_gx"], [cpd_y_vals, cpd_gy_vals]):
    #     grid.ax_marg_y.plot(cpd_vals, y_range, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=0.6,
    #                         label=style[key]["legend"])

    # Plot marginal x
    pdf_x = gmm.pdf_x(X=x_range)
    grid.ax_marg_x.fill_between(x_range, 0, pdf_x, color=style["pdf"]["fill"], alpha=0.4)
    grid.ax_marg_x.plot(x_range, pdf_x, color=style["pdf"]["color"], linestyle="-", alpha=1.0, linewidth=0.6,
                        label=style["pdf"]["legend"])
    grid.ax_marg_x.set_ylim([0, None])
    # Plot marginal y
    pdf_y = gmm.pdf_y(Y=y_range)
    grid.ax_marg_y.fill_betweenx(y_range, 0, pdf_y, color=style["pdf"]["fill"], alpha=0.4)
    grid.ax_marg_y.plot(pdf_y, y_range, color=style["pdf"]["color"], linestyle="-", alpha=1.0, linewidth=0.6,
                        label=style["pdf"]["legend"])
    grid.ax_marg_y.set_xlim([0, None])

    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=8)
    grid.fig.suptitle(r"Analytic $p(x)p(y)$, and $p(y | x)$")
    grid.fig.tight_layout()
    return grid


def plot_analytic_npmi_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                          y_samples):
    grid = sns.JointGrid(marginal_ticks=True, space=0.05)
    # Define grid for the plot
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 200)
    y_range = np.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    # Flatten the grid to evaluate joint_pdf
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # Compute the mutual information
    Pxy = gmm.joint_pdf(X=X_input, Y=Y_input)
    npmi_flat = gmm.normalized_pointwise_mutual_information(X=X_input, Y=Y_input)
    pmi = npmi_flat * -np.log(Pxy)
    MI = np.sum(Pxy * pmi)
    Z = npmi_flat.reshape(X_grid.shape)
    contour = grid.ax_joint.contourf(
        X_grid,
        Y_grid,
        Z,
        cmap=sns.color_palette("magma", as_cmap=True),
        levels=35,
        # vmin=-1,
        # vmax=1
        )
    cbar = plt.colorbar(contour, ax=grid.ax_joint, orientation='vertical')

    # Select a random sample to test the conditional expectation
    x_t, y_t = x_samples[0], y_samples[0]
    g = G.elements[-1]
    gx_t, gy_t = (rep_X(g) @ [x_t]).squeeze(), (rep_Y(g) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=0.7)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.7)
    grid.ax_joint.axvline(gx_t, color='g', alpha=0.7)
    grid.ax_joint.axhline(gy_t, color='g', alpha=0.7)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=5, alpha=0.5)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=5, alpha=0.5)
    # Set limits
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    # Do plot of conditional probability density on the test conditions x and gx
    n_samples_cpd = len(y_range)
    mi_x_vals = gmm.normalized_pointwise_mutual_information(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    mi_gx_vals = gmm.normalized_pointwise_mutual_information(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    mi_y_vals = gmm.normalized_pointwise_mutual_information(X=x_range, Y=np.repeat(y_t, len(x_range)))
    mi_gy_vals = gmm.normalized_pointwise_mutual_information(X=x_range, Y=np.repeat(gy_t, len(x_range)))

    # Plot filled distributions with lines
    style = {
        "mi_x":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$NPMI(y,x)$",
            },
        "mi_gx": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$NPMI(y, g \;\triangleright_{\mathcal{X}}\; x)$",
            },
        "mi_y":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$NPMI(y,x)$",
            },
        "mi_gy": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$NPMI(g \;\triangleright_{\mathcal{Y}}\; y, x)$",
            },
        "pdf":   {
            "color":  "blue",
            "fill":   "lightblue",
            "legend": r"$p(y)$",
            }
        }

    for key, mi_vals in zip(["mi_x", "mi_gx"], [mi_x_vals, mi_gx_vals]):
        grid.ax_marg_y.plot(mi_vals, y_range, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=1,
                            label=style[key]["legend"])

    for key, mi_vals in zip(["mi_y", "mi_gy"], [mi_y_vals, mi_gy_vals]):
        grid.ax_marg_x.plot(x_range, mi_vals, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=1,
                            label=style[key]["legend"])

    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=8)
    grid.ax_marg_x.legend(loc="upper left", fontsize=8)
    grid.ax_marg_y.axvline(0, color='gray', alpha=0.5)
    grid.ax_marg_x.axhline(0, color='gray', alpha=0.5)
    # grid.ax_marg_y.axvline(-1, color='gray', alpha=0.5)
    # grid.ax_marg_x.axhline(-1, color='gray', alpha=0.5)
    # grid.ax_marg_y.axvline(1, color='gray', alpha=0.5)
    # grid.ax_marg_x.axhline(1, color='gray', alpha=0.5)
    grid.fig.suptitle(r"Analytic Normalized MI $ln\left(\frac{p(x,y)}{p(y)p(x)}\right)$" + f" - MI={MI:.3f}")
    grid.fig.tight_layout()
    return grid

def plot_analytic_pmd_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                          y_samples):
    grid = sns.JointGrid(marginal_ticks=True, space=0.05)
    # Define grid for the plot
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 200)
    y_range = np.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    # Flatten the grid to evaluate joint_pdf
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # Compute the mutual information
    Pxy = gmm.joint_pdf(X=X_input, Y=Y_input)
    pmd = gmm.pointwise_mutual_dependency(X=X_input, Y=Y_input)
    Z = pmd.reshape(X_grid.shape)
    contour = grid.ax_joint.contourf(
        X_grid,
        Y_grid,
        Z,
        cmap=sns.color_palette("magma", as_cmap=True),
        levels=35,
        # vmin=-1,
        # vmax=1
        )
    cbar = plt.colorbar(contour, ax=grid.ax_joint, orientation='vertical')

    # Select a random sample to test the conditional expectation
    x_t, y_t = x_samples[0], y_samples[0]
    g = G.elements[-1]
    gx_t, gy_t = (rep_X(g) @ [x_t]).squeeze(), (rep_Y(g) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=0.7)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.7)
    grid.ax_joint.axvline(gx_t, color='g', alpha=0.7)
    grid.ax_joint.axhline(gy_t, color='g', alpha=0.7)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=5, alpha=0.5)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=5, alpha=0.5)
    # Set limits
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    # Do plot of conditional probability density on the test conditions x and gx
    n_samples_cpd = len(y_range)
    mi_x_vals =  gmm.pointwise_mutual_dependency(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    mi_gx_vals = gmm.pointwise_mutual_dependency(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    mi_y_vals =  gmm.pointwise_mutual_dependency(X=x_range, Y=np.repeat(y_t, len(x_range)))
    mi_gy_vals = gmm.pointwise_mutual_dependency(X=x_range, Y=np.repeat(gy_t, len(x_range)))

    # Plot filled distributions with lines
    style = {
        "mi_x":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$PMD(y,x)$",
            },
        "mi_gx": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$PMD(y, g \;\triangleright_{\mathcal{X}}\; x)$",
            },
        "mi_y":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$PMD(y,x)$",
            },
        "mi_gy": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$PMD(g \;\triangleright_{\mathcal{Y}}\; y, x)$",
            },
        "pdf":   {
            "color":  "blue",
            "fill":   "lightblue",
            "legend": r"$p(y)$",
            }
        }

    for key, mi_vals in zip(["mi_x", "mi_gx"], [mi_x_vals, mi_gx_vals]):
        grid.ax_marg_y.plot(mi_vals, y_range, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=1,
                            label=style[key]["legend"])

    for key, mi_vals in zip(["mi_y", "mi_gy"], [mi_y_vals, mi_gy_vals]):
        grid.ax_marg_x.plot(x_range, mi_vals, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=1,
                            label=style[key]["legend"])

    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=8)
    grid.ax_marg_x.legend(loc="upper left", fontsize=8)
    # grid.ax_marg_y.axvline(0, color='gray', alpha=0.5)
    # grid.ax_marg_x.axhline(0, color='gray', alpha=0.5)
    grid.ax_marg_y.axvline(1, color='gray', alpha=0.5)
    grid.ax_marg_x.axhline(1, color='gray', alpha=0.5)
    grid.fig.suptitle(r"Analytic Normalized PMD $\frac{p(x,y)}{p(y)p(x)}$")
    grid.fig.tight_layout()
    return grid


def get_model(cfg: DictConfig, x_type, y_type, lat_type) -> torch.nn.Module:
    embedding_dim = lat_type.size
    if cfg.model.lower() == "encp":  # Equivariant NCP
        from NCP.models.equiv_ncp import ENCP
        from NCP.nn.equiv_layers import EMLP

        kwargs = dict(out_type=lat_type,
                      hidden_layers=cfg.embedding.hidden_layers,
                      activation=cfg.embedding.activation,
                      hidden_units=cfg.embedding.hidden_units,
                      bias=False)
        χ_embedding = EMLP(in_type=x_type, **kwargs)
        y_embedding = EMLP(in_type=y_type, **kwargs)
        eNCPop = ENCP(embedding_x=χ_embedding,
                      embedding_y=y_embedding,
                      gamma_orthogonality=cfg.gamma,
                      truncated_op_bias=cfg.truncated_op_bias,
                      )

        return eNCPop
    elif cfg.model.lower() == "ncp":  # NCP
        from NCP.mysc.utils import class_from_name
        from NCP.models.ncp import NCP
        from NCP.nn.layers import MLP

        activation = class_from_name('torch.nn', cfg.embedding.activation)
        kwargs = dict(output_shape=embedding_dim,
                      n_hidden=cfg.embedding.hidden_layers,
                      layer_size=cfg.embedding.hidden_units,
                      activation=activation,
                      bias=False)
        fx = MLP(input_shape=x_type.size, **kwargs)
        fy = MLP(input_shape=y_type.size, **kwargs)
        ncp = NCP(embedding_x=fx,
                  embedding_y=fy,
                  embedding_dim=embedding_dim,
                  gamma_orthogonality=cfg.gamma,
                  truncated_op_bias=cfg.truncated_op_bias,
                  )
        return ncp
    elif cfg.model.lower() == "drf":  # Density Ratio Fitting
        from NCP.models.density_ratio_fitting import DRF
        from NCP.nn.layers import MLP
        from NCP.mysc.utils import class_from_name

        activation = class_from_name('torch.nn', cfg.embedding.activation)
        n_layers = cfg.embedding.hidden_layers
        # NCP_params = 2(nh^2 * nl)   ->  DRF_params = (sqrt(2)nh)^2 * nl = 2(nh^2 * nl)
        n_hidden_units = int(cfg.embedding.hidden_units * 2)
        embedding = MLP(input_shape=x_type.size + y_type.size,  # z = (x,y)
                        output_shape=1,
                        n_hidden=n_layers,
                        # Ensure NCP model and DRF have approximately equal number of parameters.
                        layer_size=[n_hidden_units] * (n_layers - 1) + [cfg.embedding.embedding_dim],
                        activation=activation,
                        bias=False)
        drf = DRF(embedding=embedding, gamma=cfg.gamma)

        return drf
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


def gmm_dataset(n_samples: int, gmm: SymmGaussianMixture, rep_X: Representation, rep_Y: Representation, device='cpu'):
    from NCP.mysc.symm_algebra import symmetric_moments

    x_samples, y_samples = gmm.simulate(n_samples=n_samples)
    x_mean, x_var = symmetric_moments(x_samples, rep_X)
    y_mean, y_var = symmetric_moments(y_samples, rep_Y)

    # Train, val, test splitting
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    n_samples = len(x_samples)
    n_train, n_val, n_test = np.asarray(np.array([train_ratio, val_ratio, test_ratio]) * n_samples, dtype=int)
    train_samples = x_samples[:n_train], y_samples[:n_train]
    val_samples = x_samples[n_train:n_train + n_val], y_samples[n_train:n_train + n_val]
    test_samples = x_samples[n_train + n_val:], y_samples[n_train + n_val:]

    X_c = (x_samples - x_mean.numpy()) / np.sqrt(x_var.numpy())
    Y_c = (y_samples - y_mean.numpy()) / np.sqrt(y_var.numpy())
    x_train, x_val, x_test = X_c[:n_train], X_c[n_train:n_train + n_val], X_c[n_train + n_val:]
    y_train, y_val, y_test = Y_c[:n_train], Y_c[n_train:n_train + n_val], Y_c[n_train + n_val:]

    X_train = torch.atleast_2d(torch.from_numpy(x_train).float()).to(device=device)
    Y_train = torch.atleast_2d(torch.from_numpy(y_train).float()).to(device=device)
    X_val = torch.atleast_2d(torch.from_numpy(x_val).float())
    Y_val = torch.atleast_2d(torch.from_numpy(y_val).float())
    X_test = torch.atleast_2d(torch.from_numpy(x_test).float())
    Y_test = torch.atleast_2d(torch.from_numpy(y_test).float())

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    return (train_samples, val_samples, test_samples), (x_mean, y_mean), (x_var, y_var), (
        train_dataset, val_dataset, test_dataset)


def plot_npmi_error_distribution(NPMI_gt, NPMI):
    from matplotlib.colors import Normalize
    # Create a JointGrid for the 2D KDE plot with marginals
    # Set PLT backed to AGG
    plt.switch_backend('agg')
    grid = sns.JointGrid(x=NPMI,
                         y=NPMI_gt,
                         marginal_ticks=True,
                         space=0.05,  # Space between the joint and marginal axes
                         # height=8,  # Size of the figure in Inches
                         # ratio=50  # Ratio of joint to marginal axes
                         )
    min_val, max_val = np.min(NPMI_gt), np.max(NPMI_gt)
    limit = max(abs(min_val), abs(max_val)) * 1.1
    contour = grid.plot_joint(sns.kdeplot,
                              fill=True,
                              cmap=sns.cubehelix_palette(as_cmap=True),
                              levels=15,
                              bw_adjust=0.6,  # Adjust bandwidth
                              # cut=0,  # Reduce cut to avoid distortions at the edges
                              clip=[(-limit, limit), (-limit, limit)],  # Clipping range for the KDE
                              cbar=True,
                              )
    # Plot scatter plot of the data with very mild alpha
    # grid.plot_joint(sns.scatterplot, alpha=0.1)

    grid.ax_joint.set_xlim([-limit, limit])
    grid.ax_joint.set_ylim([-limit, limit])
    # Plot a straight line with slope 1 crossing y at 0
    # Set axis limits and labels
    grid.ax_joint.set_xlabel('NPMI')
    grid.ax_joint.set_ylabel('NPMI_gt')
    grid.ax_joint.set_aspect('equal', adjustable='box')  # Ensure aspect ratio is the same for both axes
    grid.ax_joint.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    grid.fig.suptitle('2D Distribution of NPMI Error', y=1.02)
    grid.ax_joint.plot([-limit, limit], [-limit, limit], color='darkred', linestyle='-', linewidth=2)
    grid.plot_marginals(sns.kdeplot,
                        fill=True,
                        color='firebrick',
                        alpha=0.6,
                        bw_adjust=0.4,  # Adjust bandwidth
                        )
    # Set the limits for the marginal axes to match the joint axes
    grid.ax_marg_x.set_xlim(grid.ax_joint.get_xlim())
    grid.ax_marg_y.set_ylim(grid.ax_joint.get_ylim())
    return grid.fig


@torch.no_grad()
def measure_analytic_pmi_error(gmm, nn_model, x_samples, y_samples, x_type, y_type, x_mean, y_mean, x_var, y_var,
                               plot=False, samples=1000):
    G = x_type.fibergroup
    n_total_samples = samples if samples != 'all' else min(len(x_samples), 2048)
    n_samples_per_g = int(n_total_samples // G.order())

    device = next(nn_model.parameters()).device
    dtype = next(nn_model.parameters()).dtype

    chosen_samples_idx = np.random.choice(len(x_samples), n_samples_per_g, replace=False)

    prod_idx = np.arange(n_samples_per_g)
    # Get all pairs of samples
    X_idx, Y_idx = np.meshgrid(prod_idx, prod_idx)
    X_idx_flat = X_idx.flatten()
    Y_idx_flat = Y_idx.flatten()
    X_np = np.atleast_2d(x_samples[chosen_samples_idx][X_idx_flat])
    Y_np = np.atleast_2d(y_samples[chosen_samples_idx][Y_idx_flat])
    # Compute the marginal probabilities.
    P_X = gmm.pdf_x(x_samples)
    P_Y = gmm.pdf_y(y_samples)
    P_X = P_X[X_idx_flat]
    P_Y = P_Y[Y_idx_flat]
    # Compute the joint probability
    P_XY = gmm.joint_pdf(X_np, Y_np)  # Compute the group orbit of the data, to compute the equivariance error
    G_X_np, G_Y_np = [X_np], [Y_np]
    # Perform data augmentation using the group actions
    for g in G.elements[1:]:  # first element is the identity
        G_X_np.append(np.einsum("ij,...j->...i", x_type.representation(g), X_np))
        G_Y_np.append(np.einsum("ij,...j->...i", y_type.representation(g), Y_np))
    X_np = np.vstack(G_X_np)
    Y_np = np.vstack(G_Y_np)
    # Duplicate probabilities
    P_X = np.tile(P_X, G.order())
    P_Y = np.tile(P_Y, G.order())
    P_XY = np.tile(P_XY, G.order())

    # Normalize the data for the NN estimation of the point-wise mutual dependency
    X_c = ((torch.Tensor(X_np) - x_mean) / torch.sqrt(x_var)).to(device=device)
    Y_c = ((torch.Tensor(Y_np) - y_mean) / torch.sqrt(y_var)).to(device=device)
    # Compute the estimate of the NCP model of the mutual information _______________
    from NCP.models.equiv_ncp import ENCP
    _x, _y = (x_type(X_c), y_type(Y_c)) if isinstance(nn_model, ENCP) else (X_c, Y_c)
    pmd_xy = nn_model.pointwise_mutual_dependency(_x, _y).cpu().numpy()  # k_r(x,y) ≈ p(x,y) / p(x)p(y)
    # Compute the analytic PMD _______________________________________
    pmd_xy_gt = gmm.pointwise_mutual_dependency(X=X_np, Y=Y_np)  # p(x,y) / p(x)p(y)
    # Compute the _____ metric error between the two estimates
    PMD_err = ((pmd_xy_gt - pmd_xy) * np.sqrt(P_X * P_Y))
    n = n_samples_per_g ** 2  # Number of unique pairs. Without counting orbit
    G_PMD_err_mat = [PMD_err[i * n: i * n + n].reshape(X_idx.shape) for i in
                     range(G.order())]  # [PMD_mse_mat]_ij = (k_r(xi,yj) - k(xi,yj))^2 * p(xi)p(yj)
    G_PMD_spectral_norm = [np.linalg.norm(G_PMD_err_mat[i], ord=2) for i in range(G.order())]
    PMD_spectral_norm = np.mean(G_PMD_spectral_norm)  # Largest singular value. Spectral norm
    # Compute the error on PMI = ln(PMD)
    PMI_gt = np.log(pmd_xy_gt)
    PMI = np.log(np.clip(pmd_xy, a_min=1e-5, a_max=None))
    # Compute the normalized NPMI = ln(p(x,y)/p(x)p(y)) / -ln(P(X,Y))
    NPMI_gt = PMI_gt / -np.log(P_XY)
    NPMI = PMI / -np.log(P_XY)
    assert np.all((-1 <= NPMI_gt)) and np.all(
        NPMI_gt <= 1), f"NPMI not in [-1, 1] min:{NPMI_gt.min()} max:{NPMI_gt.max()}"
    # Since k(x, y) = k(g.x, g.y) for all g in G, we want to compute the variance of the estimate under g-action
    PMD_xy_g_var = np.var(pmd_xy.reshape(G.order(), -1), axis=0)  # Var[ k_r(g.x, g.y) ] for all g in G
    metrics = {"PMD/mse":           (PMD_err ** 2).sum(),
               "PMD/equiv_err":     PMD_xy_g_var.mean(),
               "PMD/spectral_norm": PMD_spectral_norm,
               }
    # Sample 4 random conditioning values of x
    range_n_samples = 100
    n_cond_points = 4
    cond_idx = np.random.choice(len(X_np), n_cond_points, replace=False)
    X_cond_np = X_np[cond_idx]
    X_c_cond = X_c[cond_idx]

    # X_range_np = np.linspace(x_max, x_max, range_n_samples)
    fig_cde, fig_pmd = None, None
    if y_samples.shape[-1] == 1:
        x_max, y_max = max(np.abs(x_samples)), max(np.abs(y_samples))
        Y_range_np = np.linspace(-y_max, y_max, range_n_samples)
        p_Y_range_np = gmm.pdf_y(Y_range_np)
        P_Y_range = torch.from_numpy(p_Y_range_np).to(device=device, dtype=dtype)
        Y_c_range = ((torch.from_numpy(Y_range_np) - y_mean) / torch.sqrt(y_var)).to(device=device, dtype=dtype)

        CDFs_mse, CDFs_gt, CDFs = [], [], []
        for x_cond, x_c_cond in zip(X_cond_np, X_c_cond):  # Comptue p(y | x) for each conditioning value
            cdf_gt = gmm.pdf(X=np.broadcast_to(x_cond, Y_range_np.shape), Y=Y_range_np)

            if isinstance(nn_model, ENCP):
                _x = x_type(torch.broadcast_to(x_c_cond, Y_c_range.shape)).to(device=device, dtype=dtype)
                _y = y_type(Y_c_range).to(device=device, dtype=dtype)
            else:
                _x, _y = (torch.broadcast_to(x_c_cond, Y_range_np.shape), Y_c_range)

            pmd = nn_model.pointwise_mutual_dependency(_x, _y)  # k_r(x,y) ≈ p(x,y) / p(x)p(y)

            cdf = (pmd * P_Y_range).cpu().numpy()  # k(x,y) * p(y) = p(y | x)
            CDFs_gt.append(cdf_gt)
            CDFs.append(cdf)
            CDFs_mse.append(((cdf - cdf_gt) ** 2).mean())

        metrics["CDF/mse"] = np.mean(CDFs_mse)

        if plot:
            # Plot the 4 CDFs each pair in a separate subplot
            import plotly.graph_objects as go
            fig_cde = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=[r"$p(y/x_{i})$" for i in range(4)])
            colors = {
                "p(y)":              "blue",
                r"$\hat{p}(y | x)$": "red",
                r"$p(y | x)$":       "green"
                }

            for i, (cdf, cdf_gt) in enumerate(zip(CDFs, CDFs_gt)):
                fig_cde.add_trace(
                    go.Scatter(x=Y_range_np.squeeze(), y=p_Y_range_np.squeeze(), mode='lines', name="p(y)",
                               line=dict(dash='dash', color=colors["p(y)"]),
                               legendgroup='p(y)', showlegend=(i == 0)), row=i + 1, col=1)
                fig_cde.add_trace(
                    go.Scatter(x=Y_range_np.squeeze(), y=cdf.squeeze(), mode='lines', name=r"$\hat{p}(y | x)$",
                               line=dict(color=colors[r"$\hat{p}(y | x)$"]),
                               legendgroup=r"$\hat{p}(y | x)$", showlegend=(i == 0)), row=i + 1, col=1)
                fig_cde.add_trace(
                    go.Scatter(x=Y_range_np.squeeze(), y=cdf_gt.squeeze(), mode='lines', name=r"$p(y | x)$",
                               line=dict(color=colors[r"$p(y | x)$"]),
                               legendgroup=r"$p(y | x)$", showlegend=(i == 0)), row=i + 1, col=1)
                # Add shaded area for negative values of cdf
                fig_cde.add_trace(go.Scatter(x=Y_range_np.squeeze(), y=np.minimum(cdf.squeeze(), 0), mode='lines',
                                             fill='tozeroy',
                                             fillcolor=colors[r"$\hat{p}(y | x)$"],
                                             opacity=.5,
                                             line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=i + 1, col=1)

            fig_cde.update_layout(template="plotly_white")
    if plot:
        npmi_gt = np.reshape(NPMI_gt[:n_samples_per_g ** 2], (n_samples_per_g, n_samples_per_g))
        npmi = np.reshape(NPMI[:n_samples_per_g ** 2], (n_samples_per_g, n_samples_per_g))
        # Take the n_total_samples from the joint sampling, in the digagonal of the matrix of pairwise pairings.
        npmi_gt_joint = np.diag(npmi_gt)
        npmi_joint = np.diag(npmi)
        # Sample n_samples_per_g from the off diagonal of the matrix of pairwise pairings.
        npmi_gt_prod = npmi_gt[np.triu_indices(n_samples_per_g, k=1)]
        npmi_prod = npmi[np.triu_indices(n_samples_per_g, k=1)]
        prod_idx = np.random.choice(len(npmi_gt_prod), n_samples_per_g, replace=False)
        npmi_gt_prod = npmi_gt_prod[prod_idx]  # Sample some points from the joint
        npmi_prod = npmi_prod[prod_idx]
        npmi_gt = np.concatenate([npmi_gt_joint, npmi_gt_prod])
        npmi = np.concatenate([npmi_joint, npmi_prod])

        fig_pmd = plot_npmi_error_distribution(npmi_gt, npmi)
        return metrics, fig_pmd, fig_cde

    return metrics


def get_symmetry_group(cfg: DictConfig):
    group_label = cfg.symm_group
    # group_label = "C{N}" -> CyclicGroup(N)
    if group_label[0] == "C" and group_label[1:].isdigit():
        N = int(group_label[1:])
        G = escnn.group.CyclicGroup(N)
    elif group_label[0] == "D" and group_label[1:].isdigit():
        N = int(group_label[1:])
        G = escnn.group.DihedralGroup(N)
    elif group_label.lower() == "ico":
        G = escnn.group.Icosahedral()
    elif group_label.lower() == "octa":
        G = escnn.group.Octahedral()
    else:
        raise ValueError(f"Group {group_label} not recognized")

    log.info(f"Symmetry Group G: {G} of order {G.order()}")
    return G


@hydra.main(config_path='cfg', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    # Ensure HydraConfig is initialized
    run_id = HydraConfig.get().job.override_dirname if HydraConfig.initialized() else ""

    # Symmetry group G. Symmetry subgroup H. H2G: H -> G. G2H: G -> H
    G = get_symmetry_group(cfg)
    if cfg.regular_multiplicity > 0:
        rep_X = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Χ
        rep_Y = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Y
    elif isinstance(G, escnn.group.CyclicGroup) and G.order() == 2 and cfg.regular_multiplicity == 0:
        rep_X = G.representations['irrep_1']  # ρ_Χ
        rep_Y = G.representations['irrep_1']  # ρ_Y
    else:
        raise ValueError(f"G={G} Hx={cfg.x_symm_subgroup_id} Hy={cfg.y_symm_subgroup_id} {cfg.regular_multiplicity}")

    # GENERATE the training data _______________________________________________________
    seed_everything(cfg.gmm.seed)  # Get same GMM for all training seeds.
    gmm = SymmGaussianMixture(
        n_kernels=cfg.gmm.n_kernels,
        rep_X=rep_X,
        rep_Y=rep_Y,
        means_std=cfg.gmm.means_std,
        random_seed=cfg.gmm.seed,  # Reproducibility
        x_subgroup_id=cfg.x_symm_subgroup_id,
        y_subgroup_id=cfg.y_symm_subgroup_id,
        )
    seed_everything(seed)  # Random/Selected seed for weight initialization and training.
    (train_samples, val_samples, test_samples), (x_mean, y_mean), (x_var, y_var), datasets = gmm_dataset(
        cfg.gmm.n_samples, gmm, rep_X, rep_Y, device=cfg.device if cfg.data_on_device else 'cpu'
        )

    # x_train, y_train = train_samples
    x_val, y_val = val_samples
    x_test, y_test = test_samples

    # Define the Input and Latent types for ESCNN
    x_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_X])
    y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_Y])
    lat_type = FieldType(
        gspace=escnn.gspaces.no_base_space(G),
        representations=[G.regular_representation] * ceil(cfg.embedding['embedding_dim'] / G.order())
        )

    train_ds, val_ds, test_ds = datasets

    # ESCNN equivariagnt models expect GeometricTensors.
    def geom_tensor_collate_fn(batch) -> [GeometricTensor, GeometricTensor]:
        x_batch, y_batch = default_collate(batch)
        return GeometricTensor(x_batch, x_type), GeometricTensor(y_batch, y_type)

    # Plot the GT joint PDF and MI _______________________________________________________
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    gmm_plot_path = pathlib.Path(run_path).parent.parent / f"joint_pdf-std{cfg.gmm.means_std}.png"

    if not gmm_plot_path.exists():
        if x_type.size == 1 and y_type.size == 1:
            x_samples, y_samples = gmm.simulate(n_samples=5000)
            grid = plot_analytic_joint_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(pathlib.Path(run_path).parent.parent / f"joint_pdf-std{cfg.gmm.means_std}.png")
            grid = plot_analytic_prod_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(pathlib.Path(run_path).parent.parent / f"prod_pdf-std{cfg.gmm.means_std}.png")
            grid = plot_analytic_npmi_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(pathlib.Path(run_path).parent.parent / f"normalized_mutual_information-std{cfg.gmm.means_std}.png")
            grid = plot_analytic_pmd_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(pathlib.Path(run_path).parent.parent / f"pointwise_mutual_dependency-std{cfg.gmm.means_std}.png")

    # Get the model ______________________________________________________________________
    nnPME = get_model(cfg, x_type, y_type, lat_type)
    print(nnPME)
    # Print the number of parameters
    n_params = sum(p.numel() for p in nnPME.parameters())
    log.info(f"Number of parameters: {n_params}")
    nnPME.to(device=cfg.device)
    # Define the dataloaders
    from NCP.models.equiv_ncp import ENCP
    collate_fn = geom_tensor_collate_fn if isinstance(nnPME, ENCP) else default_collate
    train_dataloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the Lightning module ________________________________________________________
    ncp_lightning_module = TrainingModule(
        model=nnPME,
        optimizer_fn=torch.optim.Adam,
        optimizer_kwargs={"lr": cfg.lr},
        loss_fn=nnPME.loss,
        val_metrics=lambda x: measure_analytic_pmi_error(
            gmm, nnPME, x_val, y_val, x_type, y_type, x_mean, y_mean, x_var, y_var
            ),
        test_metrics=lambda x: measure_analytic_pmi_error(
            gmm, nnPME, x_test, y_test, x_type, y_type, x_mean, y_mean, x_var, y_var
            ),
        )

    # Define the logger and callbacks
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.exp_name, log_model=False, config=run_cfg)
    ckpt_call = ModelCheckpoint(
        dirpath=run_path, filename="best", monitor='loss/val', save_top_k=1, save_last=True, mode='min'
        )
    early_call = EarlyStopping(monitor='||k(x,y) - k_r(x,y)||/val', patience=cfg.patience, mode='min')

    trainer = lightning.Trainer(accelerator='gpu',
                                devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
                                max_epochs=cfg.max_epochs, logger=logger,
                                enable_progress_bar=True,
                                log_every_n_steps=25,
                                check_val_every_n_epoch=50,
                                callbacks=[ckpt_call, early_call],
                                fast_dev_run=10 if cfg.debug else False,
                                )

    torch.set_float32_matmul_precision('medium')
    last_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / ckpt_call.CHECKPOINT_NAME_LAST).with_suffix(
        ckpt_call.FILE_EXTENSION)
    trainer.fit(ncp_lightning_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=last_ckpt_path if last_ckpt_path.exists() else None,
                )

    ncp_lightning_module.to(device="cpu")
    # Loads the best model.
    trainer.test(ncp_lightning_module, dataloaders=test_dataloader)
    # Flush the logger.
    logger.finalize(trainer.state)

    log.info(f"Logging test metrics ... ")
    m, fig_pmd, fig_cde = measure_analytic_pmi_error(
        gmm, nnPME, x_test, y_test, x_type, y_type, x_mean, y_mean, x_var, y_var,
        plot=True, samples='all',
        )
    run_desc = f"{pathlib.Path(run_path).parent.name}/{pathlib.Path(run_path).name}"
    if fig_cde is not None:  # Plot
        fig_cde.update_layout(title_text=run_desc, title_font=dict(size=10))
        # Get the str of the name of the current and parent directories
        pio.write_html(fig_cde, file=pathlib.Path(run_path) / f"conditional_pdf.html", auto_open=False)
        pio.write_image(fig_cde, file=pathlib.Path(run_path) / f"conditional_pdf.png", scale=1.5)
    if fig_pmd is not None:
        print("Plotting NPMI")
        # set title PLT fig
        fig_pmd.suptitle(run_desc)
        fig_pmd.tight_layout()
        fig_pmd.savefig(pathlib.Path(run_path) / f"NPMI_pdf.png", dpi=100)
        print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error("An error occurred", exc_info=True)
