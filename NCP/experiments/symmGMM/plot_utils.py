# Created by danfoa at 21/01/25
import numpy as np
import seaborn as sns
import torch
from escnn.group import Group, Representation
from matplotlib import pyplot as plt

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture

# Define constants for plot styles
PLOT_STYLE = {
    "cpd_x":  {
        "color":  "red",
        "fill":   "pink",
        "legend": r"$p(y | x)$",
        "expect": r"$E[y|x]$",
        },
    "cpd_gx": {
        "color":  "green",
        "fill":   "lightgreen",
        "legend": r"$p(y | g \;\triangleright_{\mathcal{X}}\; x)$",
        "expect": r"$E[y|g \;\triangleright_{\mathcal{X}}\; x]$",
        },
    "pdf_y":    {
        "color":  "lightblue",
        "fill":   "lightblue",
        "legend": r"$p(y)$",
        },
    "pdf_x":    {
        "color":  "lightblue",
        "fill":   "lightblue",
        "legend": r"$p(x)$",
        "legend": r"$p(x)$",
        },
    "npmi_x": {
        "color":  "red",
        "legend": r"$NPMI(x, y)$",
        },
    "npmi_gx": {
        "color":  "green",
        "legend": r"$NPMI(g \;\triangleright_{\mathcal{X}}\; x, y)$",
        },
    "npmi_y": {
        "color":  "red",
        "legend": r"$NPMI(x, y)$",
        },
    "npmi_gy": {
        "color":  "green",
        "legend": r"$NPMI(x, g \;\triangleright_{\mathcal{Y}}\; y)$",
        },
    "pmd_x": {
        "color":  "red",
        "legend": r"$\kappa(x, y)$",
        },
    "pmd_gx": {
        "color":  "green",
        "legend": r"$\kappa(g \;\triangleright_{\mathcal{X}}\; x, y)$",
        },
    "pmd_y": {
        "color":  "red",
        "legend": r"$\kappa(x, y)$",
        },
    "pmd_gy": {
        "color":  "green",
        "legend": r"$\kappa(x, g \;\triangleright_{\mathcal{Y}}\; y)$",
        },
}

PLOT_SIZE = 3
PLOT_LEVELS = 15
PLOT_CMAP = "Blues"
PLOT_LINEWIDTH = 1
PLOT_MARKERSIZE = 8
PLOT_ALPHA = 0.7
PLOT_FONT_SIZE = 7
LEGEND_FONT_SIZE = 5
LEGEND_BORDER_PAD = 1
EXPECTATION_MARKER = "D"
LEGEND_FRAME_ALPHA = 0.8

def plot_analytic_joint_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                           y_samples):
    grid = sns.JointGrid(space=0.1, height=PLOT_SIZE)
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
    joint_contour = grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap=PLOT_CMAP, levels=PLOT_LEVELS)

    # Select a random sample to test the conditional expectation
    x_t, y_t = -1, 1
    g = G.elements[-1]
    rep_X = gmm.rep_X
    rep_Y = gmm.rep_Y
    gx_t, gy_t = (rep_X(gmm.G2Hx(g)) @ [x_t]).squeeze(), (rep_Y(gmm.G2Hy(g)) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axvline(gx_t, color='g', alpha=PLOT_ALPHA)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA, markeredgecolor='white', markeredgewidth=PLOT_LINEWIDTH)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA, markeredgecolor='white', markeredgewidth=PLOT_LINEWIDTH)
    # Set limits
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])

    # Do plot of conditional probability density on the test conditions x and gx
    n_samples_cpd = len(y_range)
    cpd_y_vals = gmm.pdf(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    cpd_gy_vals = gmm.pdf(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    # Compute the Expectation of the CDF
    E_y = np.sum(y_range * cpd_y_vals) / np.sum(cpd_y_vals)
    E_gy = np.sum(y_range * cpd_gy_vals) / np.sum(cpd_gy_vals)
    P_E_y = gmm.pdf(X=np.repeat(x_t, n_samples_cpd)[:2], Y=(y_range * 0 + E_y)[:2])[0]
    P_E_gy = gmm.pdf(X=np.repeat(gx_t, n_samples_cpd)[:2], Y=(y_range * 0 + E_gy)[:2])[0]

    for key, cpd_vals, expect_y, Pey in zip(["cpd_x", "cpd_gx"], [cpd_y_vals, cpd_gy_vals ], [E_y, E_gy], [P_E_y, P_E_gy]):
        grid.ax_marg_y.plot(cpd_vals, y_range, color=PLOT_STYLE[key]["color"], linestyle="-", alpha=0.9, linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"])
        # Print a marker at the expected position using the color of the conditional distribution
        grid.ax_marg_y.plot(0, expect_y, EXPECTATION_MARKER, color=PLOT_STYLE[key]["color"], markersize=5, alpha=0.9,
                            label=PLOT_STYLE[key]["expect"])

    # Plot marginal x
    pdf_x = gmm.pdf_x(X=x_range)
    grid.ax_marg_x.fill_between(x_range, 0, pdf_x, color=PLOT_STYLE["pdf_x"]["fill"], alpha=0.6, label=PLOT_STYLE["pdf_x"]["legend"])
    grid.ax_marg_x.set_ylim([0, None])
    # Plot marginal y
    pdf_y = gmm.pdf_y(Y=y_range)
    grid.ax_marg_y.fill_betweenx(y_range, 0, pdf_y, color=PLOT_STYLE["pdf_y"]["fill"], alpha=0.6, label=PLOT_STYLE["pdf_y"]["legend"])
    grid.ax_marg_y.set_xlim([0, None])

    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    # grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    # grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")
    # Remove ticks from joint x and y axes
    grid.ax_joint.set_xticks([])
    grid.ax_joint.set_yticks([])
    # Remove borders from lower and left margins of the joint plot
    grid.ax_joint.spines['bottom'].set_visible(False)
    grid.ax_joint.spines['left'].set_visible(False)
    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.2), borderaxespad=0, framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    grid.ax_marg_x.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.1), borderaxespad=0, framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    return grid

def plot_analytic_prod_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                          y_samples):
    grid = sns.JointGrid(space=0.0, height=PLOT_SIZE)
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
    grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap=PLOT_CMAP, levels=PLOT_LEVELS)

    # Select a random sample to test the conditional expectation
    sample_idx = np.random.choice(len(x_samples))
    x_t, y_t = -1, 1
    P_x_t = gmm.pdf_x(X=np.repeat(x_t, 2))[0]
    P_y_t = gmm.pdf_y(Y=np.repeat(y_t, 2))[0]

    g = G.elements[-1]
    rep_X = gmm.rep_X
    rep_Y = gmm.rep_Y
    gx_t, gy_t = (rep_X(gmm.G2Hx(g)) @ [x_t]).squeeze(), (rep_Y(gmm.G2Hy(g)) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(y_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axvline(gx_t, color='g', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(gy_t, color='g', alpha=PLOT_ALPHA)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
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
    # Plot marginal x
    pdf_x = gmm.pdf_x(X=x_range)
    grid.ax_marg_x.fill_between(x_range, 0, pdf_x, color=PLOT_STYLE["pdf_x"]["fill"], alpha=0.4)
    grid.ax_marg_x.plot(x_range, pdf_x, color=PLOT_STYLE["pdf_x"]["color"], linestyle="-", alpha=1.0, linewidth=PLOT_LINEWIDTH,
                        label=PLOT_STYLE["pdf_x"]["legend"])
    # Print a marker at the expected position using the color of the conditional distribution
    grid.ax_marg_x.plot(x_t, 0, 'ro', markersize=5, alpha=0.5)
    grid.ax_marg_x.plot(gx_t, 0, 'go', markersize=5, alpha=0.5)
    # Print a horizontal line at the probability of x_t
    grid.ax_marg_x.plot([x_t, x_t], [0, P_x_t], 'r-', alpha=0.5)
    grid.ax_marg_x.plot([gx_t, gx_t], [0, P_x_t], 'g-', alpha=0.5)

    grid.ax_marg_x.set_ylim([0, None])
    # Plot marginal y
    pdf_y = gmm.pdf_y(Y=y_range)
    grid.ax_marg_y.fill_betweenx(y_range, 0, pdf_y, color=PLOT_STYLE["pdf_y"]["fill"], alpha=0.4)
    grid.ax_marg_y.plot(pdf_y, y_range, color=PLOT_STYLE["pdf_y"]["color"], linestyle="-", alpha=1.0, linewidth=PLOT_LINEWIDTH,
                        label=PLOT_STYLE["pdf_y"]["legend"])
    # Print a marker at the expected position using the color of the conditional distribution
    grid.ax_marg_y.plot(0, y_t, 'ro', markersize=5, alpha=0.5)
    grid.ax_marg_y.plot(0, gy_t, 'go', markersize=5, alpha=0.5)
    # Print a vertical line at the probability of y_t
    grid.ax_marg_y.plot([0, P_y_t], [y_t, y_t], 'r-', alpha=0.5)
    grid.ax_marg_y.plot([0, P_y_t], [gy_t, gy_t], 'g-', alpha=0.5)

    grid.ax_marg_y.set_xlim([0, None])

    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    # grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    # grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")
    # Remove ticks from joint x and y axes
    grid.ax_joint.set_xticks([])
    grid.ax_joint.set_yticks([])
    # Remove borders from lower and left margins of the joint plot
    grid.ax_joint.spines['bottom'].set_visible(False)
    grid.ax_joint.spines['left'].set_visible(False)
    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.2), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    grid.ax_marg_x.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.1), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    return grid

def plot_analytic_npmi_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                          y_samples):
    grid = sns.JointGrid(space=0.0, height=PLOT_SIZE)
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 200)
    y_range = np.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    Pxy = gmm.joint_pdf(X=X_input, Y=Y_input)
    npmi_flat = gmm.normalized_pointwise_mutual_information(X=X_input, Y=Y_input)
    Z = npmi_flat.reshape(X_grid.shape)
    grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap=sns.color_palette("magma", as_cmap=True), levels=PLOT_LEVELS)

    x_t, y_t = -1, 1
    g = G.elements[-1]
    rep_X = gmm.rep_X
    rep_Y = gmm.rep_Y
    gx_t, gy_t = (rep_X(gmm.G2Hx(g)) @ [x_t]).squeeze(), (rep_Y(gmm.G2Hy(g)) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(y_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axvline(gx_t, color='g', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(gy_t, color='g', alpha=PLOT_ALPHA)
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    n_samples_cpd = len(y_range)
    npmi_x_vals = gmm.normalized_pointwise_mutual_information(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    npmi_gx_vals = gmm.normalized_pointwise_mutual_information(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    npmi_y_vals = gmm.normalized_pointwise_mutual_information(X=x_range, Y=np.repeat(y_t, len(x_range)))
    npmi_gy_vals = gmm.normalized_pointwise_mutual_information(X=x_range, Y=np.repeat(gy_t, len(x_range)))

    for key, mi_vals in zip(["npmi_x", "npmi_gx"], [npmi_x_vals, npmi_gx_vals]):
        grid.ax_marg_y.plot(mi_vals, y_range, color=PLOT_STYLE[key]["color"], linestyle="-", alpha=0.9, linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"])

    for key, mi_vals in zip(["npmi_y", "npmi_gy"], [npmi_y_vals, npmi_gy_vals]):
        grid.ax_marg_x.plot(x_range, mi_vals, color=PLOT_STYLE[key]["color"], linestyle="-", alpha=0.9, linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"])

    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    # grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    # grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")
    # Remove ticks from joint x and y axes
    grid.ax_joint.set_xticks([])
    grid.ax_joint.set_yticks([])
    # Remove borders from lower and left margins of the joint plot
    grid.ax_joint.spines['bottom'].set_visible(False)
    grid.ax_joint.spines['left'].set_visible(False)
    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.2), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    grid.ax_marg_x.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.1), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    return grid

def plot_analytic_pmd_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                         y_samples):
    grid = sns.JointGrid(space=0.0, height=PLOT_SIZE)
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 500)
    y_range = np.linspace(-y_max, y_max, 500)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # Pxy = gmm.joint_pdf(X=X_input, Y=Y_input)
    pmd = gmm.pointwise_mutual_dependency(X=X_input, Y=Y_input)
    Z = pmd.reshape(X_grid.shape)
    grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap=sns.color_palette("magma", as_cmap=True), levels=PLOT_LEVELS * 2)

    
    x_t, y_t = -1, 1
    g = G.elements[-1]
    rep_X = gmm.rep_X
    rep_Y = gmm.rep_Y
    gx_t, gy_t = (rep_X(gmm.G2Hx(g)) @ [x_t]).squeeze(), (rep_Y(gmm.G2Hy(g)) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(y_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axvline(gx_t, color='g', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(gy_t, color='g', alpha=PLOT_ALPHA)
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    n_samples_cpd = len(y_range)
    pmd_x_vals = gmm.pointwise_mutual_dependency(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    pmd_gx_vals = gmm.pointwise_mutual_dependency(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    pmd_y_vals = gmm.pointwise_mutual_dependency(X=x_range, Y=np.repeat(y_t, len(x_range)))
    pmd_gy_vals = gmm.pointwise_mutual_dependency(X=x_range, Y=np.repeat(gy_t, len(x_range)))

    for key, mi_vals in zip(["pmd_x", "pmd_gx"], [pmd_x_vals, pmd_gx_vals]):
        grid.ax_marg_y.plot(mi_vals, y_range, color=PLOT_STYLE[key]["color"], linestyle="-", alpha=0.9, linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"])

    for key, mi_vals in zip(["pmd_y", "pmd_gy"], [pmd_y_vals, pmd_gy_vals]):
        grid.ax_marg_x.plot(x_range, mi_vals, color=PLOT_STYLE[key]["color"], linestyle="-", alpha=0.9, linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"])

    # Plot constant line at 1 for both marginals
    grid.ax_marg_y.axvline(1, color='k', linestyle='-', alpha=0.3)
    grid.ax_marg_x.axhline(1, color='k', linestyle='-', alpha=0.3)

    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    # grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    # grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")
    # Remove ticks from joint x and y axes
    grid.ax_joint.set_xticks([])
    grid.ax_joint.set_yticks([])
    # Remove borders from lower and left margins of the joint plot
    grid.ax_joint.spines['bottom'].set_visible(False)
    grid.ax_joint.spines['left'].set_visible(False)
    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.2), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    grid.ax_marg_x.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.1), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    return grid

def plot_pmd_err_2D(gmm: SymmGaussianMixture,
                    nn_model: torch.nn.Module,
                    G: Group,
                    x_samples,
                    y_samples,
                    x_mean, x_var, x_type,
                    y_mean, y_var, y_type,
                    x_lims, y_lims
                    ):
    device = next(nn_model.parameters()).device
    grid = sns.JointGrid(space=0.0, height=PLOT_SIZE)
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    if x_lims is not None:
        x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    else:
        x_max, y_max = np.max(np.abs(x_lims)), np.max(np.abs(y_lims))
    x_range = np.linspace(-x_max, x_max, 500)
    y_range = np.linspace(-y_max, y_max, 500)

    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # Pxy = gmm.joint_pdf(X=X_input, Y=Y_input)
    pmd = gmm.pointwise_mutual_dependency(X=X_input, Y=Y_input)
    from NCP.models.equiv_ncp import ENCP
    def get_pmd_pred(x, y):
        X_c = ((torch.Tensor(x) - x_mean) / torch.sqrt(x_var)).to(device=device)
        Y_c = ((torch.Tensor(y) - y_mean) / torch.sqrt(y_var)).to(device=device)
        _x, _y = (x_type(X_c), y_type(Y_c)) if isinstance(nn_model, ENCP) else (X_c, Y_c)
        pmd_pred = nn_model.pointwise_mutual_dependency(_x, _y).cpu().numpy()  # k_r(x,y) ≈ p(x,y) / p(x)p(y)
        return pmd_pred
    # Compute prediction
    pmd_pred = get_pmd_pred(X_input, Y_input)
    pmd_err = pmd - pmd_pred
    max_pmd_gt = np.max(pmd)
    Z = pmd_err.reshape(X_grid.shape)
    contour = grid.ax_joint.contourf(X_grid,
                           Y_grid,
                           Z,
                           cmap=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True).reversed(),
                           levels=PLOT_LEVELS
                           )

    # Plot constant line at 1 for both marginals
    grid.ax_marg_y.axvline(1, color='k', linestyle='-', alpha=0.2,)
    grid.ax_marg_x.axhline(1, color='k', linestyle='-', alpha=0.2,)
    grid.ax_marg_y.axvline(0, color='b', linestyle='-', alpha=0.2,)
    grid.ax_marg_x.axhline(0, color='b', linestyle='-', alpha=0.2,)

    # Add a colorbar that does not overlap
    colorbar_axes = grid.fig.add_axes([
        grid.ax_marg_y.get_position().x1,  # To the right of the joint axes
        grid.ax_joint.get_position().y0,  # Align bottom with joint axes
        0.02,  # Width of the colorbar
        grid.ax_marg_y.get_position().height  # Same height as joint axes
        ])
    colorbar = plt.colorbar(contour, cax=colorbar_axes)

    # # Add a new axis at the bottom-left corner of the joint axes
    # joint_axes_position = grid.ax_joint.get_position()
    # bottom_axis_height = 0.2  # Adjust height as needed
    # bottom_axis = grid.fig.add_axes([
    #     joint_axes_position.x0,  # Align to joint axes' left edge
    #     joint_axes_position.y0 - bottom_axis_height,  # Below the joint axes
    #     joint_axes_position.width,  # Same width as joint axes
    #     bottom_axis_height  # Height
    #     ])

    x_t, y_t = -1, 1
    g = G.elements[-1]
    rep_X = gmm.rep_X
    rep_Y = gmm.rep_Y
    gx_t, gy_t = (rep_X(gmm.G2Hx(g)) @ [x_t]).squeeze(), (rep_Y(gmm.G2Hy(g)) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(y_t, color='r', alpha=PLOT_ALPHA)
    grid.ax_joint.axvline(gx_t, color='g', alpha=PLOT_ALPHA)
    grid.ax_joint.axhline(gy_t, color='g', alpha=PLOT_ALPHA)
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=PLOT_MARKERSIZE, alpha=PLOT_ALPHA)
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    n_samples_cpd = len(y_range)
    pmd_x_vals = gmm.pointwise_mutual_dependency(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    pmd_x_vals_pred = get_pmd_pred(np.repeat(x_t, n_samples_cpd)[:, None], y_range[:, None])
    pmd_gx_vals = gmm.pointwise_mutual_dependency(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    pmd_gx_vals_pred = get_pmd_pred(np.repeat(gx_t, n_samples_cpd)[:, None], y_range[:, None])
    pmd_y_vals = gmm.pointwise_mutual_dependency(X=x_range, Y=np.repeat(y_t, len(x_range)))
    pmd_y_vals_pred = get_pmd_pred(x_range[:, None], np.repeat(y_t, len(x_range))[:, None])
    pmf_gy_vals = gmm.pointwise_mutual_dependency(X=x_range, Y=np.repeat(gy_t, len(x_range)))
    pmf_gy_vals_pred = get_pmd_pred(x_range[:, None], np.repeat(gy_t, len(x_range))[:, None])

    for key, pmd_x, pmd_x_pred in zip(["npmi_x", "npmi_gx"], [pmd_x_vals, pmd_gx_vals], [pmd_x_vals_pred, pmd_gx_vals_pred]):
        grid.ax_marg_y.plot(pmd_x, y_range, color=PLOT_STYLE[key]["color"], linestyle="-", alpha=0.9,
                            linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"])
        grid.ax_marg_y.plot(pmd_x_pred, y_range, color=PLOT_STYLE[key]["color"], linestyle="--", alpha=0.9,
                            linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"] + " pred")

    for key, pmd_y, pmd_y_pred in zip(["npmi_y", "npmi_gy"], [pmd_y_vals, pmf_gy_vals], [pmd_y_vals_pred, pmf_gy_vals_pred]):
        grid.ax_marg_x.plot(x_range, pmd_y, color=PLOT_STYLE[key]["color"], linestyle="-", alpha=0.9,
                            linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"])
        grid.ax_marg_x.plot(x_range, pmd_y_pred, color=PLOT_STYLE[key]["color"], linestyle="--", alpha=0.9,
                            linewidth=PLOT_LINEWIDTH,
                            label=PLOT_STYLE[key]["legend"] + " pred")


    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    # grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    # grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")
    # Remove ticks from joint x and y axes
    grid.ax_joint.set_xticks([])
    grid.ax_joint.set_yticks([])
    # Remove borders from lower and left margins of the joint plot
    grid.ax_joint.spines['bottom'].set_visible(False)
    grid.ax_joint.spines['left'].set_visible(False)
    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.2), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    grid.ax_marg_x.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.0, 1.1), borderaxespad=0,
                          framealpha=LEGEND_FRAME_ALPHA, borderpad=LEGEND_BORDER_PAD)
    return grid

def plot_pmd_error_distribution(pmd_gt, pmd):
    # Create a JointGrid for the 2D KDE plot with marginals
    # Set PLT backed to AGG
    max_pmd = np.max(pmd_gt)
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(6, 6))
    # contour = sns.kdeplot(
    #     x=pmd_gt,
    #     y=pmd,
    #     fill=True,
    #     cmap=sns.cubehelix_palette(as_cmap=True),
    #     levels=10,
    #     bw_adjust=0.4,
    #     clip=((0, None), (0, None))
    #     )
    # Plot samples in very very transparent markers
    plt.scatter(pmd_gt, pmd, alpha=0.1)
    # Plot a line from 0,0 to max_pmd, max_pmd
    plt.plot([0, max_pmd], [0, max_pmd], 'k--')
    # Set axes to be of equal aspect ratio and limits from 0 to max_pmd
    plt.axis('equal')
    plt.xlim([0, max_pmd])
    # plt.colorbar(contour.collections[0], label='Density')
    plt.xlabel('PMD ground truth')
    plt.ylabel(r'PMD prediction')
    plt.title('PMD regression error') # Ensure aspect ratio is the same for both axes
    # Set the limits for the marginal axes to match the joint axes
    return fig
