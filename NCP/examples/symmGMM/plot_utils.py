# Created by danfoa at 21/01/25
import numpy as np
import seaborn as sns
from escnn.group import Group, Representation
from matplotlib import pyplot as plt

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture


def plot_analytic_joint_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                           y_samples):
    grid = sns.JointGrid(space=0.0, height=5)
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
    grid.ax_joint.axvline(gx_t, color='g', alpha=0.7)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=8, alpha=0.7, markeredgecolor='white', markeredgewidth=1)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=8, alpha=0.7, markeredgecolor='white', markeredgewidth=1)
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
    # Compute the Expectation of the CDF
    E_y = np.sum(y_range * cpd_y_vals) / np.sum(cpd_y_vals)
    E_gy = np.sum(y_range * cpd_gy_vals) / np.sum(cpd_gy_vals)
    P_E_y = gmm.pdf(X=np.repeat(x_t, n_samples_cpd)[:2], Y=(y_range * 0 + E_y)[:2])[0]
    P_E_gy = gmm.pdf(X=np.repeat(gx_t, n_samples_cpd)[:2], Y=(y_range * 0 + E_gy)[:2])[0]
    style = {
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
        "pdf":    {
            "color":  "lightblue",
            "fill":   "lightblue",
            "legend": r"$p(y)$",
            }
        }

    for key, cpd_vals, expect_y, Pey in zip(["cpd_x", "cpd_gx"], [cpd_y_vals, cpd_gy_vals ], [E_y, E_gy], [P_E_y, P_E_gy]):
        grid.ax_marg_y.plot(cpd_vals, y_range, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=1,
                            label=style[key]["legend"])
        # Print a marker at the expected position using the color of the conditional distribution
        grid.ax_marg_y.plot(0, expect_y, 'o', color=style[key]["color"], markersize=5, alpha=0.9,
                            label=style[key]["expect"])

    # Plot marginal x
    pdf_x = gmm.pdf_x(X=x_range)
    grid.ax_marg_x.fill_between(x_range, 0, pdf_x, color=style["pdf"]["fill"], alpha=0.6, label=r"$p(x)$")
    grid.ax_marg_x.set_ylim([0, None])
    # Plot marginal y
    pdf_y = gmm.pdf_y(Y=y_range)
    grid.ax_marg_y.fill_betweenx(y_range, 0, pdf_y, color=style["pdf"]["fill"], alpha=0.6, label=r"$p(y)$")
    grid.ax_marg_y.set_xlim([0, None])

    # Remove ticks from joint x and y axes
    grid.ax_joint.set_xticks([])
    grid.ax_joint.set_yticks([])
    # Remove borders from lower and left margins of the joint plot
    grid.ax_joint.spines['bottom'].set_visible(False)
    grid.ax_joint.spines['left'].set_visible(False)
    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=7,  borderaxespad=0)
    grid.ax_marg_x.legend(loc="upper left", fontsize=7,  borderaxespad=0)
    grid.fig.suptitle(r"$p(x,y)$, $p(y | x)$, $p(x)$, $p(y)$")
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
    mi_x_vals = gmm.pointwise_mutual_dependency(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    mi_gx_vals = gmm.pointwise_mutual_dependency(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    mi_y_vals = gmm.pointwise_mutual_dependency(X=x_range, Y=np.repeat(y_t, len(x_range)))
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


def plot_npmi_error_distribution(NPMI_gt, NPMI_err):
    # Create a JointGrid for the 2D KDE plot with marginals
    # Set PLT backed to AGG
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6))
    contour = sns.kdeplot(
        x=NPMI_gt,
        y=NPMI_err,
        fill=True,
        cmap=sns.cubehelix_palette(as_cmap=True),
        levels=15,
        bw_adjust=0.6,
        )
    plt.colorbar(contour.collections[0], label='Density')
    plt.xlabel('NPMI')
    plt.ylabel(r'$NPMI$ - $NPMI_{\text{pred}}$')
    plt.title('2D Distribution of NPMI Error') # Ensure aspect ratio is the same for both axes
    # Set the limits for the marginal axes to match the joint axes
    return fig
