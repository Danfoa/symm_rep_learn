# Created by danfoa at 22/01/25
import escnn
import numpy as np
from escnn.group import directsum
from plotly.subplots import make_subplots

from symm_rep_learn.cde_fork.density_simulation.symmGMM import SymmGaussianMixture

if __name__ == "__main__":
    n_gaussians = 30
    # G = escnn.group.octa_group()
    # G = escnn.group.ico_group()
    G = escnn.group.DihedralGroup(6)
    print(G.order())
    m = 1
    # rep_X = directsum([G.standard_representation]
    # ,`* 1)
    # rep_Y = directsum([G.standard_representation] * 1)
    rep_X = directsum([G.regular_representation] * m)
    rep_Y = directsum([G.regular_representation] * m)

    for i in range(15):
        # seed = gmm_seed = np.random.randint(0, 1000),
        seed = i
        gmm = SymmGaussianMixture(
            n_kernels=n_gaussians,
            rep_X=rep_X,
            rep_Y=rep_Y,
            mean_max_norm=1,
            gmm_seed=seed,
        )
        n_sampels = 10000
        x, y = gmm.simulate(n_sampels)
        prod_idx = np.arange(n_sampels)
        # Get all pairs of samples
        X_idx, Y_idx = np.meshgrid(prod_idx, prod_idx)
        prod_samples = np.random.choice(n_sampels**2, n_sampels, replace=False)
        X_idx_flat = X_idx.flatten()[prod_samples]
        Y_idx_flat = Y_idx.flatten()[prod_samples]
        x_prod = np.atleast_2d(x[X_idx_flat])
        y_prod = np.atleast_2d(y[Y_idx_flat])
        x_full = np.concatenate([x, x_prod], axis=0)
        y_full = np.concatenate([y, y_prod], axis=0)
        del x_prod, y_prod
        pmd = gmm.pointwise_mutual_dependency(x_full, y_full)
        percentiles = [25, 50, 75, 85, 90, 95]
        pmd_perctiles = [np.percentile(pmd, p) for p in percentiles]
        MI = gmm.MI(x_full, y_full)

        desc = (
            f"- {seed}: MI: {MI:.4f}, MaxPMI (95%): {pmd_perctiles[-1]:.3f} stdPMI: {pmd.std():.3f}, "
            f"Pecentiles: {[f'{p}%: {pmd_perctiles[i]:.3f}' for i, p in enumerate(percentiles)]}"
        )
        print(desc)
        # Garbage collect x and y

        # Garbage collect x and y

        # Do a 3D plot for X using plotly interactive
        import numpy as np
        import plotly.graph_objs as go

        x, y = x[:5000], y[:5000]
        means_x = gmm.means_x
        means_y = gmm.means_y

        p_x = gmm.pdf_x(x)
        p_y = gmm.pdf_y(y)

        if gmm.ndim_x == 3:
            # Create subplots
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])
            # Calculate the radius for each center
            radius_x = np.linalg.norm(means_x, axis=1)
            radius_y = np.linalg.norm(means_y, axis=1)
            # Plot the means for X with color based on radius
            fig.add_trace(
                go.Scatter3d(
                    x=means_x[:, 0],
                    y=means_x[:, 1],
                    z=means_x[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=radius_x, colorscale="Viridis", colorbar=dict(title="Radius")),
                    name="Means X",
                ),
                row=1,
                col=1,
            )  # Plot the samples in transparent blue for X
            fig.add_trace(
                go.Scatter3d(
                    x=x[:, 0], y=x[:, 1], z=x[:, 2], mode="markers", marker=dict(size=3, opacity=0.5), name="Samples X"
                ),
                row=1,
                col=1,
            )
            # Plot the means for Y
            # Plot the means for Y with color based on radius
            fig.add_trace(
                go.Scatter3d(
                    x=means_y[:, 0],
                    y=means_y[:, 1],
                    z=means_y[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=radius_y, colorscale="Viridis", colorbar=dict(title="Radius")),
                    name="Means Y",
                ),
                row=1,
                col=2,
            )
            # Plot the samples in transparent blue for Y
            fig.add_trace(
                go.Scatter3d(
                    x=y[:, 0], y=y[:, 1], z=y[:, 2], mode="markers", marker=dict(size=3, opacity=0.5), name="Samples Y"
                ),
                row=1,
                col=2,
            )  # Set MI as title
            fig.update_layout(title=desc, font=dict(size=12))
            fig.show()
