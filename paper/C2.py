# Created by danfoa at 22/01/25
import escnn
import numpy as np
from escnn.group import directsum
from plotly.subplots import make_subplots

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture

if __name__ == "__main__":
    from lightning import seed_everything
    from matplotlib import pyplot as plt

    n_gaussians = 2
    # G = escnn.group.octa_group()
    G = escnn.group.ico_group()
    print(G.order())
    rep_X = directsum([G.standard_representation] * 1)
    rep_Y = directsum([G.standard_representation] * 1)
    # rep_X = G.standard_representation
    # rep_Y = G.standard_representation

    for _ in range(10):
        seed = gmm_seed = np.random.randint(0, 1000),
        gmm = SymmGaussianMixture(n_kernels=n_gaussians,
                                  rep_X=rep_X,
                                  rep_Y=rep_Y,
                                  mean_max_norm=6,
                                  gmm_seed=seed,
                                  )
        n_sampels = 100000
        x, y = gmm.simulate(n_sampels)
        pmi = gmm.pointwise_mutual_information(x, y)
        p_xy = gmm.joint_pdf(x, y)
        MI = (p_xy * pmi).sum()
        print(f"- {seed}: Mutual information {n_sampels} Samples: {MI}")
        # Garbage collect x and y

        # Do a 3D plot for X using plotly interactive
        import plotly.graph_objs as go
        import numpy as np

        x, y = x[:1000], y[:1000]
        means_x = gmm.means_x
        means_y = gmm.means_y

        p_x = gmm.pdf_x(x)
        p_y = gmm.pdf_y(y)

        if gmm.ndim_x == 3:
            # Create subplots
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
            # Calculate the radius for each center
            radius_x = np.linalg.norm(means_x, axis=1)
            radius_y = np.linalg.norm(means_y, axis=1)
            # Plot the means for X with color based on radius
            fig.add_trace(go.Scatter3d(
                x=means_x[:, 0], y=means_x[:, 1], z=means_x[:, 2], mode='markers',
                marker=dict(size=5, color=radius_x, colorscale='Viridis', colorbar=dict(title='Radius')),
                name='Means X'), row=1, col=1)  # Plot the samples in transparent blue for X
            fig.add_trace(
                go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers', marker=dict(size=3, opacity=0.5),
                             name='Samples X'), row=1, col=1)
            # Plot the means for Y
            # Plot the means for Y with color based on radius
            fig.add_trace(go.Scatter3d(
                x=means_y[:, 0], y=means_y[:, 1], z=means_y[:, 2], mode='markers',
                marker=dict(size=5, color=radius_y, colorscale='Viridis', colorbar=dict(title='Radius')),
                name='Means Y'), row=1, col=2)
            # Plot the samples in transparent blue for Y
            fig.add_trace(
                go.Scatter3d(x=y[:, 0], y=y[:, 1], z=y[:, 2], mode='markers', marker=dict(size=3, opacity=0.5),
                             name='Samples Y'), row=1, col=2)  # Set MI as title
            fig.update_layout(title=f"seed {seed} Mutual information {n_sampels} Samples: {MI}")
            fig.show()
