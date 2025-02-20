# Created by danfoa at 22/01/25
import escnn
import numpy as np
from escnn.group import directsum

from symm_rep_learn.cde_fork.density_simulation.symmGMM import SymmGaussianMixture
from symm_rep_learn.examples.symmGMM.plot_utils import plot_analytic_joint_2D

if __name__ == "__main__":
    n_gaussians = 10
    # G = escnn.group.octa_group()
    G = escnn.group.DihedralGroup(6)
    # G = escnn.group.CyclicGroup(2)
    m = 1
    # rep_X = directsum([G.representations['irrep_1'] ] * m)
    # rep_Y = directsum([G.representations['irrep_1'] ] * m)
    rep_X = directsum([G.regular_representation] * m)
    rep_Y = directsum([G.regular_representation] * m)

    for _ in range(10):
        seed = gmm_seed = (np.random.randint(0, 1000),)
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
        MI = gmm.MI(x_full, y_full)

        print(f"- {seed}: Mutual information {n_sampels}~p(x,y) {n_sampels}~p(x)p(y) Samples: {MI}")
        # Garbage collect x and y

        if gmm.ndim_x == 1 and gmm.ndim_y == 1:
            print("d")
            g = plot_analytic_joint_2D(gmm, G, rep_X, rep_Y, x, y)
            g.fig.suptitle(f"Mutual information {MI:.3f} Seed {seed} Samples {n_sampels}")
            g.fig.show()
