# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 11/05/25
import numpy as np
from escnn.group import DihedralGroup

from symm_rep_learn.mysc.rep_theory_utils import isotypic_decomp_rep

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(10)

    G = DihedralGroup(3)
    rep_R2 = G.representations["irrep_1,1"]

    # G = CyclicGroup(6)
    # rep_R2 = G.representations["irrep_2"]
    #
    # G = CyclicGroup(3)
    # rep_R2 = G.representations["irrep_1"]

    for g in G.elements:
        print(f" {g} :\n {rep_R2(g)}")

    # Degine the G-invariant probability distribution as a GMM with 3 kernels
    n_kernels = 15
    gmm_means = np.random.rand(n_kernels, 2)
    gmm_covs = np.array([np.eye(2) * np.random.rand() * 0.2 for _ in range(n_kernels)])
    G_gmm_means = np.concatenate([np.einsum("ik,jk->ji", rep_R2(g), gmm_means) for g in G.elements])
    G_gmm_covs = np.concatenate([gmm_covs] * len(G.elements))
    print("GMM means:")

    # Make XY grid and plot the G-invariant probability distribution
    def pdf(x, means, covs):
        probs = np.zeros(x.shape[0])
        for mean, cov in zip(means, covs):
            diff = x - mean  # (n_samples, x_dim)
            inv_cov = np.linalg.inv(cov)  # (x_dim, x_dim)
            mahalanobis = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)  # Vectorized Mahalanobis distance
            probs += np.exp(-0.5 * mahalanobis) / np.sqrt((2 * np.pi) ** len(mean) * np.linalg.det(cov))
        return probs

    # Define a random function of the 2D plane using the GMM
    fn_n_kernels = 10
    fn_means = np.random.rand(fn_n_kernels, 2)
    fn_covs = np.array([np.eye(2) * np.random.rand() * 0.4 for _ in range(fn_n_kernels)])

    def G_fn(x):
        # Computes the group orbit of the function.
        results = []
        for g in G.elements:
            g_inv_x = np.einsum("ab,nb->na", rep_R2(~g), x)
            results.append(pdf(g_inv_x, fn_means, fn_covs))
        return results

    # Make a grid of points
    x = np.linspace(-1.5, 1.5, 100)
    y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T  # Shape (n_samples, 2)
    grid_pdf = pdf(grid_points, G_gmm_means, G_gmm_covs)

    # pip install plotly
    import numpy as np

    # ─── assume you already have X, Y, grid_pdf, G_fn (list of len = |G|)
    Z0 = np.zeros_like(X)  # flat plane χ
    Z_pdf = pdf(grid_points, G_gmm_means, G_gmm_covs)
    Z_pdf /= Z_pdf.max()
    Z_pdf = Z_pdf.reshape(X.shape)  # ←  make it 2-D
    Z_func = []
    for g in G.elements:
        Z_func.append(np.asarray([a.reshape(X.shape) for a in G_fn(np.einsum("ab,nb->na", rep_R2(g), grid_points))]))
    reg_rep = isotypic_decomp_rep(G.regular_representation)
    Z_func_iso = np.einsum("ab,gbcd->gacd", reg_rep.change_of_basis_inv, Z_func)

    # # --- build the figure ---------------------------------------------------------
    # fig = go.Figure()
    # # Set plotly style to "plotly_white"
    # fig.update_layout(template="plotly_white")
    # dz = Z_pdf.max() * 10
    # # 2. marginal P(z)
    # fig.add_trace(
    #     go.Surface(
    #         x=X,
    #         y=Y,
    #         z=np.zeros_like(X),  # <-- stays at z = 0
    #         surfacecolor=Z_pdf,  # colour by the marginal
    #         colorscale="Blues",
    #         showscale=False,
    #         opacity=1.0,  # looks like a 2-D map
    #         name="P(z) contour-map",
    #     )
    # )
    #
    # # 3. orbit of your function
    # for k, Zk in enumerate(Z_funcs, start=1):
    #     fig.add_trace(
    #         go.Surface(
    #             x=X,
    #             y=Y,
    #             # z=Zk + dz * (k) + 5,
    #             z=0 * Zk + dz * (10 * k) + 5,
    #             surfacecolor=Zk,  # colour by the marginal
    #             colorscale="BuPu",  # if k % 2 else "Cividis",
    #             showscale=False,
    #             opacity=0.8,
    #             name=f"f^{k}(x)",
    #             contours=dict(
    #                 x=dict(show=True, usecolormap=False, color="rgba(10,10,10,0.2)", width=1),
    #                 y=dict(show=True, usecolormap=False, color="rgba(10,10,10,0.2)", width=1),
    #             ),
    #         )
    #     )
    #
    # # --- cosmetics ---------------------------------------------------------------
    # fig.update_layout(
    #     scene=dict(xaxis_title="x₁", yaxis_title="x₂", aspectratio=dict(x=1, y=1)),
    #     title="Stacked 3-D view of P(z) and group-orbit functions",
    # )
    #
    # fig.show()

    # --- plotting parameters -------------------------------------------------
    import matplotlib.pyplot as plt
    from matplotlib import cm

    cell_cm = 2.5  # physical size of *one* square (cm)
    nrows, ncols = int(G.order()), int(G.order()) * 2
    figsize_in = (cell_cm / 2.54 * ncols, cell_cm / 2.54 * nrows)

    cmap_funcs = cm.get_cmap("bone_r")
    cmap_pdf = cm.get_cmap("Blues")

    vmin, vmax = np.min(Z_func), np.max(Z_func)

    # helper to draw the three reflection axes and keep them INSIDE the extent
    def draw_d3_axes(ax, **kw):
        radius = 1.5  # = max(|x|,|y|) of your grid
        for θ in (0, 120, 240):
            t = np.deg2rad(θ + 60)
            ax.plot([0, radius * np.cos(t)], [0, radius * np.sin(t)], **kw, clip_on=False)

    # -------------------------------------------------------------------------
    # FIGURE 1 : six orbit functions in a 1×6 strip
    fig1, axes = plt.subplots(nrows, ncols, figsize=figsize_in, sharex=True, sharey=True)

    for i, g in enumerate(G.elements):
        for j, g in enumerate(G.elements):
            ax = axes[i, j + G.order()]
            Z = Z_func_iso[i][j]
            cmap = "Reds" if j == 0 else ("PuRd" if j == 1 else "GnBu")
            ax.imshow(
                Z,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            draw_d3_axes(ax, color="k", lw=0.8, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(x.min(), x.max())  # make sure nothing expands
            ax.set_ylim(y.min(), y.max())

    for i, g in enumerate(G.elements):
        for j, g in enumerate(G.elements):
            ax = axes[i, j]
            Z = Z_func[i][j]
            ax.imshow(
                Z,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="Greys",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            draw_d3_axes(ax, color="k", lw=0.8, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(x.min(), x.max())  # make sure nothing expands
            ax.set_ylim(y.min(), y.max())

    # make the figure border coincide with the image block
    fig1.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig1.tight_layout(pad=0)

    fig1.savefig("D3_fn_action.png", dpi=300)
    # -------------------------------------------------------------------------
    # ── FIGURE 2 : G-invariant PDF as a contour map ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(cell_cm * 4 / 2.54, cell_cm * 4 / 2.54))  # ~10 cm square

    # pick a dense set of levels for a smooth fill; adjust as you like
    levels = np.linspace(Z_pdf.min(), Z_pdf.max(), 15)

    # filled contours (vector graphics)
    cf = ax2.contourf(X, Y, Z_pdf, levels=levels, cmap=cmap_pdf, antialiased=True)

    # optional: thin isolines every Nth level for extra definition
    # ax2.contour(X, Y, Z_pdf, levels=levels[::10], colors="k", linewidths=0.3)

    draw_d3_axes(ax2, color="k", lw=1.0, alpha=0.8)

    # keep the panel perfectly square & border-to-border
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())
    ax2.set_aspect("equal", adjustable="box")

    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout(pad=0)

    # higher dpi because this one is “bigger”
    fig2.savefig("D3-inv-pdf.png", dpi=600, bbox_inches="tight")

    plt.show()
