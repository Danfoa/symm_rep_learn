# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 11/08/25
import numpy as np
from matplotlib import pyplot as plt


def scatter_with_density(x, y, ax=None, bins=200, cmap="Blues", alpha_points=0.15, s=6):
    """Scatter plot with 2D histogram density in the background.

    Args:
        x: 1D array-like of shape (N,)
        y: 1D array-like of shape (N,)
        ax: Matplotlib axis
        bins: number of bins per dimension for the 2D histogram
        cmap: colormap name
        alpha_points: alpha for foreground points
        s: marker size
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # 2D histogram as background density
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    counts = counts.T  # for correct orientation in imshow
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(counts, extent=extent, origin="lower", cmap=cmap, aspect="auto", alpha=0.6)

    # foreground scatter
    ax.scatter(x, y, s=s, c="k", alpha=alpha_points)
    return fig, ax


def plot_conditional_expectation(
    x,
    y,
    x_grid,
    y_true,
    y_pred,
    ax=None,
    label_pred="NCP E[Y|X]",
    label_true="True E[Y|X]",
    color_pred="crimson",
    color_true="black",
):
    """Plot conditional expectation curves on top of a background data density.

    Args:
        x, y: arrays of observations used for the background
        x_grid: 1D array of x locations for curves
        y_true: 1D array of true conditional expectation at x_grid
        y_pred: 1D array of predicted conditional expectation at x_grid
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    scatter_with_density(x, y, ax=ax)
    ax.plot(x_grid, y_true, color=color_true, lw=2.0, label=label_true)
    ax.plot(x_grid, y_pred, color=color_pred, lw=2.0, ls="--", label=label_pred)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return fig, ax


def layout(fig, suptitle=None, tight=True):
    if suptitle:
        fig.suptitle(suptitle)
    if tight:
        fig.tight_layout()
    return fig


# --- cCDF utilities -----------------------------------------------------------------------------


def _to_1d(a):
    """Return a flattened 1D numpy array from numpy/torch/list input (handles (m,1))."""
    try:
        import torch  # lazy import to avoid hard dependency when unused

        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(a)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.squeeze().ravel()


def empirical_cdf_at(y_values, support):
    """Empirical CDF evaluated at `support` from the sample `y_values`.

    Args:
        y_values: 1D array-like of samples (e.g., standardized train targets)
        support: 1D array-like of support points where to evaluate the CDF

    Returns:
        cdf: 1D numpy array of size len(support)
    """
    y = _to_1d(y_values)
    s = _to_1d(support)
    y_sorted = np.sort(y)
    return np.searchsorted(y_sorted, s, side="right") / max(1, y_sorted.size)


def plot_support_vlines(ax, support, color="lightgray", alpha=0.35, lw=0.6, round_decimals=12):
    """Draw vertical lines at unique support positions to visualize discretization."""
    s = _to_1d(support)
    s_unique = np.unique(np.round(s, decimals=round_decimals))
    ax.vlines(s_unique, ymin=0, ymax=1, colors=color, alpha=alpha, lw=lw)
    return s_unique


def plot_marginal_cdf_on_support(
    support,
    model_marginal_cdf,
    y_train,
    ax=None,
    label_model="NCP marginal CDF (internal)",
    label_emp="Empirical CDF (train)",
    color_model="tab:orange",
    color_emp="tab:blue",
    markersize_model=3,
    markersize_emp=2.5,
    show_vlines=True,
    vlines_kwargs=None,
):
    """Plot marginal CDF (internal) and empirical CDF evaluated on the same support.

    - Accepts train data and the discretization support.
    - Shows markers at each support point and (optionally) vertical lines for the grid.
    """
    s = _to_1d(support)
    c_model = _to_1d(model_marginal_cdf)
    assert s.size == c_model.size, "Support and model marginal CDF must have the same length"

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5))
    else:
        fig = ax.figure

    c_emp = empirical_cdf_at(y_train, s)

    # Markers only to emphasize 1-1 with support
    ax.plot(s, c_model, linestyle="None", marker="o", markersize=markersize_model, color=color_model, label=label_model)
    ax.plot(
        s, c_emp, linestyle="None", marker="x", markersize=markersize_emp, color=color_emp, alpha=0.9, label=label_emp
    )

    # Support vlines
    if show_vlines:
        vkw = {"color": "lightgray", "alpha": 0.35, "lw": 0.6}
        if vlines_kwargs:
            vkw.update(vlines_kwargs)
        plot_support_vlines(ax, s, **vkw)

    ax.set_xlabel("y (standardized)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    return fig, ax


def plot_conditional_cdf_on_support(
    support,
    ccdf_pred,
    *,
    ax=None,
    label_pred="Pred CCDF (NCP)",
    color_pred="crimson",
    lw_pred=2.0,
    y_train=None,
    model_marginal_cdf=None,
    label_marginal_emp="Empirical CDF (train)",
    label_marginal_model="Marginal CDF",
    color_marginal_emp="tab:blue",
    color_marginal_model="tab:orange",
    show_vlines=True,
    vlines_kwargs=None,
    gt_ccdf=None,
    label_gt="True CCDF (MC)",
    color_gt="black",
    lw_gt=1.7,
    ls_gt="--",
):
    """Plot a conditional CDF curve on the given discretization support and optionally:
    - overlay marginal CDF (empirical from y_train and/or provided model marginal),
    - overlay a ground-truth CCDF if available,
    - draw vertical lines at support locations.

    Args:
        support: 1D array-like support points (standardized y-space)
        ccdf_pred: 1D array-like predicted conditional CDF at `support`
        y_train: optional train y to compute empirical marginal CDF at `support`
        model_marginal_cdf: optional model marginal CDF at `support`
        gt_ccdf: optional ground-truth CCDF at `support`
    """
    s = _to_1d(support)
    c_pred = _to_1d(ccdf_pred)
    assert s.size == c_pred.size, "Support and predicted CCDF must have the same length"

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
    else:
        fig = ax.figure

    # Predicted conditional (line for readability)
    ax.plot(s, c_pred, label=label_pred, color=color_pred, lw=lw_pred)

    # Optional overlays
    if gt_ccdf is not None:
        c_gt = _to_1d(gt_ccdf)
        assert c_gt.size == s.size, "Support and gt CCDF must have the same length"
        ax.plot(s, c_gt, label=label_gt, color=color_gt, lw=lw_gt, ls=ls_gt)

    if y_train is not None:
        c_emp = empirical_cdf_at(y_train, s)
        ax.plot(s, c_emp, label=label_marginal_emp, color=color_marginal_emp, ls=":", lw=1.6)

    if model_marginal_cdf is not None:
        c_m = _to_1d(model_marginal_cdf)
        assert c_m.size == s.size, "Support and model marginal CDF must have the same length"
        ax.plot(s, c_m, label=label_marginal_model, color=color_marginal_model, ls=":", lw=1.6)

    # Support vlines
    if show_vlines:
        vkw = {"color": "lightgray", "alpha": 0.35, "lw": 0.6}
        if vlines_kwargs:
            vkw.update(vlines_kwargs)
        plot_support_vlines(ax, s, **vkw)

    ax.set_xlabel("y (standardized)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    return fig, ax


def plot_ccdf_comparison_panel(
    support,
    ccdf_series,
    *,
    y_train=None,
    model_marginal_cdf=None,
    labels=None,
    colors=None,
    ncols=None,
    figsize=(8, 2.5),
    sharey=True,
    show_vlines=True,
):
    """Plot a panel of conditional CDFs for multiple x's on the same support.

    Args:
        support: 1D support array
        ccdf_series: list of 1D arrays (one per x) or a 2D array shape (k, m)
        y_train: optional train y to overlay empirical marginal CDF
        model_marginal_cdf: optional model marginal CDF at support
        labels: list of titles/labels per series
        colors: list of colors per series
        ncols: number of subplot columns (defaults to len(series))
    """
    s = _to_1d(support)
    C = np.asarray([_to_1d(c) for c in ccdf_series])
    if C.ndim == 1:
        C = C[None, :]
    k, m = C.shape
    assert m == s.size, "Each ccdf curve must match support length"

    if ncols is None:
        ncols = k
    nrows = int(np.ceil(k / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey, squeeze=False)
    axs = axs.ravel()

    for i in range(k):
        lab = labels[i] if labels and i < len(labels) else f"Series {i}"
        col = colors[i] if colors and i < len(colors) else None
        ax = axs[i]
        plot_conditional_cdf_on_support(
            s,
            C[i],
            ax=ax,
            label_pred="Pred CCDF (NCP)",
            color_pred=col or "crimson",
            y_train=y_train,
            model_marginal_cdf=model_marginal_cdf,
            show_vlines=show_vlines,
        )
        ax.set_title(lab)

    # Hide any unused subplots
    for j in range(k, nrows * ncols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    return fig, axs[:k]


def plot_expectations_with_quantiles(
    x_train,
    y_train,
    x_grid,
    expectations,
    *,
    fig=None,
    ax=None,
    add_background=True,
    background_kwargs=None,
    true_quantiles=None,
    est_quantiles=None,
    true_label="True PI",
    true_color="green",
    true_alpha=0.15,
    est_alpha=0.18,
    figsize=(6, 3),
    legend=True,
):
    """Plot expectation curves and optional quantile regions over a data-density background.

    Args:
        x_train, y_train: arrays for background density (standardized space).
        x_grid: 1D array of x locations corresponding to curves.
        expectations: dict[label -> 1D array] mapping each label to its E[Y|X] over `x_grid`.
        fig, ax: optional Matplotlib figure/axes to draw on (for incremental updates).
        add_background: if True, draw the scatter+density background.
        background_kwargs: optional kwargs forwarded to `scatter_with_density`.
        true_quantiles: optional tuple (q_lo_true, q_hi_true), each 1D array over `x_grid`.
        est_quantiles: optional dict[label -> (q_lo, q_hi)] or a single tuple (q_lo, q_hi).
        true_label, true_color, true_alpha: styling for the true quantile band.
        est_alpha: alpha for estimated quantile bands.
        figsize: figure size if `fig`/`ax` not provided.
        legend: whether to add a legend.

    Returns:
        (fig, ax)
    """
    # Normalize arrays
    Xg = _to_1d(x_grid)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if add_background and x_train is not None and y_train is not None:
        bkw = {"bins": 200, "cmap": "Blues", "alpha_points": 0.15, "s": 6}
        if background_kwargs:
            bkw.update(background_kwargs)
        scatter_with_density(_to_1d(x_train), _to_1d(y_train), ax=ax, **bkw)

    # True quantile band first (so lines are on top)
    if true_quantiles is not None:
        qlo_t, qhi_t = true_quantiles
        qlo_t = _to_1d(qlo_t)
        qhi_t = _to_1d(qhi_t)
        ax.fill_between(Xg, qlo_t, qhi_t, color=true_color, alpha=true_alpha, label=true_label)

    # Estimated quantile bands (one or many)
    if est_quantiles is not None:
        if isinstance(est_quantiles, dict):
            for lbl, (qlo, qhi) in est_quantiles.items():
                qlo = _to_1d(qlo)
                qhi = _to_1d(qhi)
                ax.fill_between(Xg, qlo, qhi, alpha=est_alpha, label=lbl)
        else:
            # assume tuple
            qlo, qhi = est_quantiles
            qlo = _to_1d(qlo)
            qhi = _to_1d(qhi)
            ax.fill_between(Xg, qlo, qhi, alpha=est_alpha, label="Estimated PI")

    # Plot expectations
    for lbl, ycurve in expectations.items():
        ax.plot(Xg, _to_1d(ycurve), lw=2.0, label=lbl)

    ax.set_xlabel("X (standardized)")
    ax.set_ylabel("Y (standardized)")
    if legend:
        ax.legend()
    return fig, ax


# --- Minimal live training plot utilities -------------------------------------------------------


class LiveLossPlotter:
    """Minimal live-updating plot for training/validation loss in notebooks.

    Usage:
        plotter = LiveLossPlotter(title="Model training", plot_freq=5)
        for epoch in range(E):
            # after training epoch
            plotter.update(epoch, train_loss=tr)
            # at validation check
            plotter.update(epoch, train_loss=tr, val_loss=vl)
    """

    def __init__(self, title="Training", ylabel="Loss", figsize=(4.5, 2.4), plot_freq=1):
        self.title = title
        self.ylabel = ylabel
        self.plot_freq = max(1, int(plot_freq))

        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def _plot(self):
        self.ax.cla()
        # Train curve (all points)
        self.ax.plot(self.epochs, self.train_losses, label="train", color="tab:blue")
        # Validation curve only at available points (mask NaNs so line connects)
        if len(self.val_losses) > 0:
            import numpy as _np

            e = _np.asarray(self.epochs)
            v = _np.asarray(self.val_losses, dtype=float)
            m = ~_np.isnan(v)
            if m.any():
                self.ax.plot(e[m], v[m], label="val", color="tab:orange", marker="o", ms=3)
        self.ax.set_title(self.title)
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="best", fontsize=8)

    def update(self, epoch, train_loss=None, val_loss=None, force=False):
        epoch = int(epoch)
        # If called twice for same epoch, update last entry instead of appending
        if self.epochs and epoch == self.epochs[-1]:
            if train_loss is not None:
                self.train_losses[-1] = float(train_loss)
            if val_loss is not None:
                self.val_losses[-1] = float(val_loss)
        else:
            self.epochs.append(epoch)
            self.train_losses.append(np.nan if train_loss is None else float(train_loss))
            self.val_losses.append(np.nan if val_loss is None else float(val_loss))

        # Redraw only at plot_freq steps or when a val point is provided, or when forced
        should_redraw = force or (epoch % self.plot_freq == 0) or (val_loss is not None)
        if not should_redraw:
            return

        self._plot()
        try:
            from IPython.display import clear_output, display

            clear_output(wait=True)
            display(self.fig)
        except Exception:
            # Fallback if not in IPython
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        # Final redraw to ensure the last state is visible before closing
        self._plot()
        try:
            from IPython.display import display

            display(self.fig)
        except Exception:
            pass
        plt.close(self.fig)
