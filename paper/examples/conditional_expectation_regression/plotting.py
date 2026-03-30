# Created  at 11/08/25
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from utils import dataframe_to_markdown, log_metrics


def scatter_with_density(x, y, ax=None, bins=200, cmap="Blues", alpha_points=0.10, s=6):
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
    # ax.plot(
    # s, c_emp, linestyle="None", marker="x", markersize=markersize_emp, color=color_emp, alpha=0.9, label=label_emp
    # )

    # Support vlines
    if show_vlines:
        vkw = {"color": "lightgray", "alpha": 0.25, "lw": 0.6}
        if vlines_kwargs:
            vkw.update(vlines_kwargs)
        plot_support_vlines(ax, s, **vkw)

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
    label_marginal_emp="cCDF GT",
    label_marginal_model="CDF",
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
        # ax.plot(s, c_emp, label=label_marginal_emp, color=color_marginal_emp, ls=":", lw=1.6)

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

    # ax.set_xlabel("y (standardized)")
    # ax.set_ylabel("CDF")
    # ax.legend(fontsize=8)
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
    expectations=None,
    *,
    fig=None,
    ax=None,
    add_background=True,
    background_kwargs=None,
    true_quantiles=None,
    est_quantiles=None,
    quantile_colors=None,
    exp_colors=None,
    true_label="True CI",
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
        expectations: optional dict[label -> 1D array] mapping each label to its E[Y|X] over `x_grid`.
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
        bkw = {"bins": 200, "cmap": "Blues", "s": 6}
        if background_kwargs:
            bkw.update(background_kwargs)
        scatter_with_density(_to_1d(x_train), _to_1d(y_train), ax=ax, alpha_points=0.05, **bkw)

    # True quantile band first (so lines are on top)
    if true_quantiles is not None:
        qlo_t, qhi_t = true_quantiles
        qlo_t = _to_1d(qlo_t)
        qhi_t = _to_1d(qhi_t)
        ax.fill_between(
            Xg, qlo_t, qhi_t, color=true_color, alpha=true_alpha, label=true_label, edgecolor=true_color, linewidth=1.0
        )

    # Estimated quantile bands (one or many)
    if est_quantiles is not None:
        if isinstance(est_quantiles, tuple):
            est_quantiles = {"CI": est_quantiles}  # convert single tuple to dict with label "CI"

        if quantile_colors is None:
            quantile_colors = plt.cm.Set1(np.linspace(0, 1, len(est_quantiles)))
        for (lbl, (qlo, qhi)), color in zip(est_quantiles.items(), quantile_colors):
            qlo = _to_1d(qlo)
            qhi = _to_1d(qhi)
            ax.fill_between(Xg, qlo, qhi, alpha=est_alpha, label=lbl, color=color, edgecolor=color, linewidth=1.0)
            # Plot lines on edges of the CI
            ax.plot(Xg, qlo, color=color, lw=1.0, alpha=0.5)
            ax.plot(Xg, qhi, color=color, lw=1.0, alpha=0.5)

    if expectations:
        if exp_colors is None:
            exp_colors = plt.cm.Set1(np.linspace(0, 1, len(expectations)))
        for i, (lbl, ycurve) in enumerate(expectations.items()):
            color = exp_colors[i] if i < len(exp_colors) else None
            ax.plot(Xg, _to_1d(ycurve), lw=1.0, label=lbl, color=color)

    ax.set_xlabel(r"$\mathcal{X}$", fontsize=10)
    ax.set_ylabel(r"$\mathcal{Y}$", fontsize=10)
    if legend:
        ax.legend(fontsize=8, framealpha=1.0, loc="upper right")
    return fig, ax


# --- Reporting and summary utilities -----------------------------------------------------------


def dataframe_to_markdown(df, index=False, float_formats=None, default_float_fmt=".2f"):
    """Render a DataFrame as a GitHub-flavoured Markdown table."""

    if float_formats is None:
        float_formats = {}

    df_fmt = df.copy()

    if index:
        df_fmt = df_fmt.reset_index()

    numeric_cols = df_fmt.select_dtypes(include="number").columns
    for col in numeric_cols:
        fmt = float_formats.get(col, default_float_fmt)
        df_fmt[col] = df_fmt[col].map(lambda v, f=fmt: f"{v:{f}}")

    df_fmt = df_fmt.astype(str)
    header = " | ".join(df_fmt.columns)
    separator = " | ".join(["---"] * len(df_fmt.columns))
    rows = [" | ".join(row) for row in df_fmt.to_numpy().tolist()]
    lines = [f"| {header} |", f"| {separator} |"] + [f"| {row} |" for row in rows]
    return "\n".join(lines)


def plot_condexp_metrics_panels(
    df_results,
    mae_by_model=None,
    output_path=None,
    model_colors=None,
    figsize=(12, 4),
    show=False,
):
    """Plot CI-size and coverage-error comparison panels."""
    title_fs = 9
    label_fs = 10
    tick_fs = 8
    legend_fs = 8

    if model_colors is None:
        model_colors = {"NCP": "tab:green", "NCPaug": "tab:purple", "eNCP": "tab:blue", "MLP": "tab:orange"}

    coverage_suffix = " Coverage (%)"
    models = [
        col[: -len(coverage_suffix)]
        for col in df_results.columns
        if col.endswith(coverage_suffix) and col != "Desired Coverage (%)"
    ]
    if not models:
        raise ValueError("No model coverage columns found in `df_results`.")

    ncols = 2
    width_ratios = [1.1, 1.1]
    marker_cycle = ["o-", "s-", "^-", "d-", "x-"]

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = np.atleast_1d(axes)

    if "$\\alpha$" in df_results.columns:
        alpha_col = "$\\alpha$"
    elif "Actual $\\alpha$" in df_results.columns:
        alpha_col = "Actual $\\alpha$"
    elif "Actual alpha" in df_results.columns:
        alpha_col = "Actual alpha"
    else:
        alpha_col = "Alpha"
    if alpha_col not in df_results.columns:
        raise KeyError("`df_results` must contain one of {'$\\\\alpha$', 'Actual $\\\\alpha$', 'Actual alpha', 'Alpha'}.")

    for i, model in enumerate(models):
        col = f"{model} CI Size"
        if col not in df_results:
            continue
        style = marker_cycle[i % len(marker_cycle)]
        axes[0].plot(
            df_results[alpha_col],
            df_results[col],
            style,
            label=model,
            markersize=6,
            color=model_colors.get(model),
        )
    axes[0].set_xlabel(r"$\alpha$ (desired CI coverage = $100(1-\alpha)\%$)", fontsize=label_fs)
    axes[0].set_ylabel("CI Size", fontsize=label_fs)
    axes[0].set_title("CI Size", fontsize=title_fs)
    axes[0].legend(fontsize=legend_fs)
    axes[0].tick_params(axis="both", labelsize=tick_fs)
    axes[0].grid(True, alpha=0.3)

    for i, model in enumerate(models):
        col = f"{model} Coverage Error"
        if col not in df_results:
            continue
        style = marker_cycle[i % len(marker_cycle)]
        axes[1].plot(
            df_results[alpha_col],
            df_results[col],
            style,
            label=model,
            markersize=6,
            color=model_colors.get(model),
        )
    axes[1].set_xlabel(r"$\alpha$ (desired CI coverage = $100(1-\alpha)\%$)", fontsize=label_fs)
    axes[1].set_ylabel("Coverage Error (%)", fontsize=label_fs)
    axes[1].set_title("Coverage Error", fontsize=title_fs)
    axes[1].legend(fontsize=legend_fs)
    axes[1].tick_params(axis="both", labelsize=tick_fs)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(fig)

    return fig, axes


def log_condexp_summary_metrics(
    metrics_dir,
    sample_size,
    seed,
    df_results,
    mae_by_model=None,
    prefix="condexp_summary_metrics",
):
    """Persist run-level summary metrics to a uniquely named CSV file."""

    if mae_by_model is None:
        mae_by_model = {}

    coverage_error_suffix = " Coverage Error"
    coverage_models = [
        col[: -len(coverage_error_suffix)] for col in df_results.columns if col.endswith(coverage_error_suffix)
    ]
    model_order = list(dict.fromkeys(list(mae_by_model.keys()) + coverage_models))

    rows = []
    for model in model_order:
        mae = mae_by_model.get(model, np.nan)
        rows.append({
            "sample_size": int(sample_size),
            "seed": int(seed),
            "model": model,
            "mae": float(mae),
            "coverage_error": float(df_results[f"{model} Coverage Error"].mean())
            if f"{model} Coverage Error" in df_results
            else np.nan,
            "ci_size": float(df_results[f"{model} CI Size"].mean()) if f"{model} CI Size" in df_results else np.nan,
        })

    return log_metrics(
        metrics_dir=metrics_dir,
        sample_size=sample_size,
        seed=seed,
        rows=rows,
        prefix=prefix,
    )


def log_condexp_alpha_metrics(
    metrics_dir,
    sample_size,
    seed,
    df_results,
    prefix="condexp_metrics",
    train_size=None,
    coverage_eval_n=None,
):
    """Persist per-alpha uncertainty metrics for each model to CSV.

    Expected columns in ``df_results``:
    - ``$\alpha$`` (or legacy ``Actual $\alpha$`` / ``Actual alpha`` / ``Alpha``)
    - ``Desired Coverage (%)``
    - ``<model> Coverage (%)``
    - ``<model> Coverage Error``
    - ``<model> CI Size``
    """

    if train_size is None:
        train_size = int(sample_size)

    coverage_suffix = " Coverage (%)"
    models = [
        col[: -len(coverage_suffix)]
        for col in df_results.columns
        if col.endswith(coverage_suffix) and col != "Desired Coverage (%)"
    ]
    if not models:
        raise ValueError("No model coverage columns found in `df_results`.")

    if "$\\alpha$" in df_results.columns:
        alpha_col = "$\\alpha$"
    elif "Actual $\\alpha$" in df_results.columns:
        alpha_col = "Actual $\\alpha$"
    elif "Actual alpha" in df_results.columns:
        alpha_col = "Actual alpha"
    else:
        alpha_col = "Alpha"
    if alpha_col not in df_results.columns:
        raise KeyError("`df_results` must contain one of {'$\\\\alpha$', 'Actual $\\\\alpha$', 'Actual alpha', 'Alpha'}.")

    rows = []
    for _, row in df_results.iterrows():
        alpha = float(row[alpha_col])
        desired = float(row["Desired Coverage (%)"])
        for model in models:
            cov_col = f"{model} Coverage (%)"
            err_col = f"{model} Coverage Error"
            ci_col = f"{model} CI Size"

            rows.append(
                {
                    "sample_size": int(sample_size),
                    "train_size": int(train_size),
                    "seed": int(seed),
                    "alpha": alpha,
                    "desired_coverage_pct": desired,
                    "model": model,
                    "coverage_pct": float(row[cov_col]) if cov_col in df_results.columns else np.nan,
                    "coverage_error_pct": float(row[err_col]) if err_col in df_results.columns else np.nan,
                    "ci_size": float(row[ci_col]) if ci_col in df_results.columns else np.nan,
                    "coverage_eval_n": int(coverage_eval_n) if coverage_eval_n is not None else np.nan,
                }
            )

    return log_metrics(
        metrics_dir=metrics_dir,
        sample_size=sample_size,
        seed=seed,
        rows=rows,
        prefix=prefix,
    )
