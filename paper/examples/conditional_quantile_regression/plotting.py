"""Plotting utilities for the conditional quantile regression experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def plot_quantiles(
    y_c,
    low_q,
    up_q,
    ax=None,
    label="pred",
    alpha=0.05,
    gt=True,
    quantile_color="red",
    color=None,
):
    """Plots the empirical quantiles and the predicted quantiles."""

    q_up_gt, q_lo_gt = np.quantile(y_c, 1 - (alpha / 2), axis=0), np.quantile(y_c, alpha / 2, axis=0)

    y0_low, y1_low = low_q[0], low_q[1]
    y0_up, y1_up = up_q[0], up_q[1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(y_c[:, 0], y_c[:, 1], alpha=0.05, color=color)
    ax.scatter(low_q[0], low_q[1], color=quantile_color)
    ax.scatter(up_q[0], up_q[1], color=quantile_color)

    pred_rect = Rectangle(
        (y0_low, y1_low),
        y0_up - y0_low,
        y1_up - y1_low,
        edgecolor=quantile_color,
        facecolor="none",
        alpha=0.8,
        linewidth=2.0,
    )
    ax.add_patch(pred_rect)
    ax.text(y0_low, y1_up * 1.05, label, color=quantile_color, verticalalignment="bottom", fontweight="bold")

    ax.scatter(q_lo_gt[0], q_lo_gt[1], color="black")
    ax.scatter(q_up_gt[0], q_up_gt[1], color="black")

    if gt:
        gt_rect = Rectangle(
            (q_lo_gt[0], q_lo_gt[1]),
            q_up_gt[0] - q_lo_gt[0],
            q_up_gt[1] - q_lo_gt[1],
            edgecolor="black",
            facecolor="none",
            alpha=0.8,
            linewidth=2.0,
        )
        ax.add_patch(gt_rect)
        ax.text(
            q_lo_gt[0],
            q_lo_gt[1] * 0.8,
            f"GT{int((1 - alpha) * 100):d}%",
            color="black",
            verticalalignment="top",
            horizontalalignment="left",
            fontweight="bold",
        )
    return fig, ax


def _ensure_output_path(output_path: Path | str | None):
    if output_path is None:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_marginal_summary(
    y_samples,
    x_samples,
    *,
    alpha=0.05,
    fig_size=(4, 3),
    output_path: Path | str | None = None,
):
    """Plot marginal quantiles of Y and histogram of X."""

    y_np = np.asarray(y_samples)
    x_np = np.asarray(x_samples).ravel()
    q_up, q_lo = np.quantile(y_np, 1 - (alpha / 2), axis=0), np.quantile(y_np, alpha / 2, axis=0)

    fig, axs = plt.subplots(ncols=2, figsize=(fig_size[0] * 2, fig_size[1]))
    plot_quantiles(y_np, q_lo, q_up, ax=axs[0], label="", gt=True)
    axs[0].set_title("Marginal of Y = (Y_0, Y_1)")
    axs[1].hist(x_np, bins=100, color="tab:blue", alpha=0.7)
    axs[1].set_title("Histogram of X")
    fig.tight_layout()

    path = _ensure_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=250)

    return fig, axs


def plot_conditional_samples(
    dataset_fn: Callable[[float], np.ndarray],
    x_values: Iterable[float],
    *,
    alpha=0.05,
    base_samples=None,
    colors=None,
    fig_size=(4, 3),
    output_path: Path | str | None = None,
):
    """Plot conditional quantile rectangles for several conditioning values."""

    x_values = list(x_values)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(x_values)))

    if base_samples is None:
        raise ValueError("base_samples must be provided to set the scatter background.")

    y_base = np.asarray(base_samples)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(y_base[:, 0], y_base[:, 1], alpha=0.01, color="lightsteelblue")

    for x_cond, color in zip(x_values, colors):
        y_cond = dataset_fn(float(x_cond))
        q_up, q_lo = np.quantile(y_cond, 1 - (alpha / 2), axis=0), np.quantile(y_cond, alpha / 2, axis=0)
        plot_quantiles(
            y_cond, q_lo, q_up, ax=ax, label=f"X={float(x_cond):.1f}", gt=False, quantile_color="black", color=color
        )

    fig.tight_layout()

    path = _ensure_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=250)

    return fig, ax


def plot_model_quantile_prediction(
    y_cond,
    q_lo,
    q_up,
    *,
    x_value,
    alpha,
    label="Model",
    fig_size=(4, 3),
    output_path: Path | str | None = None,
):
    """Plot model quantile predictions against empirical samples at a fixed x."""

    fig, ax = plt.subplots(figsize=fig_size)
    plot_quantiles(y_cond, q_lo, q_up, ax=ax, label=f"{label}-pred", alpha=alpha)
    fig.suptitle(rf"{label} prediction for X={float(x_value):.2f}")

    path = _ensure_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=250)

    return fig, ax


def plot_basis_functions_x(
    x_grid,
    fx,
    *,
    title="Learned basis functions for x",
    fig_size=(4, 3),
    palette: str | None = "tab20",
    color_offset: int = 0,
    output_path: Path | str | None = None,
):
    """Plot one-dimensional basis functions over x with optional Matplotlib palette."""

    fig, ax = plt.subplots(figsize=fig_size)
    x_np = np.asarray(x_grid)
    fx_np = np.asarray(fx)
    cmap = plt.get_cmap(palette) if palette is not None else None
    for idx in range(fx_np.shape[1]):
        color = None
        if cmap is not None:
            color = cmap((color_offset + idx) % cmap.N)
        ax.plot(x_np, fx_np[:, idx], label=f"f_{idx}(x)", color=color)
    ax.set_title(title)
    fig.tight_layout()

    path = _ensure_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=250)

    return fig, ax


def plot_basis_functions_y(
    hy_images,
    *,
    ncols=4,
    fig_size=(6, 6),
    suptitle="Learned basis functions for y",
    cmap="viridis",
    output_path: Path | str | None = None,
):
    """Plot 2D basis functions over a y-grid."""

    hy_np = np.asarray(hy_images)
    embedding_dim = hy_np.shape[-1]
    nrows = int(np.ceil(embedding_dim / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size[0], fig_size[1]), sharex=True, sharey=True)
    axs = np.atleast_1d(axs).reshape(nrows, ncols)

    for i in range(embedding_dim):
        ax = axs[i // ncols, i % ncols]
        cmap_i = cmap if isinstance(cmap, str) else cmap[i]
        ax.imshow(hy_np[:, :, i], cmap=cmap_i)
        ax.set_title(f"h_{i}(y)", fontsize=8)
        ax.axis("off")

    for j in range(embedding_dim, nrows * ncols):
        fig.delaxes(axs[j // ncols, j % ncols])

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = _ensure_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=250)

    return fig, axs


def plot_ccdf_diagnostics(
    support,
    ccdf,
    marginal_cdf,
    *,
    x_value,
    colors=("green", "purple"),
    fig_size=(12, 3),
    output_path: Path | str | None = None,
):
    """Plot CCDF and marginal CDF diagnostics for y0 and y1 at a fixed x."""

    fig, axs = plt.subplots(1, 3, figsize=(fig_size[0], fig_size[1]))

    axs[0].axis("off")
    for idx, color in enumerate(colors):
        axs_idx = axs[idx + 1]
        axs_idx.vlines(
            support[:, idx],
            marginal_cdf[:, idx].min(),
            marginal_cdf[:, idx].max(),
            color="lightgray",
            alpha=0.2,
            lw=0.6,
        )
        axs_idx.plot(support[:, idx], ccdf[:, idx], label=rf"$CCDF_{{y_{idx}}}$", color=color)
        axs_idx.plot(support[:, idx], marginal_cdf[:, idx], label=rf"$CDF_{{y_{idx}}}$", color=color, linestyle="--")
        axs_idx.set_title(rf"Pred CCDF $y_{idx}$ given $X={float(x_value):.2f}$", fontsize=8)
        axs_idx.legend()

    fig.tight_layout()

    path = _ensure_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=250)

    return fig, axs


def plot_quantile_comparison_grid(
    x_values: Iterable[float],
    y_samples_fn: Callable[[float], np.ndarray],
    predictors: Dict[str, Callable[[float], Tuple[np.ndarray, np.ndarray, str]]],
    *,
    alpha,
    support,
    ccdf_fns: Dict[str, Callable[[float], np.ndarray]],
    marginal_cdf,
    background=None,
    fig_size=(4, 3),
    output_path: Path | str | None = None,
):
    """Plot quantile comparison and CCDF diagnostics for multiple conditioning points."""

    x_values = list(x_values)
    n_points = len(x_values)
    fig, axs = plt.subplots(n_points, 3, figsize=(fig_size[0] * 3, fig_size[1] * n_points))
    axs = np.atleast_1d(axs).reshape(n_points, 3)

    background_np = None if background is None else np.asarray(background)

    for i, x_cond in enumerate(x_values):
        y_samples = y_samples_fn(float(x_cond))
        ax_quant = axs[i, 0]

        if background_np is not None:
            ax_quant.scatter(background_np[:, 0], background_np[:, 1], alpha=0.01, color="lightsteelblue")

        for name, predictor in predictors.items():
            q_lo, q_up, color = predictor(float(x_cond))
            plot_quantiles(y_samples, q_lo, q_up, ax=ax_quant, label=name, alpha=alpha, quantile_color=color)
        ax_quant.set_title(f"X={float(x_cond):.2f}")

        ccdf_pred = ccdf_fns["ncp"](float(x_cond))
        eccdf_pred = ccdf_fns["encp"](float(x_cond))

        axs[i, 1].vlines(
            support[:, 0], marginal_cdf[:, 0].min(), marginal_cdf[:, 0].max(), color="lightgray", alpha=0.2, lw=0.6
        )
        axs[i, 1].plot(support[:, 0], ccdf_pred[:, 0], label="NCP", color="red")
        axs[i, 1].plot(support[:, 0], eccdf_pred[:, 0], label="eNCP", color="purple")
        axs[i, 1].plot(support[:, 0], marginal_cdf[:, 0], label="CDF", color="green")
        axs[i, 1].set_title(rf"Pred CCDF $y_0$ | X={float(x_cond):.2f}")

        axs[i, 2].vlines(
            support[:, 1], marginal_cdf[:, 1].min(), marginal_cdf[:, 1].max(), color="lightgray", alpha=0.2, lw=0.6
        )
        axs[i, 2].plot(support[:, 1], ccdf_pred[:, 1], label="NCP", color="red")
        axs[i, 2].plot(support[:, 1], eccdf_pred[:, 1], label="eNCP", color="purple")
        axs[i, 2].plot(support[:, 1], marginal_cdf[:, 1], label="CDF", color="green")
        axs[i, 2].set_title(rf"Pred CCDF $y_1$ | X={float(x_cond):.2f}")

        if i == 0:
            axs[i, 0].legend()
            axs[i, 1].legend()
            axs[i, 2].legend()

    fig.tight_layout()

    path = _ensure_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=300)

    return fig, axs


__all__ = [
    "plot_basis_functions_x",
    "plot_basis_functions_y",
    "plot_ccdf_diagnostics",
    "plot_conditional_samples",
    "plot_marginal_summary",
    "plot_model_quantile_prediction",
    "plot_quantile_comparison_grid",
    "plot_quantiles",
]
