from __future__ import annotations

from copy import copy
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from paper.examples.utils import training_curve_plot_path


def _figure_to_rgb(
    fig,
    *,
    target_width_px: float | None = None,
    target_height_px: float | None = None,
    min_dpi: float = 200.0,
) -> np.ndarray:
    width_in, height_in = fig.get_size_inches()
    dpi_candidates = [float(fig.dpi), float(min_dpi)]
    if target_width_px is not None and width_in > 0.0:
        dpi_candidates.append(float(target_width_px) / float(width_in))
    if target_height_px is not None and height_in > 0.0:
        dpi_candidates.append(float(target_height_px) / float(height_in))
    fig.set_dpi(max(dpi_candidates))
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buffer.reshape(height, width, 4)[..., :3].copy()


def plot_jointgrid_triptych(
    items,
    *,
    figsize=(15.5, 5.5),
    title_fontsize: int = 11,
    figure_dpi: int = 200,
    raster_min_dpi: int = 220,
):
    fig, axes = plt.subplots(1, len(items), figsize=figsize, dpi=figure_dpi)
    axes = np.atleast_1d(axes)
    target_width_px = figure_dpi * figsize[0] / max(len(items), 1)
    target_height_px = figure_dpi * figsize[1]

    for ax, (title, grid_or_factory) in zip(axes, items):
        grid = grid_or_factory() if callable(grid_or_factory) else grid_or_factory
        source_fig = grid.fig if hasattr(grid, "fig") else grid
        image = _figure_to_rgb(
            source_fig,
            target_width_px=target_width_px,
            target_height_px=target_height_px,
            min_dpi=raster_min_dpi,
        )
        ax.imshow(image, interpolation="none")
        ax.set_title(title, fontsize=title_fontsize)
        ax.axis("off")
        plt.close(source_fig)

    fig.tight_layout()
    return fig, axes


def plot_pmd_error_panel(
    error_grids: dict[str, dict],
    *,
    model_order: list[str] | None = None,
    ncols: int = 2,
    figsize=(11.5, 8.2),
    levels: int = 21,
    cmap=None,
    colorbar_label: str = r"$|\kappa(x, y) - \widehat{\kappa}(x, y)|$",
):
    if not error_grids:
        raise ValueError("error_grids must not be empty")

    if model_order is None:
        model_order = list(error_grids.keys())

    n_models = len(model_order)
    nrows = ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    if cmap is None:
        cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = copy(cmap)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    all_errors = np.concatenate(
        [
            np.asarray(error_grids[name]["abs_pmd_err"], dtype=float)[
                np.isfinite(np.asarray(error_grids[name]["abs_pmd_err"], dtype=float))
            ]
            for name in model_order
        ],
        axis=0,
    )
    vmax = float(np.nanmax(all_errors))
    if vmax == 0.0:
        vmax = 1e-6
    mesh = None

    for ax, model_name in zip(axes_flat, model_order):
        grid = error_grids[model_name]
        mesh = ax.pcolormesh(
            grid["x_edges"],
            grid["y_edges"],
            np.ma.masked_invalid(np.asarray(grid["abs_pmd_err"], dtype=float)),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            shading="flat",
        )
        ax.set_xlim(grid["x_limits"])
        ax.set_ylim(grid["y_limits"])
        ax.set_box_aspect(1)
        ax.set_title(model_name)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    for ax in axes_flat[n_models:]:
        ax.axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes_flat[:n_models], fraction=0.025, pad=0.02)
        cbar.set_label(colorbar_label)

    return fig, axes, {"vmax": vmax}


def plot_saved_training_curves_grid(
    checkpoint_paths_by_size: dict[int, dict[str, Path | str]],
    *,
    size_order: list[int],
    model_order: list[str],
    row_labels: dict[int, str] | None = None,
    figsize=(14.8, 11.5),
    missing_text: str = "curve unavailable",
    title_fontsize: int = 10,
    row_label_fontsize: int = 10,
):
    if not checkpoint_paths_by_size:
        raise ValueError("checkpoint_paths_by_size must not be empty")

    fig, axes = plt.subplots(
        len(size_order),
        len(model_order),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_idx, sample_size in enumerate(size_order):
        for col_idx, model_name in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            checkpoint_path = checkpoint_paths_by_size[sample_size][model_name]
            curve_path = training_curve_plot_path(checkpoint_path)
            if curve_path.exists():
                ax.imshow(plt.imread(curve_path))
            else:
                ax.text(0.5, 0.5, missing_text, ha="center", va="center", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(model_name, fontsize=title_fontsize)
            if col_idx == 0 and row_labels is not None:
                ax.set_ylabel(row_labels[sample_size], fontsize=row_label_fontsize)

    fig.tight_layout()
    return fig, axes


def plot_sample_efficiency_grid(
    results_df: pd.DataFrame,
    *,
    model_order: list[str],
    train_size_col: str = "train_samples",
    joint_metric_col: str = "joint_expected_abs_pmd_error",
    product_metric_col: str = "product_expected_abs_pmd_error",
    panels: tuple[str, ...] = ("joint", "product"),
    figsize=(10.8, 3.8),
):
    panel_options = {
        "joint": (joint_metric_col, r"Evaluation under $(X, Y) \sim P_{XY}$"),
        "product": (product_metric_col, r"Evaluation under $(X, Y) \sim P_X \otimes P_Y$"),
    }
    if not panels:
        raise ValueError("panels must contain at least one entry")
    invalid_panels = [panel for panel in panels if panel not in panel_options]
    if invalid_panels:
        raise ValueError(f"unsupported panels: {invalid_panels}")

    metric_cols = [panel_options[panel][0] for panel in panels]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    colors = sns.color_palette("colorblind", n_colors=len(model_order))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    style_map = {
        model_name: {"color": colors[idx], "marker": markers[idx % len(markers)]}
        for idx, model_name in enumerate(model_order)
    }

    ymax = float(
        np.nanmax(
            results_df[metric_cols].to_numpy(dtype=float),
        )
    )
    ymax = 1.05 * ymax if np.isfinite(ymax) and ymax > 0.0 else 1.0
    xticks = sorted(results_df[train_size_col].astype(int).unique().tolist())

    panel_specs = [
        (ax, *panel_options[panel])
        for ax, panel in zip(axes, panels)
    ]
    for ax, metric_col, title in panel_specs:
        for model_name in model_order:
            df_model = results_df[results_df["model"] == model_name].sort_values(train_size_col)
            style = style_map[model_name]
            ax.plot(
                df_model[train_size_col],
                df_model[metric_col],
                marker=style["marker"],
                linewidth=2,
                color=style["color"],
                label=model_name,
            )
        ax.set_title(title)
        ax.set_xlabel("Number of training samples")
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(xticks)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\mathbb{E}\!\left[\,|\kappa(X, Y) - \widehat{\kappa}(X, Y)|\,\right]$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_order), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig, axes


def plot_reference_density_pair(
    reference_grid: dict[str, np.ndarray | float],
    *,
    figsize=(8.2, 4.2),
    levels: int = 21,
    cmap="Blues",
):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    entries = [
        (r"Joint density $p_{XY}(x, y)$", "joint_density"),
        (r"Product density $p_X(x)\,p_Y(y)$", "product_density"),
    ]

    for ax, (title, key) in zip(axes, entries):
        ax.contourf(
            reference_grid["X_grid"],
            reference_grid["Y_grid"],
            reference_grid[key],
            levels=int(levels),
            cmap=cmap,
        )
        ax.set_xlim(reference_grid["x_limits"])
        ax.set_ylim(reference_grid["y_limits"])
        ax.set_box_aspect(1)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")

    axes[0].set_ylabel(r"$y$")
    fig.tight_layout()
    return fig, axes


def _pack_square_axes_grid(
    fig,
    axes,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float = 0.0,
    hspace: float = 0.0,
):
    axes = np.asarray(axes)
    nrows, ncols = axes.shape
    fig_width, fig_height = fig.get_size_inches()
    available_width = right - left
    available_height = top - bottom
    available_width_in = available_width * fig_width
    available_height_in = available_height * fig_height
    wspace_in = wspace * fig_width
    hspace_in = hspace * fig_height
    panel_width_in = (available_width_in - wspace_in * (ncols - 1)) / ncols
    panel_height_in = (available_height_in - hspace_in * (nrows - 1)) / nrows
    panel_size_in = min(panel_width_in, panel_height_in)
    panel_width = panel_size_in / fig_width
    panel_height = panel_size_in / fig_height

    total_width = ncols * panel_width + wspace * (ncols - 1)
    total_height = nrows * panel_height + hspace * (nrows - 1)
    x_start = left + 0.5 * (available_width - total_width)
    y_start = bottom + 0.5 * (available_height - total_height)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            x0 = x_start + col_idx * (panel_width + wspace)
            y0 = y_start + (nrows - 1 - row_idx) * (panel_height + hspace)
            axes[row_idx, col_idx].set_position([x0, y0, panel_width, panel_height])


def plot_pmd_error_grid(
    error_grids_by_size: dict[int, dict[str, dict]],
    *,
    size_order: list[int],
    model_order: list[str],
    row_labels: dict[int, str] | None = None,
    figsize=(14.8, 11.8),
    levels: int = 21,
    cmap=None,
    colorbar_label: str = r"$|\kappa(x, y) - \widehat{\kappa}(x, y)|$",
):
    if not error_grids_by_size:
        raise ValueError("error_grids_by_size must not be empty")

    if cmap is None:
        cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = copy(cmap)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    fig, axes = plt.subplots(
        len(size_order),
        len(model_order),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    all_errors = np.concatenate(
        [
            np.asarray(error_grids_by_size[sample_size][model_name]["abs_pmd_err"], dtype=float)[
                np.isfinite(np.asarray(error_grids_by_size[sample_size][model_name]["abs_pmd_err"], dtype=float))
            ]
            for sample_size in size_order
            for model_name in model_order
        ],
        axis=0,
    )
    vmax = float(np.nanmax(all_errors))
    if vmax == 0.0:
        vmax = 1e-6

    mesh = None
    for row_idx, sample_size in enumerate(size_order):
        for col_idx, model_name in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            grid = error_grids_by_size[sample_size][model_name]
            masked_err = np.ma.masked_invalid(np.asarray(grid["abs_pmd_err"], dtype=float))
            mesh = ax.pcolormesh(
                grid["x_edges"],
                grid["y_edges"],
                masked_err,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                shading="flat",
            )
            ax.set_xlim(grid["x_limits"])
            ax.set_ylim(grid["y_limits"])
            if row_idx == 0:
                ax.set_title(model_name)
            if row_idx == len(size_order) - 1:
                ax.set_xlabel(r"$x$")
            if col_idx == 0:
                ylabel = r"$y$"
                if row_labels is not None:
                    ylabel = f"{row_labels[sample_size]}\n{ylabel}"
                ax.set_ylabel(ylabel)

    _pack_square_axes_grid(
        fig,
        axes,
        left=0.08,
        right=0.89,
        bottom=0.08,
        top=0.95,
        wspace=0.0,
        hspace=0.0,
    )
    if mesh is not None:
        cax = fig.add_axes([0.91, 0.12, 0.016, 0.76])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(colorbar_label)

    return fig, axes, {"vmax": vmax}


def plot_conditional_expectation_grid(
    reference_grid: dict[str, np.ndarray | float],
    panel_specs: list[dict],
    *,
    figsize=(12.4, 4.4),
    levels: int = 21,
    cmap="Blues",
):
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, len(panel_specs), wspace=0.18)
    legend_handles: dict[str, Line2D] = {}
    first_joint_ax = None

    for idx, spec in enumerate(panel_specs):
        inner = outer[0, idx].subgridspec(
            2,
            2,
            height_ratios=[1.0, 4.0],
            width_ratios=[4.0, 1.0],
            hspace=0.0,
            wspace=0.0,
        )
        ax_top = fig.add_subplot(inner[0, 0], sharex=first_joint_ax)
        ax_joint = fig.add_subplot(inner[1, 0], sharex=first_joint_ax, sharey=first_joint_ax)
        ax_right = fig.add_subplot(inner[1, 1], sharey=first_joint_ax)
        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")
        if first_joint_ax is None:
            first_joint_ax = ax_joint

        ax_joint.contourf(
            reference_grid["X_grid"],
            reference_grid["Y_grid"],
            reference_grid["joint_density"],
            levels=int(levels),
            cmap=cmap,
        )
        ax_joint.plot(
            spec["x_values"],
            spec["ground_truth"],
            color="black",
            linewidth=2.0,
            linestyle="--" if "prediction" in spec else "-",
        )
        legend_handles.setdefault(
            r"$\mathbb{E}[Y \mid X=x]$",
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label=r"$\mathbb{E}[Y \mid X=x]$"),
        )
        if "prediction" in spec:
            label = spec.get("prediction_label", "Prediction")
            color = spec.get("prediction_color", sns.color_palette("colorblind")[0])
            ax_joint.plot(
                spec["x_values"],
                spec["prediction"],
                color=color,
                linewidth=2.0,
            )
            legend_handles.setdefault(
                label,
                Line2D([0], [0], color=color, linewidth=2.0, label=label),
            )

        ax_top.fill_between(
            reference_grid["x_centers"],
            0.0,
            reference_grid["marginal_x"],
            color="lightblue",
            alpha=0.65,
        )
        ax_right.fill_betweenx(
            reference_grid["y_centers"],
            0.0,
            reference_grid["marginal_y"],
            color="lightblue",
            alpha=0.65,
        )

        ax_joint.set_xlim(reference_grid["x_limits"])
        ax_joint.set_ylim(reference_grid["y_limits"])
        ax_joint.set_box_aspect(1)
        ax_joint.set_title(spec["title"])
        ax_joint.set_xlabel(r"$x$")
        if idx == 0:
            ax_joint.set_ylabel(r"$y$")
        else:
            ax_joint.tick_params(labelleft=False)

        ax_top.set_xlim(reference_grid["x_limits"])
        ax_top.set_ylim(bottom=0.0)
        ax_top.axis("off")

        ax_right.set_ylim(reference_grid["y_limits"])
        ax_right.set_xlim(left=0.0)
        ax_right.axis("off")

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.03),
        )
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.90, wspace=0.18)
    return fig


def plot_pmd_surface_grid(
    pmd_grids_by_size: dict[int, dict[str, dict[str, np.ndarray | float]]],
    *,
    size_order: list[int],
    model_order: list[str],
    row_labels: dict[int, str] | None = None,
    true_title: str = r"True $\kappa(x, y)$",
    figsize=(17.5, 11.8),
    cmap=None,
    colorbar_label: str = r"$\kappa(x, y)$",
):
    if not pmd_grids_by_size:
        raise ValueError("pmd_grids_by_size must not be empty")

    if cmap is None:
        cmap = sns.color_palette("magma", as_cmap=True)

    ncols = 1 + len(model_order)
    fig, axes = plt.subplots(
        len(size_order),
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    values = np.concatenate(
        [
            np.asarray(pmd_grids_by_size[sample_size][model_name]["pmd_true"], dtype=float).reshape(-1)
            for sample_size in size_order
            for model_name in model_order
        ]
        + [
            np.asarray(pmd_grids_by_size[sample_size][model_name]["pmd_pred"], dtype=float).reshape(-1)
            for sample_size in size_order
            for model_name in model_order
        ]
    )
    finite_values = values[np.isfinite(values)]
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if vmin == vmax:
        vmax = vmin + 1e-6

    mesh = None
    for row_idx, sample_size in enumerate(size_order):
        reference_grid = pmd_grids_by_size[sample_size][model_order[0]]
        ax_true = axes[row_idx, 0]
        mesh = ax_true.pcolormesh(
            reference_grid["x_edges"],
            reference_grid["y_edges"],
            np.asarray(reference_grid["pmd_true"], dtype=float),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        ax_true.set_xlim(reference_grid["x_limits"])
        ax_true.set_ylim(reference_grid["y_limits"])
        if row_idx == 0:
            ax_true.set_title(true_title)
        if row_idx == len(size_order) - 1:
            ax_true.set_xlabel(r"$x$")
        ylabel = r"$y$"
        if row_labels is not None:
            ylabel = f"{row_labels[sample_size]}\n{ylabel}"
        ax_true.set_ylabel(ylabel)

        for col_idx, model_name in enumerate(model_order, start=1):
            ax = axes[row_idx, col_idx]
            grid = pmd_grids_by_size[sample_size][model_name]
            mesh = ax.pcolormesh(
                grid["x_edges"],
                grid["y_edges"],
                np.asarray(grid["pmd_pred"], dtype=float),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="flat",
            )
            ax.set_xlim(grid["x_limits"])
            ax.set_ylim(grid["y_limits"])
            if row_idx == 0:
                ax.set_title(model_name)
            if row_idx == len(size_order) - 1:
                ax.set_xlabel(r"$x$")

    _pack_square_axes_grid(
        fig,
        axes,
        left=0.08,
        right=0.89,
        bottom=0.08,
        top=0.95,
        wspace=0.0,
        hspace=0.0,
    )
    if mesh is not None:
        cax = fig.add_axes([0.91, 0.12, 0.016, 0.76])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(colorbar_label)
    return fig, axes, {"vmin": vmin, "vmax": vmax}
