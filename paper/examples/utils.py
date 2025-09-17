"""Shared utilities for paper experiment notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

__all__ = [
    "LiveLossPlotter",
    "dataframe_to_markdown",
    "log_metrics",
    "plot_sample_efficiency",
]


def dataframe_to_markdown(
    df: pd.DataFrame,
    *,
    index: bool = False,
    float_formats: dict[str, str] | None = None,
    default_float_fmt: str = ".2f",
) -> str:
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


class LiveLossPlotter:
    """Minimal live-updating plot for training/validation loss in notebooks."""

    def __init__(self, title: str = "Training", ylabel: str = "Loss", figsize=(4.5, 2.4), plot_freq: int = 1):
        self.title = title
        self.ylabel = ylabel
        self.plot_freq = max(1, int(plot_freq))

        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def _plot(self) -> None:
        self.ax.cla()
        self.ax.plot(self.epochs, self.train_losses, label="train", color="tab:blue")

        if len(self.val_losses) > 0:
            import numpy as _np

            epochs = _np.asarray(self.epochs)
            vals = _np.asarray(self.val_losses, dtype=float)
            mask = ~_np.isnan(vals)
            if mask.any():
                self.ax.plot(epochs[mask], vals[mask], label="val", color="tab:orange", marker="o", ms=3)

        self.ax.set_title(self.title)
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="best", fontsize=8)

    def update(self, epoch: int, train_loss: float | None = None, val_loss: float | None = None, force: bool = False) -> None:
        epoch = int(epoch)
        if self.epochs and epoch == self.epochs[-1]:
            if train_loss is not None:
                self.train_losses[-1] = float(train_loss)
            if val_loss is not None:
                self.val_losses[-1] = float(val_loss)
        else:
            self.epochs.append(epoch)
            self.train_losses.append(np.nan if train_loss is None else float(train_loss))
            self.val_losses.append(np.nan if val_loss is None else float(val_loss))

        should_redraw = force or (epoch % self.plot_freq == 0)
        if not should_redraw:
            return

        self._plot()
        try:
            from IPython.display import clear_output, display

            clear_output(wait=True)
            display(self.fig)
        except Exception:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self) -> None:
        self._plot()
        try:
            from IPython.display import display

            display(self.fig)
        except Exception:
            pass
        plt.close(self.fig)


def log_metrics(
    *,
    metrics_dir: Path | str,
    sample_size: int,
    seed: int,
    rows: list[dict],
    prefix: str = "experiment_metrics",
):
    """Persist run-level summary metrics to a uniquely named CSV file."""

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df_summary = pd.DataFrame(rows)
    csv_path = metrics_dir / f"{prefix}_N={sample_size}_seed={seed}.csv"
    df_summary.to_csv(csv_path, index=False)
    return csv_path, df_summary


def plot_sample_efficiency(
    *,
    metrics_dir: Path | str,
    metric: str,
    prefix: str = "experiment_metrics",
    output_path: Path | str | None = None,
    model_order: list[str] | None = None,
    model_colors: dict[str, str] | None = None,
    ax=None,
    figsize=(6, 4),
    show: bool = False,
):
    """Plot sample-efficiency curves by averaging metrics across seeds for each sample size."""

    metrics_dir = Path(metrics_dir)
    csv_paths = sorted(metrics_dir.glob(f"{prefix}_N=*_*"))
    if not csv_paths:
        raise FileNotFoundError(f"No metrics files found in {metrics_dir} matching prefix '{prefix}'.")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if metric not in df.columns:
            raise KeyError(f"Metric '{metric}' not present in file {path}.")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["sample_size"] = data["sample_size"].astype(int)
    data["seed"] = data["seed"].astype(int)

    agg = (
        data.groupby(["model", "sample_size"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.rename(columns={"mean": "value_mean", "std": "value_std", "count": "value_count"}, inplace=True)
    agg["value_sem"] = agg["value_std"].fillna(0.0) / np.sqrt(np.maximum(agg["value_count"], 1))

    if model_order is None:
        model_order = sorted(agg["model"].unique())

    if model_colors is None:
        color_cycle = plt.cm.tab10(range(len(model_order)))
        model_colors = {model: color_cycle[i] for i, model in enumerate(model_order)}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for model in model_order:
        df_model = agg[agg["model"] == model].sort_values("sample_size")
        if df_model.empty:
            continue
        ax.plot(
            df_model["sample_size"],
            df_model["value_mean"],
            marker="o",
            label=model,
            color=model_colors.get(model),
            linewidth=2,
        )
        ax.fill_between(
            df_model["sample_size"],
            df_model["value_mean"] - df_model["value_sem"],
            df_model["value_mean"] + df_model["value_sem"],
            color=model_colors.get(model),
            alpha=0.15,
        )

    ax.set_xlabel("Number of training samples (N)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Sample efficiency ({metric})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(fig)

    return fig, ax, agg
