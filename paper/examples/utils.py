"""Shared utilities for paper experiment notebooks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from symm_learning.stats import var_mean as symm_var_mean
from tqdm import tqdm

__all__ = [
    "LiveLossPlotter",
    "checkpoint_exists",
    "dataframe_to_markdown",
    "display_saved_training_curve",
    "fit_or_load_ncp_like",
    "make_condexp_1d_dataset",
    "make_condexp_1d_dataset_incorrect_conditional",
    "make_condexp_1d_dataset_incorrect_conditional_biased",
    "make_positive_bias_train_val_split",
    "load_checkpoint",
    "log_metrics",
    "plot_sample_efficiency",
    "plot_saved_training_curves_panel",
    "save_checkpoint",
    "save_training_curve_plot",
    "split_standardize_tensors",
    "split_x_side_masks",
    "training_curve_plot_path",
    "true_condexp_1d",
    "true_condexp_1d_incorrect_conditional_biased",
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

    def update(
        self, epoch: int, train_loss: float | None = None, val_loss: float | None = None, force: bool = False
    ) -> None:
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

    agg = data.groupby(["model", "sample_size"], as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
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


def _f_center_1d(x: torch.Tensor) -> torch.Tensor:
    """Deterministic center function used by the 1D conditional-expectation synthetic dataset."""

    return 0.5 * torch.cos(2.0 / 3.0 * math.pi * x) + 0.2 * torch.cos(8.0 / 3.0 * math.pi * x) + 0.25


def _exp_scale_param_1d(absx: torch.Tensor) -> torch.Tensor:
    """Exponential noise scale used in the skewed regime of the 1D synthetic dataset."""

    return 0.1 + 0.1 * (torch.cos(absx * 2.0) ** 2)


def make_condexp_1d_dataset(n: int = 20_000, seed: int = 10, x=None):
    """Generate the 1D synthetic dataset used in Appendix G.4-style experiments.

    The construction matches the previous `conditional_expectation_regression_1D.ipynb` dataset:
    - |x| <= 1: skewed regime with exponential noise
    - 1 < |x| <= 2: symmetric heteroscedastic regime
    - |x| > 2: bimodal heteroscedastic regime
    """

    rng = np.random.default_rng(int(seed))
    torch_gen = torch.Generator().manual_seed(int(seed))

    if x is None:
        x_np = rng.uniform(-3.0, 3.0, size=(int(n), 1)).astype(np.float32)
    else:
        if np.isscalar(x):
            x_np = np.full((int(n), 1), float(x), dtype=np.float32)
        else:
            x_arr = np.asarray(x, dtype=np.float32)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            if x_arr.shape[1] != 1:
                raise ValueError(f"x must be shape (n,) or (n,1), got {x_arr.shape}")
            if x_arr.shape[0] != int(n):
                n = int(x_arr.shape[0])
            x_np = x_arr

    x_t = torch.from_numpy(x_np)
    absx = torch.abs(x_t)

    sigma_sym = 0.04 + 0.05 * np.cos(6 * absx) ** 2 + 0.15 * torch.sin(9 * absx) ** 2
    sigma_bi = 0.03 + 0.1 * np.cos(5 * absx) ** 2 + 0.06 * np.cos(4 * absx) ** 2
    base_noise = torch.randn(x_t.shape, generator=torch_gen, dtype=torch.float32)
    scale_param = _exp_scale_param_1d(absx)
    u = torch.from_numpy(rng.uniform(0, 1, size=x_t.shape)).to(torch.float32)
    eps_exp = -scale_param * torch.log(u)
    eps_sym = base_noise * sigma_sym
    s = torch.from_numpy(rng.choice([-1.0, 1.0], size=(n, 1))).to(torch.float32)
    a = 0.6 * (absx - 2.0).clamp(min=0)
    eps_bi = torch.randn(x_t.shape, generator=torch_gen, dtype=torch.float32) * sigma_bi

    y_zone1 = _f_center_1d(x_t) + eps_exp
    y_zone2 = _f_center_1d(x_t) + eps_sym
    y_zone3 = s * a + eps_bi
    y_t = torch.where(absx <= 1.0, y_zone1, torch.where(absx <= 2.0, y_zone2, y_zone3))
    return x_t, y_t


def make_condexp_1d_dataset_incorrect_conditional(
    n: int = 20_000,
    seed: int = 10,
    x=None,
    c_rhs: float = 4.0,
):
    """Generate a 1D dataset with incorrect conditional symmetry on the right side.

    This perturbation keeps the marginal sampling of x unchanged and modifies only the conditional
    heteroscedastic-noise amplitude on the right half-space (x > 0) for:
    - symmetric regime: 1 < |x| <= 2
    - bimodal regime: |x| > 2

    The skewed exponential regime (|x| <= 1) is left unchanged.
    """

    rng = np.random.default_rng(int(seed))
    torch_gen = torch.Generator().manual_seed(int(seed))

    if x is None:
        x_np = rng.uniform(-3.0, 3.0, size=(int(n), 1)).astype(np.float32)
    else:
        if np.isscalar(x):
            x_np = np.full((int(n), 1), float(x), dtype=np.float32)
        else:
            x_arr = np.asarray(x, dtype=np.float32)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            if x_arr.shape[1] != 1:
                raise ValueError(f"x must be shape (n,) or (n,1), got {x_arr.shape}")
            if x_arr.shape[0] != int(n):
                n = int(x_arr.shape[0])
            x_np = x_arr

    x_t = torch.from_numpy(x_np)
    absx = torch.abs(x_t)

    sigma_sym = 0.04 + 0.05 * np.cos(6 * absx) ** 2 + 0.15 * torch.sin(9 * absx) ** 2
    sigma_bi = 0.03 + 0.1 * np.cos(5 * absx) ** 2 + 0.06 * np.cos(4 * absx) ** 2

    # Increase only right-side heteroscedastic noise in zones 2 and 3.
    rhs_mask = (x_t > 0.0) & (absx > 1.0)
    rhs_scale = torch.ones_like(x_t)
    rhs_scale[rhs_mask] = float(c_rhs)

    base_noise = torch.randn(x_t.shape, generator=torch_gen, dtype=torch.float32)
    scale_param = _exp_scale_param_1d(absx)
    u = torch.from_numpy(rng.uniform(0, 1, size=x_t.shape)).to(torch.float32)
    eps_exp = -scale_param * torch.log(u)  # unchanged in zone 1

    eps_sym = base_noise * sigma_sym * rhs_scale
    s = torch.from_numpy(rng.choice([-1.0, 1.0], size=(n, 1))).to(torch.float32)
    a = 0.6 * (absx - 2.0).clamp(min=0)
    eps_bi = torch.randn(x_t.shape, generator=torch_gen, dtype=torch.float32) * sigma_bi * rhs_scale

    y_zone1 = _f_center_1d(x_t) + eps_exp
    y_zone2 = _f_center_1d(x_t) + eps_sym
    y_zone3 = s * a + eps_bi
    y_t = torch.where(absx <= 1.0, y_zone1, torch.where(absx <= 2.0, y_zone2, y_zone3))
    return x_t, y_t


def make_condexp_1d_dataset_incorrect_conditional_biased(
    n: int = 20_000,
    seed: int = 10,
    x=None,
    slope_rhs: float = 1.0,
):
    """Generate a 1D dataset with a right-side mean-shift perturbation in y|x.

    The perturbation keeps x-sampling unchanged and adds a deterministic linear offset on:
    - symmetric regime: 1 < x <= 2
    - bimodal regime: x > 2

    Offset: y <- y + b * (x - 1), where b = slope_rhs.
    The skewed exponential regime (|x| <= 1) is unchanged.
    """

    rng = np.random.default_rng(int(seed))
    torch_gen = torch.Generator().manual_seed(int(seed))

    if x is None:
        x_np = rng.uniform(-3.0, 3.0, size=(int(n), 1)).astype(np.float32)
    else:
        if np.isscalar(x):
            x_np = np.full((int(n), 1), float(x), dtype=np.float32)
        else:
            x_arr = np.asarray(x, dtype=np.float32)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            if x_arr.shape[1] != 1:
                raise ValueError(f"x must be shape (n,) or (n,1), got {x_arr.shape}")
            if x_arr.shape[0] != int(n):
                n = int(x_arr.shape[0])
            x_np = x_arr

    x_t = torch.from_numpy(x_np)
    absx = torch.abs(x_t)

    sigma_sym = 0.04 + 0.05 * np.cos(6 * absx) ** 2 + 0.15 * torch.sin(9 * absx) ** 2
    sigma_bi = 0.03 + 0.1 * np.cos(5 * absx) ** 2 + 0.06 * np.cos(4 * absx) ** 2
    base_noise = torch.randn(x_t.shape, generator=torch_gen, dtype=torch.float32)
    scale_param = _exp_scale_param_1d(absx)
    u = torch.from_numpy(rng.uniform(0, 1, size=x_t.shape)).to(torch.float32)
    eps_exp = -scale_param * torch.log(u)
    eps_sym = base_noise * sigma_sym
    s = torch.from_numpy(rng.choice([-1.0, 1.0], size=(n, 1))).to(torch.float32)
    a = 0.6 * (absx - 2.0).clamp(min=0)
    eps_bi = torch.randn(x_t.shape, generator=torch_gen, dtype=torch.float32) * sigma_bi

    rhs_shift = float(slope_rhs) * torch.clamp(x_t - 1.0, min=0.0)

    y_zone1 = _f_center_1d(x_t) + eps_exp
    y_zone2 = _f_center_1d(x_t) + eps_sym + rhs_shift
    y_zone3 = s * a + eps_bi + rhs_shift
    y_t = torch.where(absx <= 1.0, y_zone1, torch.where(absx <= 2.0, y_zone2, y_zone3))
    return x_t, y_t


@torch.no_grad()
def true_condexp_1d(x) -> torch.Tensor:
    """True conditional expectation E[Y|X=x] for the shared 1D synthetic dataset."""

    x = torch.as_tensor(x, dtype=torch.float32)
    absx = torch.abs(x)
    expected_exp_noise = _exp_scale_param_1d(absx)
    zone1_exp = _f_center_1d(x) + expected_exp_noise
    zone2_exp = _f_center_1d(x)
    zone3_exp = torch.zeros_like(x)
    return torch.where(absx <= 1.0, zone1_exp, torch.where(absx <= 2.0, zone2_exp, zone3_exp))


@torch.no_grad()
def true_condexp_1d_incorrect_conditional_biased(x, slope_rhs: float = 1.0) -> torch.Tensor:
    """True E[Y|X=x] for the right-side mean-shift perturbed dataset."""

    x = torch.as_tensor(x, dtype=torch.float32)
    base = true_condexp_1d(x)
    rhs_shift = float(slope_rhs) * torch.clamp(x - 1.0, min=0.0)
    return base + rhs_shift


def split_standardize_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int | None = None,
    x_rep=None,
    y_rep=None,
):
    """Random split and standardization for tensor datasets.

    Returns a dictionary containing raw splits, standardized splits, and train statistics.
    """

    n = int(x.shape[0])
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("`train_ratio` must be in (0, 1).")
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("`val_ratio` must be in [0, 1].")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("`train_ratio + val_ratio` must be <= 1.")

    if seed is None:
        idx = torch.randperm(n)
    else:
        gen = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(n, generator=gen)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    idx_tr, idx_val, idx_te = idx[:train_end], idx[train_end:val_end], idx[val_end:]

    x_tr, y_tr = x[idx_tr], y[idx_tr]
    x_val, y_val = x[idx_val], y[idx_val]
    x_te, y_te = x[idx_te], y[idx_te]

    if x_rep is not None:
        x_var, x_mean_vec = symm_var_mean(x_tr, x_rep)
        x_mean = x_mean_vec.reshape(1, -1)
        x_std = torch.sqrt(torch.clamp(x_var, min=0.0)).reshape(1, -1) + 1e-8
    else:
        x_mean, x_std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-8

    if y_rep is not None:
        y_var, y_mean_vec = symm_var_mean(y_tr, y_rep)
        y_mean = y_mean_vec.reshape(1, -1)
        y_std = torch.sqrt(torch.clamp(y_var, min=0.0)).reshape(1, -1) + 1e-8
    else:
        y_mean, y_std = y_tr.mean(0, keepdim=True), y_tr.std(0, keepdim=True) + 1e-8

    x_tr_c = (x_tr - x_mean) / x_std
    y_tr_c = (y_tr - y_mean) / y_std
    x_val_c = (x_val - x_mean) / x_std
    y_val_c = (y_val - y_mean) / y_std
    x_te_c = (x_te - x_mean) / x_std
    y_te_c = (y_te - y_mean) / y_std

    return {
        "x_tr": x_tr,
        "y_tr": y_tr,
        "x_val": x_val,
        "y_val": y_val,
        "x_te": x_te,
        "y_te": y_te,
        "x_tr_c": x_tr_c,
        "y_tr_c": y_tr_c,
        "x_val_c": x_val_c,
        "y_val_c": y_val_c,
        "x_te_c": x_te_c,
        "y_te_c": y_te_c,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "idx_tr": idx_tr,
        "idx_val": idx_val,
        "idx_te": idx_te,
    }


def make_positive_bias_train_val_split(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    train_ratio: float = 0.85,
    seed: int | None = None,
    threshold: float = 0.0,
    x_rep=None,
    y_rep=None,
):
    """Create train/val splits from the positive side of X only (extrinsic marginal-bias setup)."""

    x_flat = x.reshape(-1)
    pos_mask = x_flat > float(threshold)
    x_pos, y_pos = x[pos_mask], y[pos_mask]
    if x_pos.shape[0] < 2:
        raise ValueError("Not enough positive-side samples to split train/val.")

    val_ratio = 1.0 - float(train_ratio)
    return split_standardize_tensors(
        x_pos,
        y_pos,
        train_ratio=float(train_ratio),
        val_ratio=val_ratio,
        seed=seed,
        x_rep=x_rep,
        y_rep=y_rep,
    )


def split_x_side_masks(x, *, threshold: float = 0.0):
    """Boolean masks for left/right/full regions in one-dimensional x."""

    x_flat = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    left = x_flat < float(threshold)
    right = ~left
    full = torch.ones_like(left, dtype=torch.bool)
    return {"left": left, "right": right, "full": full}


def training_curve_plot_path(checkpoint_path: Path | str) -> Path:
    """Return the companion PNG path for a model checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix(".training_curve.png")


def save_training_curve_plot(plotter: LiveLossPlotter | None, checkpoint_path: Path | str) -> Path | None:
    """Persist the latest live training-curve figure next to its checkpoint."""

    if plotter is None or not hasattr(plotter, "fig"):
        return None

    out_path = training_curve_plot_path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter._plot()
    except Exception:
        pass
    plotter.fig.savefig(out_path, dpi=440, bbox_inches="tight")
    return out_path


def display_saved_training_curve(checkpoint_path: Path | str, title: str, figsize=(4.6, 2.5)):
    """Display a saved training-curve PNG for a checkpoint."""

    img_path = training_curve_plot_path(checkpoint_path)
    fig, ax = plt.subplots(figsize=figsize)
    if img_path.exists():
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.text(
            0.5,
            0.5,
            "No saved training curve figure found.\nThis checkpoint predates curve export.",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(img_path, dpi=440, bbox_inches="tight")
        print(f"Saved placeholder training-curve figure to {img_path}")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig, ax


def save_checkpoint(
    model,
    optimizer,
    best_val_loss: float,
    epoch: int,
    checkpoint_path: Path | str,
    *,
    plotter: LiveLossPlotter | None = None,
    extra_state: dict | None = None,
) -> Path:
    """Save model checkpoint and current training-curve figure."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_val_loss": float(best_val_loss),
        "epoch": int(epoch),
    }
    if extra_state:
        checkpoint.update(extra_state)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    if plotter is not None:
        save_training_curve_plot(plotter, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path: Path | str, *, device: str | torch.device = "cpu"):
    """Load model and optimizer state from checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def checkpoint_exists(checkpoint_path: Path | str) -> bool:
    """Check whether a checkpoint file exists on disk."""

    return Path(checkpoint_path).exists()


@torch.no_grad()
def ncp_val_objective(model, dataloader, *, device: str | torch.device = "cpu"):
    """Compute validation objective and averaged metric dictionary for NCP/eNCP-style models."""

    metrics = {}
    model.eval()
    total, n = 0.0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        fx, hy = model(xb, yb)
        loss, batch_metrics = model.loss(fx, hy)
        total += float(loss.item())
        n += 1
        for key, value in batch_metrics.items():
            metrics.setdefault(key, []).append(value)

    for key, values in metrics.items():
        metrics[key] = float(np.mean(values))

    return total / max(1, n), metrics


def fit_or_load_ncp_like(
    *,
    model,
    train_loader,
    val_loader,
    optimizer,
    checkpoint_path: Path | str,
    device: str | torch.device = "cpu",
    train_epochs: int = 2500,
    check_every: int = 10,
    patience: int = 50,
    plot_freq: int = 100,
    desc: str = "Training",
    plot_title: str = "Training",
    val_metric: str = "||k(x,y) - k_r(x,y)||",
    checkpoint_meta: dict | None = None,
    show_curve_on_load: bool = True,
    enable_plots: bool = True,
):
    """Train an NCP/eNCP-like model with early stopping or load it from checkpoint.

    Returns:
        (best_state_dict_on_cpu, best_validation_objective)
    """

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(model, optimizer, checkpoint_path, device=device)
        best_val = float(checkpoint.get("best_val_loss", np.nan))
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"Loaded - best val objective: {best_val:.5f}")
        if show_curve_on_load and enable_plots:
            display_saved_training_curve(checkpoint_path, title=plot_title)
    else:
        print(f"{desc} from scratch...")
        best_val = float("inf")
        patience_counter = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        pbar = tqdm(range(train_epochs), desc=desc)
        plotter = LiveLossPlotter(title=plot_title, plot_freq=plot_freq) if enable_plots else None

        for epoch in pbar:
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                fx, hy = model(xb, yb)
                loss, metrics = model.loss(fx, hy)
                loss.backward()
                optimizer.step()

            if epoch % check_every == 0 or epoch == train_epochs - 1:
                vm, val_metrics = ncp_val_objective(model, val_loader, device=device)
                pbar.set_postfix(loss=float(loss.item()), val=vm)
                train_loss = float(metrics.get(val_metric, loss.item()))
                val_loss = float(val_metrics.get(val_metric, vm))
                if plotter is not None:
                    plotter.update(epoch, train_loss=train_loss, val_loss=val_loss)
                if vm < best_val:
                    best_val = vm
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    patience_counter = 0
                    save_checkpoint(
                        model,
                        optimizer,
                        best_val,
                        epoch,
                        checkpoint_path,
                        plotter=plotter,
                        extra_state=checkpoint_meta,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= int(patience):
                        print(f"Early stopping at epoch {epoch}")
                        break

        if plotter is not None:
            save_training_curve_plot(plotter, checkpoint_path)
            plotter.close()
        print(f"Best val objective: {best_val:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    return best_state, best_val


def plot_saved_training_curves_panel(
    curve_specs: list[tuple[str, Path | str]],
    *,
    figsize=(14.8, 3.0),
    missing_text: str = "curve unavailable",
    title_fontsize: int = 8,
):
    """Plot a horizontal panel of saved training curves for multiple checkpoints."""

    if not curve_specs:
        raise ValueError("`curve_specs` must contain at least one (label, checkpoint_path) pair.")

    fig, axes = plt.subplots(1, len(curve_specs), figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, (model_name, checkpoint_path) in zip(axes, curve_specs):
        curve_path = training_curve_plot_path(checkpoint_path)
        if curve_path.exists():
            ax.imshow(plt.imread(curve_path))
        else:
            ax.text(0.5, 0.5, missing_text, ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(model_name, fontsize=title_fontsize)
    fig.tight_layout()
    return fig, axes
