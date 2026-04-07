from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import escnn
import numpy as np
import torch
from escnn.group import directsum
from matplotlib import pyplot as plt
from symm_learning.models import MLP, eMLP, iMLP
from symm_learning.stats import var_mean
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from paper.examples.utils import (
    LiveLossPlotter,
    checkpoint_exists,
    display_saved_training_curve,
    load_checkpoint,
    save_checkpoint,
    save_training_curve_plot,
)
from paper.experiments.symmetricGMM.symmGMM import SymmGaussianMixture
from symm_rep_learn.models.density_ratio_fitting import DRF, InvDRF
from symm_rep_learn.models.neural_conditional_probability.encp import ENCP
from symm_rep_learn.models.neural_conditional_probability.ncp import NCP

VAL_METRIC = "||k(x,y) - k_r(x,y)||"


@dataclass
class C2ScalarSetup:
    G: escnn.group.Group
    rep_x: escnn.group.Representation
    rep_y: escnn.group.Representation
    gmm: SymmGaussianMixture


@dataclass
class DatasetBundle:
    x: torch.Tensor
    y: torch.Tensor
    x_c: torch.Tensor
    y_c: torch.Tensor
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    x_var: torch.Tensor
    y_var: torch.Tensor
    x_std: torch.Tensor
    y_std: torch.Tensor
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_train_c: torch.Tensor
    y_train_c: torch.Tensor
    x_val_c: torch.Tensor
    y_val_c: torch.Tensor
    x_test_c: torch.Tensor
    y_test_c: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    pairwise_train_loader: DataLoader
    pairwise_val_loader: DataLoader
    pairwise_test_loader: DataLoader


@dataclass
class PlotDomain:
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]


class SilentLossPlotter(LiveLossPlotter):
    """Training-curve recorder that never emits live notebook output."""

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

        if force or (epoch % self.plot_freq == 0):
            self._plot()
            self.fig.canvas.draw()

    def close(self) -> None:
        self._plot()
        plt.close(self.fig)


def build_c2_scalar_gmm(
    *,
    n_kernels: int,
    means_max_norm: float,
    sampling_seed: int,
    gmm_seed: int,
) -> C2ScalarSetup:
    G = escnn.group.CyclicGroup(2)
    rep_x = G.representations["irrep_1"]
    rep_y = G.representations["irrep_1"]
    gmm = SymmGaussianMixture(
        n_kernels=n_kernels,
        rep_X=rep_x,
        rep_Y=rep_y,
        mean_max_norm=means_max_norm,
        sampling_seed=sampling_seed,
        gmm_seed=gmm_seed,
    )
    return C2ScalarSetup(G=G, rep_x=rep_x, rep_y=rep_y, gmm=gmm)


def build_dataset_bundle(
    *,
    gmm: SymmGaussianMixture,
    rep_x,
    rep_y,
    n_total_samples: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
    pairwise_batch_size: int,
) -> DatasetBundle:
    test_ratio = 1.0 - train_ratio - val_ratio
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    x_np, y_np = gmm.simulate(n_samples=int(n_total_samples))
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()

    x_var, x_mean = var_mean(x, rep_x)
    y_var, y_mean = var_mean(y, rep_y)
    x_mean = x_mean.to(dtype=torch.float32)
    y_mean = y_mean.to(dtype=torch.float32)
    x_var = x_var.to(dtype=torch.float32)
    y_var = y_var.to(dtype=torch.float32)
    x_std = torch.sqrt(x_var)
    y_std = torch.sqrt(y_var)

    x_c = (x - x_mean) / x_std
    y_c = (y - y_mean) / y_std

    n_train = int(train_ratio * n_total_samples)
    n_val = int(val_ratio * n_total_samples)
    n_test = int(n_total_samples) - n_train - n_val

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[-n_test:], y[-n_test:]

    x_train_c, y_train_c = x_c[:n_train], y_c[:n_train]
    x_val_c, y_val_c = x_c[n_train : n_train + n_val], y_c[n_train : n_train + n_val]
    x_test_c, y_test_c = x_c[-n_test:], y_c[-n_test:]

    train_dataset = TensorDataset(x_train_c, y_train_c)
    val_dataset = TensorDataset(x_val_c, y_val_c)
    test_dataset = TensorDataset(x_test_c, y_test_c)

    pairwise_train_dataset = TensorDataset(x_train_c, y_train_c)
    pairwise_val_dataset = TensorDataset(x_val_c, y_val_c)
    pairwise_test_dataset = TensorDataset(x_test_c, y_test_c)

    return DatasetBundle(
        x=x,
        y=y,
        x_c=x_c,
        y_c=y_c,
        x_mean=x_mean,
        y_mean=y_mean,
        x_var=x_var,
        y_var=y_var,
        x_std=x_std,
        y_std=y_std,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        x_train_c=x_train_c,
        y_train_c=y_train_c,
        x_val_c=x_val_c,
        y_val_c=y_val_c,
        x_test_c=x_test_c,
        y_test_c=y_test_c,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        pairwise_train_loader=DataLoader(pairwise_train_dataset, batch_size=pairwise_batch_size, shuffle=True),
        pairwise_val_loader=DataLoader(pairwise_val_dataset, batch_size=pairwise_batch_size, shuffle=False),
        pairwise_test_loader=DataLoader(pairwise_test_dataset, batch_size=pairwise_batch_size, shuffle=False),
    )


def checkpoint_tag(
    *,
    n_kernels: int,
    means_max_norm: float,
    n_total_samples: int,
    seed: int,
) -> str:
    return f"nk={n_kernels}_norm={means_max_norm:.2f}_N={n_total_samples}_seed={seed}".replace(".", "p")


def checkpoint_path_for_model(
    *,
    checkpoint_dir: Path | str,
    model_name: str,
    n_kernels: int,
    means_max_norm: float,
    n_total_samples: int,
    seed: int,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    tag = checkpoint_tag(
        n_kernels=n_kernels,
        means_max_norm=means_max_norm,
        n_total_samples=n_total_samples,
        seed=seed,
    )
    return checkpoint_dir / f"{model_name.lower()}_{tag}.pth"


def build_checkpoint_paths_by_size(
    *,
    checkpoint_dir: Path | str,
    model_order: list[str],
    sample_sizes: list[int],
    n_kernels: int,
    means_max_norm: float,
    seed: int,
) -> dict[int, dict[str, Path]]:
    return {
        int(n_total_samples): {
            model_name: checkpoint_path_for_model(
                checkpoint_dir=checkpoint_dir,
                model_name=model_name,
                n_kernels=n_kernels,
                means_max_norm=means_max_norm,
                n_total_samples=int(n_total_samples),
                seed=seed,
            )
            for model_name in model_order
        }
        for n_total_samples in sample_sizes
    }


def draw_joint_and_product_samples(
    *,
    gmm: SymmGaussianMixture,
    n_joint: int,
    n_product: int,
) -> dict[str, np.ndarray]:
    x_joint, y_joint = gmm.simulate(n_samples=int(n_joint))
    x_product, _ = gmm.simulate(n_samples=int(n_product))
    _, y_product = gmm.simulate(n_samples=int(n_product))
    return {
        "x_joint": x_joint,
        "y_joint": y_joint,
        "x_product": x_product,
        "y_product": y_product,
    }


def _mesh_inputs(domain: PlotDomain) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_grid, Y_grid = np.meshgrid(domain.x_centers, domain.y_centers)
    X_input = np.column_stack([X_grid.ravel()])
    Y_input = np.column_stack([Y_grid.ravel()])
    return X_grid, Y_grid, X_input, Y_input


def _axis_bounds(
    values: np.ndarray,
    *,
    support_quantile: float,
    use_extrema: bool = False,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot estimate plotting support from an empty sample.")

    if use_extrema:
        lower = float(np.min(values))
        upper = float(np.max(values))
    else:
        support_quantile = float(support_quantile)
        tail_mass = max(0.0, 1.0 - support_quantile)
        lower = float(np.quantile(values, 0.5 * tail_mass))
        upper = float(np.quantile(values, 1.0 - 0.5 * tail_mass))

    if not np.isfinite(lower) or not np.isfinite(upper):
        lower = float(np.min(values))
        upper = float(np.max(values))
    if lower == upper:
        pad = 1.0 if lower == 0.0 else 0.1 * abs(lower)
        lower -= pad
        upper += pad
    return lower, upper


def build_product_support_domain_from_samples(
    *,
    x_product: np.ndarray,
    y_product: np.ndarray,
    grid_size: int = 180,
    support_quantile: float = 0.95,
    use_extrema: bool = False,
) -> PlotDomain:
    x_lower, x_upper = _axis_bounds(
        x_product,
        support_quantile=support_quantile,
        use_extrema=use_extrema,
    )
    y_lower, y_upper = _axis_bounds(
        y_product,
        support_quantile=support_quantile,
        use_extrema=use_extrema,
    )

    x_edges = np.linspace(x_lower, x_upper, int(grid_size) + 1)
    y_edges = np.linspace(y_lower, y_upper, int(grid_size) + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return PlotDomain(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        x_limits=(float(x_edges[0]), float(x_edges[-1])),
        y_limits=(float(y_edges[0]), float(y_edges[-1])),
    )


def build_product_support_domain(
    *,
    gmm: SymmGaussianMixture,
    n_samples: int,
    grid_size: int = 180,
    support_quantile: float = 0.95,
    use_extrema: bool = False,
) -> PlotDomain:
    product_samples = draw_joint_and_product_samples(gmm=gmm, n_joint=1, n_product=n_samples)
    return build_product_support_domain_from_samples(
        x_product=product_samples["x_product"],
        y_product=product_samples["y_product"],
        grid_size=grid_size,
        support_quantile=support_quantile,
        use_extrema=use_extrema,
    )


def _activation_from_name(name: str) -> torch.nn.Module:
    activation = getattr(torch.nn, name, None)
    if activation is None:
        raise ValueError(f"Unknown torch activation '{name}'")
    return activation()


def _latent_representation(group: escnn.group.Group, embedding_dim: int):
    reg_rep = group.regular_representation
    return directsum([reg_rep] * max(1, ceil(int(embedding_dim) / reg_rep.size)))


def build_pmd_model(
    model_name: str,
    *,
    rep_x,
    rep_y,
    embedding_dim: int,
    hidden_units: int,
    hidden_layers: int,
    activation: str,
    orth_reg: float,
    centering_reg: float,
    momentum: float,
) -> torch.nn.Module:
    act = _activation_from_name(activation)
    hidden = [int(hidden_units)] * int(hidden_layers)
    model_key = model_name.lower()

    if model_key == "ncp":
        embedding_x = MLP(in_dim=rep_x.size, out_dim=embedding_dim, hidden_units=hidden, activation=act, bias=False)
        embedding_y = MLP(in_dim=rep_y.size, out_dim=embedding_dim, hidden_units=hidden, activation=act, bias=False)
        return NCP(
            embedding_x=embedding_x,
            embedding_y=embedding_y,
            embedding_dim_x=embedding_dim,
            embedding_dim_y=embedding_dim,
            orth_reg=orth_reg,
            centering_reg=centering_reg,
            momentum=momentum,
        )

    if model_key == "encp":
        lat_rep = _latent_representation(rep_x.group, embedding_dim)
        embedding_x = eMLP(
            in_rep=rep_x,
            out_rep=lat_rep,
            hidden_units=hidden,
            activation=act,
            bias=False,
        )
        embedding_y = eMLP(
            in_rep=rep_y,
            out_rep=lat_rep,
            hidden_units=hidden,
            activation=act,
            bias=False,
        )
        return ENCP(
            embedding_x=embedding_x,
            embedding_y=embedding_y,
            orth_reg=orth_reg,
            centering_reg=centering_reg,
            momentum=momentum,
        )

    if model_key == "drf":
        drf_hidden = [int(hidden_units) * 2] * max(0, int(hidden_layers) - 1) + [int(embedding_dim)]
        embedding = MLP(
            in_dim=rep_x.size + rep_y.size,
            out_dim=1,
            hidden_units=drf_hidden,
            activation=act,
            bias=False,
        )
        return DRF(embedding=embedding, gamma=orth_reg)

    if model_key == "idrf":
        xy_rep = directsum([rep_x, rep_y])
        embedding = iMLP(
            in_rep=xy_rep,
            out_dim=1,
            hidden_units=hidden,
            activation=act,
            bias=False,
        )
        return InvDRF(embedding=embedding, gamma=orth_reg)

    raise ValueError(f"Unsupported PMD model '{model_name}'")


def _compute_loss_and_metrics(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor):
    out = model(xb, yb)
    if isinstance(out, tuple):
        loss, metrics = model.loss(*out)
    elif isinstance(out, dict):
        loss, metrics = model.loss(**out)
    else:
        loss, metrics = model.loss(out)
    return loss, metrics


@torch.no_grad()
def pmd_val_objective(model, dataloader, *, device: str | torch.device = "cpu"):
    metrics = {}
    model.eval()
    total, n = 0.0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        loss, batch_metrics = _compute_loss_and_metrics(model, xb, yb)
        total += float(loss.item())
        n += 1
        for key, value in batch_metrics.items():
            metrics.setdefault(key, []).append(float(value))

    for key, values in metrics.items():
        metrics[key] = float(np.mean(values))

    return total / max(1, n), metrics


def fit_or_load_pmd_model(
    *,
    model,
    train_loader,
    val_loader,
    optimizer,
    checkpoint_path: Path | str,
    device: str | torch.device = "cpu",
    train_epochs: int = 250,
    check_every: int = 5,
    patience: int = 20,
    accumulation_steps: int = 1,
    plot_freq: int = 50,
    desc: str = "Training",
    plot_title: str = "Training",
    val_metric: str = VAL_METRIC,
    checkpoint_meta: dict | None = None,
    show_curve_on_load: bool = False,
    enable_plots: bool = False,
    store_curve: bool = True,
    load_checkpoint_if_available: bool = True,
    save_checkpoint_on_improve: bool = True,
    force_retrain: bool = False,
):
    checkpoint_path = Path(checkpoint_path)

    if load_checkpoint_if_available and checkpoint_exists(checkpoint_path) and not force_retrain:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(model, optimizer, checkpoint_path, device=device)
        best_val = float(checkpoint.get("best_val_loss", np.nan))
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"Loaded - best val objective: {best_val:.5f}")
        if show_curve_on_load:
            display_saved_training_curve(checkpoint_path, title=plot_title)
    else:
        best_val = float("inf")
        patience_counter = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        plotter = None
        if store_curve or enable_plots:
            plotter_cls = LiveLossPlotter if enable_plots else SilentLossPlotter
            plotter = plotter_cls(title=plot_title, plot_freq=plot_freq)
        progress = tqdm(range(int(train_epochs)), desc=desc)
        accumulation_steps = max(1, int(accumulation_steps))

        for epoch in progress:
            model.train()
            optimizer.zero_grad()
            num_batches = len(train_loader)
            for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
                xb, yb = xb.to(device), yb.to(device)
                loss, train_metrics = _compute_loss_and_metrics(model, xb, yb)
                (loss / accumulation_steps).backward()
                if batch_idx % accumulation_steps == 0 or batch_idx == num_batches:
                    optimizer.step()
                    optimizer.zero_grad()

            if epoch % int(check_every) == 0 or epoch == int(train_epochs) - 1:
                vm, val_metrics = pmd_val_objective(model, val_loader, device=device)
                progress.set_postfix(loss=float(loss.item()), val=vm)
                train_loss = float(train_metrics.get(val_metric, loss.item()))
                val_loss = float(val_metrics.get(val_metric, vm))
                if plotter is not None:
                    plotter.update(epoch, train_loss=train_loss, val_loss=val_loss)
                if vm < best_val:
                    best_val = vm
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    patience_counter = 0
                    if save_checkpoint_on_improve:
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

        if plotter is not None and save_checkpoint_on_improve:
            save_training_curve_plot(plotter, checkpoint_path)
        if plotter is not None:
            plotter.close()
        print(f"Best val objective: {best_val:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    return best_state, best_val


@torch.no_grad()
def predict_pmd(
    *,
    model: torch.nn.Module,
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> np.ndarray:
    device = next(model.parameters()).device
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)

    x_mean = x_mean.to(device=device, dtype=torch.float32)
    y_mean = y_mean.to(device=device, dtype=torch.float32)
    x_std = x_std.to(device=device, dtype=torch.float32)
    y_std = y_std.to(device=device, dtype=torch.float32)

    x_c = (x_t - x_mean) / x_std
    y_c = (y_t - y_mean) / y_std
    pred = model.pointwise_mutual_dependency(x_c, y_c)
    return pred.detach().cpu().numpy()


@torch.no_grad()
def estimate_expected_pmd_error(
    *,
    gmm: SymmGaussianMixture,
    model: torch.nn.Module,
    x_mean: torch.Tensor,
    y_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_std: torch.Tensor,
    n_joint: int | None = None,
    n_product: int | None = None,
    evaluation_samples: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    if evaluation_samples is None:
        if n_joint is None or n_product is None:
            raise ValueError("Either evaluation_samples or both n_joint and n_product must be provided.")
        evaluation_samples = draw_joint_and_product_samples(gmm=gmm, n_joint=n_joint, n_product=n_product)

    x_joint = evaluation_samples["x_joint"]
    y_joint = evaluation_samples["y_joint"]
    x_prod = evaluation_samples["x_product"]
    y_prod = evaluation_samples["y_product"]

    gt_joint = gmm.pointwise_mutual_dependency(X=x_joint, Y=y_joint)
    pred_joint = predict_pmd(
        model=model,
        x=x_joint,
        y=y_joint,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    gt_prod = gmm.pointwise_mutual_dependency(X=x_prod, Y=y_prod)
    pred_prod = predict_pmd(
        model=model,
        x=x_prod,
        y=y_prod,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    joint_abs_err = np.abs(gt_joint - pred_joint)
    product_abs_err = np.abs(gt_prod - pred_prod)

    return {
        "joint_expected_abs_pmd_error": float(np.mean(joint_abs_err)),
        "product_expected_abs_pmd_error": float(np.mean(product_abs_err)),
    }


def fit_conditional_expectation_decoder(
    *,
    model: torch.nn.Module,
    y_train_c: torch.Tensor,
    rep_y=None,
    batch_size: int = 4096,
) -> torch.nn.Module:
    decoder_loader = DataLoader(
        TensorDataset(y_train_c, y_train_c),
        batch_size=min(int(batch_size), len(y_train_c)),
        shuffle=False,
    )
    if isinstance(model, ENCP):
        decoder = model.fit_linear_decoder(
            train_dataloader=decoder_loader,
            lstsq=False,
            z_rep=rep_y,
        )
    else:
        decoder = model.fit_linear_decoder(
            train_dataloader=decoder_loader,
            lstsq=False,
        )
    decoder.eval()
    return decoder


@torch.no_grad()
def predict_conditional_expectation(
    *,
    model: torch.nn.Module,
    x: torch.Tensor | np.ndarray,
    decoder: torch.nn.Module,
    x_mean: torch.Tensor,
    y_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_std: torch.Tensor,
) -> np.ndarray:
    device = next(model.parameters()).device
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)

    x_mean = x_mean.to(device=device, dtype=torch.float32)
    y_mean = y_mean.to(device=device, dtype=torch.float32)
    x_std = x_std.to(device=device, dtype=torch.float32)
    y_std = y_std.to(device=device, dtype=torch.float32)

    x_c = (x_t - x_mean) / x_std
    y_c_pred = model.conditional_expectation(x=x_c, hy2zy=decoder)
    y_pred = y_mean + y_std * y_c_pred
    return y_pred.detach().cpu().numpy()


@torch.no_grad()
def compute_density_reference_grid(
    *,
    gmm: SymmGaussianMixture,
    G: escnn.group.Group,
    domain: PlotDomain,
) -> dict[str, np.ndarray | float]:
    X_grid, Y_grid, X_input, Y_input = _mesh_inputs(domain)

    joint_density = gmm.joint_pdf(X=X_input, Y=Y_input).reshape(X_grid.shape)
    product_density = (gmm.pdf_x(X=X_input) * gmm.pdf_y(Y=Y_input)).reshape(X_grid.shape)
    marginal_x = gmm.pdf_x(X=np.column_stack([domain.x_centers]))
    marginal_y = gmm.pdf_y(Y=np.column_stack([domain.y_centers]))

    x_t, y_t = -1.0, 1.0
    g = G.elements[-1]
    gx_t = float((gmm.rep_X(gmm.G2Hx(g)) @ [x_t]).squeeze())
    gy_t = float((gmm.rep_Y(gmm.G2Hy(g)) @ [y_t]).squeeze())

    return {
        "X_grid": X_grid,
        "Y_grid": Y_grid,
        "x_edges": domain.x_edges,
        "y_edges": domain.y_edges,
        "x_centers": domain.x_centers,
        "y_centers": domain.y_centers,
        "x_limits": domain.x_limits,
        "y_limits": domain.y_limits,
        "joint_density": joint_density,
        "product_density": product_density,
        "marginal_x": marginal_x,
        "marginal_y": marginal_y,
        "x_t": x_t,
        "y_t": y_t,
        "gx_t": gx_t,
        "gy_t": gy_t,
    }


@torch.no_grad()
def compute_conditional_expectation_curve(
    *,
    gmm: SymmGaussianMixture,
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    domain: PlotDomain,
    x_mean: torch.Tensor,
    y_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_std: torch.Tensor,
) -> dict[str, np.ndarray | tuple[float, float]]:
    x_values = np.column_stack([domain.x_centers])
    ground_truth = gmm.mean_(x_values).reshape(-1)
    prediction = predict_conditional_expectation(
        model=model,
        x=x_values,
        decoder=decoder,
        x_mean=x_mean,
        y_mean=y_mean,
        x_std=x_std,
        y_std=y_std,
    ).reshape(-1)
    return {
        "x_values": domain.x_centers,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "x_limits": domain.x_limits,
    }


@torch.no_grad()
def compute_pmd_error_grid(
    *,
    gmm: SymmGaussianMixture,
    model: torch.nn.Module,
    G: escnn.group.Group,
    domain: PlotDomain,
    x_mean: torch.Tensor,
    y_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_std: torch.Tensor,
    x_product: np.ndarray,
    y_product: np.ndarray,
    support_quantile: float = 0.995,
    apply_density_mask: bool = False,
) -> dict[str, np.ndarray | float]:
    x_product = np.asarray(x_product, dtype=float).reshape(-1)
    y_product = np.asarray(y_product, dtype=float).reshape(-1)
    X_grid, Y_grid, X_input, Y_input = _mesh_inputs(domain)

    pmd_gt = gmm.pointwise_mutual_dependency(X=X_input, Y=Y_input).reshape(X_grid.shape)
    pmd_pred = predict_pmd(
        model=model,
        x=X_input,
        y=Y_input,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    ).reshape(X_grid.shape)
    abs_err_grid = np.abs(pmd_gt - pmd_pred)

    in_bounds = (
        (x_product >= domain.x_limits[0])
        & (x_product <= domain.x_limits[1])
        & (y_product >= domain.y_limits[0])
        & (y_product <= domain.y_limits[1])
    )
    if not np.any(in_bounds):
        raise ValueError("No product samples fall inside the requested plotting domain.")

    product_density_grid = (
        gmm.pdf_x(X=np.column_stack([X_grid.ravel()])) * gmm.pdf_y(Y=np.column_stack([Y_grid.ravel()]))
    ).reshape(X_grid.shape)
    support_mask = np.ones_like(abs_err_grid, dtype=bool)
    density_floor = float("nan")
    if apply_density_mask:
        product_density_samples = gmm.pdf_x(X=np.column_stack([x_product[in_bounds]])) * gmm.pdf_y(
            Y=np.column_stack([y_product[in_bounds]])
        )
        tail_mass = float(np.clip(1.0 - support_quantile, 0.0, 0.25))
        density_floor = float(np.quantile(product_density_samples.reshape(-1), tail_mass))
        support_mask = product_density_grid >= density_floor
        abs_err_grid = np.where(support_mask, abs_err_grid, np.nan)

    x_t, y_t = -1.0, 1.0
    g = G.elements[-1]
    gx_t = float((gmm.rep_X(gmm.G2Hx(g)) @ [x_t]).squeeze())
    gy_t = float((gmm.rep_Y(gmm.G2Hy(g)) @ [y_t]).squeeze())

    return {
        "x_edges": domain.x_edges,
        "y_edges": domain.y_edges,
        "x_centers": domain.x_centers,
        "y_centers": domain.y_centers,
        "x_limits": domain.x_limits,
        "y_limits": domain.y_limits,
        "pmd_true": pmd_gt,
        "pmd_pred": pmd_pred,
        "abs_pmd_err": abs_err_grid,
        "product_density": product_density_grid,
        "support_mask": support_mask,
        "density_floor": density_floor,
        "x_t": x_t,
        "y_t": y_t,
        "gx_t": gx_t,
        "gy_t": gy_t,
    }
