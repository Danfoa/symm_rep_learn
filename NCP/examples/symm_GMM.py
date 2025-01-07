# Created by danfoa at 18/12/24
import pathlib

import escnn
import hydra
import lightning
import numpy as np
import seaborn as sns
import torch
from escnn.group import Group, Representation
from escnn.nn import FieldType, GeometricTensor
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate, TensorDataset

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture
from NCP.models.ncp_lightning_module import NCPModule


def plot_analytic_joint_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples, y_samples):
    grid = sns.JointGrid()
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 200)
    y_range = np.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    # Flatten the grid to evaluate joint_pdf
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # p(x,y) -- Joint PDF
    # Compute the joint PDF for each point on the grid
    Z_flat = gmm.joint_pdf(X=X_input, Y=Y_input)
    Z = Z_flat.reshape(X_grid.shape)
    joint_contour = grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap="Blues", levels=15)

    # Select a random sample to test the conditional expectation
    x_t, y_t = x_samples[0], y_samples[0]
    g = G.elements[-1]
    gx_t, gy_t = (rep_X(g) @ [x_t]).squeeze(), (rep_Y(g) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=0.7)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.7)
    grid.ax_joint.axvline(gx_t, color='g', alpha=0.7)
    grid.ax_joint.axhline(gy_t, color='g', alpha=0.7)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=5, alpha=0.7)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=5, alpha=0.7)
    # Set limits
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    # Do plot of conditional probability density on the test conditions x and gx
    n_samples_cpd = len(y_range)
    cpd_y_vals = gmm.pdf(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    cpd_gy_vals = gmm.pdf(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    # Plot filled distributions with lines
    style = {
        "cpd_x":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$p(y | x)$",
            },
        "cpd_gx": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$p(y | g \;\triangleright_{\mathcal{X}}\; x)$",
            },
        "pdf":    {
            "color":  "blue",
            "fill":   "lightblue",
            "legend": r"$p(y)$",
            }
        }

    for key, cpd_vals in zip(["cpd_x", "cpd_gx"], [cpd_y_vals, cpd_gy_vals]):
        grid.ax_marg_y.plot(cpd_vals, y_range, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=0.6,
                            label=style[key]["legend"])

    # Plot marginal x
    pdf_x = gmm.pdf_x(X=x_range)
    grid.ax_marg_x.fill_between(x_range, 0, pdf_x, color=style["pdf"]["fill"], alpha=0.4)
    grid.ax_marg_x.plot(x_range, pdf_x, color=style["pdf"]["color"], linestyle="-", alpha=1.0, linewidth=0.6,
                        label=style["pdf"]["legend"])
    grid.ax_marg_x.set_ylim([0, None])
    # Plot marginal y
    pdf_y = gmm.pdf_y(Y=y_range)
    grid.ax_marg_y.fill_betweenx(y_range, 0, pdf_y, color=style["pdf"]["fill"], alpha=0.4)
    grid.ax_marg_y.plot(pdf_y, y_range, color=style["pdf"]["color"], linestyle="-", alpha=1.0, linewidth=0.6,
                        label=style["pdf"]["legend"])
    grid.ax_marg_y.set_xlim([0, None])

    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=8)
    grid.fig.suptitle(r"Analytic $p(x,y)$, and $p(y | x)$")
    grid.fig.tight_layout()
    return grid


def plot_analytic_mi_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation, x_samples,
                        y_samples):
    grid = sns.JointGrid()
    # Define grid for the plot
    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = np.max(np.abs(x_samples)), np.max(np.abs(y_samples))
    x_range = np.linspace(-x_max, x_max, 200)
    y_range = np.linspace(-y_max, y_max, 200)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    # Flatten the grid to evaluate joint_pdf
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    X_input = np.column_stack([X_flat])
    Y_input = np.column_stack([Y_flat])
    # Compute the mutual information
    mi_flat = gmm.mutual_information(X=X_input, Y=Y_input)
    Z = mi_flat.reshape(X_grid.shape)
    grid.ax_joint.contourf(X_grid, Y_grid, Z, sns.color_palette("mako", as_cmap=True), levels=35)

    # Select a random sample to test the conditional expectation
    x_t, y_t = x_samples[0], y_samples[0]
    g = G.elements[-1]
    gx_t, gy_t = (rep_X(g) @ [x_t]).squeeze(), (rep_Y(g) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=0.7)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.7)
    grid.ax_joint.axvline(gx_t, color='g', alpha=0.7)
    grid.ax_joint.axhline(gy_t, color='g', alpha=0.7)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=5, alpha=0.5)
    grid.ax_joint.plot(gx_t, gy_t, 'go', markersize=5, alpha=0.5)
    # Set limits
    grid.ax_joint.set_xlim([-x_max, x_max])
    grid.ax_joint.set_ylim([-y_max, y_max])
    # Customizing labels
    grid.ax_joint.set_xlabel(r"$\mathcal{X}$")
    grid.ax_joint.set_ylabel(r"$\mathcal{Y}$")
    grid.ax_marg_x.set_xlabel(r"$p(\textnormal{x})$")
    grid.ax_marg_y.set_ylabel(r"$p(\textnormal{y})$")

    # Do plot of conditional probability density on the test conditions x and gx
    n_samples_cpd = len(y_range)
    mi_x_vals = gmm.mutual_information(X=np.repeat(x_t, n_samples_cpd), Y=y_range)
    mi_gx_vals = gmm.mutual_information(X=np.repeat(gx_t, n_samples_cpd), Y=y_range)
    mi_y_vals = gmm.mutual_information(X=x_range, Y=np.repeat(y_t, len(x_range)))
    mi_gy_vals = gmm.mutual_information(X=x_range, Y=np.repeat(gy_t, len(x_range)))

    # Plot filled distributions with lines
    style = {
        "mi_x":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$MI(y,x)$",
            },
        "mi_gx": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$MI(y, g \;\triangleright_{\mathcal{X}}\; x)$",
            },
        "mi_y":  {
            "color":  "red",
            "fill":   "pink",
            "legend": r"$MI(y,x)$",
            },
        "mi_gy": {
            "color":  "green",
            "fill":   "lightgreen",
            "legend": r"$MI(g \;\triangleright_{\mathcal{Y}}\; y, x)$",
            },
        "pdf":   {
            "color":  "blue",
            "fill":   "lightblue",
            "legend": r"$p(y)$",
            }
        }

    for key, mi_vals in zip(["mi_x", "mi_gx"], [mi_x_vals, mi_gx_vals]):
        grid.ax_marg_y.plot(mi_vals, y_range, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=1,
                            label=style[key]["legend"])
    for key, mi_vals in zip(["mi_y", "mi_gy"], [mi_y_vals, mi_gy_vals]):
        grid.ax_marg_x.plot(x_range, mi_vals, color=style[key]["color"], linestyle="-", alpha=0.9, linewidth=1,
                            label=style[key]["legend"])

    # Add legend
    grid.ax_marg_y.legend(loc="upper left", fontsize=8)
    grid.ax_marg_x.legend(loc="upper left", fontsize=8)
    grid.ax_marg_y.axvline(0, color='gray', alpha=0.5)
    grid.ax_marg_x.axhline(0, color='gray', alpha=0.5)
    grid.fig.suptitle(r"Analytic mutual information $\frac{p(x,y)}{p(y)p(x)}$")
    grid.fig.tight_layout()
    return grid


def get_model(cfg: DictConfig, x_type, y_type, lat_type) -> torch.nn.Module:
    embedding_dim = lat_type.size
    if cfg.model.lower() == "encp":  # Equivariant NCP
        from NCP.models.equiv_ncp import ENCPOperator
        from NCP.nn.equiv_layers import EMLP

        kwargs = dict(out_type=lat_type,
                      hidden_layers=cfg.embedding.hidden_layers,
                      hidden_units=cfg.embedding.hidden_units, bias=False)
        χ_embedding = EMLP(in_type=x_type, **kwargs)
        y_embedding = EMLP(in_type=y_type, **kwargs)
        eNCPop = ENCPOperator(x_fns=χ_embedding, y_fns=y_embedding, gamma=cfg.gamma)

        return eNCPop
    elif cfg.model.lower() == "ncp":  # NCP
        from NCP.mysc.utils import class_from_name
        from NCP.models.ncp import NCP
        from NCP.nn.layers import MLP

        activation = class_from_name('torch.nn', cfg.embedding.activation)
        kwargs = dict(output_shape=embedding_dim,
                      n_hidden=cfg.embedding.hidden_layers,
                      layer_size=cfg.embedding.hidden_units,
                      activation=activation)
        fx = MLP(input_shape=x_type.size, **kwargs)
        fy = MLP(input_shape=y_type.size, **kwargs)
        ncp = NCP(fx, fy, embedding_dim=embedding_dim, gamma=cfg.gamma * lat_type.size)
        return ncp
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


def gmm_dataset(n_samples: int, gmm: SymmGaussianMixture, rep_X: Representation, rep_Y: Representation):
    from NCP.mysc.symm_algebra import symmetric_moments

    x_samples, y_samples = gmm.simulate(n_samples=n_samples)
    x_mean, x_var = symmetric_moments(x_samples, rep_X)
    y_mean, y_var = symmetric_moments(y_samples, rep_Y)

    # Train, val, test splitting
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    n_samples = len(x_samples)
    n_train, n_val, n_test = np.asarray(np.array([train_ratio, val_ratio, test_ratio]) * n_samples, dtype=int)
    X = (x_samples - x_mean.numpy()) / np.sqrt(x_var.numpy())
    Y = (y_samples - y_mean.numpy()) / np.sqrt(y_var.numpy())
    x_train, x_val, x_test = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    y_train, y_val, y_test = Y[:n_train], Y[n_train:n_train + n_val], Y[n_train + n_val:]

    X_train = torch.atleast_2d(torch.from_numpy(x_train).float())
    Y_train = torch.atleast_2d(torch.from_numpy(y_train).float())
    X_val = torch.atleast_2d(torch.from_numpy(x_val).float())
    Y_val = torch.atleast_2d(torch.from_numpy(y_val).float())
    X_test = torch.atleast_2d(torch.from_numpy(x_test).float())
    Y_test = torch.atleast_2d(torch.from_numpy(y_test).float())

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    return (x_samples, y_samples), (x_mean, y_mean), (x_var, y_var), (train_dataset, val_dataset, test_dataset)


@torch.no_grad()
def measure_analytic_mi_error(gmm, ncp, x_samples, y_samples, x_type, y_type, x_mean, y_mean, x_var, y_var, plot=False):
    G = x_type.fibergroup
    n_total_samples = 1000
    n_samples_per_g = int(n_total_samples // G.order())

    device = next(ncp.parameters()).device
    dtype = next(ncp.parameters()).dtype

    idx = np.random.choice(len(x_samples), n_samples_per_g, replace=False)

    x_max, y_max = max(np.abs(x_samples)), max(np.abs(y_samples))
    X_np = np.atleast_2d(x_samples)[idx]
    Y_np = np.atleast_2d(y_samples)[idx]
    G_X_np, G_Y_np = [X_np], [Y_np]
    # Perform data augmentation using the group actions
    for g in G.elements[1:]:
        G_X_np.append(np.einsum("ij,...j->...i", x_type.representation(g), X_np))
        G_Y_np.append(np.einsum("ij,...j->...i", y_type.representation(g), Y_np))
    X_np = np.vstack(G_X_np)
    Y_np = np.vstack(G_Y_np)

    X_c = ((torch.Tensor(X_np) - x_mean) / torch.sqrt(x_var)).to(device=device)
    Y_c = ((torch.Tensor(Y_np) - y_mean) / torch.sqrt(y_var)).to(device=device)

    # Compute the estimate of the NCP model of the mutual information _______________
    fx, hy = ncp(x=X_c, y=Y_c)
    mi_xy = ncp.mutual_information(fx, hy)  # k_r(x,y) ≈ p(x,y) / p(x)p(y)
    # Compute the analytic mutual information _______________________________________
    mi_xy_gt = gmm.mutual_information(X=X_np, Y=Y_np)  # p(x,y) / p(x)p(y)
    # Compute the mean squared error between the two estimates
    mi_mse = ((mi_xy.cpu().numpy() - mi_xy_gt) ** 2).mean()
    # Since k(x, y) = k(g.x, g.y) for all g in G, we want to compute the variance of the estimate under g-action
    mi_xy_g_var = mi_xy.view(G.order(), -1).var(dim=0)

    # Sample 4 random conditioning values of x
    range_n_samples = 100
    n_cond_points = 4
    cond_idx = np.random.choice(len(X_np), n_cond_points, replace=False)
    X_cond_np = X_np[cond_idx]
    X_c_cond = X_c[cond_idx]

    # X_range_np = np.linspace(-x_max, x_max, range_n_samples)
    Y_range_np = np.linspace(-y_max, y_max, range_n_samples)
    p_Y_range_np = gmm.pdf_y(Y_range_np)
    P_Y_range = torch.from_numpy(p_Y_range_np).to(device=device, dtype=dtype)
    Y_c_range = ((torch.from_numpy(Y_range_np) - y_mean) / torch.sqrt(y_var)).to(device=device, dtype=dtype)

    CDFs_mse, CDFs_gt, CDFs = [], [], []
    for x_cond, x_c_cond in zip(X_cond_np, X_c_cond):  # Comptue p(y | x) for each conditioning value
        cdf_gt = gmm.pdf(X=np.broadcast_to(x_cond, Y_range_np.shape), Y=Y_range_np)
        fx, hy = ncp(x=torch.broadcast_to(x_c_cond, Y_c_range.shape), y=Y_c_range)
        mi_xy = ncp.mutual_information(fx, hy)  # k_r(x,y) ≈ p(x,y) / p(x)p(y)

        cdf = (mi_xy * P_Y_range).cpu().numpy()  # k(x,y) * p(y) = p(y | x)
        CDFs_gt.append(cdf_gt)
        CDFs.append(cdf)
        CDFs_mse.append(((cdf - cdf_gt) ** 2).mean())

    metrics = {"MI/mse": mi_mse,
               "MI/equiv_err": mi_xy_g_var.mean(),
               "CDF/mse": np.mean(CDFs_mse)}

    if plot:
        # Plot the 4 CDFs each pair in a separate subplot
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        for i, (cdf, cdf_gt) in enumerate(zip(CDFs, CDFs_gt)):
            axs[i].plot(Y_range_np, p_Y_range_np, label="p(y)", linestyle="--")
            axs[i].plot(Y_range_np, cdf, label=r"$\hat{p}_{\text{ncp}}(y | x)$")
            axs[i].plot(Y_range_np, cdf_gt, label=r"$p(y | x)$")
            axs[i].set_title(f"Conditioning value {i}")
            axs[i].legend()
        fig.tight_layout()
        return metrics, fig

    return metrics


@hydra.main(config_path='cfg', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed > 0 else np.random.randint(0, 1000)
    seed_everything(seed)

    C2 = escnn.group.CyclicGroup(2)  # Reflection group = Cyclic group of order 2
    # rep_X = C2.regular_representation  # ρ_Χ
    # rep_Y = C2.regular_representation  # ρ_Y
    rep_X = C2.representations['irrep_1']  # ρ_Χ
    rep_Y = C2.representations['irrep_1']  # ρ_Y
    rep_X.name, rep_Y.name = "rep_X", "rep_Y"

    gmm = SymmGaussianMixture(n_kernels=cfg.gmm.n_kernels, rep_X=rep_X, rep_Y=rep_Y, means_std=2.0, random_seed=10)

    # x_samples, y_samples = gmm.simulate(n_samples=5000)
    # grid = plot_analytic_joint_2D(gmm, G=C2, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
    # plt.show()
    # grid = plot_analytic_mi_2D(gmm, G=C2, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
    # plt.show()

    (x_samples, y_samples), (x_mean, y_mean), (x_var, y_var), datasets = gmm_dataset(cfg.n_samples, gmm, rep_X, rep_Y)
    train_ds, val_ds, test_ds = datasets

    # Define the Input and Latent types for ESCNN
    x_type = FieldType(gspace=escnn.gspaces.no_base_space(C2), representations=[rep_X])
    y_type = FieldType(gspace=escnn.gspaces.no_base_space(C2), representations=[rep_Y])
    lat_type = FieldType(
        gspace=escnn.gspaces.no_base_space(C2),
        representations=[C2.regular_representation] * (cfg.embedding['embedding_dim'] // C2.order())
        )

    # ESCNN equivariant models expect GeometricTensors.
    def geom_tensor_collate_fn(batch) -> [GeometricTensor, GeometricTensor]:
        x_batch, y_batch = default_collate(batch)
        return GeometricTensor(x_batch, x_type), GeometricTensor(y_batch, y_type)

    # Get the model
    ncp_op = get_model(cfg, x_type, y_type, lat_type)

    # Define the dataloaders
    from escnn.nn import EquivariantModule
    collate_fn = geom_tensor_collate_fn if isinstance(ncp_op, EquivariantModule) else default_collate
    train_dataloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    lightning_module = NCPModule(
        model=ncp_op,
        optimizer_fn=torch.optim.Adam,
        optimizer_kwargs={"lr": cfg.lr},
        loss_fn=ncp_op.loss,
        val_metrics=lambda x: measure_analytic_mi_error(
            gmm, ncp_op, x_samples, y_samples, x_type, y_type, x_mean, y_mean, x_var, y_var
            ),
        )

    pathlib.Path("lightning_logs").mkdir(exist_ok=True)
    logger = WandbLogger(save_dir="lightning_logs",
                         project=cfg.exp_name,
                         log_model=False,
                         config=OmegaConf.to_container(cfg, resolve=True))
    # logger.watch(ncp_op, log="all", log_graph=False)

    trainer = lightning.Trainer(accelerator='auto',
                                max_epochs=cfg.max_epochs, logger=logger,
                                enable_progress_bar=True,
                                log_every_n_steps=50,
                                check_val_every_n_epoch=20,
                                )

    torch.set_float32_matmul_precision('medium')
    trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    m, f = measure_analytic_mi_error(
        gmm, ncp_op, x_samples, y_samples, x_type, y_type, x_mean, y_mean, x_var, y_var, plot=True
        )
    f.show()

    a = trainer.test(lightning_module, dataloaders=test_dataloader)
    print(a)


if __name__ == '__main__':
    main()
