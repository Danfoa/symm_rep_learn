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
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate, TensorDataset

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture
from NCP.models.ncp_lightning_module import NCPModule


def plot_analytic_joint_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation):
    grid = sns.JointGrid()
    # Define grid for the plot
    x_samples, y_samples = gmm.simulate(n_samples=5000)
    assert x_samples.shape[-1] == 1 and y_samples.shape[-1] == 1, "Only 1D samples are supported for this plot"

    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = max(abs(x_samples.max()), abs(x_samples.max())), max(abs(y_samples.max()), abs(y_samples.max()))
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
    grid.ax_joint.axvline(x_t, color='r', alpha=0.2)
    grid.ax_joint.axvline(gx_t, color='r', alpha=0.2)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.2)
    grid.ax_joint.axhline(gy_t, color='r', alpha=0.2)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=4, alpha=0.5)
    grid.ax_joint.plot(gx_t, gy_t, 'ro', markersize=4, alpha=0.5)
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


def plot_analytic_mi_2D(gmm: SymmGaussianMixture, G: Group, rep_X: Representation, rep_Y: Representation):
    grid = sns.JointGrid()
    # Define grid for the plot
    x_samples, y_samples = gmm.simulate(n_samples=5000)
    assert x_samples.shape[-1] == 1 and y_samples.shape[-1] == 1, "Only 1D samples are supported for this plot"

    x_samples = x_samples.squeeze()
    y_samples = y_samples.squeeze()
    x_max, y_max = max(abs(x_samples.max()), abs(x_samples.max())), max(abs(y_samples.max()), abs(y_samples.max()))
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
    grid.ax_joint.contourf(X_grid, Y_grid, Z, cmap=sns.color_palette("rocket", as_cmap=True), levels=35)

    # Select a random sample to test the conditional expectation
    x_t, y_t = x_samples[0], y_samples[0]
    g = G.elements[-1]
    gx_t, gy_t = (rep_X(g) @ [x_t]).squeeze(), (rep_Y(g) @ [y_t]).squeeze()
    grid.ax_joint.axvline(x_t, color='r', alpha=0.2)
    grid.ax_joint.axvline(gx_t, color='r', alpha=0.2)
    grid.ax_joint.axhline(y_t, color='r', alpha=0.2)
    grid.ax_joint.axhline(gy_t, color='r', alpha=0.2)
    # Draw red point on the selected sample
    grid.ax_joint.plot(x_t, y_t, 'ro', markersize=4, alpha=0.5)
    grid.ax_joint.plot(gx_t, gy_t, 'ro', markersize=4, alpha=0.5)
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
    grid.fig.suptitle(r"Analytic mutual information $\frac{p(x,y)}{p(y)p(x)}$")
    grid.fig.tight_layout()
    return grid


def get_model(cfg: DictConfig, x_type, y_type, lat_type) -> torch.nn.Module:
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
        kwargs = dict(output_shape=lat_type.size,
                      n_hidden=cfg.embedding.hidden_layers,
                      layer_size=cfg.embedding.hidden_units,
                      activation=activation)
        fx = MLP(input_shape=x_type.size, **kwargs)
        fy = MLP(input_shape=y_type.size, **kwargs)
        ncp = NCP(fx, fy, embedding_dim=lat_type.size, gamma=cfg.gamma * lat_type.size)
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

    return (x_samples, y_samples), (train_dataset, val_dataset, test_dataset)


@hydra.main(config_path='cfg', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    C2 = escnn.group.CyclicGroup(2)  # Reflection group = Cyclic group of order 2
    # rep_X = C2.regular_representation  # ρ_Χ
    # rep_Y = C2.regular_representation  # ρ_Y
    rep_X = C2.representations['irrep_1']  # ρ_Χ
    rep_Y = C2.representations['irrep_1']  # ρ_Y
    rep_X.name, rep_Y.name = "rep_X", "rep_Y"

    gmm = SymmGaussianMixture(n_kernels=6, rep_X=rep_X, rep_Y=rep_Y, means_std=2.0, random_seed=10)

    grid = plot_analytic_joint_2D(gmm, G=C2, rep_X=rep_X, rep_Y=rep_Y)
    plt.show()

    (x_samples, y_samples), (train_ds, val_ds, test_ds) = gmm_dataset(cfg.n_samples, gmm, rep_X, rep_Y)

    # Plot y vs x
    # plt.scatter(x_samples, y_samples, alpha=0.5, color='C0', label='Data')
    # plt.show()

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
        )

    pathlib.Path("lightning_logs").mkdir(exist_ok=True)
    logger = WandbLogger(save_dir="lightning_logs",
                         project="NCP-GMM-C2",
                         log_model=False,
                         config=OmegaConf.to_container(cfg, resolve=True))
    # logger.watch(ncp_op, log="all", log_graph=False)

    trainer = lightning.Trainer(accelerator='auto',
                                max_epochs=500, logger=logger,
                                enable_progress_bar=True,
                                log_every_n_steps=5,
                                check_val_every_n_epoch=5)

    torch.set_float32_matmul_precision('medium')
    trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    a = trainer.test(lightning_module, dataloaders=test_dataloader)
    print(a)


if __name__ == '__main__':
    main()
