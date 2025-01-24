# Created by danfoa at 18/12/24
import pathlib
from math import ceil

import escnn
import hydra
import lightning
import numpy as np
import pandas as pd
import torch
from escnn.group import directsum, Representation
from escnn.nn import FieldType, GeometricTensor
from hydra.core.hydra_config import HydraConfig
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from plotly.subplots import make_subplots
import plotly.io as pio

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, default_collate, TensorDataset

from NCP.cde_fork.density_simulation.symmGMM import SymmGaussianMixture
from NCP.examples.symmGMM.plot_utils import (plot_analytic_joint_2D, plot_analytic_npmi_2D, plot_analytic_pmd_2D,
                                             plot_analytic_prod_2D, plot_pmd_error_distribution)
from NCP.models.ncp_lightning_module import TrainingModule

import logging

from NCP.nn.equiv_layers import IMLP

log = logging.getLogger(__name__)


def get_model(cfg: DictConfig, x_type, y_type, lat_type) -> torch.nn.Module:
    embedding_dim = lat_type.size
    if cfg.model.lower() == "encp":  # Equivariant NCP
        from NCP.models.equiv_ncp import ENCP
        from NCP.nn.equiv_layers import EMLP

        kwargs = dict(out_type=lat_type,
                      hidden_layers=cfg.embedding.hidden_layers,
                      activation=cfg.embedding.activation,
                      hidden_units=cfg.embedding.hidden_units,
                      bias=False)
        χ_embedding = EMLP(in_type=x_type, **kwargs)
        y_embedding = EMLP(in_type=y_type, **kwargs)
        eNCPop = ENCP(embedding_x=χ_embedding,
                      embedding_y=y_embedding,
                      gamma=cfg.gamma,
                      truncated_op_bias=cfg.truncated_op_bias,
                      )

        return eNCPop
    elif cfg.model.lower() == "ncp":  # NCP
        from NCP.mysc.utils import class_from_name
        from NCP.models.ncp import NCP
        from NCP.nn.layers import MLP

        activation = class_from_name('torch.nn', cfg.embedding.activation)
        kwargs = dict(output_shape=embedding_dim,
                      n_hidden=cfg.embedding.hidden_layers,
                      layer_size=cfg.embedding.hidden_units,
                      activation=activation,
                      bias=False)
        fx = MLP(input_shape=x_type.size, **kwargs)
        fy = MLP(input_shape=y_type.size, **kwargs)
        ncp = NCP(embedding_x=fx,
                  embedding_y=fy,
                  embedding_dim=embedding_dim,
                  gamma=cfg.gamma,
                  truncated_op_bias=cfg.truncated_op_bias,
                  )
        return ncp
    elif cfg.model.lower() == "drf":  # Density Ratio Fitting
        from NCP.models.density_ratio_fitting import DRF
        from NCP.nn.layers import MLP
        from NCP.mysc.utils import class_from_name

        activation = class_from_name('torch.nn', cfg.embedding.activation)
        n_layers = cfg.embedding.hidden_layers
        n_hidden_units = int(cfg.embedding.hidden_units * 2)
        embedding = MLP(input_shape=x_type.size + y_type.size,  # z = (x,y)
                        output_shape=1,
                        n_hidden=n_layers,
                        # Ensure NCP model and DRF have approximately equal number of parameters.
                        layer_size=[n_hidden_units] * (n_layers - 1) + [cfg.embedding.embedding_dim],
                        activation=activation,
                        bias=False)
        drf = DRF(embedding=embedding, gamma=cfg.gamma)

        return drf
    elif cfg.model.lower() == "idrf":  # Density Ratio Fitting
        from NCP.models.inv_density_ratio_fitting import InvDRF
        from NCP.mysc.utils import class_from_name

        xy_reps = x_type.representations + y_type.representations
        in_type = FieldType(x_type.gspace, xy_reps)
        imlp = IMLP(in_type=in_type,
                    out_dim=1,  # Scalar PMD value
                    hidden_layers=cfg.embedding.hidden_layers,
                    activation=cfg.embedding.activation,
                    hidden_units=cfg.embedding.hidden_units,
                    bias=False)
        idrf = InvDRF(embedding=imlp, gamma=cfg.gamma)
        return idrf
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


def gmm_dataset(cfg: DictConfig, gmm: SymmGaussianMixture, rep_X: Representation, rep_Y: Representation, device='cpu'):
    from NCP.mysc.symm_algebra import symmetric_moments

    total_samples = cfg.gmm.n_total_samples
    x_samples, y_samples = gmm.simulate(n_samples=total_samples)
    MI = gmm.MI(x_samples, y_samples),
    log.info(f"\n\n MI estimation: {MI}  with {total_samples} samples\n\n")
    x_mean, x_var = symmetric_moments(x_samples, rep_X)
    y_mean, y_var = symmetric_moments(y_samples, rep_Y)
    # Train, val, test splitting
    assert 0.0 < cfg.train_samples_ratio <= 0.7, f'Invalid train_samples_ratio: {cfg.train_samples_ratio}'
    train_ratio, val_ratio, test_ratio = cfg.train_samples_ratio, 0.15, 0.15

    log.info(f"Train: {train_ratio} Val: {val_ratio} Test: {test_ratio}")
    log.info(f"Train samples: {train_ratio * total_samples} Val samples: {val_ratio * total_samples} "
             f"Test samples: {test_ratio * total_samples}")
    n_samples = len(x_samples)
    n_train, n_val, n_test = np.asarray(np.array([train_ratio, val_ratio, test_ratio]) * n_samples, dtype=int)
    # Take the train samples from the front of the array
    train_samples = x_samples[:n_train], y_samples[:n_train]
    # Take the Val and test sampeles from the back of the array
    val_samples = x_samples[-(n_val + n_test):-n_test], y_samples[-(n_val + n_test):-n_test]
    test_samples = x_samples[-n_test:], y_samples[-n_test:]

    X_c = (x_samples - x_mean.numpy()) / np.sqrt(x_var.numpy())
    Y_c = (y_samples - y_mean.numpy()) / np.sqrt(y_var.numpy())
    x_train, x_val, x_test = X_c[:n_train], X_c[n_train:n_train + n_val], X_c[n_train + n_val:]
    y_train, y_val, y_test = Y_c[:n_train], Y_c[n_train:n_train + n_val], Y_c[n_train + n_val:]

    X_train = torch.atleast_2d(torch.from_numpy(x_train).float()).to(device=device)
    Y_train = torch.atleast_2d(torch.from_numpy(y_train).float()).to(device=device)
    X_val = torch.atleast_2d(torch.from_numpy(x_val).float())
    Y_val = torch.atleast_2d(torch.from_numpy(y_val).float())
    X_test = torch.atleast_2d(torch.from_numpy(x_test).float())
    Y_test = torch.atleast_2d(torch.from_numpy(y_test).float())

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    return (train_samples, val_samples, test_samples), (x_mean, y_mean), (x_var, y_var), (
        train_dataset, val_dataset, test_dataset)


@torch.no_grad()
def measure_analytic_pmi_error(gmm, nn_model, x_samples, y_samples,
                               x_type, y_type, x_mean, y_mean, x_var, y_var,
                               plot=False, samples=1000, save_data_path=None, debug=False):
    from NCP.models.equiv_ncp import ENCP

    prev_device = next(nn_model.parameters()).device
    dtype = next(nn_model.parameters()).dtype

    G = x_type.fibergroup
    if samples == 'all':
        n_total_samples = len(x_samples)
        device = 'cpu'
    else:
        n_total_samples = samples if samples != 'all' else len(x_samples)
        device = prev_device

    n_samples_per_g = int(n_total_samples // G.order())
    chosen_samples_idx = np.random.choice(len(x_samples), n_samples_per_g, replace=False)
    X = x_samples[chosen_samples_idx]  # Random sample from p(x,y)  #
    Y = y_samples[chosen_samples_idx]  # Random sample from p(x,y)
    p_X, p_Y = gmm.pdf_x(X), gmm.pdf_y(Y)
    # Perform data augmentation using the group actions
    # G_X, G_Y = {G.identity: X}, {G.identity: Y}
    # for g in G.elements[1:]:  # first element is the identity
    #     G_X[g] = np.einsum("ij,...j->...i", x_type.representation(g), X)
    #     G_Y[g] = np.einsum("ij,...j->...i", y_type.representation(g), Y)

    # Compute the NxN indices of all pairs of x and y samples
    prod_idx = np.arange(n_samples_per_g)
    X_idx, Y_idx = np.meshgrid(prod_idx, prod_idx)
    X_idx_flat, Y_idx_flat = X_idx.flatten(), Y_idx.flatten()
    # Get N^**2 samples from the product and joint distributions (of the original data)
    X_pairs = np.atleast_2d(X[X_idx_flat])
    Y_pairs = np.atleast_2d(Y[Y_idx_flat])
    # Compute the marginal and joint probabilities.
    P_joint_XY_pairs = gmm.joint_pdf(X_pairs, Y_pairs)
    P_joint_XY_pairs_mat = P_joint_XY_pairs.reshape(X_idx.shape)  # P_joint_XY_pairs_mat_ij = p(x_i, y_j)
    # Diagonal P_XY_pairs_mat is the p(x,y) of samples from the joint.
    if debug:
        assert np.allclose(np.diag(P_joint_XY_pairs_mat), gmm.joint_pdf(X, Y), rtol=1e-5,
                           atol=1e-5), "Error in joint PDF computation"
        x_0, y_1 = X[[0], :], Y[[1], :]
        p_x0y1 = gmm.joint_pdf(x_0, y_1)
        assert np.allclose(P_joint_XY_pairs_mat[1, 0], gmm.joint_pdf(x_0, y_1), rtol=1e-5, atol=1e-5), \
            f"{P_joint_XY_pairs_mat[1, 0]} != {gmm.joint_pdf(x_0, y_1)}"

    # ==============================================================================================================
    # Compute the NN estimate of the PMD for the entire group orbit in a single pass ______

    # Sample elements of the group to compute the PMD invariance error. Identity + 5 random elements
    G_elements = [G.identity] + np.random.choice(G.elements[1:], min(G.order() - 1, 5), replace=False).tolist()
    G_X, G_Y = [], []
    for g in G_elements:
        if hasattr(gmm, "G2H"):
            g_X = X_pairs if g == G.identity else np.einsum("ij,...j->...i", gmm.rep_X(gmm.G2H(g)), X_pairs)
            g_Y = Y_pairs if g == G.identity else np.einsum("ij,...j->...i", gmm.rep_Y(gmm.G2H(g)), Y_pairs)
        else:
            g_X = X_pairs if g == G.identity else np.einsum("ij,...j->...i", gmm.rep_X(g), X_pairs)
            g_Y = Y_pairs if g == G.identity else np.einsum("ij,...j->...i", gmm.rep_Y(g), Y_pairs)

        G_X.append(g_X)
        G_Y.append(g_Y)
    G_X = np.concatenate(G_X, axis=0)
    G_Y = np.concatenate(G_Y, axis=0)
    # Normalize the data for the NN estimation of the point-wise mutual dependency
    X_c = ((torch.Tensor(G_X) - x_mean) / torch.sqrt(x_var)).to(device=device)
    Y_c = ((torch.Tensor(G_Y) - y_mean) / torch.sqrt(y_var)).to(device=device)
    _x, _y = (x_type(X_c), y_type(Y_c)) if isinstance(nn_model, ENCP) else (X_c, Y_c)
    pmd_xy_pred = nn_model.pointwise_mutual_dependency(_x, _y).cpu().numpy()  # k_r(x,y) ≈ p(x,y) / p(x)p(y)
    G_pmd_xy_pred_mat = {}  # g : k_r(g.x, g.y)  for all pairs of x and y  (N x N)
    for i, g in enumerate(G_elements):
        start, end = i * n_samples_per_g ** 2, (i + 1) * n_samples_per_g ** 2
        G_pmd_xy_pred_mat[g] = pmd_xy_pred[start: end].reshape(X_idx.shape)

    # ==============================================================================================================
    # Compute the analytic PMD k(x,y) = p(x,y) / p(x)p(y)
    pmd_xy_gt_mat = np.einsum("ij,i,j->ij", P_joint_XY_pairs_mat, 1 / p_Y, 1 / p_X)  # p(x,y) / p(x)p(y)
    G_pmd_xy_err_mat = {g: pmd_xy_gt_mat - G_pmd_xy_pred_mat[g] for g in G_elements}

    if debug:
        # Check we compute appropriately the PMD
        pmd_xy_gt2 = gmm.pointwise_mutual_dependency(X=X_pairs, Y=Y_pairs)  # p(x,y) / p(x)p(y)
        assert np.allclose(pmd_xy_gt2.reshape(X_idx.shape), pmd_xy_gt_mat, rtol=1e-5, atol=1e-5)
        # Check reshaping is not breaking ordering
        pmd_xy_err = pmd_xy_gt2 - G_pmd_xy_pred_mat[G.identity].flatten()
        pmd_xy_err_mat2 = pmd_xy_err.reshape(X_idx.shape)
        assert np.allclose(G_pmd_xy_err_mat[G.identity], pmd_xy_err_mat2, rtol=1e-5, atol=1e-5), \
            f"Max error: {np.max(G_pmd_xy_err_mat[G.identity] - pmd_xy_err_mat2)}"

    # ==============================================================================================================
    #  Approximate the operator norm from the Gram matrix of errors.
    #  | E - Er |_op = sup_||f||_2 [ (E - Er) f ] ≈ max_sval( k_mat - k_pred_mat )
    G_pmd_xy_err_tensor = np.stack([G_pmd_xy_err_mat[g] for g in G_elements], axis=0)  # (|G|, N, N)
    # PMD_mse = E_p(x)p(y) (k(x,y) - k_r(x,y))^2 = (Σ_ij (k(x_i, y_j) - k_r(x_i, y_j))^2 * p(x)p(y))/ Σ_ij p(x_i)p(y_j)
    G_pmd_err_mat = np.einsum('gij,i,j->gij', G_pmd_xy_err_tensor, np.sqrt(p_Y / p_Y.sum()), np.sqrt(p_X / p_Y.sum()))
    pmd_mse = (G_pmd_err_mat ** 2).sum()
    Op_norms = np.linalg.norm(G_pmd_xy_err_tensor, ord=2, axis=(1, 2))
    Op_norm = np.max(Op_norms)  # Largest singular value. Spectral norm
    # Since k(x, y) = k(g.x, g.y) for all g in G, we want to compute the variance of the estimate under g-action
    pmd_G_var = np.var([G_pmd_xy_pred_mat[g] for g in G_elements], axis=0).mean()

    # Compute the error on PMI = ln(PMD)
    pmi_gt_mat = np.log(pmd_xy_gt_mat)
    npmi_gt_mat = pmi_gt_mat / (-np.log(P_joint_XY_pairs_mat))
    if debug:
        pmi_gt2 = gmm.pointwise_mutual_information(X=X_pairs, Y=Y_pairs)
        npmi_gt2 = gmm.normalized_pointwise_mutual_information(X=X_pairs, Y=Y_pairs)
        assert np.allclose(pmi_gt2.reshape(X_idx.shape), pmi_gt_mat, rtol=1e-5, atol=1e-5)
        assert np.allclose(npmi_gt2.reshape(X_idx.shape), npmi_gt_mat, rtol=1e-5, atol=1e-5)

    G_pmi = {g: np.log(np.clip(G_pmd_xy_pred_mat[g], a_min=1e-5, a_max=None)) for g in G_elements}
    G_pmi_err_tensor = np.stack([pmi_gt_mat - G_pmi[g] for g in G_elements], axis=0)
    pmi_mse = np.einsum('gij,i,j->', G_pmi_err_tensor ** 2, p_Y / p_Y.sum(), p_X / p_Y.sum())
    G_npmi = {g: G_pmi[g] / (-1 * np.log(P_joint_XY_pairs_mat)) for g in G_elements}
    G_npmi_err_tensor = np.stack([npmi_gt_mat - G_npmi[g] for g in G_elements], axis=0)
    npmi_mse = np.einsum('gij,i,j->', G_npmi_err_tensor ** 2, p_Y / p_Y.sum(), p_X / p_Y.sum())

    # Compute estimate of the Mutual information from the samples.
    _P_xy_norm = P_joint_XY_pairs_mat / P_joint_XY_pairs_mat.sum()
    G_MI = [(_P_xy_norm * G_pmd_xy_pred_mat[g]).sum() for g in G_elements]
    MI = np.mean(G_MI)
    MI_gt = np.sum(_P_xy_norm * pmi_gt_mat)
    if debug:
        assert pmi_gt_mat.max() > MI, f"Expectation cannot be larger than the maximum value of the PMI"
    metrics = {"PMD/mse":            pmd_mse,
               "PMD/invariance_err": pmd_G_var,
               "PMD/spectral_norm":  Op_norm,
               "PMI/NPMI/mse":       npmi_mse,  # Numerically unstable.
               "PMI/mse":            pmi_mse,
               "MI/gt":              MI_gt,
               "MI/err":             MI_gt - MI,
               }
    # Sample 4 random conditioning values of x
    range_n_samples = 100
    n_cond_points = 4
    cond_idx = np.random.choice(len(X_pairs), n_cond_points, replace=False)
    X_cond_np = X_pairs[cond_idx]
    X_c_cond = X_c[cond_idx]

    # X_range_np = np.linspace(x_max, x_max, range_n_samples)
    fig_cde, fig_pmd = None, None
    if y_samples.shape[-1] == 1:
        x_max, y_max = max(np.abs(x_samples)), max(np.abs(y_samples))
        Y_range_np = np.linspace(-y_max, y_max, range_n_samples)
        p_Y_range_np = gmm.pdf_y(Y_range_np)
        P_Y_range = torch.from_numpy(p_Y_range_np).to(device=device, dtype=dtype)
        Y_c_range = ((torch.from_numpy(Y_range_np) - y_mean) / torch.sqrt(y_var)).to(device=device, dtype=dtype)

        CDFs_mse, CDFs_gt, CDFs = [], [], []
        for x_cond, x_c_cond in zip(X_cond_np, X_c_cond):  # Comptue p(y | x) for each conditioning value
            cdf_gt = gmm.pdf(X=np.broadcast_to(x_cond, Y_range_np.shape), Y=Y_range_np)

            if isinstance(nn_model, ENCP):
                _x = x_type(torch.broadcast_to(x_c_cond, Y_c_range.shape)).to(device=device, dtype=dtype)
                _y = y_type(Y_c_range).to(device=device, dtype=dtype)
            else:
                _x, _y = (torch.broadcast_to(x_c_cond, Y_range_np.shape), Y_c_range)

            pmd = nn_model.pointwise_mutual_dependency(_x, _y)  # k_r(x,y) ≈ p(x,y) / p(x)p(y)

            cdf = (pmd * P_Y_range).cpu().numpy()  # k(x,y) * p(y) = p(y | x)
            CDFs_gt.append(cdf_gt)
            CDFs.append(cdf)
            CDFs_mse.append(((cdf - cdf_gt) ** 2).mean())

        metrics["CDF/mse"] = np.mean(CDFs_mse)

        if plot:
            # Plot the 4 CDFs each pair in a separate subplot
            import plotly.graph_objects as go
            fig_cde = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=[rf"$p(y \mid x_{k})$" for k in range(4)])
            colors = {
                "p(y)":              "blue",
                r"$\hat{p}(y | x)$": "red",
                r"$p(y | x)$":       "green"
                }

            for i, (cdf, cdf_gt) in enumerate(zip(CDFs, CDFs_gt)):
                fig_cde.add_trace(
                    go.Scatter(x=Y_range_np.squeeze(), y=p_Y_range_np.squeeze(), mode='lines', name="p(y)",
                               line=dict(dash='dash', color=colors["p(y)"]),
                               legendgroup='p(y)', showlegend=(i == 0)), row=i + 1, col=1)
                fig_cde.add_trace(
                    go.Scatter(x=Y_range_np.squeeze(), y=cdf.squeeze(), mode='lines', name=r"$\hat{p}(y | x)$",
                               line=dict(color=colors[r"$\hat{p}(y | x)$"]),
                               legendgroup=r"$\hat{p}(y | x)$", showlegend=(i == 0)), row=i + 1, col=1)
                fig_cde.add_trace(
                    go.Scatter(x=Y_range_np.squeeze(), y=cdf_gt.squeeze(), mode='lines', name=r"$p(y | x)$",
                               line=dict(color=colors[r"$p(y | x)$"]),
                               legendgroup=r"$p(y | x)$", showlegend=(i == 0)), row=i + 1, col=1)
                # Add shaded area for negative values of cdf
                fig_cde.add_trace(go.Scatter(x=Y_range_np.squeeze(), y=np.minimum(cdf.squeeze(), 0), mode='lines',
                                             fill='tozeroy',
                                             fillcolor=colors[r"$\hat{p}(y | x)$"],
                                             opacity=.5,
                                             line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=i + 1, col=1)

            fig_cde.update_layout(template="plotly_white")

    # Return to original device.
    nn_model.to(device=prev_device)

    if plot:

        # Take the n_total_samples from the joint sampling, in the digagonal of the matrix of pairwise pairings.
        G_pmd_gt, G_pmd_pred = [], []
        for g in G_elements:
            pmd_gt_joint = np.diag(pmd_xy_gt_mat)
            pmd_pred_joint = np.diag(G_pmd_xy_pred_mat[G.identity])
            prod_idx = np.random.choice(len(pmd_gt_joint), n_samples_per_g, replace=False)

            pmd_gt_prod = pmd_xy_gt_mat[np.triu_indices(n_samples_per_g, k=1)][prod_idx]
            pmd_pred_prod = G_pmd_xy_pred_mat[G.identity][np.triu_indices(n_samples_per_g, k=1)][prod_idx]

            pmd_gt = np.concatenate([pmd_gt_joint, pmd_gt_prod])
            pmd_pred = np.concatenate([pmd_pred_joint, pmd_pred_prod])

            G_pmd_gt.append(pmd_gt)
            G_pmd_pred.append(pmd_pred)

        G_pmd_gt = np.concatenate(G_pmd_gt)
        G_pmd_pred = np.concatenate(G_pmd_pred)

        if save_data_path is not None:
            save_path = pathlib.Path(save_data_path)
            log.info(f"Saving NPMI data to {save_path.absolute()}")
            np.savez(save_path / "npmi_data.npz", pmd_gt=G_pmd_gt, pmd_pred=G_pmd_pred)
        fig_pmd = plot_pmd_error_distribution(pmd_gt, pmd_pred)
        return metrics, fig_pmd, fig_cde

    return metrics


def get_symmetry_group(cfg: DictConfig):
    group_label = cfg.symm_group
    rep_X, rep_Y = None, None

    # group_label = "C{N}" -> CyclicGroup(N)
    if group_label[0] == "C" and group_label[1:].isdigit():
        N = int(group_label[1:])
        G = escnn.group.CyclicGroup(N)
        if cfg.regular_multiplicity > 0:
            rep_X = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Χ
            rep_Y = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Y
    elif group_label[0] == "D" and group_label[1:].isdigit():
        N = int(group_label[1:])
        G = escnn.group.DihedralGroup(N)
        if cfg.regular_multiplicity > 0:
            rep_X = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Χ
            rep_Y = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Y
    elif group_label.lower() == "ico":
        G = escnn.group.ico_group()
        if cfg.regular_multiplicity == 0:
            rep_X = G.standard_representation
            rep_Y = G.standard_representation
        else:
            rep_X = directsum([G.standard_representation] * cfg.regular_multiplicity)
            rep_Y = directsum([G.standard_representation] * cfg.regular_multiplicity)
    elif group_label.lower() == "octa":
        G, _, _ = escnn.group.O3().subgroup((False, "octa"))
        if cfg.regular_multiplicity == 0:
            rep_X = G.standard_representation
            rep_Y = G.standard_representation
        else:
            rep_X = directsum([G.standard_representation] * cfg.regular_multiplicity)
            rep_Y = directsum([G.standard_representation] * cfg.regular_multiplicity)
    else:
        raise ValueError(f"Group {group_label} not recognized")

    log.info(f"Symmetry Group G: {G} of order {G.order()}")

    if rep_Y is None or rep_X is None:
        if cfg.regular_multiplicity > 0:
            rep_X = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Χ
            rep_Y = directsum([G.regular_representation] * cfg.regular_multiplicity)  # ρ_Y
        elif isinstance(G, escnn.group.CyclicGroup) and G.order() == 2 and cfg.regular_multiplicity == 0:
            rep_X = G.representations['irrep_1']  # ρ_Χ
            rep_Y = G.representations['irrep_1']  # ρ_Y
        elif rep_Y is None or rep_X is None:
            raise ValueError(
                f"G={G} Hx={cfg.x_symm_subgroup_id} Hy={cfg.y_symm_subgroup_id} {cfg.regular_multiplicity}")

    return G, rep_X, rep_Y


@hydra.main(config_path='cfg', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)

    # Symmetry group G. Symmetry subgroup H. H2G: H -> G. G2H: G -> H
    G, rep_X, rep_Y = get_symmetry_group(cfg)

    # GENERATE the training data _______________________________________________________
    seed_everything(cfg.gmm.seed)  # Get same GMM for all training seeds.
    gmm = SymmGaussianMixture(
        n_kernels=cfg.gmm.n_kernels,
        rep_X=rep_X,
        rep_Y=rep_Y,
        mean_max_norm=cfg.gmm.means_max_norm,
        sampling_seed=seed,  # Each seed gets different training samples.
        gmm_seed=cfg.gmm.seed,  # Same GMM model for all seeds.
        x_subgroup_id=cfg.x_symm_subgroup_id,
        y_subgroup_id=cfg.y_symm_subgroup_id,
        )
    seed_everything(seed)  # Random/Selected seed for weight initialization and training.
    (train_samples, val_samples, test_samples), (x_mean, y_mean), (x_var, y_var), datasets = gmm_dataset(
        cfg, gmm, rep_X, rep_Y, device=cfg.device if cfg.data_on_device else 'cpu'
        )

    # x_train, y_train = train_samples
    x_val, y_val = val_samples
    x_test, y_test = test_samples

    # Define the Input and Latent types for ESCNN
    x_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_X])
    y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[rep_Y])
    lat_type = FieldType(
        gspace=escnn.gspaces.no_base_space(G),
        representations=[G.regular_representation] * ceil(cfg.embedding['embedding_dim'] / G.order())
        )

    train_ds, val_ds, test_ds = datasets

    # ESCNN equivariagnt models expect GeometricTensors.
    def geom_tensor_collate_fn(batch) -> [GeometricTensor, GeometricTensor]:
        x_batch, y_batch = default_collate(batch)
        return GeometricTensor(x_batch, x_type), GeometricTensor(y_batch, y_type)

    # Plot the GT joint PDF and MI _______________________________________________________
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    gmm_plot_path = pathlib.Path(run_path).parent.parent / f"joint_pdf.png"

    if not gmm_plot_path.exists():
        if x_type.size == 1 and y_type.size == 1:
            x_samples, y_samples = gmm.simulate(n_samples=5000)
            grid = plot_analytic_joint_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(pathlib.Path(run_path).parent.parent / f"joint_pdf.png")
            grid = plot_analytic_prod_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(pathlib.Path(run_path).parent.parent / f"prod_pdf.png")
            grid = plot_analytic_npmi_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(
                pathlib.Path(run_path).parent.parent / f"normalized_mutual_information.png")
            grid = plot_analytic_pmd_2D(gmm, G=G, rep_X=rep_X, rep_Y=rep_Y, x_samples=x_samples, y_samples=y_samples)
            grid.fig.savefig(
                pathlib.Path(run_path).parent.parent / f"pointwise_mutual_dependency.png")

    # Get the model ______________________________________________________________________
    nnPME = get_model(cfg, x_type, y_type, lat_type)
    print(nnPME)
    # Print the number of parameters
    n_params = sum(p.numel() for p in nnPME.parameters())
    log.info(f"Number of parameters: {n_params}")
    nnPME.to(device=cfg.device)
    # Define the dataloaders
    from NCP.models.equiv_ncp import ENCP
    collate_fn = geom_tensor_collate_fn if isinstance(nnPME, ENCP) else default_collate
    train_dataloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # Define the Lightning module ________________________________________________________
    ncp_lightning_module = TrainingModule(
        model=nnPME,
        optimizer_fn=torch.optim.Adam,
        optimizer_kwargs={"lr": cfg.lr},
        loss_fn=nnPME.loss,
        val_metrics=lambda x: measure_analytic_pmi_error(
            gmm, nnPME, x_val, y_val, x_type, y_type, x_mean, y_mean, x_var, y_var, debug=cfg.debug
            ),
        test_metrics=lambda x: measure_analytic_pmi_error(
            gmm, nnPME, x_test, y_test, x_type, y_type, x_mean, y_mean, x_var, y_var
            ),
        )

    # Define the logger and callbacks
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.exp_name, log_model=False, config=run_cfg)
    scaled_saved_freq = int(5 * cfg.gmm.n_total_samples // cfg.batch_size)
    BEST_CKPT_NAME, LAST_CKPT_NAME = "best", ModelCheckpoint.CHECKPOINT_NAME_LAST
    ckpt_call = ModelCheckpoint(
        dirpath=run_path, filename=BEST_CKPT_NAME, monitor='loss/val', save_top_k=1, save_last=True, mode='min',
        every_n_epochs=scaled_saved_freq,
        )

    # Fix for all runs independent on the train_ratio chosen. This way we compare on effective number of "epochs"
    max_steps = int(cfg.gmm.n_total_samples * cfg.max_epochs // cfg.batch_size)
    scaled_patience = int(cfg.patience * cfg.gmm.n_total_samples // cfg.batch_size)
    early_call = EarlyStopping(monitor='||k(x,y) - k_r(x,y)||/val', patience=scaled_patience, mode='min')

    trainer = lightning.Trainer(accelerator='gpu',
                                devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
                                max_steps=max_steps,
                                logger=logger,
                                enable_progress_bar=True,
                                log_every_n_steps=25,
                                check_val_every_n_epoch=50,
                                callbacks=[ckpt_call, early_call],
                                fast_dev_run=10 if cfg.debug else False,
                                )

    torch.set_float32_matmul_precision('medium')
    last_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / LAST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
    trainer.fit(ncp_lightning_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=last_ckpt_path if last_ckpt_path.exists() else None,
                )

    ncp_lightning_module.to(device="cpu")  # Do testing in CPU as the testing rutine is memory intensive
    # Loads the best model.
    best_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / BEST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
    test_logs = trainer.test(ncp_lightning_module,
                             dataloaders=test_dataloader,
                             ckpt_path=best_ckpt_path if best_ckpt_path.exists() else None,
                             )
    test_metrics = test_logs[0]  # dict: metric_name -> value
    # Save the testing matrics in a csv file using pandas.
    test_metrics_path = pathlib.Path(run_path) / "test_metrics.csv"
    pd.DataFrame(test_metrics, index=[0]).to_csv(test_metrics_path, index=False)

    # Flush the logger.
    logger.finalize(trainer.state)
    # Wand sync
    logger.experiment.finish()

    log.info(f"Logging test metrics ... ")
    m, fig_pmd, fig_cde = measure_analytic_pmi_error(
        gmm, nnPME, x_test, y_test, x_type, y_type, x_mean, y_mean, x_var, y_var,
        plot=True, samples='all', save_data_path=run_path,
        )
    run_desc = f"{pathlib.Path(run_path).parent.name}/{pathlib.Path(run_path).name}"
    if fig_cde is not None:  # Plot
        fig_cde.update_layout(title_text=run_desc, title_font=dict(size=10))
        # Get the str of the name of the current and parent directories
        pio.write_html(fig_cde, file=pathlib.Path(run_path) / f"conditional_pdf.html", auto_open=False)
        pio.write_image(fig_cde, file=pathlib.Path(run_path) / f"conditional_pdf.png", scale=1.5)
    if fig_pmd is not None:
        print("Plotting NPMI")
        # set title PLT fig
        fig_pmd.suptitle(run_desc)
        fig_pmd.tight_layout()
        fig_pmd.savefig(pathlib.Path(run_path) / f"NPMI_pdf.png", dpi=100)
        print()


if __name__ == '__main__':
    main()
