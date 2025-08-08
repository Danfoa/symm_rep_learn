"""Real World Experiment of G-Equivariant Regression in Robotics."""

from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
import math
import pathlib
import time

import escnn
import hydra
import numpy as np
import pandas as pd
import torch
from escnn.nn import FieldType
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader

import paper.experiments.dynamics.ordered_mnist as ordered_mnist
from paper.experiments.dynamics.dynamics_dataset import TrajectoryDataset
from paper.experiments.dynamics.model.unet import UNet
from symm_rep_learn.inference.ncp import NCPRegressor
from symm_rep_learn.models.equiv_ncp import ENCP
from symm_rep_learn.models.lightning_modules import SupervisedTrainingModule, TrainingModule
from symm_rep_learn.models.ncp import NCP
from symm_rep_learn.nn.layers import Lambda, ResidualEncoder

log = logging.getLogger(__name__)


def get_model(cfg: DictConfig, state_type: FieldType) -> torch.nn.Module:
    if cfg.model.lower() == "encp":  # Equivariant NCP
        from symm_rep_learn.models.equiv_ncp import ENCP

        fx = ordered_mnist.SO2SCNNEncoder(
            channels=cfg.architecture.hidden_units,
            batch_norm=cfg.architecture.batch_norm,
            flatten_img=cfg.flat_embedding,
        )
        eNCPop = ENCP(
            embedding_x=fx,
            embedding_y=fx,
            orth_reg=cfg.gamma,
            centering_reg=cfg.gamma_centering,
            momentum=cfg.momentum,
        )
        return eNCPop
    elif cfg.model.lower() == "ncp":  # NCP
        from symm_rep_learn.models.img_evol_op import ImgEvolutionOperator

        # Channels of the last (latent) image representation are the basis functions.
        # fx = ordered_mnist.CNNEncoderSimple(num_classes=cfg.architecture.embedding_dim)
        # fx = ordered_mnist.UNetEncoder(
        #     out_channels=cfg.architecture.embedding_dim, hidden_channels=cfg.architecture.hidden_units[:2]
        # )
        embedding_dim = cfg.architecture.embedding_dim
        fx = UNet(in_channels=1, out_channels=cfg.architecture.embedding_dim)

        if cfg.architecture.residual_encoder:
            fx = ResidualEncoder(fx, in_dim=1)  # Append Gray-scale image to the latent representation
            embedding_dim += 1  # Increase the embedding dimension by 1 for the residual image
        # fx = ordered_mnist.CNNEncoder(
        #     channels=cfg.architecture.hidden_units,
        #     batch_norm=cfg.architecture.batch_norm,
        #     flat_img=cfg.flat_embedding,
        #     embedding_dim=cfg.architecture.embedding_dim,
        # )
        ncp = ImgEvolutionOperator(
            embedding_state=fx,
            state_embedding_dim=embedding_dim,
            orth_reg=cfg.gamma,
            centering_reg=cfg.gamma_centering,
            momentum=cfg.momentum,
            self_adjoint=cfg.architecture.self_adjoint,
        )
        return ncp
    else:
        raise ValueError(f"Model {cfg.model} not recognized")


def get_dataset(cfg: DictConfig):
    oracle_ckpt_path = ordered_mnist.oracle_ckpt_path
    assert oracle_ckpt_path.exists(), "Need to run ordered_mnist.py first to train oracle."

    data_path = ordered_mnist.data_path
    if not data_path.exists():
        print("Data directory not found, preprocessing data.")
        ordered_mnist.make_dataset(n_classes=5)

    ordered_MNIST = ordered_mnist.load_from_disk(str(data_path))

    torch_kwargs = dict(dtype=torch.float32, device=cfg.device)
    ds_kwargs = dict(
        past_frames=cfg.dataset.past_frames,
        future_frames=cfg.dataset.future_frames,
        time_lag=1,
        shuffle=True,
        **torch_kwargs,
    )

    sup_train_ds = ordered_MNIST["train"]
    sup_val_ds = ordered_MNIST["validation"]
    sup_test_ds = ordered_MNIST["test"]

    train_ds = TrajectoryDataset(trajectories=[sup_train_ds["image"]], **ds_kwargs)
    val_ds = TrajectoryDataset(trajectories=[sup_val_ds["image"]], **ds_kwargs)
    test_ds = TrajectoryDataset(trajectories=[sup_test_ds["image"]], **ds_kwargs)

    # Configure the observation space representations ------------------------------------------------------------------
    import escnn

    r2_act = escnn.gspaces.rot2dOnR2(N=-1)  # 2D grid with SO(2) symmetry
    G = r2_act.fibergroup  # The group SO(2)
    # The input image is a scalar field, corresponding to the trivial representation
    state_type = escnn.nn.FieldType(r2_act, [r2_act.trivial_repr])

    if not (cfg.dataset.past_frames == 1 and cfg.dataset.future_frames == 1):
        raise ValueError("Single frame prediction supported only for now.")

    # Create the CNN encoder model
    oracle_classifier = ordered_mnist.SO2SteerableCNN(n_classes=cfg.dataset.n_classes)

    state_dict = torch.load(oracle_ckpt_path)["state_dict"]
    # Remove "model." prefix from the state_dict keys
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    # Load the state_dict into the oracle classifier
    oracle_classifier.load_state_dict(state_dict, strict=True)
    oracle_classifier.eval()

    return ((train_ds, val_ds, test_ds), (sup_train_ds, sup_val_ds, sup_test_ds), state_type, oracle_classifier)


@torch.no_grad()
def decoder_collect_fn(batch, augment: bool = True, split: str = "train"):
    """Collect latent representations and images for decoder training."""
    p_imgs, f_imgs = ordered_mnist.traj_collate_fn(batch, augment=augment, split=split)
    imgs = torch.cat([p_imgs, f_imgs], dim=0)  # Concatenate past and future images
    return imgs, imgs


def evolve_images(imgs: torch.Tensor, ncp_model: NCP):
    """Evolve images using NCP model."""
    device = next(ncp_model.parameters()).device
    fx_c, _ = ncp_model(x=imgs.to(device), y=None)  # fx: (B, r_x, H, W)
    # Evolve state observations
    hy_cond_x = ncp_model.evolve_latent_state(fx_c=fx_c)
    return fx_c, hy_cond_x  # fx: (B, r_x, H, W), hy_cond_x: (B, r_y, H, W)


def classify_images(imgs: torch.Tensor, ncp_model: NCP, oracle_classifier: torch.nn.Module, decoder: torch.nn.Module):
    z_imgs, z_next_pred = evolve_images(imgs, ncp_model)  # (B, r, W, H), (B, r, W, H)
    rec_imgs = decoder(z_imgs)  # (B, 1, W, H)
    pred_next_imgs = decoder(z_next_pred)  # (B, 1, W, H)

    # Get the predicted labels using the oracle classifier
    rec_logits = oracle_classifier(rec_imgs)
    pred_next_logits = oracle_classifier(pred_next_imgs)

    return z_imgs, z_next_pred, rec_imgs, pred_next_imgs, rec_logits, pred_next_logits


def reconstruction_metrics(
    ncp_model: NCP,
    decoder: torch.nn.Module,
    oracle_classifier: torch.nn.Module,
    dataset,
    augment: bool,
    split: str,
    plot_kwargs: dict = None,
    current_epoch: int = 0,
):
    """Compute the NCP future forcasting capability using the oracle classifier of the ordered MNIST dataset.

    We take images from the dataset. Compute their latent representations using the NCP encoder, and evolve them in time
    via the approximated conditional expectation operator. We then use the (frozen) decoder to take the forcasted latent
    representations and reconstruct the images. Finally, we evaluate the prediction capabilities in terms of the oracle classifier
    by computing the accuracy of the predicted labels.

    """

    samples = len(dataset["image"])
    batch_size = max(samples // 4, 128)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: ordered_mnist.collate_fn(x, augment=augment, split=split),
    )

    ncp_device = next(ncp_model.parameters()).device
    oracle_classifier.to(device=ncp_device)
    decoder.to(device=ncp_device)

    metrics = {}
    for imgs, labels in dataloader:
        batch_metrics = {}
        next_labels = (labels + 1) % 5  # The next label is the current label + 1, modulo 5

        _, _, _, _, rec_logits, pred_next_logits = classify_images(imgs, ncp_model, oracle_classifier, decoder)

        # Compute reconstruction loss and accuracy.
        class_rec_loss, rec_metrics = ordered_mnist.classification_loss_metrics(
            y_true=labels, y_pred=rec_logits.to(device=labels.device)
        )
        batch_metrics["class_rec_loss"] = class_rec_loss.item()
        batch_metrics.update({f"rec_{k}": v for k, v in rec_metrics.items()})
        # Compute the accuracy of the predicted labels
        pred_loss, next_pred_metrics = ordered_mnist.classification_loss_metrics(
            y_true=next_labels, y_pred=pred_next_logits.to(device=next_labels.device)
        )
        batch_metrics["class_pred_loss"] = pred_loss.item()
        batch_metrics.update({f"class_pred_{k}": v for k, v in next_pred_metrics.items()})

        for key, value in batch_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
    # Average the metrics over the batches
    for key, value in metrics.items():
        metrics[key] = np.mean(value).item()

    if current_epoch % plot_kwargs["plot_every_n_epochs"] == 0:
        imgs, next_imgs = plot_kwargs["samples"]
        z_imgs, z_next_pred, rec_imgs, pred_next_imgs, rec_logits, pred_next_logits = classify_images(
            imgs, ncp_model, oracle_classifier, decoder
        )
        rec_labels = rec_logits.argmax(dim=1)  # (B,)
        pred_next_labels = pred_next_logits.argmax(dim=1)  # (B,)
        # Plot the images in a 4 x n_cols grid
        n_cols = 10
        n_rows = 4
        fig = ordered_mnist.plot_predictions_images(
            imgs[:n_cols],
            rec_imgs[:n_cols],
            next_imgs[:n_cols],
            pred_next_imgs[:n_cols],
            pred_next_labels[:n_cols],
            rec_labels[:n_cols],
            n_rows=n_rows,
            n_cols=n_cols,
            save_path=pathlib.Path(plot_kwargs["path"]) / f"test_examples_{current_epoch:d}.png",
        )

    return metrics


@torch.no_grad()
def linear_reconstruction_metrics(
    ncp_model: NCP,
    oracle_classifier: torch.nn.Module,
    train_ds: torch.utils.data.Dataset,
    eval_dl: torch.utils.data.DataLoader,
    augment: bool,
    split: str,
    plot_kwargs: dict = None,
    current_epoch: int = 0,
    plot=False,
):
    """Compute the NCP future forcasting capability using the oracle classifier of the ordered MNIST dataset.

    We take images from the dataset. Compute their latent representations using the NCP encoder, and evolve them in time
    via the approximated conditional expectation operator. We then use forcasted latent
    representations and linearly reconstruct the images.

    To train this linear decoder, we need to:
    1. Compute the latent representations of images in the training set.
    3. Linearly regress the original image. To do this, we need to have the expected mean image
    """
    ncp_device = next(ncp_model.parameters()).device
    ncp_model.eval()

    # Compute the training data for the linear regressor.
    train_dl = DataLoader(
        train_ds,
        batch_size=max(len(train_ds) // 4, 128),
        shuffle=False,
        collate_fn=lambda x: decoder_collect_fn(x, augment=augment, split="train"),
    )

    lin_dec = ncp_model.fit_linear_decoder(train_dataloader=train_dl)

    metrics = {}
    for batch_idx, (imgs, next_imgs) in enumerate(eval_dl):
        batch_metrics = {}
        z_imgs, z_next_imgs_pred = evolve_images(imgs, ncp_model)
        _, z_next_imgs_gt = ncp_model(y=next_imgs.to(device=ncp_device))

        rec_imgs = lin_dec(z_imgs)
        pred_next_imgs = lin_dec(z_next_imgs_pred)

        batch_metrics["latent_lin_pred_err"] = torch.nn.functional.mse_loss(
            input=z_next_imgs_pred.cpu(), target=z_next_imgs_gt.cpu(), reduction="mean"
        ).item()
        batch_metrics["lin_rec_err"] = torch.nn.functional.mse_loss(
            input=rec_imgs.cpu(), target=imgs.cpu(), reduction="mean"
        ).item()
        batch_metrics["lin_pred_err"] = torch.nn.functional.mse_loss(
            input=pred_next_imgs.cpu(), target=next_imgs.cpu(), reduction="mean"
        ).item()

        for key, value in batch_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)

        # Optional plotting:
        if (plot or current_epoch % plot_kwargs["plot_every_n_epochs"] == 0) and batch_idx == 0:
            # Plot the images in a 4 x n_cols grid
            n_cols = 10
            n_rows = 4
            plot_name = f"test_examples_lin_{current_epoch:d}.png" if not plot else "test_examples_lin.png"
            ordered_mnist.plot_predictions_images(
                imgs[:n_cols],
                rec_imgs[:n_cols],
                next_imgs[:n_cols],
                pred_next_imgs[:n_cols],
                None,
                None,
                n_rows=n_rows,
                n_cols=n_cols,
                save_path=pathlib.Path(plot_kwargs["path"]) / plot_name,
            )
            plot = False  # Only plot once per epoch

    # Average the metrics over the batches
    for key, value in metrics.items():
        metrics[key] = np.mean(value).item()

    return metrics


@hydra.main(config_path="cfg", config_name="ordered_mnist", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed if cfg.seed > 0 else np.random.randint(0, 1000)
    seed_everything(seed)
    batch_size = cfg.optim.batch_size

    # Load dataset______________________________________________________________________
    datasets, sup_datasets, state_type, oracle_classifier = get_dataset(cfg)
    train_ds, val_ds, test_ds = datasets
    sup_train_ds, sup_val_ds, sup_test_ds = sup_datasets

    # Get the model_____________________________________________________________________
    ncp_model = get_model(cfg, state_type=state_type)
    print(ncp_model)
    n_trainable_params = sum(p.numel() for p in ncp_model.parameters())
    log.info(f"No. trainable parameters: {n_trainable_params}")
    is_equiv_model = isinstance(ncp_model, ENCP)

    # Define the dataloaders_____________________________________________________________
    train_dataloader = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=cfg.dataset.augment, split="train"),
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size,
        shuffle=True,
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=cfg.dataset.augment, split="val"),
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size,
        shuffle=False,
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=cfg.dataset.augment, split="test"),
    )

    # Define the logger and callbacks
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.proj_name, log_model=False, config=run_cfg)
    # logger.watch(model, log="gradients")
    BEST_CKPT_NAME, LAST_CKPT_NAME = "best", ModelCheckpoint.CHECKPOINT_NAME_LAST
    VAL_METRIC = "||k(x,y) - k_r(x,y)||/val" if not cfg.optim.regression_loss else "loss/val"
    ckpt_call = ModelCheckpoint(
        dirpath=run_path,
        filename=BEST_CKPT_NAME,
        monitor=VAL_METRIC,
        save_top_k=1,
        save_last=True,
        mode="min",
        every_n_epochs=5,
    )
    # Fix for all runs independent on the train_ratio chosen. This way we compare on effective number of "epochs"
    check_val_every_n_epoch = 5  # max(5, int(cfg.optim.max_epochs // cfg.optim.check_val_n_times))
    effective_patience = cfg.optim.patience // check_val_every_n_epoch
    early_call = EarlyStopping(VAL_METRIC, patience=effective_patience, mode="min")
    last_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / LAST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
    best_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / BEST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)

    plot_kwargs = dict(
        samples=next(iter(test_dataloader)), path=run_path, plot_every_n_epochs=check_val_every_n_epoch * 2
    )
    # Define the Lightning module ______________________________________________________
    lightning_module = TrainingModule(
        model=ncp_model,
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": cfg.optim.lr},
        loss_fn=ncp_model.loss if not cfg.optim.regression_loss else ncp_model.regression_loss,
        val_metrics=lambda **kwargs: linear_reconstruction_metrics(
            ncp_model=ncp_model,
            oracle_classifier=oracle_classifier,
            train_ds=train_ds,
            eval_dl=val_dataloader,
            augment=cfg.dataset.augment,
            split="val",
            plot_kwargs=plot_kwargs,
            **kwargs,
        ),
        test_metrics=lambda **kwargs: linear_reconstruction_metrics(
            ncp_model=ncp_model,
            oracle_classifier=oracle_classifier,
            train_ds=train_ds,
            eval_dl=test_dataloader,
            augment=cfg.dataset.augment,
            split="test",
            plot_kwargs=plot_kwargs,
            **kwargs,
        ),
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
        max_epochs=cfg.optim.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[ckpt_call, early_call],
        fast_dev_run=5 if cfg.debug else False,
        num_sanity_val_steps=5,
        reload_dataloaders_every_n_epochs=5,
        limit_train_batches=cfg.optim.limit_train_batches,
        limit_val_batches=cfg.optim.limit_train_batches,
    )

    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=last_ckpt_path if last_ckpt_path.exists() else None,
    )

    # Loads the best model.
    test_logs = trainer.test(
        lightning_module,
        dataloaders=test_dataloader,
        ckpt_path=best_ckpt_path if best_ckpt_path.exists() else None,
    )

    test_metrics = test_logs[0]  # dict: metric_name -> value
    # Save the testing matrices in a csv file using pandas.
    test_metrics_path = pathlib.Path(run_path) / f"test_metrics_{trainer.current_epoch:d}.csv"
    pd.DataFrame(test_metrics, index=[0]).to_csv(test_metrics_path, index=False)
    ncp_model.eval()

    # # Train CNN decoder from the learned image representation. ===============================================
    # # Keep the model in the target device
    # ncp_model.to(device=cfg.device)

    # if isinstance(ncp_model, ENCP):
    #     decoder = ordered_mnist.SO2SCNNDecoder(
    #         in_type=ncp_model._embedding_x.out_type,
    #         spatial_size=7,
    #         channels=list(reversed(cfg.architecture.hidden_units)),
    #         flat_img=cfg.flat_embedding,
    #     )
    # else:
    #     # decoder = ordered_mnist.CNNDecoder(
    #     #     channels=list(reversed(cfg.architecture.hidden_units)),
    #     #     spatial_size=7,  # TODO: make this dynamic
    #     #     flat_img=cfg.flat_embedding,
    #     #     embedding_dim=cfg.architecture.embedding_dim,
    #     # )
    #     # decoder = ordered_mnist.CNNDecoderSimple(num_classes=cfg.architecture.embedding_dim)
    #     decoder = UNet(in_channels=ncp_model.dim_fx, out_channels=1)

    # def img_reconstruction_loss(y, y_gt) -> tuple[torch.Tensor, dict]:
    #     metrics = {}
    #     y = y.tensor if isinstance(y, escnn.nn.GeometricTensor) else y
    #     y_gt = y_gt.tensor if isinstance(y_gt, escnn.nn.GeometricTensor) else y_gt
    #     assert y.shape == y_gt.shape, f"Shapes do not match: {y.shape} != {y_gt.shape}"
    #     # Truncate the predictions to have the range of 0, 1 using sigmoid
    #     mse_loss = torch.nn.functional.mse_loss(y, y_gt, reduction="mean")
    #     return mse_loss, metrics

    # val_batch_size = cfg.optim.val_batch_size
    # rec_train_dataloader = DataLoader(
    #     train_ds,
    #     batch_size,
    #     shuffle=True,
    #     collate_fn=lambda x: decoder_collect_fn(x, augment=cfg.dataset.augment, split="train"),
    # )
    # rec_val_dataloader = DataLoader(
    #     val_ds,
    #     val_batch_size,
    #     shuffle=True,
    #     collate_fn=lambda x: decoder_collect_fn(x, augment=cfg.dataset.augment, split="val"),
    # )
    # rec_test_dataloader = DataLoader(
    #     test_ds,
    #     val_batch_size,
    #     shuffle=True,
    #     collate_fn=lambda x: decoder_collect_fn(x, augment=cfg.dataset.augment, split="test"),
    # )

    # decoder_module = SupervisedTrainingModule(
    #     model=decoder,
    #     optimizer_fn=Adam,
    #     optimizer_kwargs={"lr": cfg.optim.lr},
    #     loss_fn=img_reconstruction_loss,
    #     metrics_prefix="decoder/",
    #     val_metrics=lambda **kwargs: reconstruction_metrics(
    #         ncp_model=ncp_model,
    #         decoder=decoder,
    #         oracle_classifier=oracle_classifier,
    #         dataset=sup_val_ds,
    #         augment=cfg.dataset.augment,
    #         split="val",
    #         plot_kwargs=plot_kwargs,
    #         **kwargs,
    #     ),
    #     test_metrics=lambda **kwargs: reconstruction_metrics(
    #         ncp_model=ncp_model,
    #         decoder=decoder,
    #         oracle_classifier=oracle_classifier,
    #         dataset=sup_test_ds,
    #         augment=cfg.dataset.augment,
    #         split="test",
    #         plot_kwargs=plot_kwargs,
    #         **kwargs,
    #     ),
    # )

    # effective_patience = cfg.optim.patience // check_val_every_n_epoch
    # dec_early_call = EarlyStopping("decoder/loss/val", patience=effective_patience, mode="min")
    # dec_ckpt_call = ModelCheckpoint(
    #     dirpath=pathlib.Path(run_path) / "decoder",
    #     filename=BEST_CKPT_NAME,
    #     monitor="decoder/loss/val",
    #     save_last=True,
    #     mode="min",
    #     every_n_epochs=5,
    # )
    # dec_best_ckpt_path = (pathlib.Path(dec_ckpt_call.dirpath) / BEST_CKPT_NAME).with_suffix(
    #     dec_ckpt_call.FILE_EXTENSION
    # )

    # decoder_trainer = Trainer(
    #     accelerator="gpu",
    #     devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
    #     max_epochs=cfg.optim.max_epochs * 2,
    #     logger=logger,
    #     enable_progress_bar=True,
    #     log_every_n_steps=10,
    #     check_val_every_n_epoch=check_val_every_n_epoch,
    #     callbacks=[dec_ckpt_call, dec_early_call],
    #     fast_dev_run=25 if cfg.debug else False,
    #     num_sanity_val_steps=5,
    #     reload_dataloaders_every_n_epochs=10,
    #     limit_train_batches=cfg.optim.limit_train_batches,
    # )

    # # Freeze the encoder and oracle classifier
    # ncp_model.eval()
    # oracle_classifier.eval()
    # # Train the decoder
    # decoder_trainer.fit(
    #     decoder_module,
    #     train_dataloaders=rec_train_dataloader,
    #     val_dataloaders=rec_val_dataloader,
    #     ckpt_path=dec_best_ckpt_path if dec_best_ckpt_path.exists() else None,
    # )

    # # Test the decoder
    # decoder_trainer.test(
    #     decoder_module,
    #     dataloaders=rec_test_dataloader,
    #     ckpt_path=dec_best_ckpt_path if dec_best_ckpt_path.exists() else None,
    # )

    # decoder.eval()
    # Plot 5 examples from the test set of [past_image, rec_image, future_image, pred_image] =========================
    device = "cpu"
    ncp_model.to(device=device)
    oracle_classifier.to(device=device)

    # Plot predictions.
    linear_reconstruction_metrics(
        ncp_model=ncp_model,
        oracle_classifier=oracle_classifier,
        train_ds=train_ds,
        eval_dl=test_dataloader,
        augment=cfg.dataset.augment,
        split="test",
        plot_kwargs=plot_kwargs,
        current_epoch=trainer.current_epoch,
        plot=True,  # Plot only once per epoch
    )
    # Wand sync
    logger.experiment.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
