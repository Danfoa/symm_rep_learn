"""Real World Experiment of G-Equivariant Regression in Robotics."""

from __future__ import annotations  # Support new typing structure in 3.8 and 3.9

import logging
import pathlib

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
from symm_rep_learn.models.equiv_ncp import ENCP
from symm_rep_learn.models.lightning_modules import SupervisedTrainingModule, TrainingModule
from symm_rep_learn.models.ncp import NCP
import math

log = logging.getLogger(__name__)


def get_model(cfg: DictConfig, state_type: FieldType) -> torch.nn.Module:
    if cfg.model.lower() == "encp":  # Equivariant NCP
        pass
        # import escnn
        # from escnn.nn import FieldType
        # from symm_learning.models.emlp import EMLP

        # from symm_rep_learn.models.equiv_ncp import ENCP

        # G = state_type.representation.group

        # reg_rep = G.regular_representation

        # kwargs = dict(
        #     hidden_layers=cfg.architecture.hidden_layers,
        #     activation=cfg.architecture.activation,
        #     hidden_units=cfg.architecture.hidden_units,
        #     bias=False,
        # )
        # if cfg.architecture.residual_encoder:
        #     from symm_rep_learn.nn.equiv_layers import ResidualEncoder

        #     lat_rep = [reg_rep] * max(1, math.ceil(cfg.architecture.embedding_dim // reg_rep.size))
        #     lat_x_type = FieldType(
        #         gspace=escnn.gspaces.no_base_space(G), representations=list(y_type.representations) + lat_rep
        #     )
        #     lat_y_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=lat_rep)
        #     x_embedding = EMLP(in_type=state_type, out_type=lat_x_type, **kwargs)
        #     y_embedding = ResidualEncoder(encoder=EMLP(in_type=y_type, out_type=lat_y_type, **kwargs), in_type=y_type)
        #     assert y_embedding.out_type.size == x_embedding.out_type.size
        # else:
        #     lat_type = FieldType(
        #         gspace=escnn.gspaces.no_base_space(G),
        #         representations=[reg_rep] * max(1, math.ceil(cfg.architecture.embedding_dim // reg_rep.size)),
        #     )
        #     x_embedding = EMLP(in_type=state_type, out_type=lat_type, **kwargs)
        #     y_embedding = EMLP(in_type=y_type, out_type=lat_type, **kwargs)
        # eNCPop = ENCP(
        #     embedding_x=x_embedding,
        #     embedding_y=y_embedding,
        #     gamma=cfg.gamma,
        #     gamma_centering=cfg.gamma_centering,
        #     learnable_change_of_basis=cfg.learnable_change_basis,
        # )

        # return eNCPop
    elif cfg.model.lower() == "ncp":  # NCP
        from symm_rep_learn.models.evol_op import EvolutionOperator

        # Channels of the last (latent) image representation are the basis functions.
        embedding_dim = cfg.architecture.hidden_units[-1]
        fx = ordered_mnist.CNNEncoder(channels=cfg.architecture.hidden_units, batch_norm=cfg.architecture.batch_norm)
        ncp = EvolutionOperator(
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


def decoder_collect_fn(batch, ncp_model: NCP, augment: bool = False, split: str = "train"):
    imgs = torch.utils.data.default_collate(batch)
    imgs = ordered_mnist.pre_process_images(imgs)

    if augment:
        imgs = ordered_mnist.augment_image(imgs.squeeze(2), split=split)
    else:
        imgs = imgs.squeeze(2)

    encoder_device = next(ncp_model.parameters()).device

    lat_imgs, _ = ncp_model(x=imgs.to(device=encoder_device))
    #     x = lat_imgs, y = imgs
    return lat_imgs.detach(), imgs


def evolve_latent_representations(lat_imgs: torch.Tensor, ncp: NCP):
    """Evolve the latent representations in time using the NCP."""
    next_lat_imgs = ncp.data_norm_y.mean[None] * 0 + torch.einsum("bx,xy->by", lat_imgs, ncp.truncated_operator)
    return next_lat_imgs


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
    # return {}

    # def rec_collect_fn(batch, augment: bool = False, split: str = "train"):
    #     batch = torch.utils.data.default_collate(batch)
    #     imgs = batch["image"]
    #     imgs = ordered_mnist.pre_process_images(imgs)
    #     labels = batch["label"]
    #     if augment:
    #         imgs = ordered_mnist.augment_image(imgs.squeeze(2), split=split)
    #         if split == "test" or split == "val":  # Append aug images to original images in batch dimension
    #             labels = torch.cat((labels, labels), dim=0)
    #     return imgs, labels

    # samples = len(dataset["image"])
    # batch_size = max(samples // 4, 128)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=lambda x: rec_collect_fn(x, augment=augment, split=split),
    # )

    # ncp_device = next(ncp.parameters()).device
    # oracle_classifier.to(device=ncp_device)
    # decoder.to(device=ncp_device)

    # metrics = {}
    # for x, labels in dataloader:
    #     batch_metrics = {}
    #     # Get the latent representations of the images
    #     z_pred = ncp.encode_x(x.to(device=ncp_device))  # (B * W * H,  r)
    #     # Evolve the latent representations in time using the NCP
    #     z_next_pred = evolve_latent_representations(z_pred, ncp)  # (B * W * H, r)
    #     # Decode the evolved latent representations to images
    #     decoder.to(z_next_pred.device)
    #     x_next_pred = decoder(z_next_pred)  # (B, 1, W, H)
    #     x_next_labels = (labels + 1) % 5
    #     # Reconstruct the present images.
    #     x_rec = decoder(z_pred)  # (B, 1, W, H)
    #     x_rec_labels = labels

    #     # Get the predicted labels using the oracle classifier
    #     # x_next_logits_pred = oracle_classifier(x_next_pred)  # (B,)
    #     # x_rec_logits_pred = oracle_classifier(x_rec)  # (B,)

    #     # Compute reconstruction loss and accuracy.
    #     # rec_loss, rec_metrics = ordered_mnist.classification_loss_metrics(
    #     # y_true=x_rec_labels, y_pred=x_rec_logits_pred.to(device=x_rec_labels.device)
    #     # )
    #     # batch_metrics["class_rec_loss"] = rec_loss.item()
    #     # batch_metrics.update({f"rec_{k}": v for k, v in rec_metrics.items()})
    #     # Compute the accuracy of the predicted labels
    #     # pred_loss, next_pred_metrics = ordered_mnist.classification_loss_metrics(
    #     # y_true=x_next_labels, y_pred=x_next_logits_pred.to(device=x_next_labels.device)
    #     # )
    #     # batch_metrics["class_pred_loss"] = pred_loss.item()
    #     # batch_metrics.update({f"class_pred_{k}": v for k, v in next_pred_metrics.items()})

    #     for key, value in batch_metrics.items():
    #         if key not in metrics:
    #             metrics[key] = []
    #         metrics[key].append(value)
    # # Average the metrics over the batches
    # for key, value in metrics.items():
    #     metrics[key] = np.mean(value).item()
    # return metrics

    if current_epoch % plot_kwargs["plot_every_n_epochs"] == 0:
        past_imgs, future_imgs = plot_kwargs["samples"]  # (B, 1, W, H) , # (B, 1, W, H)
        # Get the latent representations of the images
        device = next(ncp_model.parameters()).device
        lat_past_imgs, _ = ncp_model(x=past_imgs.to(device=device))
        rec_past_imgs = decoder(lat_past_imgs)  # (B, 1, W, H)
        # Evolve the latent representations in time using the NCP
        pred_lat_future_imgs = evolve_latent_representations(lat_past_imgs, ncp_model)
        pred_future_imgs = decoder(pred_lat_future_imgs)  # (B, 1, W, H)
        # Get the labels of the future images
        # pred_rec_labels = oracle_classifier(rec_past_imgs).argmax(dim=1)  # (B,)
        # pred_futue_labels = oracle_classifier(pred_future_imgs).argmax(dim=1)  # (B,)
        # Plot the images in a 4 x n_cols grid
        n_cols = 10
        n_rows = 4
        fig = ordered_mnist.plot_predictions_images(
            past_imgs[:n_cols],
            rec_past_imgs[:n_cols],
            future_imgs[:n_cols],
            pred_future_imgs[:n_cols],
            # pred_futue_labels[:n_cols],
            # pred_rec_labels[:n_cols],
            torch.zeros(n_cols),
            torch.zeros(n_cols),
            n_rows=n_rows,
            n_cols=n_cols,
            save_path=pathlib.Path(plot_kwargs["path"]) / f"test_examples_{current_epoch:d}.png",
        )

    return {}


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
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=cfg.dataset.augment_train, split="train"),
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size,
        shuffle=True,
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=cfg.dataset.augment_val, split="val"),
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size,
        shuffle=False,
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=cfg.dataset.augment_test, split="test"),
    )

    # Define the Lightning module ______________________________________________________
    lightning_module = TrainingModule(
        model=ncp_model,
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": cfg.optim.lr},
        loss_fn=ncp_model.loss if hasattr(ncp_model, "loss") else None,
        # val_metrics=lambda _: inference_metrics(
        #     model,
        #     x_cond=past_val,
        #     y_gt=future_val,
        #     y_train=future_train,
        #     x_type=state_type,
        #     y_type=state_type,
        #     alpha=cfg.alpha,
        #     lstsq=cfg.lstsq,
        # ),
        # test_metrics=lambda _: inference_metrics(
        #     model,
        #     x_cond=past_test,
        #     y_gt=future_test,
        #     y_train=future_train,
        #     x_type=state_type,
        #     y_type=state_type,
        #     alpha=cfg.alpha,
        #     lstsq=cfg.lstsq,
        # ),
    )

    # Define the logger and callbacks
    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Run path: {run_path}")
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(save_dir=run_path, project=cfg.proj_name, log_model=False, config=run_cfg)
    # logger.watch(model, log="gradients")
    BEST_CKPT_NAME, LAST_CKPT_NAME = "best", ModelCheckpoint.CHECKPOINT_NAME_LAST
    VAL_METRIC = "||k(x,y) - k_r(x,y)||/val"  # Best low-rank approximation.
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
    check_val_every_n_epoch = cfg.optim.check_val_every_n_epochs
    effective_patience = cfg.optim.patience // check_val_every_n_epoch
    early_call = EarlyStopping(VAL_METRIC, patience=effective_patience, mode="min")

    trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
        max_epochs=cfg.optim.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=25,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[ckpt_call, early_call],
        fast_dev_run=5 if cfg.debug else False,
        num_sanity_val_steps=5,
        reload_dataloaders_every_n_epochs=20,
        limit_train_batches=cfg.optim.limit_train_batches,
    )

    torch.set_float32_matmul_precision("medium")
    last_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / LAST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
    trainer.fit(
        lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=last_ckpt_path if last_ckpt_path.exists() else None,
    )

    # Loads the best model.
    best_ckpt_path = (pathlib.Path(ckpt_call.dirpath) / BEST_CKPT_NAME).with_suffix(ckpt_call.FILE_EXTENSION)
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

    # Train CNN decoder from the learned image representation. ===============================================
    # Keep the model in the target device
    ncp_model.to(device=cfg.device)
    decoder = ordered_mnist.CNNDecoder(
        channels=list(reversed(cfg.architecture.hidden_units)), spatial_size=ncp_model._embedding_x.spatial_size
    )

    def img_reconstruction_loss(y, y_gt) -> tuple[torch.Tensor, dict]:
        metrics = {}
        # Truncate the predictions to have the range of 0, 1 using sigmoid
        y_truncated = y  # torch.sigmoid(y)
        mse_loss = torch.nn.functional.mse_loss(y_truncated, y_gt, reduction="mean")
        return mse_loss, metrics

    val_batch_size = cfg.optim.val_batch_size
    rec_train_dataloader = DataLoader(
        sup_train_ds["image"], batch_size, shuffle=True, collate_fn=lambda x: decoder_collect_fn(x, ncp_model)
    )
    rec_val_dataloader = DataLoader(
        sup_val_ds["image"], val_batch_size, shuffle=True, collate_fn=lambda x: decoder_collect_fn(x, ncp_model)
    )
    rec_test_dataloader = DataLoader(
        sup_test_ds["image"], val_batch_size, shuffle=True, collate_fn=lambda x: decoder_collect_fn(x, ncp_model)
    )

    plot_kwargs = dict(
        samples=next(iter(test_dataloader)), path=run_path, plot_every_n_epochs=cfg.optim.max_epochs // 5
    )
    decoder_module = SupervisedTrainingModule(
        model=decoder,
        optimizer_fn=Adam,
        optimizer_kwargs={"lr": cfg.optim.lr},
        loss_fn=img_reconstruction_loss,
        metrics_prefix="decoder/",
        val_metrics=lambda **kwargs: reconstruction_metrics(
            ncp_model=ncp_model,
            decoder=decoder,
            oracle_classifier=oracle_classifier,
            dataset=sup_val_ds,
            augment=cfg.dataset.augment_val,
            split="val",
            plot_kwargs=plot_kwargs,
            **kwargs,
        ),
        test_metrics=lambda **kwargs: reconstruction_metrics(
            ncp_model=ncp_model,
            decoder=decoder,
            oracle_classifier=oracle_classifier,
            dataset=sup_test_ds,
            augment=cfg.dataset.augment_test,
            split="test",
            plot_kwargs=plot_kwargs,
            **kwargs,
        ),
    )

    effective_patience = cfg.optim.patience // check_val_every_n_epoch
    dec_early_call = EarlyStopping("decoder/loss/val", patience=effective_patience, mode="min")
    dec_ckpt_call = ModelCheckpoint(
        dirpath=pathlib.Path(run_path) / "decoder",
        filename=BEST_CKPT_NAME,
        monitor="decoder/loss/val",
        save_last=True,
        mode="min",
        every_n_epochs=5,
    )
    dec_best_ckpt_path = (pathlib.Path(dec_ckpt_call.dirpath) / BEST_CKPT_NAME).with_suffix(
        dec_ckpt_call.FILE_EXTENSION
    )

    decoder_trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.device] if cfg.device != -1 else cfg.device,  # -1 for all available GPUs
        max_epochs=cfg.optim.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=25,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[dec_ckpt_call, dec_early_call],
        fast_dev_run=25 if cfg.debug else False,
        num_sanity_val_steps=5,
        reload_dataloaders_every_n_epochs=20,
    )

    # Freeze the encoder and oracle classifier
    ncp_model.eval()
    oracle_classifier.eval()
    # Train the decoder
    decoder_trainer.fit(
        decoder_module,
        train_dataloaders=rec_train_dataloader,
        val_dataloaders=rec_val_dataloader,
        ckpt_path=dec_best_ckpt_path if dec_best_ckpt_path.exists() else None,
    )

    # Test the decoder
    decoder_trainer.test(
        decoder_module,
        dataloaders=rec_test_dataloader,
        ckpt_path=dec_best_ckpt_path if dec_best_ckpt_path.exists() else None,
    )

    decoder.eval()
    # Plot 5 examples from the test set of [past_image, rec_image, future_image, pred_image] =========================
    device = "cpu"
    ncp_model.to(device=device)
    oracle_classifier.to(device=device)
    decoder.to(device=device)

    for past_imgs, future_imgs in test_dataloader:  # (B, 1, W, H) , # (B, 1, W, H)
        # Get the latent representations of the images
        lat_past_imgs, _ = ncp_model(x=past_imgs.to(device=device))
        rec_past_imgs = decoder(lat_past_imgs)  # (B, 1, W, H)
        # Evolve the latent representations in time using the NCP
        pred_lat_future_imgs = evolve_latent_representations(lat_past_imgs, ncp_model)
        pred_future_imgs = decoder(pred_lat_future_imgs)  # (B, 1, W, H)
        # Get the labels of the future images
        # pred_rec_labels = oracle_classifier(rec_past_imgs).argmax(dim=1)  # (B,)
        # pred_futue_labels = oracle_classifier(pred_future_imgs).argmax(dim=1)  # (B,)
        # Plot the images in a 4 x n_cols grid
        n_cols = 10
        n_rows = 4
        fig = ordered_mnist.plot_predictions_images(
            past_imgs[:n_cols],
            rec_past_imgs[:n_cols],
            future_imgs[:n_cols],
            pred_future_imgs[:n_cols],
            # pred_futue_labels[:n_cols],
            # pred_rec_labels[:n_cols],
            torch.zeros(n_cols),
            torch.zeros(n_cols),
            n_rows=n_rows,
            n_cols=n_cols,
            save_path=pathlib.Path(run_path) / f"test_examples_{trainer.current_epoch:d}.png",
        )
        if cfg.debug:
            fig.show()

        break
    # Wand sync
    logger.experiment.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("An error occurred", exc_info=True)
