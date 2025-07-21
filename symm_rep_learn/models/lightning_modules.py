# Created by danfoa at 26/12/24
from __future__ import annotations

import time
from copy import deepcopy

import lightning
import torch

from symm_rep_learn.models.multivariateCQR import MultivariateCQR
from symm_rep_learn.mysc.utils import flatten_dict


class TrainingModule(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_fn: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        loss_fn: torch.nn.Module | callable,
        test_metrics: callable = None,  # Callable at the end of testing
        val_metrics: callable = None,  # Callable at the end of validation
    ):
        super(TrainingModule, self).__init__()
        self._test_metrics = test_metrics
        self._val_metrics = val_metrics

        self.model = model
        self._optimizer = optimizer_fn
        _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
        if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
            self.lr = _tmp_opt_kwargs.pop("lr")
            self.opt_kwargs = _tmp_opt_kwargs
        else:
            self.lr = 1e-3
            raise Warning(
                "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing "
                "it to the optimizer_kwargs argument."
            )
        assert loss_fn is not None, "Loss function must be specified."
        self.loss_fn = loss_fn
        self.train_loss = []
        self.val_loss = []
        self._n_train_samples = torch.tensor(0).to(dtype=torch.int64)

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self._optimizer(self.parameters(), **kw)

    def on_train_start(self) -> None:
        self._n_train_samples *= 0

    def training_step(self, batch, batch_idx):
        out = self.model(*batch)
        if isinstance(out, tuple):
            loss, metrics = self.loss_fn(*out)
        elif isinstance(out, dict):
            loss, metrics = self.loss_fn(**out)
        else:
            loss, metrics = self.loss_fn(out)

        batch_dim = self.get_batch_dim(batch)
        self._n_train_samples += batch_dim
        self.log("loss/train", loss, prog_bar=True, batch_size=batch_dim)
        self.log("train_samples [k]", self._n_train_samples / 1000, on_step=True, on_epoch=False)
        self.log_metrics(metrics, suffix="train", batch_size=batch_dim)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(*batch)
        if isinstance(out, tuple):
            loss, metrics = self.loss_fn(*out)
        elif isinstance(out, dict):
            loss, metrics = self.loss_fn(**out)
        else:
            loss, metrics = self.loss_fn(out)

        self.log("loss/val", loss, prog_bar=True, batch_size=self.get_batch_dim(batch))
        self.log_metrics(metrics, suffix="val", batch_size=self.get_batch_dim(batch))
        return loss

    def test_step(self, batch, batch_idx):
        out = self.model(*batch)
        if isinstance(out, tuple):
            loss, metrics = self.loss_fn(*out)
        elif isinstance(out, dict):
            loss, metrics = self.loss_fn(**out)
        else:
            loss, metrics = self.loss_fn(out)

        self.log("loss/test", loss, prog_bar=True, batch_size=self.get_batch_dim(batch))
        self.log_metrics(metrics, suffix="test", batch_size=self.get_batch_dim(batch))
        return loss

    def on_train_epoch_start(self) -> None:
        self._time = time.time()

    def on_train_epoch_end(self) -> None:
        if hasattr(self, "_time"):
            epoch_time = time.time() - self._time
            self.log("epoch_time", epoch_time, on_epoch=True)

    def on_validation_start(self) -> None:
        self._val_metrics_run = False

    def on_validation_epoch_end(self) -> None:
        if self._val_metrics is not None and self._val_metrics_run is False:
            metrics = self._val_metrics(None)
            self.log_metrics(metrics, suffix="val", batch_size=None)
            self._val_metrics_run = True

    def on_test_start(self) -> None:
        self._test_metrics_run = False

    def on_test_epoch_end(self) -> None:
        if self._test_metrics is not None and self._test_metrics_run is False:
            metrics = self._test_metrics(None)
            self.log_metrics(metrics, suffix="test", batch_size=None)

    @torch.no_grad()
    def log_metrics(self, metrics: dict, suffix="", batch_size=None, **kwargs):
        flat_metrics = flatten_dict(metrics)
        for k, v in flat_metrics.items():
            name = f"{k}/{suffix}"
            self.log(name, v, batch_size=batch_size, **kwargs)

    def get_batch_dim(self, batch):
        return batch[0].shape[0]

    def on_fit_end(self):
        pass


class SupervisedTrainingModule(TrainingModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_fn: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        loss_fn: torch.nn.Module | callable,
        test_metrics: callable = None,  # Callable at the end of testing
        val_metrics: callable = None,  # Callable at the end of validation
    ):
        super(SupervisedTrainingModule, self).__init__(
            model=model,
            optimizer_fn=optimizer_fn,
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=torch.nn.MSELoss(reduce=True, reduction="mean") if loss_fn is None else loss_fn,
            test_metrics=test_metrics,
            val_metrics=val_metrics,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss, metrics = self.loss_fn(y, y_pred)
        batch_dim = self.get_batch_dim(batch)
        self._n_train_samples += batch_dim
        self.log("loss/train", loss, prog_bar=True, batch_size=batch_dim)
        self.log_metrics(metrics, suffix="train", batch_size=batch_dim)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss, metrics = self.loss_fn(y, y_pred)

        self.log("loss/val", loss, prog_bar=True, batch_size=self.get_batch_dim(batch))
        self.log_metrics(metrics, suffix="val", batch_size=self.get_batch_dim(batch))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss, metrics = self.loss_fn(y, y_pred)

        self.log("loss/test", loss, prog_bar=True, batch_size=self.get_batch_dim(batch))
        self.log_metrics(metrics, suffix="test", batch_size=self.get_batch_dim(batch))
        return loss


class CQRLightningModule(TrainingModule):
    def __init__(
        self,
        model: MultivariateCQR,
        optimizer_fn: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        loss_fn: callable,
        **kwargs,
    ):
        super(CQRLightningModule, self).__init__(
            model=model, optimizer_fn=optimizer_fn, optimizer_kwargs=optimizer_kwargs, loss_fn=loss_fn, **kwargs
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        q_lo, q_up = self.model(x)
        loss, metrics = self.loss_fn(q_lo, q_up, target=y)

        batch_dim = self.get_batch_dim(batch)
        self.log("loss/train", loss, prog_bar=True, batch_size=batch_dim)
        self.log_metrics(metrics, suffix="train", batch_size=batch_dim)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        q_lo, q_up = self.model(x)
        loss, metrics = self.loss_fn(q_lo, q_up, target=y)

        batch_dim = self.get_batch_dim(batch)
        self.log("loss/val", loss, prog_bar=True, batch_size=batch_dim)
        self.log_metrics(metrics, suffix="val", batch_size=batch_dim)

    def test_step(self, batch, batch_idx):
        x, y = batch
        q_lo, q_up = self.model(x)
        loss, metrics = self.loss_fn(q_lo, q_up, target=y)

        batch_dim = self.get_batch_dim(batch)
        self.log("loss/test", loss, prog_bar=True, batch_size=batch_dim)
        self.log_metrics(metrics, suffix="test", batch_size=batch_dim)
