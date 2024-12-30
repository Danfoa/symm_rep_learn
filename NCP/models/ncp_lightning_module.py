# Created by danfoa at 26/12/24
from __future__ import annotations

from copy import deepcopy

import lightning
import torch

from NCP.models.equiv_ncp import ENCPOperator
from NCP.mysc.utils import flatten_dict


class NCPModule(lightning.LightningModule):
    def __init__(
            self,
            model: ENCPOperator,
            optimizer_fn: torch.optim.Optimizer,
            optimizer_kwargs: dict,
            loss_fn: torch.nn.Module | callable,
            ):
        super(NCPModule, self).__init__()
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
        self.loss_fn = loss_fn if loss_fn is not None else model.loss
        self.train_loss = []
        self.val_loss = []

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self._optimizer(self.parameters(), **kw)

    @torch.no_grad()
    def log_metrics(self, metrics: dict, suffix='', batch_size=None):
        flat_metrics = flatten_dict(metrics)
        for k, v in flat_metrics.items():
            name = f"{k}/{suffix}"
            self.log(name, v, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        outputs = self.model(*batch)
        loss, metrics = self.loss_fn(*outputs)

        self.log("loss/train", loss, batch_size=self.get_batch_dim(batch))
        self.log_metrics(metrics, suffix="train", batch_size=self.get_batch_dim(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(*batch)
        loss, metrics = self.loss_fn(*outputs)

        self.log("loss/val", loss, batch_size=self.get_batch_dim(batch))
        self.log_metrics(metrics, suffix="val", batch_size=self.get_batch_dim(batch))
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(*batch)
        loss, metrics = self.loss_fn(*outputs)

        self.log("loss/test", loss, batch_size=self.get_batch_dim(batch))
        self.log_metrics(metrics, suffix="test", batch_size=self.get_batch_dim(batch))
        return loss

    def get_batch_dim(self, batch):
        return batch[0].shape[0]

    def on_fit_end(self):
        pass
