from copy import deepcopy
import torch
import lightning as L

from normflows import ConditionalNormalizingFlow

from NCP.nn.losses import NFLoss

class NFModule(L.LightningModule):
    def __init__(
            self,
            model: ConditionalNormalizingFlow,
            optimizer_fn: torch.optim.Optimizer,
            optimizer_kwargs: dict,
    ):
        super(NFModule, self).__init__()
        self.model = model
        self._optimizer = optimizer_fn
        _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
        if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
            self.lr = _tmp_opt_kwargs.pop("lr")
            self.opt_kwargs = _tmp_opt_kwargs
        else:
            self.lr = 1e-3
            raise Warning(
                "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing it to the optimizer_kwargs argument."
            )
        self.loss_fn = NFLoss()
        self.train_loss = []
        self.val_loss = []

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self._optimizer(self.parameters(), **kw)

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.batch = batch

        X, Y = batch
        loss = self.loss_fn
        l = loss(X, Y, self.model)
        self.log('train_loss', l, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss.append(l.detach().cpu().numpy())

        return l

    def validation_step(self, batch, batch_idx):
        X, Y = batch

        loss = self.loss_fn
        l = loss(X, Y, self.model)
        self.log('val_loss', l, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.append(l.detach().cpu().numpy())
        return l