import logging
from copy import deepcopy

import ml_confs
import schnetpack
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from NCP.models.ncp import NCPOperator
from NCP.nn.layers import SingularLayer
from NCP.utils import cross_cov, random_split


class NCPLoss:
    @staticmethod
    def __call__(representations: list[torch.Tensor], NCP):
        X1, X2, Y1, Y2 = random_split(*representations, 2)
        U1 = NCP.S(X1)
        U2 = NCP.S(X2)
        V1 = NCP.S(Y1)
        V2 = NCP.S(Y2)

        # centered covariance matrices
        cov_U1 = robust_cov(U1.T)
        cov_U2 = robust_cov(U2.T)
        cov_V1 = robust_cov(V1.T)
        cov_V2 = robust_cov(V2.T)

        cov_U1V1 = cross_cov(U1.T, V1.T, centered=True)
        cov_U2V2 = cross_cov(U2.T, V2.T, centered=True)

        rewards = (0.5 * (torch.sum(cov_U1 * cov_V2) + torch.sum(cov_U2 * cov_V1))
                - torch.trace(cov_U1V1) - torch.trace(cov_U2V2))

        d = X1.shape[-1]
        U1_mean = U1.mean(axis=0, keepdims=True)
        U2_mean = U2.mean(axis=0, keepdims=True)
        V1_mean = V1.mean(axis=0, keepdims=True)
        V2_mean = V2.mean(axis=0, keepdims=True)

        # uncentered covariance matrices
        uc_cov_U1 = cov_U1 + U1_mean @ U1_mean.T
        uc_cov_U2 = cov_U2 + U2_mean @ U2_mean.T
        uc_cov_V1 = cov_V1 + V1_mean @ V1_mean.T
        uc_cov_V2 = cov_V2 + V2_mean @ V2_mean.T

        reg = 0.5 * (
                torch.sum(uc_cov_U1 * uc_cov_U2) - torch.trace(uc_cov_U1) - torch.trace(uc_cov_U2)
                + torch.sum(uc_cov_V1 * uc_cov_V2) - torch.trace(uc_cov_V1) - torch.trace(uc_cov_V2)
        ) + d

        return {
            "objective": rewards + reg,
            "rewards": rewards,
            "regularization": reg,
        }


def compute_covs(encoded_X, encoded_Y):
    _norm = torch.rsqrt(torch.tensor(encoded_X.shape[0]))
    encoded_X = _norm * encoded_X
    encoded_Y = _norm * encoded_Y

    cov_X = torch.mm(encoded_X.T, encoded_X)
    cov_Y = torch.mm(encoded_Y.T, encoded_Y)
    cov_XY = torch.mm(encoded_X.T, encoded_Y)
    return cov_X, cov_Y, cov_XY

class SchNet(schnetpack.model.AtomisticModel):
    def __init__(self, configs: ml_confs.Configs):
        super().__init__(
            input_dtype_str="float32",
            postprocessors=None,
            do_postprocessing=False,
        )
        self.cutoff = configs.cutoff
        self.pwise_dist = schnetpack.atomistic.PairwiseDistances()
        self.radial_basis = schnetpack.nn.GaussianRBF(
            n_rbf=configs.n_rbf, cutoff=self.cutoff
        )
        self.net = schnetpack.representation.SchNet(
            n_atom_basis=configs.n_atom_basis,
            n_interactions=configs.n_interactions,
            radial_basis=self.radial_basis,
            cutoff_fn=schnetpack.nn.CosineCutoff(self.cutoff),
        )
        self.final_lin = torch.nn.Linear(configs.n_atom_basis, configs.n_final_features)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = self.pwise_dist(inputs)
        inputs = self.net(inputs)
        inputs["scalar_representation"] = self.final_lin(
            inputs["scalar_representation"]
        )
        return inputs

class GraphNCP(LightningModule):
    def __init__(
        self,
        model: NCPOperator,
        configs: ml_confs.Configs,
        n_atoms: int,
        optimizer: torch.optim.Optimizer,
        optimizer_kwargs={},
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["configs", "optimizer"])
        _tmp_opt_kwargs = deepcopy(optimizer_kwargs)
        if "lr" in _tmp_opt_kwargs:  # For Lightning's LearningRateFinder
            self.lr = _tmp_opt_kwargs.pop("lr")
            self.opt_kwargs = _tmp_opt_kwargs
        else:
            self.lr = 1e-3
            self.opt_kwargs = {}
            logging.warning(
                "No learning rate specified. Using default value of 1e-3. You can specify the learning rate by passing it to the optimizer_kwargs argument."
            )
        self.model = model
        self.n_atoms = n_atoms
        self.configs = configs
        self.optimizer = optimizer
        self.loss = NCPLoss()
        self.train_loss = []
        self.val_loss = []

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model.U(inputs)

    def training_step(self, batch, batch_idx):
        features = self.model.U(batch)["scalar_representation"]
        features = features.reshape(-1, self.n_atoms, self.configs.n_final_features)
        features = features.mean(dim=1)  # Linear kernel mean embedding

        representations = [features[::2], features[1::2]]
        loss = self.loss(representations, self.model)
        loss = loss['rewards'] + self.configs.gamma * loss['regularization']
        loss_slug = "NCP_loss"

        self.train_loss.append(loss.detach().cpu().numpy())

        metrics = {}
        metrics[f"train/{loss_slug}"] = loss.item()
        cov_X, cov_Y, cov_XY = compute_covs(*representations)

        cov_eigs = torch.linalg.eigvalsh(cov_X)
        top_eigs = torch.topk(cov_eigs, k=5, largest=True).values

        covXY_svals = torch.linalg.svdvals(cov_XY)
        top_svals = torch.topk(covXY_svals, k=5, largest=True).values
        for i, v in enumerate(top_eigs):
            metrics[f"train/cov_eig_{i}"] = v.item()
        for i, v in enumerate(top_svals):
            metrics[f"train/covXY_sval_{i}"] = v.item()

        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features = self.model.U(batch)["scalar_representation"]
        features = features.reshape(-1, self.n_atoms, self.configs.n_final_features)
        features = features.mean(dim=1)  # Linear kernel mean embedding

        representations = [features[::2], features[1::2]]
        loss = self.loss(representations, self.model)
        loss = loss['rewards'] + self.configs.gamma * loss['regularization']

        self.val_loss.append(loss.detach().cpu().numpy())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.configs.batch_size)
        return loss

    def configure_optimizers(self):
        kw = self.opt_kwargs | {"lr": self.lr}
        return self.optimizer(self.parameters(), **kw)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model.U, norm_type=2)
        self.log_dict(norms)


def robust_cov(X):
    C = torch.cov(X)
    Cp = 0.5*(C + C.T)
    return Cp

class SchNet_NCPOperator(NCPOperator):
    def __init__(self, U_operator: SchNet, U_operator_configs: ml_confs.Configs):

        super(NCPOperator, self).__init__()
        d = U_operator_configs.n_final_features
        self.U = U_operator(U_operator_configs)
        self.V = self.U
        self.S = SingularLayer(d)

        # buffers for centering
        self.register_buffer('_mean_Ux', torch.zeros(d))
        self.register_buffer('_mean_Vy', torch.zeros(d))

        # buffers for whitening
        self.register_buffer('_sqrt_cov_X_inv', torch.eye(d))
        self.register_buffer('_sqrt_cov_Y_inv', torch.eye(d))
        self.register_buffer('_sing_val', torch.ones(d))
        self.register_buffer('_sing_vec_l', torch.eye(d))
        self.register_buffer('_sing_vec_r', torch.eye(d))
