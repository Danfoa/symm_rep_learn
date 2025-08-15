# Created by danfoa at 19/12/24
from __future__ import annotations

import logging

import escnn.nn
import lightning
import torch
import torch.nn.functional as F
from escnn.group import directsum
from escnn.nn import FieldType, GeometricTensor
from symm_learning.nn import eDataNorm
from symm_learning.nn.disentangled import Change2DisentangledBasis
from symm_learning.representation_theory import isotypic_decomp_rep

from symm_rep_learn.models.neural_conditional_probability.ncp import NCP

log = logging.getLogger(__name__)


# Equivariant Neural Conditional Probabily (e-NCP) module ==============================================================
class ENCP(NCP):
    def __init__(
        self,
        embedding_x: escnn.nn.EquivariantModule,
        embedding_y: escnn.nn.EquivariantModule,
        **ncp_kwargs,
    ):
        self.G = embedding_x.out_type.fibergroup
        # Given any Field types of the embeddings of x and y, we need to change basis to the isotypic basis.
        embedding_x_iso = escnn.nn.SequentialModule(embedding_x, Change2DisentangledBasis(in_type=embedding_x.out_type))
        embedding_y_iso = escnn.nn.SequentialModule(embedding_y, Change2DisentangledBasis(in_type=embedding_y.out_type))
        fx_type, hy_type = embedding_x_iso[-1].out_type, embedding_y_iso[-1].out_type
        self.x_type, self.y_type = embedding_x.in_type, embedding_y.in_type

        self.rep_x_iso = isotypic_decomp_rep(fx_type.representation)
        self.rep_y_iso = isotypic_decomp_rep(hy_type.representation)

        # Intialize the NCP module
        super().__init__(
            embedding_x=embedding_x_iso,
            embedding_y=embedding_y_iso,
            embedding_dim_x=fx_type.size,
            embedding_dim_y=hy_type.size,
            **ncp_kwargs,
        )

        # Replace truncated operator with equivariant linear layer
        # Equivariant Linear layer from lat singular basis to lat singular basis.
        self.Dr = escnn.nn.Linear(in_type=fx_type, out_type=hy_type, bias=False)
        # Reinitialize the (nparams,)
        self.Dr.weights.data = torch.nn.init.uniform_(self.Dr.weights.data, a=-1, b=1)

        # Replace the DataNorm layers with equivariant DataNorm layers
        self.data_norm_x = eDataNorm(fx_type, self.data_norm_x.momentum, compute_cov=True, only_centering=True)
        self.data_norm_y = eDataNorm(hy_type, self.data_norm_y.momentum, compute_cov=True, only_centering=True)

        # Buffers for spectral normalization power iteration (2D Dr: (out_dim=hy_size, in_dim=fx_size))
        # u in R^{out_dim} (hy_type.size), v in R^{in_dim} (fx_type.size)
        u = F.normalize(self.Dr.weights.new_empty(hy_type.size).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(self.Dr.weights.new_empty(fx_type.size).normal_(0, 1), dim=0, eps=1e-12)
        self.register_buffer("_sn_u", u, persistent=True)
        self.register_buffer("_sn_v", v, persistent=True)

    def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
        x = self.x_type(x) if isinstance(x, torch.Tensor) else x
        y = self.y_type(y) if isinstance(y, torch.Tensor) else y
        fx_c, hy_c = super().forward(x, y)

        return fx_c.tensor if fx_c is not None else None, hy_c.tensor if hy_c is not None else None

    @property
    def truncated_operator(self):
        # Expand the equivariant linear layer into its dense matrix form (out_dim, in_dim)
        Dr, _ = self.Dr.expand_parameters()

        # Spectral normalization via power iteration ------------------------------
        # Copied from SpectralNorm in PyTorch
        device, dtype = Dr.device, Dr.dtype

        do_power_iteration = self.training
        eps = 1e-12
        n_power_iters = 1

        u, v = self._sn_u, self._sn_v
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(n_power_iters):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(Dr.t(), u), dim=0, eps=eps, out=v)
                    u = F.normalize(torch.mv(Dr, v), dim=0, eps=eps, out=u)
                if n_power_iters > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(Dr, v))
        Dr = Dr / sigma
        return Dr


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(5)

    x_rep = G.regular_representation  # ρ_Χ
    y_rep = directsum([G.regular_representation] * 10)  # ρ_Y
    lat_rep = directsum([G.regular_representation] * 12)  # ρ_Ζ
    x_rep.name, y_rep.name, lat_rep.name = "rep_X", "rep_Y", "rep_L2"

    type_X = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[x_rep])
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[y_rep])
    lat_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[lat_rep])

    χ_embedding = escnn.nn.Linear(type_X, lat_type)
    y_embedding = escnn.nn.Linear(type_Y, lat_type)

    model = ENCP(χ_embedding, y_embedding)

    print(model)
    n_samples = 100
    X = torch.randn(n_samples, x_rep.size)
    Y = torch.randn(n_samples, y_rep.size)
    k = model(GeometricTensor(X, type_X), GeometricTensor(Y, type_Y))
    print("Done")

    # Check all training pipeline ========================================
    from torch.utils.data import DataLoader, TensorDataset, default_collate

    batch_size = 256

    # ESCNN equivariant models expect GeometricTensors.
    def geom_tensor_collate_fn(batch) -> list[GeometricTensor]:
        x_batch, y_batch = default_collate(batch)
        return GeometricTensor(x_batch, type_X), GeometricTensor(y_batch, type_Y)

    n_samples = 1000
    X_train, X_val = torch.randn(n_samples, x_rep.size), torch.randn(int(n_samples * 0.15), x_rep.size)
    Y_train, Y_val = torch.randn(n_samples, y_rep.size), torch.randn(int(n_samples * 0.15), y_rep.size)

    # Dataset
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    # Dataloaders
    collate_fn = geom_tensor_collate_fn if isinstance(model, ENCP) else default_collate
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Simple train and backward loop test:

    model.train()
    for _ in range(100):
        for x, y in train_dataloader:
            with torch.autograd.set_detect_anomaly(True):
                fx, hy = model(x, y)
                loss, metrics = model.loss(fx, hy)
                loss.backward()
            break
        break

    # Train using lightning_______________________________________________
    from symm_rep_learn.models.lightning_modules import TrainingModule

    light_module = TrainingModule(
        model, optimizer_fn=torch.optim.Adam, optimizer_kwargs=dict(lr=1e-3), loss_fn=model.loss
    )
    trainer = lightning.Trainer(max_epochs=50)

    torch.set_float32_matmul_precision("medium")
    trainer.fit(light_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    #
    # log_std = torch.nn.Parameter(n,)
    # _std = torch.exp(log_std)
    #
    # std = torch.mean([rep_A(g) _std for g in G.elements])   #  std = 1/|G| Sum_g∈G rep_A(g) _std
