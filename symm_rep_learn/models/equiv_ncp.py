# Created by danfoa at 19/12/24
from __future__ import annotations

import logging

import escnn.nn
import lightning
import torch
from escnn.group import directsum
from escnn.nn import FieldType, GeometricTensor
from symm_learning.nn import eDataNorm
from symm_learning.nn.disentangled import Change2DisentangledBasis
from symm_learning.representation_theory import isotypic_decomp_rep

from symm_rep_learn.models.ncp import NCP
from symm_rep_learn.nn.losses import symm_orthonormality_regularization

log = logging.getLogger(__name__)


# Equivariant Neural Conditional Probabily (e-NCP) module ==============================================================
class ENCP(NCP):
    def __init__(
        self,
        embedding_x: escnn.nn.EquivariantModule,
        embedding_y: escnn.nn.EquivariantModule,
        orth_reg=0.1,
        centering_reg=0.01,
        momentum=0.9,
    ):
        self.G = embedding_x.out_type.fibergroup
        # Given any Field types of the embeddings of x and y, we need to change basis to the isotypic basis.
        embedding_x_iso = escnn.nn.SequentialModule(embedding_x, Change2DisentangledBasis(in_type=embedding_x.out_type))
        embedding_y_iso = escnn.nn.SequentialModule(embedding_y, Change2DisentangledBasis(in_type=embedding_y.out_type))

        self.x_type, self.y_type = embedding_x.in_type, embedding_y.in_type
        self.rep_x_iso = isotypic_decomp_rep(embedding_x_iso[-1].out_type.representation)
        self.rep_y_iso = isotypic_decomp_rep(embedding_y_iso[-1].out_type.representation)
        fx_type = embedding_x_iso[-1].out_type
        hy_type = embedding_y_iso[-1].out_type

        # Isotypic subspace are identified by the irrep id associated with the subspace
        self.n_iso_subspaces = len(fx_type.representations)
        self.iso_subspace_ids = [iso_rep.irreps[0] for iso_rep in fx_type.representations]
        self.iso_subspace_dims = [iso_rep.size for iso_rep in fx_type.representations]
        self.irreps_dim = {irrep_id: self.G.irrep(*irrep_id).size for irrep_id in self.iso_subspace_ids}
        self.iso_subspace_slice = [slice(s, e) for s, e in zip(fx_type.fields_start, fx_type.fields_end)]
        self.iso_irreps_multiplicities = [
            space_dim // self.irreps_dim[id] for space_dim, id in zip(self.iso_subspace_dims, self.iso_subspace_ids)
        ]
        if self.G.trivial_representation.id in self.iso_subspace_ids:
            self.idx_inv_subspace = self.iso_subspace_ids.index(self.G.trivial_representation.id)
        else:
            self.idx_inv_subspace = None

        # Intialize the NCP module
        super().__init__(
            embedding_x=embedding_x_iso,
            embedding_y=embedding_y_iso,
            embedding_dim_x=fx_type.size,
            embedding_dim_y=hy_type.size,
            orth_reg=orth_reg,
            centering_reg=centering_reg,
        )

        # Replace truncated operator with equivariant linear layer
        # Equivariant Linear layer from lat singular basis to lat singular basis.
        self._Dr = escnn.nn.Linear(in_type=fx_type, out_type=hy_type, bias=False)
        # Reinitialize the (nparams,)
        self._Dr.weights.data = torch.nn.init.uniform_(self._Dr.weights.data, a=-1, b=1)

        # Replace the DataNorm layers with equivariant DataNorm layers
        self.data_norm_x = eDataNorm(fx_type, momentum, compute_cov=True, only_centering=True)
        self.data_norm_y = eDataNorm(hy_type, momentum, compute_cov=True, only_centering=True)

    def forward(self, x: torch.Tensor = None, y: torch.Tensor = None):
        
        x = self.x_type(x) if isinstance(x, torch.Tensor) else x
        y = self.y_type(y) if isinstance(y, torch.Tensor) else y
        fx_c, hy_c = super().forward(x, y)
        
        return fx_c.tensor if fx_c is not None else None, hy_c.tensor if hy_c is not None else None

    # def orthonormality_regularization(self, fx_c: torch.Tensor, hy_c: torch.Tensor):
    #     Cfx, Chy = self.data_norm_x.cov, self.data_norm_y.cov
    #     fx_mean, hy_mean = self.data_norm_x.mean, self.data_norm_y.mean
    #     # orthonormal_reg_fx = ||Cx - I||_F^2 + 2 ||E_p(x) f(x)||_F^2
    #     orthonormal_reg_x, metrics_x = symm_orthonormality_regularization(
    #         x=fx_c, rep_x=self.rep_x_iso, Cx=Cfx, x_mean=fx_mean, var_name="x"
    #     )
    #     # orthonormal_reg_hy = ||Cy - I||_F^2 + 2 ||E_p(y) h(y)||_F^2
    #     orthonormal_reg_y, metrics_y = symm_orthonormality_regularization(
    #         x=hy_c, rep_x=self.rep_y_iso, Cx=Chy, x_mean=hy_mean, var_name="y"
    #     )

    #     metrics = metrics_x | metrics_y  # Combine metrics from both regularizations

    #     return orthonormal_reg_x, orthonormal_reg_y, metrics

    # def orthonormality_penalization(
    #     self, fx_c: GeometricTensor, hy_c: GeometricTensor, return_inner_prod=False, permutation=None
    # ):
    #     """Computes orthonormality and centering regularization penalization for a batch of feature vectors.

    #     Computes finite sample unbiased empirical estimates of the term:
    #     || Vx - I ||_F^2 = || Cx - I ||_F^2 + 2 || E_p(x) f(x) ||^2
    #                      = || ⊕_k Cx_k - I_r_k ||_F^2 + 2 || E_p(x) f^inv (x) ||^2
    #                      = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || Cx_k - I_r_k ||_F^2
    #                      = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || Cx_k ||_F^2 - 2tr(Cx_k) + r_k
    #                      = 2 || E_p(x) f^inv (x) ||^2 + Σ_k || (Dx_k ⊗ I_ρk) ||_F^2 - 2tr(Dx_k ⊗ I_ρk) + r_k
    #                      = 2 || E_p(x) f^inv (x) ||^2 + Σ_k |ρk| (||Dx_k||_F^2 - 2tr(Dx_k)) + r_k
    #     || Vy - I ||_F^2 = 2 || E_p(y) h^inv (y) ||^2 + Σ_k |ρk| (||Dy_k||_F^2 - 2tr(Dy_k)) + r_k
    #     Args:
    #         fx_c: (n_samples, r) Centered feature vectors f_c(x) = [f_c,1(x), ..., f_c,r(x)].
    #         hy_c: (n_samples, r) Centered feature vectors h_c(y) = [h_c,1(y), ..., h_c,r(y)].
    #         return_inner_prod: (bool) If True, return intermediate inner products.
    #         permutation: (torch.Tensor) Permutation tensor to shuffle the samples in the batch.

    #     Returns:
    #         Regularization term as a scalar tensor.
    #     """
    #     assert fx_c.type == self._embedding_x.out_type and hy_c.type == self._embedding_y.out_type  # Iso basis.
    #     # Project embedding into the isotypic subspaces
    #     fx_c_iso, reps_Fx_iso = self._orth_proj_isotypic_subspaces(z=fx_c), fx_c.type.representations
    #     hy_c_iso, reps_Hy_iso = self._orth_proj_isotypic_subspaces(z=hy_c), hy_c.type.representations

    #     Cx_iso_fro_2, Cy_iso_fro_2 = [], []
    #     trCx_iso, trCy_iso = [], []
    #     for k, (fx_ck, rep_x_k, hy_ck, rep_y_k) in enumerate(zip(fx_c_iso, reps_Fx_iso, hy_c_iso, reps_Hy_iso)):
    #         irrep_dim = self.irreps_dim[self.iso_subspace_ids[k]]
    #         # Flatten the realizations along irreducible subspaces, while preserving sampling from the joint dist.
    #         zx = isotypic_signal2irreducible_subspaces(fx_ck, rep_x_k)  # (n_samples * |ρ_k|, r_k / |ρ_k|)
    #         zy = isotypic_signal2irreducible_subspaces(hy_ck, rep_y_k)  # (n_samples * |ρ_k|, r_k / |ρ_k|)
    #         r_xk, r_yk = fx_ck.shape[-1], hy_ck.shape[-1]  # r_k
    #         # Compute unbiased empirical estimates ||Dx_k||_F^2
    #         Dx_k_fro_2 = cov_norm_squared_unbiased_estimation(zx, False, permutation=permutation)
    #         Dy_k_fro_2 = cov_norm_squared_unbiased_estimation(zy, False, permutation=permutation)
    #         # Trace terms without need of unbiased estimation
    #         tr_Dx_k = torch.trace(self.__getattr__(f"Dx_{k}"))  # tr(Dx_k)
    #         tr_Dy_k = torch.trace(self.__getattr__(f"Dy_{k}"))  # tr(Dy_k)
    #         #  ||Cx_k||_F^2 := |ρk| (||Dx_k||_F^2 - 2tr(Dx_k)) + r_k
    #         Cx_k_fro_2 = irrep_dim * (Dx_k_fro_2 - 2 * tr_Dx_k) + r_xk
    #         #  ||Cy_k||_F^2 := |ρk| (||Dy_k||_F^2 - 2tr(Dy_k)) + r_k
    #         Cy_k_fro_2 = irrep_dim * (Dy_k_fro_2 - 2 * tr_Dy_k) + r_yk
    #         Cx_iso_fro_2.append(Cx_k_fro_2)
    #         Cy_iso_fro_2.append(Cy_k_fro_2)
    #         trCx_iso.append(tr_Dx_k)
    #         trCy_iso.append(tr_Dy_k)
    #         # Cx_k_fro_2_biased = torch.linalg.matrix_norm(self.Cx(k)) ** 2
    #         # Cy_k_fro_2_biased = torch.linalg.matrix_norm(self.Cy(k)) ** 2

    #     Cx_I_err_fro_2 = sum(Cx_iso_fro_2)  # ||Cx - I||_F^2 = Σ_k ||Cx_k - I_r_k||_F^2,
    #     Cy_I_err_fro_2 = sum(Cy_iso_fro_2)  # ||Cy - I||_F^2 = Σ_k ||Cy_k - I_r_k||_F^2
    #     trCx = sum(trCx_iso)  # tr(Cx) = Σ_k tr(Dx_k)
    #     trCy = sum(trCy_iso)  # tr(Cy) = Σ_k tr(Dy_k)
    #     # Cx_fro_2_biased = torch.linalg.matrix_norm(self.Cx(None)) ** 2
    #     # Cy_fro_2_biased = torch.linalg.matrix_norm(self.Cy(None)) ** 2

    #     # TODO: Unbiased estimation of mean squared
    #     fx_centering_loss = (self.mean_fx**2).sum()  # ||E_p(x) (f(x_i))||^2
    #     hy_centering_loss = (self.mean_hy**2).sum()  # ||E_p(y) (h(y_i))||^2

    #     # ||Vx - I||_F^2 = ||Cx - I||_F^2 + 2||E_p(x) f(x)||^2
    #     orthonormality_fx = Cx_I_err_fro_2 + 2 * fx_centering_loss
    #     # ||Vy - I||_F^2 = ||Cy - I||_F^2 + 2||E_p(y) h(y)||^2
    #     orthonormality_hy = Cy_I_err_fro_2 + 2 * hy_centering_loss

    #     with torch.no_grad():
    #         embedding_dim_x, embedding_dim_y = self._embedding_x.out_type.size, self._embedding_y.out_type.size
    #         metrics = {
    #             "||Cx||_F^2": Cx_I_err_fro_2 / embedding_dim_x,
    #             "tr(Cx)": trCx / embedding_dim_x,
    #             "||mu_x||": torch.sqrt(fx_centering_loss),
    #             "||Vx - I||_F^2": orthonormality_fx / embedding_dim_x,
    #             #
    #             "||Cy||_F^2": Cy_I_err_fro_2 / embedding_dim_y,
    #             "tr(Cy)": trCy / embedding_dim_y,
    #             "||mu_y||": torch.sqrt(hy_centering_loss),
    #             "||Vy - I||_F^2": orthonormality_hy / embedding_dim_y,
    #         }

    #     if return_inner_prod:
    #         raise NotImplementedError("Inner products not implemented yet.")
    #     else:
    #         return orthonormality_fx, orthonormality_hy, metrics

    @property
    def truncated_operator(self):
        # D_r is diagonal and is stable (that is has eivalues <= 1)
        Dr, _ = self._Dr.expand_parameters()  # Expand the equiv lin layer into its matrix form
        eigval_max = torch.linalg.eigvalsh(Dr)[-1]
        Dr = Dr / eigval_max
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
    def geom_tensor_collate_fn(batch) -> [GeometricTensor, GeometricTensor]:
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
