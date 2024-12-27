# Created by danfoa at 19/12/24
from __future__ import annotations
import logging

import escnn.nn
import lightning
import numpy as np
import torch
from escnn.group import directsum
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from NCP.mysc.rep_theory_utils import field_type_to_isotypic_basis
from NCP.mysc.symm_algebra import isotypic_cross_cov
from NCP.nn.layers import SingularLayer

log = logging.getLogger(__name__)

class ENCPOperator(torch.nn.Module):

    def __init__(self, x_fns: EquivariantModule, y_fns: EquivariantModule, gamma=0.01):
        super(ENCPOperator, self).__init__()
        self.gamma = gamma
        # Get field type in the singular-isotypic basis
        assert x_fns.out_type == y_fns.out_type, "Embeddings field types must be the same"
        lat_singular_type = field_type_to_isotypic_basis(x_fns.out_type)
        # Take any input field-type, add a G-equivariant linear layer, parameterizing a change of basis to the
        # Iso-singular basis. (singular functions clustered by isotypic subspaces)
        x2singular = escnn.nn.Linear(in_type=x_fns.out_type, out_type=lat_singular_type, bias=False)
        y2singular = escnn.nn.Linear(in_type=y_fns.out_type, out_type=lat_singular_type, bias=False)
        self.singular_fns_x = escnn.nn.SequentialModule(x_fns, x2singular)
        self.singular_fns_y = escnn.nn.SequentialModule(y_fns, y2singular)
        self.G = self.singular_fns_x.in_type.fibergroup

        assert lat_singular_type.size == lat_singular_type.size, "Fn spaces of diff dimensionality not yet supported"
        # Isotypic subspace are identified by the irrep id associated with the subspace
        self.iso_subspaces_id = [iso_rep.irreps[0] for iso_rep in lat_singular_type.representations]
        self.iso_subspaces_dim = [iso_rep.size for iso_rep in lat_singular_type.representations]
        self.irreps_dim = {irrep_id: self.G.irrep(*irrep_id).size for irrep_id in self.iso_subspaces_id}
        self.iso_subspace_irrep_dim = [self.irreps_dim[id] for id in self.iso_subspaces_id]  # For completeness
        self.iso_irreps_multiplicities = [
            space_dim // self.irreps_dim[id] for space_dim, id in zip(self.iso_subspaces_dim, self.iso_subspaces_id)
            ]
        if self.G.trivial_representation.id in self.iso_subspaces_id:
            self.idx_inv_subspace = self.iso_subspaces_id.index(self.G.trivial_representation.id)
        else:
            self.idx_inv_subspace = None

        # Store the sval trainable parameters / degrees of freedom (dof)
        num_sval_dof = np.sum(self.iso_irreps_multiplicities)  # There is one sval per irrep
        assert num_sval_dof == len(lat_singular_type.irreps), f"{num_sval_dof} != {len(lat_singular_type.irreps)}"
        # TODO: Enable different initializations for this parameter
        self.sval_dof = SingularLayer(num_sval_dof)
        # vector storing the multiplicity of each singular value
        self.sval_multiplicities = torch.tensor(
            [self.irreps_dim[irrep_id] for irrep_id in lat_singular_type.irreps]
            )
        # TODO: Buffers for centering and whitening

    @property
    def svals(self):
        """Ensures the multiplicities of singular values required to satisfy the equivariance constraint.

        Each singular space, can be thought of being associated with an instance of an irrep of the group. The
        dimensionality of the space is hence the dimensionality of the irrep, which implies that the singular values
        have multiplicities equal to the dimensionality of the irrep.

        Returns:
            The singular values in the form of a tensor.
        """
        unique_svals = self.sval_dof.svals
        return unique_svals.repeat_interleave(repeats=self.sval_multiplicities.to(unique_svals.device))


    def forward(self, x: GeometricTensor, y: GeometricTensor):
        """Forward pass of the eNCP operator.

        Computes non-linear transformations of the input random variables x and y, and returns r-dimensional embeddings
        f(x) = [f_1(x), ..., f_r(x)] and h(y) = [h_1(y), ..., h_r(y)] representing the top r-singular functions of
        the conditional expectation operator such that E_p(y|x)[h_i(y)] = σ_i f_i(x) for i=1,...,r.

        Args:
            x: (GeometricTensor) of shape (..., d_x) representing the input x.
            y: (GeometricTensor) of shape (..., d_y) representing the input y.

        Returns:
            fx: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(X)
            hy: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(Y)
        """
        fx = self.singular_fns_x(x)  # f(x) = [f_1(x), ..., f_r(x)]
        hy = self.singular_fns_y(y)  # h(y) = [h_1(y), ..., h_r(y)]
        return fx, hy

    def exp_mutual_information(self, svals: torch.Tensor, fx: torch.Tensor, hy: torch.Tensor):
        # k(x, y) = 1 + Σ_i=1^r σ_i f_i(x) h_i(y)
        # Einsum can do this operation faster in GPU with some internal optimizations.
        k_centered = torch.einsum('i,...i,...i->...', svals, fx.tensor, hy.tensor)
        k = 1 + k_centered
        return k

    def loss(self, fx: GeometricTensor, hy: GeometricTensor):
        """ Compute the loss as in eq(5) of the Neural Conditional Probabilities for Uncertainty Quantification paper.

        Args:
            fx: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(X)
            hy: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(Y)
        Returns:
        """
        assert isinstance(fx, GeometricTensor) and isinstance(hy, GeometricTensor), \
            f"Expected Geometric Tensors got f(x): {type(fx)} and h(y): {type(hy)}"
        assert fx.type == self.singular_fns_x.out_type and hy.type == self.singular_fns_y.out_type
        device, dtype = fx.tensor.device, fx.tensor.dtype
        _tensor_kwargs = dict(device=device, dtype=dtype)
        metrics = dict()

        sqrt_svals = torch.sqrt(self.svals)

        # mi_loss, metrics = self.loss_mi(fx, hy)

        # Center functions by computing the empirical mean only from the G-invariant subspace
        fx_c, mean_fx = self.center_fns(fx)
        hy_c, mean_hy = self.center_fns(hy)
        # Multiply by the square root of the singular values
        fx_c = GeometricTensor(fx_c.tensor * sqrt_svals, fx_c.type)
        hy_c = GeometricTensor(hy_c.tensor * sqrt_svals, hy_c.type)

        # Get projections into isotypic subspaces.  fx_iso[k] = fx^(k), hy_iso[k] = hy^(k)
        fx_iso = [fx_c.tensor[..., s:e] for s, e in zip(fx_c.type.fields_start, fx_c.type.fields_end)]
        reps_Fx_iso = fx_c.type.representations
        hy_iso = [fx_c.tensor[..., s:e] for s, e in zip(hy_c.type.fields_start, hy_c.type.fields_end)]
        reps_Hy_iso = hy_c.type.representations
        sqrt_svals_iso = [sqrt_svals[s:e] for s, e in zip(fx_c.type.fields_start, fx_c.type.fields_end)]

        loss_iso, Cx_iso, Cy_iso, Cxy_iso = [], [], [], []
        orth_iso_x, orth_iso_y, cent_iso_x, cent_iso_y = [], [], [], []
        for fx_k, hy_k, rep_x_k, rep_y_k, sqrt_sval_k in zip(fx_iso, hy_iso, reps_Fx_iso, reps_Hy_iso, sqrt_svals_iso):
            Cx_k = isotypic_cross_cov(X=fx_k, Y=fx_k, rep_X=rep_x_k, rep_Y=rep_x_k, centered=False)
            Cy_k = isotypic_cross_cov(X=hy_k, Y=hy_k, rep_X=rep_y_k, rep_Y=rep_y_k, centered=False)
            Cxy_k = isotypic_cross_cov(X=fx_k, Y=hy_k, rep_X=rep_x_k, rep_Y=rep_y_k, centered=False)
            Ix, Iy = torch.eye(Cx_k.shape[0], **_tensor_kwargs), torch.eye(Cy_k.shape[0], **_tensor_kwargs)
            orth_x_k = torch.linalg.matrix_norm(Cx_k - Ix, ord='fro') ** 2  # ||C_x - I||_F^2
            orth_y_k = torch.linalg.matrix_norm(Cy_k - Iy, ord='fro') ** 2  # ||C_y - I||_F^2
            loss_k = torch.trace((Cx_k @ Cy_k) - 2 * Cxy_k)         # tr(C_x C_y - 2 C_xy)
            Cx_iso.append(Cx_k), Cy_iso.append(Cy_k), Cxy_iso.append(Cxy_k)
            orth_iso_x.append(orth_x_k), orth_iso_y.append(orth_y_k), loss_iso.append(loss_k)

        mi_loss = sum(loss_iso)                                # tr(block_diag(C1,...Cn)) = Σ_i tr(Ci)
        orth_reg = (sum(orth_iso_x) + sum(orth_iso_y))          # ||block_diag(C1,...Cn) - I||_F^2 = Σ_i ||Ci - I||_F^2
        center_reg = mean_fx.norm() ** 2 + mean_hy.norm() ** 2   # Centering regularization
        loss = mi_loss + self.gamma * (orth_reg + 2*center_reg)

        # Metrics computations
        metrics |= dict(mi_loss=mi_loss.detach(),
                        orth_reg=orth_reg.detach() / (self.singular_fns_x.out_type.size ** 2) / 2,
                        center_reg=center_reg.detach())
        for id, Cxy, Cx, Cy in zip(self.iso_subspaces_id, Cxy_iso, Cx_iso, Cy_iso):
            metrics[f"||Cxy||_F-{id}"] = torch.linalg.matrix_norm(Cxy, ord='fro').detach()**2 / Cxy.shape[0]
            metrics[f"||Cx||_F-{id}"] = torch.linalg.matrix_norm(Cx, ord='fro').detach()**2 / Cx.shape[0]
            metrics[f"||Cy||_F-{id}"] = torch.linalg.matrix_norm(Cy, ord='fro').detach()**2 / Cy.shape[0]
        return loss, metrics

    def loss_mi(self, fx: GeometricTensor, hy: GeometricTensor):
        """Compute the loss as in eq(6) of the Neural Conditional Probabilities for Uncertainty Quantification paper.
            Given:
            k(x,y) = 1 + Σ_i=1^r σ_i f_i(x) h_i(y)

            We have that the loss is equivalent to:
            L = E_(x,y)∼p(x)p(y) [k(x,y)^2] - 2 E_(x,y)∼p(x,y) [k(x,y)] + 1
        Args:
            fx: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(X)
            hy: (GeometricTensor) of shape (..., r) representing the singular functions of a subspace of L^2(Y)
        Returns:
            loss: (torch.Tensor) The loss value.
            metrics: 'normalization_err': E_(x,y)∼p(x,y) [1 - k(x,y)]. This value should be close to 0, for perfect
                normalization of the conditional probability density.
        """
        # Randomly permute the batch samples of fx, to break the joint-probability sampling structure.
        fx_p = fx[torch.randperm(fx.shape[0])]
        # Compute the mutual information from sampling of the product of the marginals p(x)p(y)
        k_prod = self.exp_mutual_information(self.svals, fx_p, hy)
        # Compute the mutual information from sampling of the joint distribution p(x,y)
        k_joint = self.exp_mutual_information(self.svals, fx, hy)
        # L = E_(x,y)∼p(x)p(y) [k(x,y)^2] - 2 E_(x,y)∼p(x,y) [k(x,y)] + 1
        loss = torch.mean(k_prod ** 2) - 2 * torch.mean(k_joint) + 1
        metrics = dict(normalization_err=(torch.mean(1 - k_joint)).detach())
        return loss, metrics

    def center_fns(self, f: GeometricTensor):
        """Centers the functions by removing the mean of their G-invariant components.

        TODO: Add running mean.

        Args:
            f: (GeometricTensor) The functions to be centered of shape (..., d), field_type assumed to be in the
            isotypic basis.

        Returns:
            f: (GeometricTensor) (..., d) The centered functions.
            mean_f: (torch.Tensor) (d,) The mean of the functions.
        """
        mean_f = torch.zeros((f.shape[-1]), device=f.tensor.device, dtype=f.tensor.dtype)
        f_c = GeometricTensor(f.tensor.clone(), f.type)

        if self.idx_inv_subspace is not None:
            inv_subspace_start = f_c.type.fields_start[self.idx_inv_subspace]
            inv_subspace_end = f_c.type.fields_end[self.idx_inv_subspace]
            f_inv = f_c.tensor[..., inv_subspace_start:inv_subspace_end]
            # Compute the mean using the batch dimension as samples
            mean_f_inv = f_inv.mean(dim=0)
            f_c.tensor[..., inv_subspace_start:inv_subspace_end] = f_inv - mean_f_inv
            mean_f[inv_subspace_start:inv_subspace_end] = mean_f_inv

        return f_c, mean_f


if __name__ == "__main__":

    G = escnn.group.DihedralGroup(5)

    x_rep = G.regular_representation                       # ρ_Χ
    y_rep = directsum([G.regular_representation] * 10)     # ρ_Y
    lat_rep = directsum([G.regular_representation] * 12)   # ρ_Ζ
    x_rep.name, y_rep.name, lat_rep.name = "rep_X", "rep_Y", "rep_L2"

    type_X = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[x_rep])
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[y_rep])
    lat_type = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[lat_rep])

    χ_embedding = escnn.nn.Linear(type_X, lat_type)
    y_embedding = escnn.nn.Linear(type_Y, lat_type)

    model = ENCPOperator(χ_embedding, y_embedding)

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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=geom_tensor_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=geom_tensor_collate_fn)

    # Train using lightning_______________________________________________
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from NCP.models.ncp_lightning_module import NCPModule

    light_module = NCPModule(model, optimizer_fn=torch.optim.Adam, optimizer_kwargs=dict(lr=1e-3), loss_fn=model.loss)
    # logger = CSVLogger(".test_logs", name="test")
    logger = WandbLogger(project="NCP-GMM-1D", log_model=False)
    logger.watch(light_module)
    trainer = lightning.Trainer(max_epochs=50, logger=logger)

    torch.set_float32_matmul_precision('medium')
    trainer.fit(light_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    # # Training
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # for epoch in range(100):
    #     for i, (x, y) in tqdm(enumerate(train_dataloader)):
    #         optimizer.zero_grad()
    #         loss, metrics = model.loss(x, y)
    #         loss.backward()
    #         optimizer.step()
    #         tqdm.write(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
    # print("Done")


