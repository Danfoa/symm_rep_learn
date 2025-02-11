# Created by danfoa at 26/12/24
from __future__ import annotations

from math import ceil
from typing import Any, List, Tuple

import escnn
import torch
from escnn.nn import EquivariantModule, FieldType, FourierPointwise, GeometricTensor

from symm_rep_learn.mysc.rep_theory_utils import isotypic_decomp_rep


class IMLP(EquivariantModule):
    """G-Invariant Multi-Layer Perceptron.

    This module is a G-invariant MLP that extracts G-invariant features from the input tensor. The input tensor
    is first processed by an EMLP module that extracts G-equivariant features. The output of the EMLP module is
    then processed by an IrrepSubspaceNormPooling module that computes the norm of the features in each G-stable
    subspace associated to individual irreducible representations. The output of the IrrepSubspaceNormPooling module
    is a tensor with G-invariant features that can be processed with any NN architecture.
    Default implementation is to add a single linear layer projecting the invariant features to the desired
    output dimension.
    """

    def __init__(
        self,
        in_type: FieldType,
        out_dim: int,  # Number of G-invariant features to extract.
        hidden_layers: int = 1,
        hidden_units: int = 128,
        activation: str = "ReLU",
        bias: bool = False,
        hidden_irreps: list | tuple = None,
    ):
        super(IMLP, self).__init__()

        self.G = in_type.fibergroup
        self.in_type = in_type

        equiv_out_type = FieldType(
            gspace=in_type.gspace,
            representations=[self.G.regular_representation] * max(1, ceil(hidden_units // self.G.order())),
        )

        self.equiv_feature_extractor = EMLP(
            in_type=in_type,
            out_type=equiv_out_type,
            hidden_layers=hidden_layers - 1,  # Last layer will be an unconstrained linear layer.
            hidden_units=hidden_units,
            activation=activation,
            bias=bias,
            hidden_irreps=hidden_irreps,
        )
        self.inv_feature_extractor = IrrepSubspaceNormPooling(in_type=self.equiv_feature_extractor.out_type)
        self.head = torch.nn.Linear(
            in_features=self.inv_feature_extractor.out_type.size, out_features=out_dim, bias=bias
        )
        self.out_type = FieldType(gspace=in_type.gspace, representations=[self.G.trivial_representation] * out_dim)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        x = self.equiv_feature_extractor(x)
        x = self.inv_feature_extractor(x)
        return self.out_type(self.head(x.tensor))

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (len(self.out_type.size),)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-4) -> List[Tuple[Any, float]]:
        self.equiv_feature_extractor.check_equivariance(atol=atol, rtol=rtol)
        self.inv_feature_extractor.check_equivariance(atol=atol, rtol=rtol)
        return super(IMLP, self).check_equivariance(atol=atol, rtol=rtol)


class EMLP(EquivariantModule):
    """
    G-Equivariant Multi-Layer Perceptron.

    This module is a G-equivariant MLP that extracts G-equivariant features from the input tensor.
    """

    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        hidden_layers: int = 1,
        hidden_units: int = 128,
        activation: str = "ReLU",
        bias: bool = True,
        hidden_irreps: list | tuple = None,
    ):
        """EMLP constructor.
        Args:
            in_type: Input field type.
            out_type: Output field type.
            hidden_layers: Number of hidden layers.
            hidden_units: Number of units in the hidden layers.
            activation: Name of the class of activation function.
            bias: Whether to include a bias term in the linear layers.
            hidden_irreps: (Optional) List of irreps to use in the hidden layers. If None, the latent representation
            is constructed from multiplicities of the regular representation.
        """
        super(EMLP, self).__init__()
        assert hidden_layers > 0, "A MLP with 0 hidden layers is equivalent to a linear layer"
        self.G = in_type.fibergroup
        self.in_type, self.out_type = in_type, out_type

        hidden_irreps = hidden_irreps or self.G.regular_representation.irreps
        hidden_irreps = tuple(set(hidden_irreps))
        signal_dim = sum(self.G.irrep(*id).size for id in hidden_irreps)
        # Number of multiplicities / signals in the hidden layers

        if isinstance(hidden_units, int):
            hidden_units = [hidden_units] * hidden_layers
        elif isinstance(hidden_units, list):
            assert len(hidden_units) == hidden_layers, "Number of hidden units must match the number of hidden layers"

        layers = []
        layer_in_type = in_type
        for i, units in enumerate(hidden_units):
            channels = int(ceil(units // signal_dim))
            layer = FourierBlock(
                in_type=layer_in_type, irreps=hidden_irreps, channels=channels, activation=activation, bias=bias
            )
            layer_in_type = layer.out_type
            layers.append(layer)

        # Head layer
        layers.append(escnn.nn.Linear(in_type=layer_in_type, out_type=out_type, bias=bias))
        self.net = escnn.nn.SequentialModule(*layers)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.net(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.net.evaluate_output_shape(input_shape)

    def extra_repr(self) -> str:
        return f"{self.G}-equivariant MLP: in={self.in_type}, out={self.out_type}"


class FourierBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        irreps: tuple | list,
        channels: int,
        activation: str,
        bias: bool = True,
        grid_kwargs: dict = None,
    ):
        super(FourierBlock, self).__init__()
        self.G = in_type.fibergroup
        self._activation = activation
        gspace = in_type.gspace
        grid_kwargs = grid_kwargs or self.get_group_kwargs(self.G)

        self.act = FourierPointwise(
            gspace,
            channels=channels,
            irreps=list(irreps),
            function=f"p_{activation.lower()}",
            inplace=True,
            **grid_kwargs,
        )

        self.in_type = in_type
        self.out_type = self.act.in_type
        self.linear = escnn.nn.Linear(in_type=in_type, out_type=self.act.in_type, bias=bias)

    def forward(self, *input):
        return self.act(self.linear(*input))

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.linear.evaluate_output_shape(input_shape)

    @staticmethod
    def get_group_kwargs(group: escnn.group.Group):
        """Configuration for sampling elements of the group to achieve equivariance."""
        grid_type = "regular" if not group.continuous else "rand"
        N = group.order() if not group.continuous else 10
        kwargs = dict()

        if isinstance(group, escnn.group.DihedralGroup):
            N = N // 2
        elif isinstance(group, escnn.group.DirectProductGroup):
            G1_args = FourierBlock.get_group_kwargs(group.G1)
            G2_args = FourierBlock.get_group_kwargs(group.G2)
            kwargs.update({f"G1_{k}": v for k, v in G1_args.items()})
            kwargs.update({f"G2_{k}": v for k, v in G2_args.items()})

        return dict(N=N, type=grid_type, **kwargs)

    def extra_repr(self) -> str:
        return f"{self.G}-FourierBlock {self._activation}: in={self.in_type.size}, out={self.out_type.size}"


class Change2IsotypicBasis(EquivariantModule):
    """Module that changes the basis of the input tensor to the isotypic basis of the input representation."""

    def __init__(self, in_type: FieldType):
        super(Change2IsotypicBasis, self).__init__()
        self.in_type = in_type
        # Compute the isotypic decomposition of the input representation
        in_rep_iso_basis = isotypic_decomp_rep(in_type.representation)
        # Get the representation per isotypic subspace
        iso_subspaces_reps = in_rep_iso_basis.attributes["isotypic_reps"]
        self.out_type = FieldType(gspace=in_type.gspace, representations=list(iso_subspaces_reps.values()))
        # Change of basis required to move from input basis to isotypic basis
        self.Qin2iso = torch.tensor(in_rep_iso_basis.change_of_basis_inv)
        I = torch.eye(self.Qin2iso.shape[-1]).to(device=self.Qin2iso.device, dtype=self.Qin2iso.dtype)
        self._is_in_iso_basis = torch.allclose(self.Qin2iso, I, atol=1e-5, rtol=1e-5)

    def forward(self, x: GeometricTensor):
        assert x.type == self.in_type, f"Expected input tensor of type {self.in_type}, got {x.type}"
        if self._is_in_iso_basis:
            return self.out_type(x.tensor)
        else:
            # Change of basis
            self.Qin2iso = self.Qin2iso.to(device=x.tensor.device, dtype=x.tensor.dtype)
            x_iso = torch.einsum("ij,...j->...i", self.Qin2iso, x.tensor)
            return self.out_type(x_iso)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def extra_repr(self) -> str:
        return f"Change of basis: {not self._is_in_iso_basis}"


class IrrepSubspaceNormPooling(EquivariantModule):
    def __init__(self, in_type: FieldType):
        super(IrrepSubspaceNormPooling, self).__init__()
        self.G = in_type.fibergroup
        self.in_type = in_type
        self.in2iso = Change2IsotypicBasis(in_type)
        self.in_type_iso = self.in2iso.out_type
        # The number of features is equal to the number of irreducible representations
        n_inv_features = sum(len(rep.irreps) for rep in self.in_type_iso.representations)
        self.out_type = FieldType(
            gspace=in_type.gspace, representations=[self.G.trivial_representation] * n_inv_features
        )

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        x_ = self.in2iso(x)
        x_iso = self._orth_proj_isotypic_subspaces(x_)

        inv_features_iso = []
        for x_k, rep_k in zip(x_iso, self.in_type_iso.representations):
            n_irrep_G_stable_spaces = len(rep_k.irreps)  # Number of G-invariant features = multiplicity of irrep
            # This basis is useful because we can apply the norm in a vectorized way
            # Reshape features to [batch, n_irrep_G_stable_spaces, num_features_per_G_stable_space]
            x_field_p = torch.reshape(x_k, (x_k.shape[0], n_irrep_G_stable_spaces, -1))
            # Compute G-invariant measures as the norm of the features in each G-stable space
            inv_field_features = torch.norm(x_field_p, dim=-1)
            # Append to the list of inv features
            inv_features_iso.append(inv_field_features)

        inv_features = torch.cat(inv_features_iso, dim=-1)
        assert (
            inv_features.shape[-1] == self.out_type.size
        ), f"Expected {self.out_type.size} features, got {inv_features.shape[-1]}"
        return self.out_type(inv_features)

    def _orth_proj_isotypic_subspaces(self, z: GeometricTensor) -> [torch.Tensor]:
        """Compute the orthogonal projection of the input tensor into the isotypic subspaces."""
        assert z.type == self.in_type_iso, f"Expected input tensor of type {self.in_type_iso}, got {z.type}"
        z_iso = [z.tensor[..., s:e] for s, e in zip(z.type.fields_start, z.type.fields_end)]
        return z_iso

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (len(self.out_type.size),)

    def extra_repr(self) -> str:
        return f"{self.G}-Irrep Norm Pooling: in={self.in_type} -> out={self.out_type}"


class EquivResidualEncoder(EquivariantModule):
    """Residual encoder for symm_rep_learn. This encoder processes batches of shape (batch_size, dim_y) and
    returns (batch_size, embedding_dim + dim_y).
    """

    def __init__(self, encoder: EquivariantModule, in_type):
        super(EquivResidualEncoder, self).__init__()
        self.encoder = encoder
        self.in_type = in_type
        self.out_type = FieldType(
            gspace=encoder.out_type.gspace, representations=in_type.representations + encoder.out_type.representations
        )

    def forward(self, input: GeometricTensor):
        embedding = self.encoder(input)
        out = torch.cat([input.tensor, embedding.tensor], dim=-1)
        return self.out_type(out)

    def decode(self, encoded_x: torch.Tensor):
        x = encoded_x[..., self.residual_dims]
        return x

    @property
    def residual_dims(self):
        return slice(0, self.in_type.size)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (len(self.out_type.size),)


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(6)
    x_rep = G.regular_representation  # ρ_Χ

    y_rep = isotypic_decomp_rep(G.regular_representation)  # ρ_Y
    y_iso_reps = tuple(y_rep.attributes["isotypic_reps"].values())

    type_X = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=[x_rep])
    type_Y = FieldType(gspace=escnn.gspaces.no_base_space(G), representations=y_iso_reps)

    emlp = EMLP(in_type=type_X, out_type=type_Y, hidden_layers=2, hidden_units=64, bias=False)
    emlp.check_equivariance(atol=1e-5, rtol=1e-5)

    print(emlp)

    n_samples = 5
    x = GeometricTensor(torch.randn(n_samples, x_rep.size), type_X)
    y = emlp(x)

    y_iso = [y.tensor[..., s:e] for s, e in zip(y.type.fields_start, y.type.fields_end)]

    for irrep_id, y_p in zip(y_rep.attributes["isotypic_reps"].keys(), y_iso):
        print(f"{irrep_id}: \n{torch.linalg.norm(y_p)}")

    # Tests for IMLP
    imlp = IMLP(in_type=type_X, out_dim=2, hidden_layers=5, hidden_units=128, bias=False)
    imlp.check_equivariance(atol=1e-6, rtol=1e-6)
    print(imlp)
