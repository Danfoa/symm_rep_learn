import math
from pathlib import Path

import escnn
import lightning
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, Pad, RandomRotation, Resize

from symm_rep_learn.models.lightning_modules import SupervisedTrainingModule

main_path = Path(__file__).parent
data_path = main_path / "data" / "ordered_mnist"
oracle_ckpt_path = data_path / "oracle.ckpt"

# TRAIN_SAMPLES = {"train": 29210, "val": 1000, "test": 4900}
TRAIN_SAMPLES = {"train": 1000, "val": 250, "test": 1000}


def make_dataset(n_classes: int, val_ratio: float = 0.2):
    """
    Create an ordered MNIST dataset with sequential digit arrangement.

    This function processes the MNIST dataset to create ordered sequences where
    digits appear in sequential order (0, 1, 2, 3, 4, etc.) rather than randomly.
    """
    # Load the full MNIST dataset into memory for faster processing
    MNIST = load_dataset("mnist", keep_in_memory=True)

    # Create separate datasets for each digit class (0-9)
    # Filter the dataset to get only examples for each specific digit
    digit_ds = []
    for i in range(n_classes):
        digit_ds.append(MNIST.filter(lambda example: example["label"] == i, keep_in_memory=True, num_proc=8))

    ordered_MNIST = DatasetDict()

    # Create ordered sequences by interleaving digits sequentially
    # This ensures digits appear in order: 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...
    for split in ["train", "test"]:
        ordered_MNIST[split] = interleave_datasets(
            [ds[split] for ds in digit_ds],
            split=split,
            stopping_strategy="all_exhausted",
        ).select(range(TRAIN_SAMPLES[f"{split}"]))

    # Split training data to create validation set
    # Use the last portion of training data as validation (no shuffling to maintain order)
    _tmp_ds = ordered_MNIST["train"].train_test_split(test_size=val_ratio, shuffle=False)
    ordered_MNIST["train"] = _tmp_ds["train"]
    ordered_MNIST["validation"] = _tmp_ds["test"]

    # Set format to PyTorch tensors for the image and label columns
    ordered_MNIST.set_format(type="torch", columns=["image", "label"])

    # Normalize pixel values from [0, 255] to [0, 1] range
    # Apply this transformation to all examples in the dataset
    ordered_MNIST = ordered_MNIST.map(
        lambda example: {"image": example["image"] / 255.0, "label": example["label"]},
        batched=True,
        keep_in_memory=True,
        num_proc=2,
    )

    # Save the processed dataset to disk for future use
    ordered_MNIST.save_to_disk(data_path)


# CNN Architecture
class CNNEncoderSimple(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNEncoderSimple, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        # fully connected layer, output num_classes classes
        self.out = torch.nn.Sequential(torch.nn.Linear(32 * 7 * 7, num_classes))
        torch.torch.nn.init.orthogonal_(self.out[0].weight)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        # Flatten the output of conv2
        X = X.view(X.size(0), -1)
        output = self.out(X)
        return output


class CNNDecoderSimple(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNNDecoderSimple, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(num_classes, 32 * 7 * 7))

        self.conv1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(16, 1, 5, 1, 2),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetEncoder(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=[16, 32], flat_img=True):
        super(UNetEncoder, self).__init__()
        self.embedding_dim = out_channels
        self.channels = [in_channels] + hidden_channels
        self.n_layers = len(self.channels) - 1  # Number of encoder/decoder layers
        self.flat_img = flat_img

        # Encoder (downsampling path) _____________________________________________________________________________
        self.encoders = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i in range(self.n_layers):
            self.encoders.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=5, stride=1, padding=2),
                    torch.nn.ReLU(inplace=True),
                )
            )
            self.pools.append(torch.nn.MaxPool2d(kernel_size=2))

        # Decoder (upsampling path) _______________________________________________________________________________
        self.upconvs = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()

        # Decoder channels: reverse the encoder channels and add intermediate channels
        decoder_channels = hidden_channels[::-1] + [hidden_channels[0] // 2]

        for i in range(self.n_layers):
            # Simple upsampling (no conv here, just like encoder has simple pooling)
            self.upconvs.append(torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

            # Decoder convolution (after skip connection) - mirrors encoder structure
            if i < self.n_layers - 1:  # Not the final layer
                in_ch = decoder_channels[i] + self.channels[self.n_layers - i]  # upsampled + skip
                out_ch = decoder_channels[i + 1]
                activation = torch.nn.ReLU(inplace=True)
            else:  # Final layer
                in_ch = decoder_channels[i] + self.channels[self.n_layers - i]  # upsampled + skip
                out_ch = out_channels
                activation = torch.nn.Sigmoid()

            self.decoders.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                    activation,
                )
            )

    def forward(self, x):
        # Encoder path - store skip connections
        skip_connections = []

        for i in range(self.n_layers):
            x = self.encoders[i](x)
            skip_connections.append(x)  # Store for skip connection
            x = self.pools[i](x)

        # Decoder path with skip connections
        for i in range(self.n_layers):
            # Upsample
            x = self.upconvs[i](x)
            skip_idx = self.n_layers - 1 - i  # Reverse order: last encoder output first
            skip = skip_connections[skip_idx]
            x = torch.cat([x, skip], dim=1)
            # Decoder convolution
            x = self.decoders[i](x)

        if self.flat_img:
            # (B, out_channels, H, W) -> (B, H, W, out_channels) -> (B * H * W, out_channels)
            x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        return x


# CNN Architecture
class CNNEncoder(torch.nn.Module):
    def __init__(
        self, channels=[32, 64, 64], batch_norm: bool = False, flat_img: bool = False, embedding_dim: int = 64
    ):
        super(CNNEncoder, self).__init__()
        self.channels = channels
        self.flatten_img = flat_img
        self.hidden_channels = channels
        self.embedding_dim = embedding_dim

        # Covolutional parameters
        conv_kwargs_low_rf = dict(kernel_size=5, stride=1, padding=2, dilation=1)
        conv_kwargs_high_rf = dict(kernel_size=5, stride=1, padding=6, dilation=3)
        # Downsampling parameters
        conv_kwargs_low_rf_d = dict(kernel_size=4, stride=2, padding=1, dilation=1)
        conv_kwargs_high_rf_d = dict(kernel_size=4, stride=2, padding=4, dilation=3)

        self.conv1_low_rf = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels[0] // 2, bias=True, **conv_kwargs_low_rf),
            torch.nn.BatchNorm2d(channels[0] // 2) if batch_norm else torch.nn.Identity(),
            torch.nn.ELU(),
        )
        self.conv1_high_rf = torch.nn.Sequential(
            torch.nn.Conv2d(1, channels[0] // 2, bias=True, **conv_kwargs_high_rf),
            torch.nn.BatchNorm2d(channels[0] // 2) if batch_norm else torch.nn.Identity(),
            torch.nn.ELU(),
        )

        self.conv2_low_rf = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1] // 2, bias=True, **conv_kwargs_low_rf_d),
            torch.nn.BatchNorm2d(channels[1] // 2) if batch_norm else torch.nn.Identity(),
            torch.nn.ELU(),
        )
        self.conv2_high_rf = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1] // 2, bias=True, **conv_kwargs_high_rf_d),
            torch.nn.BatchNorm2d(channels[1] // 2) if batch_norm else torch.nn.Identity(),
            torch.nn.ELU(),
        )

        self.conv3 = torch.nn.Sequential(  # (HC1, 14, 14) -> (HC2, 7, 7)
            torch.nn.Conv2d(channels[1], channels[2], bias=False, **conv_kwargs_low_rf_d),
            torch.nn.BatchNorm2d(channels[2]) if batch_norm else torch.nn.Identity(),
            torch.nn.ELU(),
        )
        self.spatial_size = 7

        if flat_img:
            self.head = torch.nn.Linear(
                in_features=channels[2] * self.spatial_size * self.spatial_size,
                out_features=self.embedding_dim,
            )

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add a channel dimension if needed
        # print(f"Input shape: {x.shape}")
        x_low = self.conv1_low_rf(x)
        x_high = self.conv1_high_rf(x)
        x = torch.cat([x_low, x_high], dim=1)
        x_low = self.conv2_low_rf(x)
        x_high = self.conv2_high_rf(x)
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv3(x)  # Shape: (B, hidden_channels[1], 7, 7)

        assert x.shape[2] == x.shape[3] == self.spatial_size, f"Expected size {self.spatial_size}, got {x.shape}"

        if self.flatten_img:
            # Flatten image (B, C, H, W) -> (B, H * W * C)
            x = x.view(x.shape[0], -1)
            x = self.head(x)  # Apply the linear head
            return x
        else:
            # Reshape from (B, C, H, W) to (B * H * W, C)
            B, C, H, W = x.shape
            output = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            # print(f"Flattened output shape: {output.shape}")
            return output


# A decoder which is specular to CNNEncoder, starting with reshaping from (B*H*W, C) back to (B, C, H, W)
class CNNDecoder(torch.nn.Module):
    def __init__(self, spatial_size: int, channels=[32, 16], flat_img: bool = False, embedding_dim: int = 64):
        super(CNNDecoder, self).__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.flat_img = flat_img
        self.embedding_dim = embedding_dim

        conv_kwargs_low_rf = dict(kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv1 = torch.nn.Sequential(  # (F, 7, 7) -> (HC1//2, 14, 14)
            torch.nn.ConvTranspose2d(channels[0], channels[1], bias=True, **conv_kwargs_low_rf),
            torch.nn.ELU(),
        )

        self.conv2 = torch.nn.Sequential(  # (HC1, 14, 14) -> (1, 29, 29)
            torch.nn.ConvTranspose2d(channels[1], 1, bias=True, **conv_kwargs_low_rf),
            torch.nn.Sigmoid(),
        )

        if flat_img:
            self.head = torch.nn.Linear(
                in_features=self.embedding_dim,
                out_features=channels[0] * self.spatial_size * self.spatial_size,
            )

    def forward(self, x):
        if self.flat_img:
            x = self.head(x)
            # Reshape from (B, C * H * W) to (B, C, H, W)
            x = x.view(x.shape[0], self.channels[0], self.spatial_size, self.spatial_size)
        else:
            # x shape: (B * H * W, C)
            # Infer batch size from input
            total_spatial_elements = x.size(0)
            assert total_spatial_elements % (self.spatial_size * self.spatial_size) == 0, (
                f"Expected batch size to be divisible by {self.spatial_size}x{self.spatial_size}, got {total_spatial_elements}"
            )
            batch_size = total_spatial_elements // (self.spatial_size * self.spatial_size)
            # print(f"Input shape: {x.shape}")
            # Reshape from (B * H * W, C) to (B, H, W, C) to (B, C, H, W)
            x = x.view(batch_size, self.spatial_size, self.spatial_size, self.channels[0])
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # print(f"Reshaped to: {x.shape}")
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        return x


class SO2SCNNEncoder(escnn.nn.EquivariantModule):
    """SO(2) equivariant CNN encoder for ordered MNIST."""

    def __init__(
        self, channels=[32, 64, 64], batch_norm: bool = False, flatten_img: bool = False, embedding_dim: int = 64
    ):
        super(SO2SCNNEncoder, self).__init__()
        self.channels = channels
        self.batch_norm = batch_norm
        self.flatten_img = flatten_img
        self.embedding_dim = embedding_dim
        # The model is equivariant under all planar rotations
        self.r2_act_gs = escnn.gspaces.rot2dOnR2(N=-1)
        self.G = self.r2_act_gs.fibergroup
        # The input image is a scalar field (grey values), corresponding to the trivial representation
        self.in_type = escnn.nn.FieldType(self.r2_act_gs, [self.r2_act_gs.trivial_repr])
        # Covolutional parameters
        conv_kwargs_low_rf = dict(kernel_size=5, stride=1, padding=2, dilation=1)
        conv_kwargs_high_rf = dict(kernel_size=5, stride=1, padding=6, dilation=3)
        # Downsampling parameters
        conv_kwargs_low_rf_d = dict(kernel_size=4, stride=2, padding=1, dilation=1)
        conv_kwargs_high_rf_d = dict(kernel_size=4, stride=2, padding=4, dilation=3)

        reg_channels = [max(1, math.ceil(c / 7)) for c in channels]
        reg_channels[0] = max(1, reg_channels[0] // 2)
        reg_channels[1] = max(1, reg_channels[1] // 2)

        act1 = escnn.nn.FourierELU(self.r2_act_gs, reg_channels[0], irreps=self.G.bl_irreps(3), N=16)
        block1_out_type = act1.out_type + act1.out_type
        act2 = escnn.nn.FourierELU(self.r2_act_gs, reg_channels[1], irreps=self.G.bl_irreps(3), N=16)
        block2_out_type = act2.out_type + act2.out_type
        act3 = escnn.nn.FourierELU(self.r2_act_gs, reg_channels[2], irreps=self.G.bl_irreps(3), N=16)

        self.conv1_low_rf = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, act1.in_type, bias=True, **conv_kwargs_low_rf),
            escnn.nn.IIDBatchNorm2d(act1.in_type) if batch_norm else torch.nn.Identity(),
            act1,
        )
        self.conv1_high_rf = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, act1.in_type, bias=True, **conv_kwargs_high_rf),
            escnn.nn.IIDBatchNorm2d(act1.in_type) if batch_norm else torch.nn.Identity(),
            act1,
        )

        self.conv2_low_rf = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(block1_out_type, act2.in_type, bias=True, **conv_kwargs_low_rf_d),
            escnn.nn.IIDBatchNorm2d(act2.in_type) if batch_norm else torch.nn.Identity(),
            act2,
        )
        self.conv2_high_rf = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(block1_out_type, act2.in_type, bias=True, **conv_kwargs_high_rf_d),
            escnn.nn.IIDBatchNorm2d(act2.in_type) if batch_norm else torch.nn.Identity(),
            act2,
        )

        self.conv3 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(block2_out_type, act3.in_type, bias=False, **conv_kwargs_low_rf_d),
            escnn.nn.IIDBatchNorm2d(act3.in_type) if batch_norm else torch.nn.Identity(),
            act3,
        )

        self.out_type = self.conv3.out_type  # Final output type
        self.spatial_size = 7  # Output spatial size after convolutions

        if flatten_img:
            # Head that flattens the gird feature maps. (B, C, H, W) -> (B, C*H*W)
            linear_base_space = escnn.gspaces.no_base_space(self.G)
            feat_type = escnn.nn.FieldType(linear_base_space, [act3.out_type.representation] * self.spatial_size**2)
            out_rep = self.G.spectral_regular_representation(*self.G.bl_irreps(3), name="embedding_rep")
            out_rep_multiplicity = math.ceil(self.embedding_dim // out_rep.size)
            self.out = escnn.nn.Linear(
                in_type=feat_type,
                out_type=escnn.nn.FieldType(linear_base_space, [out_rep] * out_rep_multiplicity),
                bias=False,
            )

            self.out_type = self.out.out_type  # Final output type

    def forward(self, input: torch.Tensor | escnn.nn.GeometricTensor):
        if not isinstance(input, escnn.nn.GeometricTensor):
            x = self.in_type(input)
        else:
            x = input
        assert x.type == self.in_type

        x_low = self.conv1_low_rf(x)
        x_high = self.conv1_high_rf(x)
        x = torch.cat([x_low.tensor, x_high.tensor], dim=1)
        # print(f"After conv1: {x.shape}")
        x_low = self.conv2_low_rf(self.conv2_low_rf.in_type(x))
        x_high = self.conv2_high_rf(self.conv2_high_rf.in_type(x))
        x = torch.cat([x_low.tensor, x_high.tensor], dim=1)
        # print(f"After conv2: {x.shape}")
        x = self.conv3(self.conv3.in_type(x))  # Shape: (B, hidden_channels[1], 7, 7)
        # print(f"After conv3: {x.shape}")

        B, C, H, W = x.tensor.shape
        if self.flatten_img:  # Extract the tensor and flatten correctly for group representation
            # Shape: (B, C, H, W) -> (B, H, W, C) -> (B, H*W*C)
            # This preserves pixel-wise feature grouping for equivariant features
            x_flat = x.tensor.permute(0, 2, 3, 1).reshape(B, -1)
            # print(f"flattened: {x_flat.shape}")
            output = self.out(self.out.in_type(x_flat))

            # Test that we can reconstruct the original shape
            x_rec = x_flat.view(B, H, W, C).permute(0, 3, 1, 2)
            assert torch.allclose(x_rec, x.tensor), (
                f"Error: Reconstructed shape {x_rec.shape} does not match original {x.tensor.shape}"
            )
            # Thesr the first fiber is as expectedly stored in the flat vector.
            first_fiber = x.tensor.permute(0, 2, 3, 1)[:, 0, 0, :]
            first_fiber_expected = x_flat[:, :C]
            assert torch.allclose(first_fiber, first_fiber_expected), (
                f"Error: First fiber {first_fiber.shape} does not match expected {first_fiber_expected.shape}"
            )
            return output
        else:
            output = x.tensor.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            # print(f"Flattened output shape: {output.shape}")
            return self.out_type(output)

    def evaluate_output_shape(self, input_shape):
        B = input_shape[0]
        return None


class SO2SCNNDecoder(escnn.nn.EquivariantModule):
    """SO(2) equivariant CNN decoder for ordered MNIST - counterpart to SO2SCNNEncoder."""

    def __init__(self, in_type: escnn.nn.FieldType, spatial_size: int, channels=[32, 16], flat_img: bool = True):
        super(SO2SCNNDecoder, self).__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.flat_img = flat_img

        # The model is equivariant under all planar rotations
        self.r2_act = escnn.gspaces.rot2dOnR2(N=-1)
        self.G = self.r2_act.fibergroup

        reg_channels = [max(1, math.ceil(c / 7)) for c in channels]
        act1 = escnn.nn.FourierELU(self.r2_act, reg_channels[0], irreps=self.G.bl_irreps(3), N=16)
        act2 = escnn.nn.FourierELU(self.r2_act, reg_channels[1], irreps=self.G.bl_irreps(3), N=16)
        act3 = escnn.nn.FourierPointwise(
            self.r2_act, 1, irreps=[self.r2_act.trivial_repr.id], N=16, function="p_sigmoid"
        )

        self.in_type = in_type  # Image embedding
        self.out_type = act3.out_type  # Final output type
        # if self.flat_img:
        # Define the unflattened type.
        linear_base_space = escnn.gspaces.no_base_space(self.G)
        flat_feat_type = escnn.nn.FieldType(linear_base_space, [act1.in_type.representation] * spatial_size**2)
        # Linear map from flat_embedding to flattened spatial features
        self.fc = escnn.nn.Linear(in_type=self.in_type, out_type=flat_feat_type)

        conv_kwargs_low_rf = dict(kernel_size=4, stride=2, padding=1, dilation=1)
        # First decoder block: Upsample → Activation → ConvTranspose:   fx8x8 → hc1x16x16
        self.deconv1 = escnn.nn.SequentialModule(
            escnn.nn.R2ConvTransposed(act1.in_type, act1.in_type, bias=True, **conv_kwargs_low_rf),
            act1,
        )
        # Second decoder block: Upsample → Activation → ConvTranspose:   hc1x16x16 → hc2x32x32
        self.deconv2 = escnn.nn.SequentialModule(
            escnn.nn.R2ConvTransposed(act1.out_type, act2.in_type, bias=True, **conv_kwargs_low_rf),
            act2,
        )
        # Final invariant block mapping to gray scale values.
        self.inv_map = escnn.nn.SequentialModule(
            escnn.nn.R2ConvTransposed(act2.out_type, self.out_type, kernel_size=1, bias=True),
            act3,
        )

    def forward(self, x: torch.Tensor | escnn.nn.GeometricTensor):
        if not isinstance(x, escnn.nn.GeometricTensor):
            x = self.in_type(x)  # Wrap input tensor in a GeometricTensor

        assert x.type == self.in_type

        # Flat embedding linear map to flatten spatial fibers
        spatial_features = self.fc(x).tensor
        # Unflatten spatial fibers.
        # Shape: (B, H*W*C) -> (B, H, W, C) -> (B, C, H, W)
        B = spatial_features.shape[0]
        H, W = self.spatial_size, self.spatial_size  # Target spatial dimensions after encoder's pooling
        C = self.deconv1.in_type.size
        # Reshape to spatial format and permute to (B, C, H, W)
        x_img = spatial_features.view(B, H, W, C).permute(0, 3, 1, 2)

        # Apply reverse operations: each block now contains Upsample → Activation → ConvTranspose
        x_img = self.deconv1.in_type(x_img)  # Wrap input tensor in a GeometricTensor
        x_img = self.deconv1(x_img)
        # print(f"After deconv1: {x_img.shape}")
        x_img = self.deconv2(x_img)  # 16x16 -> 32x32 (16→1 channels)
        # print(f"After deconv2: {x_img.shape}")
        x_img = self.inv_map(x_img)
        # print(f"After conv3: {x_img.shape}")
        # Extract final tensor
        output = x_img

        return output

    def evaluate_output_shape(self, input_shape):
        B = input_shape[0]
        return (B, 1, self.spatial_size, self.spatial_size)


import escnn


class SO2SteerableCNN(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(SO2SteerableCNN, self).__init__()

        # The model is equivariant under all planar rotations
        self.r2_act = escnn.gspaces.rot2dOnR2(N=-1)

        # The group SO(2)
        self.G = self.r2_act.fibergroup

        # The input image is a scalar field, corresponding to the trivial representation
        in_type = escnn.nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # We store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # We need to mask the input image since the corners are moved outside the grid under rotations
        # self.mask = escnn.nn.MaskModule(in_type, 29, margin=1)

        # convolution 1
        # first we build the non-linear layer, which also constructs the right feature type
        # we choose 8 feature fields, each transforming under the regular representation of SO(2) up to frequency 3
        # When taking the ELU non-linearity, we sample the feature fields on N=16 points
        activation1 = escnn.nn.FourierELU(self.r2_act, 8, irreps=self.G.bl_irreps(3), N=16)
        out_type = activation1.in_type
        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation1,
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 16 regular feature fields
        activation2 = escnn.nn.FourierELU(self.r2_act, 16, irreps=self.G.bl_irreps(3), N=16)
        out_type = activation2.in_type
        self.block2 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation2,
        )
        # to reduce the downsampling artifacts, we use a Gaussian smoothing filter
        self.pool1 = escnn.nn.SequentialModule(escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 32 regular feature fields
        activation3 = escnn.nn.FourierELU(self.r2_act, 32, irreps=self.G.bl_irreps(3), N=16)
        out_type = activation3.in_type
        self.block3 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation3,
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 64 regular feature fields
        activation4 = escnn.nn.FourierELU(self.r2_act, 32, irreps=self.G.bl_irreps(3), N=16)
        out_type = activation4.in_type
        self.block4 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation4,
        )
        self.pool2 = escnn.nn.SequentialModule(escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields
        activation5 = escnn.nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(3), N=16)
        out_type = activation5.in_type
        self.block5 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation5,
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields
        activation6 = escnn.nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(3), N=16)
        out_type = activation6.in_type
        self.block6 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(in_type, out_type, kernel_size=4, padding=1, bias=False),
            escnn.nn.IIDBatchNorm2d(out_type),
            activation6,
        )
        self.pool3 = escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        # number of output invariant channels
        c = 64

        # WARN: Very stupidly zeroing out most of the learned features here....
        # last 1x1 convolution layer, which maps the regular fields to c=64 invariant scalar fields
        # this is essential to provide *invariant* features in the final classification layer
        output_invariant_type = escnn.nn.FieldType(self.r2_act, c * [self.r2_act.trivial_repr])
        self.invariant_map = escnn.nn.R2Conv(out_type, output_invariant_type, kernel_size=1, bias=False)

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, n_classes),
        )

    def forward(self, input: torch.Tensor | escnn.nn.GeometricTensor):
        if not isinstance(input, escnn.nn.GeometricTensor):
            x = self.input_type(input)  # Wrap input tensor in a GeometricTensor
        else:
            x = input.tensor
            x = self.input_type(x)  # Ensure the input is wrapped in the correct type

        # mask out the corners of the input image
        # x = self.mask(x)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # Each layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        # print(f"Input shape: {x.shape}")
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        # print(f"After block1 and pool1: {x.shape}")

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        # print(f"After block2 and pool2: {x.shape}")

        x = self.block5(x)
        x = self.block6(x)
        # print(f"After block5 and block6: {x.shape}")

        # pool over the spatial dimensions
        x = self.pool3(x)
        # print(f"After pool3: {x.shape}")

        # extract invariant features
        x = self.invariant_map(x)
        # print(f"After invariant map: {x.shape}")

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layer
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x


def classification_loss_metrics(y_true, y_pred):
    """
    Loss function that returns both cross-entropy loss and accuracy metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions (logits)

    Returns:
        tuple: (loss, metrics_dict)
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(input=y_pred, target=y_true)

    with torch.no_grad():
        pred_labels = y_pred.argmax(dim=1)
        accuracy = (pred_labels == y_true).float().mean()

    metrics = {"accuracy": accuracy}
    return loss, metrics


def pre_process_images(images):
    """Define required transformations for all MNIST images."""
    return images


def augment_image(imgs, split: str, angle: float = None):
    """
    Apply rotation augmentation to a tensor of images.

    Args:
        image: Tensor of shape (H, W) or (1, H, W)
        split: Dataset split ("train", "test", "val")
        angle: Specific rotation angle in degrees. If None, sample uniformly from 0 to 360.

    Returns:
        tuple: (original_image, augmented_image) both with shape (1, 29, 29)
    """
    # Ensure image is in the right format
    resize_up = Resize(28 * 3)  # Upsample to reduce interpolation artifacts
    resize_down = Resize(28)  # Downsample back to 28x28

    if angle is not None:
        # Use specific angle
        from torchvision.transforms.functional import rotate

        rotate_fn = lambda x: rotate(x, angle, interpolation=InterpolationMode.BILINEAR)
    else:
        import kornia

        rotate_fn = kornia.augmentation.RandomRotation(degrees=(0, 360), p=1.0, keepdim=True)

    original = imgs

    # Create augmented image with rotation
    img_upsampled = resize_up(original)
    img_rotated = rotate_fn(img_upsampled)
    augmented = resize_down(img_rotated)

    if split == "test" or split == "val":  # Append aug images to original images in batch dimension
        # This provides better estimate of the equivariant component of the error.
        imgs = torch.cat((original.squeeze(2), augmented), dim=0)
    else:
        imgs = augmented.squeeze(2)

    return imgs


def flatten_img(img: torch.Tensor) -> torch.Tensor:
    """
    Flatten image preserving spatial structure.

    Args:
        img: (B, C, H, W) - batch of images with C channels

    Returns:
        x: (B*H*W, C) - flattened where each row is one pixel's features
           Spatial locations are grouped: [batch0_pixel(0,0), batch1_pixel(0,0), ..., batch0_pixel(0,1), ...]
    """
    B, C, H, W = img.shape
    # Permute to (H, W, B, C) then flatten to (H*W*B, C)
    # This groups all batch samples for each spatial location together
    x = img.permute(2, 3, 0, 1).contiguous().view(-1, C)
    return x


def unflatten_img(x: torch.Tensor, H: int, W: int, C) -> torch.Tensor:
    """
    Unflatten preserving spatial structure.

    Args:
        x: (B*H*W, C) - flattened features
        H, W: spatial dimensions
        C: number of channels

    Returns:
        img: (B, C, H, W) - reconstructed image batch
    """
    T, C = x.shape
    assert T % (H * W) == 0, f"Expected T to be divisible by {H * W}, got {T}"
    B = T // (H * W)  # Calculate batch size
    # Reshape to (H, W, B, C) then permute to (B, C, H, W)
    img = x.view(H, W, B, C).permute(2, 3, 0, 1)
    return img


def flatten_img_spatial_first(img: torch.Tensor) -> torch.Tensor:
    """
    Alternative flattening that keeps batch samples together for each spatial location.

    Args:
        img: (B, C, H, W) - batch of images

    Returns:
        x: (B*H*W, C) - flattened where spatial locations are grouped within each batch:
           [batch0_pixel(0,0), batch0_pixel(0,1), ..., batch1_pixel(0,0), batch1_pixel(0,1), ...]
    """
    B, C, H, W = img.shape
    # Reshape to (B*H*W, C) - this keeps batch samples together
    x = img.permute(0, 2, 3, 1).contiguous().view(-1, C)
    return x


def unflatten_img_spatial_first(x: torch.Tensor, H: int, W: int, C: int, B: int) -> torch.Tensor:
    """
    Unflatten for spatial_first flattening.

    Args:
        x: (B*H*W, C) - flattened features
        H, W: spatial dimensions
        C: number of channels
        B: batch size

    Returns:
        img: (B, C, H, W) - reconstructed image batch
    """
    # Reshape to (B, H, W, C) then permute to (B, C, H, W)
    img = x.view(B, H, W, C).permute(0, 3, 1, 2)
    return img


def collate_fn(batch, augment=True, split="train"):
    """
    Custom collate function that applies augmentation and returns both original and augmented images.
    SupervisedTrainingModule expects (x, y) format.
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    images = pre_process_images(images)
    if augment:
        images = augment_image(images, split=split)
        if split == "test" or split == "val":
            # Duplicate the labels for augmented images
            labels = torch.cat((labels, labels), dim=0)

    return images, labels


def traj_collate_fn(batch, augment: bool = True, split: str = "train"):
    """
    Custom collate function that applies augmentation and returns both original and augmented images.
    SupervisedTrainingModule expects (x, y) format.
    """
    imgs = torch.utils.data.default_collate(batch)
    imgs = pre_process_images(imgs)

    if augment:
        imgs = augment_image(imgs.squeeze(2), split=split)
    else:
        imgs = imgs.squeeze(2)

    present_image, future_image = imgs[:, [0]], imgs[:, [1]]

    # Ignore future image. Make the future image be 45 degrees rotated version of the present image.
    future_image = augment_image(present_image, split="train", angle=45.0)

    return present_image, future_image


def plot_predictions_images(
    past_imgs,
    rec_past_imgs,
    future_imgs,
    pred_future_imgs,
    pred_future_labels=None,
    pred_rec_labls=None,
    n_rows=4,
    n_cols=10,
    save_path=None,
):
    """
    Plot predictions in a 4-row grid showing:
    - Row 1: Past images
    - Row 2: Reconstructed past images
    - Row 3: True future images
    - Row 4: Predicted future images

    Args:
        past_imgs: Tensor of past images (B, 1, H, W)
        rec_past_imgs: Tensor of reconstructed past images (B, 1, H, W)
        future_imgs: Tensor of true future images (B, 1, H, W)
        pred_future_imgs: Tensor of predicted future images (B, 1, H, W)
        pred_future_labels: Tensor of predicted future labels (B,)
        pred_rec_labls: Tensor of predicted reconstructed labels (B,)
        n_rows: Number of rows (should be 4)
        n_cols: Number of columns
        save_path: Optional path to save the figure

    Returns:
        matplotlib.pyplot.Figure: The created figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    rec_past_imgs = rec_past_imgs.tensor if isinstance(rec_past_imgs, escnn.nn.GeometricTensor) else rec_past_imgs
    pred_future_imgs = (
        pred_future_imgs.tensor if isinstance(pred_future_imgs, escnn.nn.GeometricTensor) else pred_future_imgs
    )
    # Convert tensors to numpy and ensure they're on CPU
    past_imgs = past_imgs.detach().cpu().numpy()
    rec_past_imgs = rec_past_imgs.detach().cpu().numpy()
    future_imgs = future_imgs.detach().cpu().numpy()
    pred_future_imgs = pred_future_imgs.detach().cpu().numpy()

    # Handle optional labels
    if pred_future_labels is not None:
        pred_future_labels = pred_future_labels.detach().cpu().numpy()
    if pred_rec_labls is not None:
        pred_rec_labls = pred_rec_labls.detach().cpu().numpy()

    # Row titles
    row_titles = ["Past Images", "Reconstructed Past", "True Future", "Predicted Future"]

    for col in range(n_cols):
        # Row 0: Past images
        axes[0, col].imshow(past_imgs[col].squeeze(), cmap="gray")
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel(row_titles[0], rotation=90, size=12)

        # Row 1: Reconstructed past images
        axes[1, col].imshow(rec_past_imgs[col].squeeze(), cmap="gray")
        if pred_rec_labls is not None:
            axes[1, col].set_title(f"Rec: {pred_rec_labls[col]}", fontsize=10)
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel(row_titles[1], rotation=90, size=12)

        # Row 2: True future images
        axes[2, col].imshow(future_imgs[col].squeeze(), cmap="gray")
        axes[2, col].axis("off")
        if col == 0:
            axes[2, col].set_ylabel(row_titles[2], rotation=90, size=12)

        # Row 3: Predicted future images with labels
        axes[3, col].imshow(pred_future_imgs[col].squeeze(), cmap="gray")
        if pred_future_labels is not None:
            axes[3, col].set_title(f"Pred: {pred_future_labels[col]}", fontsize=10)
        axes[3, col].axis("off")
        if col == 0:
            axes[3, col].set_ylabel(row_titles[3], rotation=90, size=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def train_oracle(n_classes=5):
    ordered_MNIST = load_from_disk(str(data_path))
    train_dl = DataLoader(
        ordered_MNIST["train"], batch_size=512, shuffle=True, collate_fn=lambda x: collate_fn(x, split="train")
    )
    val_dl = DataLoader(
        ordered_MNIST["validation"],
        batch_size=128,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x),
    )
    test_dl = DataLoader(
        ordered_MNIST["test"],
        batch_size=128,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x),
    )

    trainer_kwargs = {
        "accelerator": "gpu",
        "max_epochs": 5,
        "log_every_n_steps": 2,
        "enable_progress_bar": True,
        "devices": 1,
        "enable_checkpointing": False,
        "logger": False,
    }

    trainer = lightning.Trainer(**trainer_kwargs)

    # Set seed for reproducibility
    lightning.seed_everything(0)

    # Create the CNN encoder model
    model = SO2SteerableCNN(n_classes=n_classes)

    # Create the supervised training module
    lightning_module = SupervisedTrainingModule(
        model=model,
        optimizer_fn=torch.optim.Adam,  # type: ignore
        optimizer_kwargs={"lr": 1e-3},
        loss_fn=classification_loss_metrics,
    )

    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=oracle_ckpt_path if oracle_ckpt_path.exists() else None,
    )

    # out = trainer.test(model=lightning_module, dataloaders=test_dl)
    # print("Test results:", out)
    # Save the model checkpoint
    oracle_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(oracle_ckpt_path)


if __name__ == "__main__":
    if not (data_path / "train").exists():
        # Check if the data directory exists, if not preprocess the data
        print("Data directory not found, preprocessing data.")
        make_dataset(n_classes=5)

    # Train the oracle classifier which is SO(2) equivariant
    train_oracle(n_classes=5)

    # Test that we can sample trajectories of consecutive digits
    ordered_MNIST = load_from_disk(str(data_path))

    from paper.experiments.dynamics.dynamics_dataset import TrajectoryDataset

    ordered_ds = TrajectoryDataset(trajectories=[ordered_MNIST["train"]["image"]], past_frames=1, future_frames=1)

    dataloader = DataLoader(ordered_ds, batch_size=128, shuffle=True, collate_fn=traj_collate_fn)

    channels = [32, 64, 50]
    ecnn_encoder = SO2SCNNEncoder(channels=channels, batch_norm=True, flatten_img=True)
    ecnn_decoder = SO2SCNNDecoder(
        in_type=ecnn_encoder.out.out_type,
        spatial_size=ecnn_encoder.spatial_size,
        channels=list(reversed(channels)),
        flat_img=True,
    )
    cnn_encoder = CNNEncoder(channels=channels, batch_norm=True, flat_img=True)
    cnn_decoder = CNNDecoder(channels=list(reversed(channels)), spatial_size=cnn_encoder.spatial_size, flat_img=True)
    so2_cnn_classifier = SO2SteerableCNN(n_classes=5)

    state_dict = torch.load(oracle_ckpt_path)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    so2_cnn_classifier.load_state_dict(state_dict, strict=True)

    # Iterate over the first 10 samples and plot currecnt and next images
    import matplotlib.pyplot as plt

    i = 0
    for batch_trajs in dataloader:
        present_batch, future_batch = batch_trajs

        x = present_batch
        x_flat = flatten_img(x)
        x_rec = unflatten_img(x_flat, H=x.shape[2], W=x.shape[3], C=x.shape[1])
        assert torch.allclose(x, x_rec), f"Reconstruction failed: {x.shape} != {x_rec.shape}"

        print(f"Present: {present_batch.shape} - Future {future_batch.shape}")
        ep_embedding = ecnn_encoder(present_batch).tensor
        ef_embedding = ecnn_encoder(future_batch).tensor
        p_embedding = cnn_encoder(present_batch)
        f_embedding = cnn_encoder(future_batch)
        # assert ep_embedding.shape == p_embedding.shape, f"{ep_embedding.shape} != {p_embedding.shape}"
        print(f"Embedding shapes: present {ep_embedding.shape}, future {ef_embedding.shape}")
        ep_rec = ecnn_decoder(ep_embedding).tensor
        ef_rec = ecnn_decoder(ef_embedding).tensor
        p_rec = cnn_decoder(p_embedding)
        f_rec = cnn_decoder(f_embedding)
        assert present_batch.shape == ep_rec.shape, f"{present_batch.shape} != {ep_rec.shape}"
        print(f"Reconstruction e-shapes: present {ep_rec.shape}, future {ef_rec.shape}")
        print(f"Reconstruction shapes: present {p_rec.shape}, future {f_rec.shape}")
        current_img = present_batch[5].numpy()
        next_img = future_batch[5].numpy()

        present_label = so2_cnn_classifier(present_batch).argmax(dim=1)[5].item()
        next_label = so2_cnn_classifier(future_batch).argmax(dim=1)[5].item()

        # Get reconstructed images for display
        current_rec_img = ep_rec[5].detach().cpu().numpy()
        next_rec_img = ef_rec[5].detach().cpu().numpy()

        # break
        plt.figure(figsize=(12, 6))

        # Original present image
        plt.subplot(2, 4, 1)
        plt.imshow(current_img.squeeze(0), cmap="gray")
        plt.title(f"Present: {present_label}")
        plt.axis("off")

        # Reconstructed present image
        plt.subplot(2, 4, 2)
        plt.imshow(current_rec_img.squeeze(0), cmap="gray")
        plt.title(f"Present Rec")
        plt.axis("off")

        # Original future image
        plt.subplot(2, 4, 3)
        plt.imshow(next_img.squeeze(0), cmap="gray")
        plt.title(f"Future: {next_label}")
        plt.axis("off")

        # Reconstructed future image
        plt.subplot(2, 4, 4)
        plt.imshow(next_rec_img.squeeze(0), cmap="gray")
        plt.title(f"Future Rec")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        i += 1
        if i >= 3:
            break
