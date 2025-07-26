import math
from pathlib import Path

import escnn
import lightning
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, Pad, RandomRotation, Resize

from symm_rep_learn.models.lightning_modules import SupervisedTrainingModule

main_path = Path(__file__).parent
data_path = main_path / "data" / "ordered_mnist"
ckpt_path = main_path / "ckpt"

# Create OmegaConf config
base_config = DictConfig(
    {
        "classes": 5,
        "train_samples": 1001,
        "val_ratio": 0.2,
        "test_samples": 1001,
        "num_rng_seeds": 20,
        "batch_size": 32,
        "eval_up_to_t": 15,
        "reduced_rank": True,
        "max_epochs": 150,
        "trial_budget": 50,
    }
)


def make_dataset(cfg: DictConfig):
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
    for i in range(cfg.classes):
        digit_ds.append(MNIST.filter(lambda example: example["label"] == i, keep_in_memory=True, num_proc=8))

    ordered_MNIST = DatasetDict()

    # Create ordered sequences by interleaving digits sequentially
    # This ensures digits appear in order: 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...
    for split in ["train", "test"]:
        ordered_MNIST[split] = interleave_datasets([ds[split] for ds in digit_ds], split=split).select(
            range(cfg[f"{split}_samples"])
        )

    # Split training data to create validation set
    # Use the last portion of training data as validation (no shuffling to maintain order)
    _tmp_ds = ordered_MNIST["train"].train_test_split(test_size=cfg.val_ratio, shuffle=False)
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
class CNNEncoder(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNEncoder, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # fully connected layer, output num_classes classes
        self.out = torch.nn.Sequential(torch.nn.Linear(32 * 7 * 7, num_classes))
        torch.nn.init.orthogonal_(self.out[0].weight)

    def forward(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(1)  # Add a channel dimension if needed
        X = self.conv1(X)
        X = self.conv2(X)
        # Flatten the output of conv2
        X = X.view(X.size(0), -1)
        output = self.out(X)
        return output


class SO2SCNNEncoder(torch.nn.Module):
    """SO(2) equivariant CNN encoder for ordered MNIST."""

    def __init__(self, embedding_dim=64, hidden_channels=[16, 32]):
        super(SO2SCNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        # The model is equivariant under all planar rotations
        self.r2_act = escnn.gspaces.rot2dOnR2(N=-1)
        self.G = self.r2_act.fibergroup

        # The input image is a scalar field (grey values), corresponding to the trivial representation
        self.in_type = escnn.nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # We need to mask the input image since the corners are moved outside the grid under rotations
        self.mask = escnn.nn.MaskModule(self.in_type, 29, margin=1)

        act1 = escnn.nn.FourierELU(self.r2_act, hidden_channels[0], irreps=self.G.bl_irreps(3), N=16)
        act2 = escnn.nn.FourierELU(self.r2_act, hidden_channels[1], irreps=self.G.bl_irreps(3), N=16)

        self.conv1 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(self.in_type, act1.in_type, kernel_size=5, stride=1, padding=2),
            act1,
            escnn.nn.PointwiseAvgPoolAntialiased(act1.out_type, sigma=0.66, stride=2),
        )
        self.conv2 = escnn.nn.SequentialModule(
            escnn.nn.R2Conv(act1.out_type, act2.in_type, kernel_size=5, stride=1, padding=2),
            act2,
            escnn.nn.PointwiseAvgPoolAntialiased(act2.out_type, sigma=0.66, stride=2),
        )

        # Head that flattens the gird feature maps. (B, C, H, W) -> (B, C*H*W)
        linear_base_space = escnn.gspaces.no_base_space(self.G)
        feat_type = escnn.nn.FieldType(linear_base_space, [act2.out_type.representation] * 8 * 8)
        out_rep = self.G.spectral_regular_representation(*self.G.bl_irreps(3), name="embedding_rep")
        out_rep_multiplicity = math.ceil(self.embedding_dim // out_rep.size)
        self.out = escnn.nn.Linear(
            in_type=feat_type,
            out_type=escnn.nn.FieldType(linear_base_space, [out_rep] * out_rep_multiplicity),
            bias=False,
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        x = self.in_type(input)
        # mask out the corners of the input image
        x = self.mask(x)
        # apply equivariant blocks
        x = self.conv1(x)
        x = self.conv2(x)
        # Extract the tensor and flatten correctly for group representation
        # Shape: (B, C, H, W) -> (B, H, W, C) -> (B, H*W*C)
        # This preserves pixel-wise feature grouping for equivariant features
        B, C, H, W = x.tensor.shape
        x_flat = x.tensor.permute(0, 2, 3, 1).reshape(B, -1)

        embedding = self.out(self.out.in_type(x_flat))

        return embedding


# A decoder which is specular to CNNEncoder, starting with a fully connected layer and then reshaping the output to a 2D image
class CNNDecoder(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNNDecoder, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(num_classes, 32 * 7 * 7))

        self.conv1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        # Remove the channel dimension
        x = x.squeeze(1)
        return x


class SO2SCNNDecoder(torch.nn.Module):
    """SO(2) equivariant CNN decoder for ordered MNIST - counterpart to SO2SCNNEncoder."""

    def __init__(self, in_type: escnn.nn.FieldType, hidden_channels=[32, 16]):
        super(SO2SCNNDecoder, self).__init__()
        self.embedding_dim = in_type.size
        self.hidden_channels = hidden_channels

        # The model is equivariant under all planar rotations
        self.r2_act = escnn.gspaces.rot2dOnR2(N=-1)
        self.G = self.r2_act.fibergroup

        self.in_type = in_type  # Image embedding
        self.output_type = escnn.nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])  #  Gray image

        act1 = escnn.nn.FourierELU(self.r2_act, hidden_channels[0], irreps=self.G.bl_irreps(3), N=16)
        act2 = escnn.nn.FourierELU(self.r2_act, hidden_channels[1], irreps=self.G.bl_irreps(3), N=16)

        # Define the unflattened type.
        linear_base_space = escnn.gspaces.no_base_space(self.G)
        flat_feat_type = escnn.nn.FieldType(linear_base_space, [act1.in_type.representation] * 8 * 8)
        # Linear map from flat_embedding to flattened spatial features
        self.fc = escnn.nn.Linear(in_type=self.in_type, out_type=flat_feat_type)

        # First decoder block: Upsample → Activation → ConvTranspose:   fx8x8 → hc1x16x16
        self.deconv1 = escnn.nn.SequentialModule(
            escnn.nn.R2Upsampling(act1.in_type, scale_factor=2),
            act1,
            escnn.nn.R2ConvTransposed(act1.in_type, act2.in_type, kernel_size=5, stride=1, padding=2),
        )
        # Second decoder block: Upsample → Activation → ConvTranspose:   hc1x16x16 → hc2x32x32
        self.deconv2 = escnn.nn.SequentialModule(
            escnn.nn.R2Upsampling(act2.in_type, scale_factor=2),
            act2,
            escnn.nn.R2ConvTransposed(act2.in_type, self.output_type, kernel_size=5, stride=1, padding=2, bias=True),
        )

    def forward(self, x: escnn.nn.GeometricTensor):
        # Unflatten embedding back to spatial features
        spatial_features = self.fc(x).tensor

        # Shape: (B, H*W*C) -> (B, H, W, C) -> (B, C, H, W)
        B = spatial_features.shape[0]
        H, W = 8, 8  # Target spatial dimensions after encoder's pooling
        C = self.deconv1.in_type.size

        # Reshape to spatial format and permute to (B, C, H, W)
        x_img = spatial_features.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # Apply reverse operations: each block now contains Upsample → Activation → ConvTranspose
        x = self.deconv1(self.deconv1.in_type(x_img))
        x = self.deconv2(x)  # 16x16 -> 32x32 (16→1 channels)

        # Extract final tensor
        output = x.tensor

        return output


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
        self.mask = escnn.nn.MaskModule(in_type, 29, margin=1)

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
            escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
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

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = self.input_type(input)

        # mask out the corners of the input image
        x = self.mask(x)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # Each layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # extract invariant features
        x = self.invariant_map(x)

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
    loss = loss_fn(y_pred, y_true)

    with torch.no_grad():
        pred_labels = y_pred.argmax(dim=1)
        accuracy = (pred_labels == y_true).float().mean()

    metrics = {"accuracy": accuracy}
    return loss, metrics


def augment_image(image):
    """
    Apply rotation augmentation to a tensor of images.

    Args:
        image: Tensor of shape (H, W) or (1, H, W)

    Returns:
        tuple: (original_image, augmented_image) both with shape (1, 29, 29)
    """
    # Ensure image is in the right format
    if image.dim() == 2:  # (H, W)
        image = image.unsqueeze(0)  # Add channel dimension -> (1, H, W)

    # Create transformation pipeline similar to the original code
    pad = Pad((0, 0, 1, 1), fill=0)  # Pad to 29x29
    resize_up = Resize(29 * 3)  # Upsample to reduce interpolation artifacts
    resize_down = Resize(29)  # Downsample back to 29x29
    rotate = RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR)

    # Apply padding first (28x28 -> 29x29)
    original = pad(image)

    # Create augmented image with random rotation
    img_upsampled = resize_up(original)
    img_rotated = rotate(img_upsampled)
    augmented = resize_down(img_rotated)

    return original, augmented


def collate_fn(batch):
    """
    Custom collate function that applies augmentation and returns both original and augmented images.
    SupervisedTrainingModule expects (x, y) format.
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    original, aug_image = augment_image(images)

    return aug_image, labels


def traj_collate_fn(batch):
    """
    Custom collate function that applies augmentation and returns both original and augmented images.
    SupervisedTrainingModule expects (x, y) format.
    """
    images = torch.utils.data.default_collate(batch)

    _, aug_image = augment_image(images.squeeze(2))

    present_image, future_image = aug_image[:, [0]], aug_image[:, [1]]
    return present_image, future_image


def train_oracle(cfg: DictConfig):
    ordered_MNIST = load_from_disk(str(data_path))
    train_dl = DataLoader(ordered_MNIST["train"], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(
        ordered_MNIST["validation"], batch_size=len(ordered_MNIST["validation"]), shuffle=False, collate_fn=collate_fn
    )
    test_dl = DataLoader(
        ordered_MNIST["test"], batch_size=len(ordered_MNIST["test"]), shuffle=False, collate_fn=collate_fn
    )

    trainer_kwargs = {
        "accelerator": "gpu",
        "max_epochs": 20,
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
    # model = CNNEncoder(num_classes=cfg.classes)
    model = SO2SteerableCNN(n_classes=cfg.classes)

    # Create the supervised training module
    lightning_module = SupervisedTrainingModule(
        model=model,
        optimizer_fn=torch.optim.Adam,  # type: ignore
        optimizer_kwargs={"lr": 1e-2},
        loss_fn=classification_loss_metrics,
    )

    trainer.fit(
        model=lightning_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path / "oracle.ckpt"
    )

    out = trainer.test(model=lightning_module, dataloaders=test_dl)
    print("Test results:", out)
    # Save the model checkpoint
    ckpt_path.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_path / "oracle.ckpt")


if __name__ == "__main__":
    cfg = base_config

    if not data_path.exists():
        # Check if the data directory exists, if not preprocess the data
        print("Data directory not found, preprocessing data.")
        make_dataset(cfg)

    # Train the oracle classifier which is SO(2) equivariant
    train_oracle(cfg)

    # Test that we can sample trajectories of consecutive digits
    ordered_MNIST = load_from_disk(str(data_path))

    from paper.experiments.dynamics.dynamics_dataset import TrajectoryDataset

    ordered_ds = TrajectoryDataset(trajectories=[ordered_MNIST["train"]["image"]], past_frames=1, future_frames=1)

    dataloader = DataLoader(ordered_ds, batch_size=128, shuffle=True, collate_fn=traj_collate_fn)

    so2_cnn_encoder = SO2SCNNEncoder(embedding_dim=64)
    so2_cnn_decoder = SO2SCNNDecoder(in_type=so2_cnn_encoder.out.out_type)
    # Iterate over the first 10 samples and plot currecnt and next images
    import matplotlib.pyplot as plt

    i = 0
    for batch_trajs in dataloader:
        present_batch, future_batch = batch_trajs

        p_embedding = so2_cnn_encoder(present_batch)
        f_embedding = so2_cnn_encoder(future_batch)
        print(f"Embedding shapes: present {p_embedding.shape}, future {f_embedding.shape}")
        p_rec = so2_cnn_decoder(p_embedding)
        f_rec = so2_cnn_decoder(f_embedding)
        print(f"Reconstruction shapes: present {p_rec.shape}, future {f_rec.shape}")
        current_img = present_batch[5].numpy()
        next_img = future_batch[5].numpy()

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(current_img.squeeze(0), cmap="gray")
        plt.title("Current Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(next_img.squeeze(0), cmap="gray")
        plt.title("Next Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        i += 1
        if i >= 3:
            break
    print("Sampled and displayed first 10 trajectory images.")
