from click import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

from paper.experiments.dynamics import ordered_mnist
from symm_rep_learn.nn.layers import Lambda


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(nn.Upsample(mode="bilinear", scale_factor=2), conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """`UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, out_channels, in_channels=3, depth=3, start_filts=32, up_mode="upsample", merge_mode="concat"):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for upsampling. Only "transpose" and "upsample" are allowed.'.format(up_mode)
            )

        if merge_mode in ("concat", "add"):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode formerging up and down paths. Only "concat" and "add" are allowed.'.format(
                    up_mode
                )
            )

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                'up_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )

        self.num_classes = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        outs = None
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


def simple_reconstruction_training(model, dataloader, device="cuda"):
    """
    Simple single epoch training to test encoder-decoder reconstruction.
    """
    model = model.to(device)
    model.train()

    # Use MSE loss for reconstruction
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_loss = 0.0
    num_batches = 0

    print("Starting single epoch reconstruction training...")

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move data to device and ensure it's offloaded after processing
        images = images.to(device)

        # Forward pass
        optimizer.zero_grad()
        reconstructed = model(images)

        # Compute reconstruction loss
        loss = criterion(reconstructed, images)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss and move to CPU to free GPU memory
        total_loss += loss.item()
        num_batches += 1

        # Offload data from GPU after processing
        images = images.cpu()
        reconstructed = reconstructed.detach().cpu()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / num_batches
    print(f"Average reconstruction loss: {avg_loss:.6f}")

    return avg_loss


def visualize_reconstruction(model, dataloader, device="cuda", num_samples=8):
    """
    Visualize original vs reconstructed images.
    """
    import matplotlib.pyplot as plt

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Get a batch of data
        images, labels = next(iter(dataloader))
        images = images.to(device)

        # Get reconstructions
        reconstructed = model(images)

        # Move to CPU for visualization
        images = images.cpu()
        reconstructed = reconstructed.cpu()

        # Plot original vs reconstructed
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(images[i].squeeze(), cmap="gray")
            axes[0, i].set_title(f"Original {labels[i].item()}")
            axes[0, i].axis("off")

            # Reconstructed image
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show(block=False)  # Non-blocking display
        plt.pause(1)  # Brief pause to ensure display


def simple_next_digit_training(model, dataloader, device="cuda"):
    """
    Simple single epoch training to predict the next digit image from current digit image.

    Args:
        model: Encoder-decoder model (takes current digit, outputs next digit prediction)
        dataloader: DataLoader that returns (present_image, future_image) tuples
        device: Device to run training on
    """
    model = model.to(device)
    model.train()

    # Use MSE loss for image prediction
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_loss = 0.0
    num_batches = 0

    print("Starting single epoch next digit prediction training...")

    MAX_EPOCHS = 50
    for _ in range(MAX_EPOCHS):  # Single epoch
        for batch_idx, (imgs, next_imgs) in enumerate(dataloader):
            # Move data to device
            imgs = imgs.to(device)
            next_imgs = next_imgs.to(device)

            # Forward pass: predict future image from present image
            optimizer.zero_grad()
            pred_next_imgs = model(imgs)
            # Compute prediction loss
            loss = criterion(pred_next_imgs, next_imgs)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Offload data from GPU after processing
            imgs = imgs.cpu()
            next_imgs = next_imgs.cpu()
            pred_next_imgs = pred_next_imgs.detach().cpu()

        print(f"Epoch {_ + 1}/{MAX_EPOCHS}, Average Loss: {total_loss / num_batches:.6f}")

        # if batch_idx >= 10:
        # break

    avg_loss = total_loss / num_batches
    print(f"Average next digit prediction loss: {avg_loss:.6f}")

    return avg_loss


def visualize_next_digit_prediction(model, dataloader, device="cuda", num_samples=8):
    """
    Visualize current digit vs predicted next digit vs true next digit.
    """
    import matplotlib.pyplot as plt

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Get a batch of trajectory data
        imgs, next_imgs = next(iter(dataloader))
        imgs = imgs.to(device)
        next_imgs = next_imgs.to(device)

        # Get predictions
        pred_next_imgs = model(imgs)

        # Move to CPU for visualization
        imgs = imgs.cpu()
        next_imgs = next_imgs.cpu()
        pred_next_imgs = pred_next_imgs.cpu()

        # Plot current, predicted next, and true next
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

        samples = np.random.choice(range(len(imgs)), num_samples, replace=False)
        imgs = imgs[samples]
        pred_next_imgs = pred_next_imgs[samples]
        next_imgs = next_imgs[samples]
        for i in range(num_samples):
            # Current image
            axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
            axes[0, i].set_title("Current")
            axes[0, i].axis("off")

            # Predicted next image
            axes[1, i].imshow(pred_next_imgs[i].squeeze(), cmap="gray")
            axes[1, i].set_title("Predicted Next")
            axes[1, i].axis("off")

            # True next image
            axes[2, i].imshow(next_imgs[i].squeeze(), cmap="gray")
            axes[2, i].set_title("True Next")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.show(block=False)  # Non-blocking display
        plt.pause(1)  # Brief pause to ensure display


def train_linear_decoder(encoder_model, dataloader, device="cuda", regularization=0):
    """
    Train a linear decoder using least squares regression.

    Treats each spatial location as a separate sample: (B * H * W, C) -> (B * H * W, 1)

    Args:
        encoder_model: Pre-trained encoder (first part of the model)
        dataloader: DataLoader that returns (present_image, future_image) tuples
        device: Device to run on
        regularization: L2 regularization parameter for ridge regression

    Returns:
        linear_transform: Learned transformation matrix (C, 1)
    """
    print("Training linear decoder using least squares...")
    print("Treating each spatial location as a separate sample: (B*H*W, C) -> (B*H*W, 1)")

    encoder_model = encoder_model.to(device)
    encoder_model.eval()

    # Collect encoded features and target values
    encoded_features = []
    target_values = []

    with torch.no_grad():
        for imgs, next_imgs in dataloader:
            imgs = imgs.to(device)
            next_imgs = next_imgs.to(device)

            # Get encoded features (B, C, H, W)
            encoded = encoder_model(imgs)
            B, C, H, W = encoded.shape

            # Reshape to treat each spatial location as a sample: (B, C, H, W) -> (B*H*W, C)
            encoded_spatial = ordered_mnist.flatten_img(encoded)

            # Reshape target images similarly: (B, 1, H, W) -> (B*H*W, 1)
            target_spatial = ordered_mnist.flatten_img(next_imgs)

            encoded_features.append(encoded_spatial.cpu())
            target_values.append(target_spatial.cpu())

    # Concatenate all batches
    X = torch.cat(encoded_features, dim=0)  # (N*H*W, C)
    Y = torch.cat(target_values, dim=0)  # (N*H*W, 1)

    print(f"Training data shape: X={X.shape}, Y={Y.shape}")

    # Solve least squares with L2 regularization: W = (X^T X + λI)^(-1) X^T Y
    # X: (N*H*W, C), Y: (N*H*W, 1), W: (C, 1)
    XtX = torch.mm(X.t(), X)  # (C, C)
    XtY = torch.mm(X.t(), Y)  # (C, 1)
    # Add regularization
    I = torch.eye(XtX.shape[0]) * regularization
    # Solve the system
    linear_transform = torch.linalg.solve(XtX + I, XtY)  # (C, 1)

    print(f"Linear transform shape: {linear_transform.shape}")

    return linear_transform


def visualize_decoder_comparison(
    encoder_model,
    dataloader,
    device="cuda",
    num_samples=8,
    title_prefix="",
    nonlinear_decoder=None,
    linear_decoder=None,
):
    """
    Unified visualization function that compares linear vs nonlinear decoders.
    Shows: Current Image, [Linear Prediction], [Nonlinear Prediction], Ground Truth

    Args:
        encoder_model: Pre-trained encoder
        dataloader: DataLoader for visualization (should be test/validation data)
        device: Device to run on
        num_samples: Number of samples to visualize
        title_prefix: Prefix for the plot title (e.g., "Before Training", "After Training")
        nonlinear_decoder: Pre-trained nonlinear decoder (optional)
        linear_decoder: Pre-trained linear decoder matrix (optional)
    """
    import matplotlib.pyplot as plt

    print(f"\n{title_prefix} - Visualizing decoder comparison...")

    # Determine which decoders to plot
    plot_linear = linear_decoder is not None
    plot_nonlinear = nonlinear_decoder is not None

    if not plot_linear and not plot_nonlinear:
        print("Warning: No decoders provided for comparison!")
        return None

    encoder_model = encoder_model.to(device)
    encoder_model.eval()

    if plot_linear:
        linear_decoder = linear_decoder.to(device)

    if plot_nonlinear:
        nonlinear_decoder = nonlinear_decoder.to(device)
        nonlinear_decoder.eval()

    with torch.no_grad():
        # Get a batch of trajectory data
        imgs, next_imgs = next(iter(dataloader))
        imgs = imgs.to(device)
        next_imgs = next_imgs.to(device)

        # Get encoded features
        encoded = encoder_model(imgs)
        B, C, H, W = encoded.shape

        predictions = {}
        row_titles = ["Current Image"]

        # Linear decoder prediction (if provided)
        if plot_linear:
            decoded = linear_decoder(encoded)
            predictions["linear"] = decoded.cpu()
            row_titles.append("Linear Prediction")

        # Nonlinear decoder prediction (if provided)
        if plot_nonlinear:
            pred_nonlinear = nonlinear_decoder(encoded)
            predictions["nonlinear"] = pred_nonlinear.cpu()
            row_titles.append("Nonlinear Prediction")

        row_titles.append("Ground Truth")

        # Move to CPU for visualization
        imgs = imgs.cpu()
        next_imgs = next_imgs.cpu()

        # Determine number of rows for plotting
        num_rows = len(row_titles)

        # Plot comparison
        fig, axes = plt.subplots(num_rows, num_samples, figsize=(num_samples * 2, num_rows * 2))
        if title_prefix:
            fig.suptitle(f"{title_prefix} - Decoder Comparison", fontsize=16)

        # Handle case where there's only one row
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        samples = np.random.choice(range(len(imgs)), num_samples, replace=False)
        imgs = imgs[samples]
        next_imgs = next_imgs[samples]

        # Sample predictions
        for key in predictions:
            predictions[key] = predictions[key][samples]

        row_idx = 0

        # Plot current images
        for i in range(num_samples):
            axes[row_idx, i].imshow(imgs[i].squeeze(), cmap="gray")
            axes[row_idx, i].axis("off")
            if i == 0:
                axes[row_idx, i].set_ylabel(row_titles[row_idx], rotation=90, size=12)
        row_idx += 1

        # Plot linear predictions (if available)
        if plot_linear:
            for i in range(num_samples):
                axes[row_idx, i].imshow(predictions["linear"][i].squeeze(), cmap="gray")
                axes[row_idx, i].axis("off")
                if i == 0:
                    axes[row_idx, i].set_ylabel(row_titles[row_idx], rotation=90, size=12)
            row_idx += 1

        # Plot nonlinear predictions (if available)
        if plot_nonlinear:
            for i in range(num_samples):
                axes[row_idx, i].imshow(predictions["nonlinear"][i].squeeze(), cmap="gray")
                axes[row_idx, i].axis("off")
                if i == 0:
                    axes[row_idx, i].set_ylabel(row_titles[row_idx], rotation=90, size=12)
            row_idx += 1

        # Plot ground truth
        for i in range(num_samples):
            axes[row_idx, i].imshow(next_imgs[i].squeeze(), cmap="gray")
            axes[row_idx, i].axis("off")
            if i == 0:
                axes[row_idx, i].set_ylabel(row_titles[row_idx], rotation=90, size=12)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)

        # Compute and print losses
        print(f"{title_prefix} Results:")

        if plot_linear:
            linear_mse = F.mse_loss(predictions["linear"], next_imgs)
            print(f"  Linear decoder MSE loss: {linear_mse.item():.6f}")

        if plot_nonlinear:
            nonlinear_mse = F.mse_loss(predictions["nonlinear"], next_imgs)
            print(f"  Nonlinear decoder MSE loss: {nonlinear_mse.item():.6f}")

        if plot_linear and plot_nonlinear:
            improvement = ((linear_mse - nonlinear_mse) / linear_mse * 100).item()
            print(f"  Performance improvement (nonlinear vs linear): {improvement:.2f}%")

        return predictions


def train_evol_op(
    evol_op,
    dataloader,
    device="cuda",
    num_epochs=50,
    lr=1e-3,
    print_every=10,
):
    """
    Train the Evolution Operator using contrastive learning on consecutive image pairs.

    The Evolution Operator learns to map latent representations of current images
    to latent representations of next images, acting as a Koopman/Transfer operator
    in the latent space.

    Args:
        evol_op: EvolutionOperator model
        dataloader: DataLoader that returns (present_image, next_image) pairs
        device: Device to train on
        num_epochs: Number of training epochs
        lr: Learning rate
        print_every: Print progress every N epochs
    """
    evol_op = evol_op.to(device)
    evol_op.train()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(evol_op.parameters(), lr=lr)

    total_loss = 0.0
    num_batches = 0

    print(f"Starting Evolution Operator training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        for batch_idx, (present_imgs, next_imgs) in enumerate(dataloader):
            # Move data to device
            present_imgs = present_imgs.to(device)
            next_imgs = next_imgs.to(device)

            # Forward pass: get embeddings for present and next images
            optimizer.zero_grad()

            # Since EvolutionOperator uses the same encoder for both x and y,
            # we pass present_imgs as x and next_imgs as y
            fx_c, hy_c = evol_op(x=present_imgs, y=next_imgs)

            # Compute loss using the inherited NCP loss function
            loss, metrics = evol_op.loss(fx_c, hy_c)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Offload data from GPU after processing
            present_imgs = present_imgs.cpu()
            next_imgs = next_imgs.cpu()

        if (epoch + 1) % print_every == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
            # Print some metrics if available
            if metrics:
                for key, value in metrics.items():
                    print(f"  {key}: {value:.6f}")

    avg_loss = total_loss / num_batches
    print(f"Average Evolution Operator training loss: {avg_loss:.6f}")

    return avg_loss


if __name__ == "__main__":
    from pathlib import Path

    # Configuration parameters
    BATCH_SIZE = 128  # Single parameter to control all dataloaders
    EMBEDDING_DIM = 32  # Embedding dimension for encoder-decoder
    DEPTH = 3  # Depth of the UNet
    AUGMENT = True  # Whether to apply data augmentation

    main_path = Path(__file__).parent.parent
    data_path = main_path / "data" / "ordered_mnist"

    # Test that we can sample trajectories of consecutive digits
    ordered_MNIST = ordered_mnist.load_from_disk(str(data_path))

    # Create encoder-decoder model
    encoder = UNet(in_channels=1, out_channels=EMBEDDING_DIM, depth=DEPTH)
    decoder = UNet(in_channels=EMBEDDING_DIM, out_channels=1, depth=DEPTH)
    model = nn.Sequential(encoder, decoder)

    # Create dataloader with parametric batch size
    train_dataloader = torch.utils.data.DataLoader(
        ordered_MNIST["train"],
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: ordered_mnist.collate_fn(x, augment=AUGMENT),
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        ordered_MNIST["test"],
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: ordered_mnist.collate_fn(x, augment=AUGMENT),  # No augmentation for reconstruction test
        shuffle=False,
    )

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train for one epoch
    # avg_loss = simple_reconstruction_training(model, train_dataloader, device=device)

    # Visualize results
    # print("Visualizing reconstruction results...")
    # visualize_reconstruction(model, test_dataloader, device=device)

    # Now test predictions of next digit images:
    print("\n" + "=" * 60)
    print("NEXT DIGIT PREDICTION TRAINING")
    print("=" * 60)

    from paper.experiments.dynamics.dynamics_dataset import TrajectoryDataset

    ordered_train_ds = TrajectoryDataset(trajectories=[ordered_MNIST["train"]["image"]], past_frames=1, future_frames=1)
    ordered_test_ds = TrajectoryDataset(trajectories=[ordered_MNIST["test"]["image"]], past_frames=1, future_frames=1)

    ordered_train_dl = torch.utils.data.DataLoader(
        ordered_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=AUGMENT),
    )
    ordered_test_dl = torch.utils.data.DataLoader(
        ordered_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: ordered_mnist.traj_collate_fn(x, augment=AUGMENT),
    )

    # Create a new encoder-decoder model for next digit prediction
    encoder = UNet(in_channels=1, out_channels=EMBEDDING_DIM, depth=DEPTH)
    decoder = UNet(in_channels=EMBEDDING_DIM, out_channels=1, depth=DEPTH)
    model = nn.Sequential(encoder, decoder)

    # Define checkpoint path
    checkpoint_path = Path(__file__).parent / f"unet_next_digit_model_emb{EMBEDDING_DIM}_depth{DEPTH}.pth"

    # Check if checkpoint exists and load it
    if checkpoint_path.exists():
        print(f"Loading existing checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully!")

        # Compare linear vs nonlinear decoder AFTER training (loaded model)
        print("\n" + "=" * 60)
        print("COMPARING LINEAR VS NONLINEAR DECODER - LOADED TRAINED MODEL")
        print("=" * 60)

        visualize_decoder_comparison(
            encoder_model=encoder,
            dataloader=ordered_test_dl,
            device=device,
            title_prefix="Non-linear decoder",
            nonlinear_decoder=decoder,
        )
    else:
        # Compare linear vs nonlinear decoder BEFORE training
        print("\n" + "=" * 60)
        print("COMPARING LINEAR VS NONLINEAR DECODER - BEFORE TRAINING")
        print("=" * 60)

        # Train linear decoder using training data (before training the full model)
        linear_decoder_before = train_linear_decoder(encoder, ordered_train_dl, device=device)

        visualize_decoder_comparison(
            encoder_model=encoder,
            dataloader=ordered_test_dl,
            device=device,
            title_prefix="Before Training",
            nonlinear_decoder=decoder,
            linear_decoder=linear_decoder_before,
        )

        # Train for next digit prediction
        print("\nTraining model to predict next digit...")
        next_digit_loss = simple_next_digit_training(model, ordered_train_dl, device=device)

        # Save the trained model
        print(f"Saving trained model to {checkpoint_path}")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "final_loss": next_digit_loss,
                "embedding_dim": EMBEDDING_DIM,
                "depth": DEPTH,
                "batch_size": BATCH_SIZE,
            },
            checkpoint_path,
        )
        print("Model saved successfully!")

        # Compare linear vs nonlinear decoder AFTER training
        print("\n" + "=" * 60)
        print("COMPARING LINEAR VS NONLINEAR DECODER - AFTER TRAINING")
        print("=" * 60)

        # Train linear decoder using training data (after training the full model)
        linear_decoder_after = train_linear_decoder(encoder, ordered_train_dl, device=device)

        visualize_decoder_comparison(
            encoder_model=encoder,
            dataloader=ordered_test_dl,
            device=device,
            title_prefix="After Training",
            nonlinear_decoder=decoder,
            linear_decoder=linear_decoder_after,
        )

    # Keep all plots open at the end of the script
    print("\nAll training and visualization complete!")
    print("Plots will remain open. Close the plot windows manually or press Ctrl+C to exit.")

    # Train a NCP model

    from symm_rep_learn.models.img_evol_op import ImgEvolutionOperator

    def lin_collate_fn(batch):
        imgs, next_imgs = ordered_mnist.traj_collate_fn(batch, augment=AUGMENT)
        return next_imgs, next_imgs

    lin_train_dl = torch.utils.data.DataLoader(
        ordered_train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lin_collate_fn
    )

    evol_op = ImgEvolutionOperator(
        embedding_state=UNet(in_channels=1, out_channels=EMBEDDING_DIM, depth=DEPTH),
        state_embedding_dim=EMBEDDING_DIM,
        orth_reg=0.001,
        centering_reg=0.00,
    )
    lin_dec: torch.nn.Conv2d = evol_op.fit_linear_decoder(
        train_dataloader=lin_train_dl,
    )
    device = next(iter(evol_op.parameters())).device
    cond_encoder = Lambda(func=lambda x: evol_op.conditional_expectation(x=x.to(device), hy2zy=None))
    visualize_decoder_comparison(
        encoder_model=cond_encoder,
        dataloader=ordered_test_dl,
        device=device,
        title_prefix="[BEFORE TRAINING] Evolution Operator Analysis",
        # nonlinear_decoder=decoder,
        linear_decoder=lin_dec,
    )

    train_evol_op(
        evol_op,
        ordered_train_dl,
        device=device,
        num_epochs=50,
        lr=1e-3,
        print_every=10,
    )

    lin_dec: torch.nn.Conv2d = evol_op.fit_linear_decoder(
        train_dataloader=lin_train_dl,
    )
    # Visualize Evolution Operator + Linear Decoder performance

    print("\n" + "=" * 60)
    print("COMPARING EVOLUTION OPERATOR + LINEAR DECODER VS NONLINEAR DECODER")
    print("=" * 60)
    visualize_decoder_comparison(
        encoder_model=cond_encoder,
        dataloader=ordered_test_dl,
        device=device,
        title_prefix="Evolution Operator Analysis",
        # nonlinear_decoder=decoder,
        linear_decoder=lin_dec,
    )

    import matplotlib.pyplot as plt

    try:
        plt.show()  # Blocking call to keep plots open
    except KeyboardInterrupt:
        pass
