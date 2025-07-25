import os
import shutil
from pathlib import Path

import lightning
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

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
        "batch_size": 64,
        "eval_up_to_t": 15,
        "reduced_rank": True,
        "max_epochs": 150,
        "trial_budget": 50,
    }
)


def make_dataset(cfg: DictConfig):
    # Data pipeline
    MNIST = load_dataset("mnist", keep_in_memory=True)
    digit_ds = []
    for i in range(cfg.classes):
        digit_ds.append(MNIST.filter(lambda example: example["label"] == i, keep_in_memory=True, num_proc=8))
    ordered_MNIST = DatasetDict()
    # Order the digits in the dataset and select only a subset of the data
    for split in ["train", "test"]:
        ordered_MNIST[split] = interleave_datasets([ds[split] for ds in digit_ds], split=split).select(
            range(cfg[f"{split}_samples"])
        )
    _tmp_ds = ordered_MNIST["train"].train_test_split(test_size=cfg.val_ratio, shuffle=False)
    ordered_MNIST["train"] = _tmp_ds["train"]
    ordered_MNIST["validation"] = _tmp_ds["test"]
    ordered_MNIST.set_format(type="torch", columns=["image", "label"])
    ordered_MNIST = ordered_MNIST.map(
        lambda example: {"image": example["image"] / 255.0, "label": example["label"]},
        batched=True,
        keep_in_memory=True,
        num_proc=2,
    )
    ordered_MNIST.save_to_disk(data_path)


# CNN Architecture
class CNNEncoder(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNEncoder, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
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


# A decoder which is specular to CNNEncoder, starting with a fully connected layer and then reshaping the output to a 2D image
class CNNDecoder(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CNNDecoder, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(num_classes, 32 * 7 * 7))

        self.conv1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, 5, 1, 2),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        # Remove the channel dimension
        x = x.squeeze(1)
        return x


def classification_loss_with_metrics(y_true, y_pred):
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


def collate_fn(batch):
    """
    Custom collate function to convert batch format from dict to tuple.
    SupervisedTrainingModule expects (x, y) format.
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return images, labels


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
    model = CNNEncoder(num_classes=cfg.classes)

    # Create the supervised training module
    lightning_module = SupervisedTrainingModule(
        model=model,
        optimizer_fn=torch.optim.Adam,  # type: ignore
        optimizer_kwargs={"lr": 1e-2},
        loss_fn=classification_loss_with_metrics,
    )

    trainer.fit(model=lightning_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

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

    train_oracle(cfg)
