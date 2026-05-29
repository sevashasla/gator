import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import tyro
from lightning import Trainer, LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import json
import matplotlib.pyplot as plt

from gator.models.gator_2view.model_gator import Gator, GatorConfig
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters


@dataclass
class Args:
    gator_config: GatorConfig
    opt_params: OptimizationParameters

    checkpoint_path: Path
    data_dir: Path = Path("/scratch/izar/mayila/imagenet_full")
    """Path to ImageNet directory with train/ and validation/ subfolders"""

    num_workers: int = 8
    seed: int = 0
    precision: Literal['16-mixed', '32'] = '16-mixed'


class GatorClassifier(LightningModule):
    def __init__(self, model: Gator, embed_dim: int, num_register_tokens: int, lr: float, total_steps: int):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.num_register_tokens = num_register_tokens
        self.total_steps = total_steps

        self.head = nn.Linear(embed_dim, 1000)
        self.criterion = nn.CrossEntropyLoss()

        # Per-epoch accumulators
        self._train_loss_sum: float = 0.0
        self._train_loss_n: int = 0

        self._val_loss_sum: float = 0.0
        self._val_correct_top1: float = 0.0
        self._val_correct_top5: float = 0.0
        self._val_n: int = 0

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.model._image_mean) / self.model._image_std
        x_enc, _, _ = self.model._forward_encoder(x, shuffle=False)
        patch_tokens = x_enc[:, self.num_register_tokens:]
        return patch_tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.float().argmax(dim=1) == y).float().mean()

        # Weighted accumulation
        n = y.size(0)
        self._train_loss_sum += loss.detach().cpu().item() * n
        self._train_loss_n += n

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  acc,  prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        if self._train_loss_n > 0:
            self.history["train_loss"].append(self._train_loss_sum / self._train_loss_n)
        self._train_loss_sum = 0.0
        self._train_loss_n = 0

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        n = y.size(0)
        correct_top1 = (logits.float().argmax(dim=1) == y).sum().item()
        correct_top5 = (logits.float().topk(5, dim=1).indices == y.unsqueeze(1)).any(dim=1).sum().item()

        # Weighted accumulation
        self._val_loss_sum += loss.detach().cpu().item() * n
        self._val_correct_top1 += correct_top1
        self._val_correct_top5 += correct_top5
        self._val_n += n

        top1 = correct_top1 / n
        top5 = correct_top5 / n
        self.log("val_loss", loss,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_top1", top1,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_top5", top5,  prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self._val_n > 0:
            self.history["val_loss"].append(self._val_loss_sum / self._val_n)
            self.history["val_top1"].append(self._val_correct_top1 / self._val_n)
            self.history["val_top5"].append(self._val_correct_top5 / self._val_n)
        self._val_loss_sum = 0.0
        self._val_correct_top1 = 0.0
        self._val_correct_top5 = 0.0
        self._val_n = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr)

        # Warmup: 10% of total steps, lr rises from 0 to self.lr linearly
        warmup_steps = max(1, int(0.1 * self.total_steps))
        cosine_steps = self.total_steps - warmup_steps

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


# Data

class ImageNetDataset(Dataset):
    """Reads label directly from the folder name (0–999), bypassing ImageFolder's
    alphabetical sorting which would assign wrong indices to numeric class names."""

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = [
            (path, int(path.parent.name))
            for path in sorted(Path(root).glob("*/*.jpg"))
        ]
        assert len(self.samples) > 0, f"No .jpg files found under {root}"
        labels = sorted({label for _, label in self.samples})
        assert labels == list(range(1000)), (
            f"Expected labels 0–999, got {labels[:5]}…{labels[-5:]}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(args: Args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = ImageNetDataset(str(args.data_dir / "train"),      transform=transform_train)
    val_ds   = ImageNetDataset(str(args.data_dir / "validation"), transform=transform_val)

    print(f"Train: {len(train_ds)} images", flush=True)
    print(f"Val  : {len(val_ds)}   images", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.opt_params.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.opt_params.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    return train_loader, val_loader


# Plotting / saving

def save_metrics(module: GatorClassifier, path: str = "metrics.json"):
    with open(path, "w") as f:
        json.dump(module.history, f, indent=2)
    print(f"Metrics saved to {path}")


def plot(module: GatorClassifier, path: str = "training_curves.png"):
    h = module.history
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, h["val_top1"])
    axes[1].set_title("Validation Top-1 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[2].plot(epochs, h["val_top5"])
    axes[2].set_title("Validation Top-5 Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(path)
    print(f"Plot saved to {path}")

# Main

def main(args: Args):
    seed_everything(args.seed)
    cudnn.benchmark = True

    train_loader, val_loader = build_dataloaders(args)

    model = Gator(args.gator_config)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = {
        k.replace("_model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("_model.")
    }

    n_model = len(model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    n_matched = n_model - len(missing)
    print(f"\n[Checkpoint loading]")
    print(f"  matched   : {n_matched} / {n_model} keys")
    print(f"  missing   : {len(missing)}  {missing[:5] if missing else ''}")
    print(f"  unexpected: {len(unexpected)}  {unexpected[:5] if unexpected else ''}")
    if n_matched == 0:
        raise RuntimeError("No keys matched ... checkpoint incompatible with model.")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    args.opt_params.update_lr()

    steps_per_epoch = len(train_loader)
    total_steps = args.opt_params.max_epoch * steps_per_epoch
    print(f"steps/epoch={steps_per_epoch}, total_steps={total_steps}", flush=True)

    module = GatorClassifier(
        model=model,
        embed_dim=args.gator_config.enc_emb_dim,
        num_register_tokens=args.gator_config.num_register_tokens,
        lr=args.opt_params.lr,
        total_steps=total_steps,
    )

    wandb_logger = WandbLogger(project="gator-small-disba")

    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/mayila/gator/gator_small_disba",
        filename="best",
        monitor="val_top1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=args.opt_params.max_epoch,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor()],
        log_every_n_steps=50,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, train_loader, val_loader)
    save_metrics(module)
    plot(module)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)