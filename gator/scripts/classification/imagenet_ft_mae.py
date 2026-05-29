import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from functools import partial

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image

import tyro
from lightning import Trainer, LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import json
import matplotlib.pyplot as plt

from gator.models.models_mae import MaskedAutoencoderViT


_IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGE_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Args:
    checkpoint_path: Path
    data_dir: Path = Path("/scratch/izar/mayila/imagenet_full")

    # MAE architecture
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 16
    img_size: int = 224

    # Training
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 60

    num_workers: int = 8
    seed: int = 0
    precision: Literal['16-mixed', '32'] = '16-mixed'


class MAEClassifier(LightningModule):
    def __init__(self, model: MaskedAutoencoderViT, embed_dim: int, lr: float, total_steps: int, warmup_steps: int):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        self.head = nn.Linear(embed_dim, 1000)
        self.criterion = nn.CrossEntropyLoss()

        self._train_loss_buf: list[float] = []
        self._val_loss_buf: list[float] = []
        self._val_top1_buf: list[float] = []
        self._val_top5_buf: list[float] = []

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        mean = _IMAGE_MEAN.to(x.device)
        std  = _IMAGE_STD.to(x.device)
        x = (x - mean) / std
        # mask_ratio=0 → no masking, all patches visible
        x_enc, _, _ = self.model.forward_encoder(x, mask_ratio=0.0)
        # CLS token is at index 0
        return x_enc[:, 0, :]  # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.float().argmax(dim=1) == y).float().mean()

        self._train_loss_buf.append(loss.detach().cpu().item())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",  acc,  prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        if self._train_loss_buf:
            self.history["train_loss"].append(sum(self._train_loss_buf) / len(self._train_loss_buf))
            self._train_loss_buf.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        top1 = (logits.float().argmax(dim=1) == y).float().mean()
        top5 = (logits.float().topk(5, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean()

        if batch_idx == 0:
            feats = self.extract_features(x)
            self.log("debug/feat_mean", feats.mean().item(), on_step=True, on_epoch=False)
            self.log("debug/feat_std",  feats.std().item(),  on_step=True, on_epoch=False)
            print(f"[VAL epoch {self.current_epoch}] feat mean={feats.mean():.4f} std={feats.std():.4f} labels min={y.min().item()} max={y.max().item()}", flush=True)

        self._val_loss_buf.append(loss.detach().cpu().item())
        self._val_top1_buf.append(top1.detach().cpu().item())
        self._val_top5_buf.append(top5.detach().cpu().item())

        self.log("val_loss", loss,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_top1", top1,  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_top5", top5,  prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self._val_loss_buf:
            self.history["val_loss"].append(sum(self._val_loss_buf) / len(self._val_loss_buf))
            self.history["val_top1"].append(sum(self._val_top1_buf) / len(self._val_top1_buf))
            self.history["val_top5"].append(sum(self._val_top5_buf) / len(self._val_top5_buf))
            self._val_loss_buf.clear()
            self._val_top1_buf.clear()
            self._val_top5_buf.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr)

        # warmup_steps = 10% of total_steps, passed from main()
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps - self.warmup_steps,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


# Data

class ImageNetDataset(Dataset):
    """Reads label directly from the folder name (0-999)"""

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = [
            (path, int(path.parent.name))
            for path in sorted(Path(root).glob("*/*.jpg"))
        ]
        assert len(self.samples) > 0, f"No .jpg files found under {root}"
        labels = sorted({label for _, label in self.samples})
        assert labels == list(range(1000)), (
            f"Expected labels 0-999, got {labels[:5]}...{labels[-5:]}"
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    return train_loader, val_loader


# Plotting / saving

def save_metrics(module: MAEClassifier, path: str = "metrics_mae.json"):
    with open(path, "w") as f:
        json.dump(module.history, f, indent=2)
    print(f"Metrics saved to {path}")


def plot(module: MAEClassifier, path: str = "training_curves_mae.png"):
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

    model = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")

    # Support both original MAE checkpoints (.pth) and Lightning checkpoints (.ckpt)
    if "state_dict" in ckpt and any(k.startswith("model.") for k in ckpt["state_dict"]):
        # Lightning checkpoint
        state_dict = {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        # Original MAE checkpoint
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    n_model   = len(model.state_dict())
    n_matched = n_model - len(missing)
    print(f"\n[CHECK] Checkpoint loading")
    print(f"  matched  : {n_matched} / {n_model} keys")
    print(f"  missing  : {len(missing)}  {missing[:5] if missing else ''}")
    print(f"  unexpected: {len(unexpected)}  {unexpected[:5] if unexpected else ''}")
    if n_matched == 0:
        raise RuntimeError("No keys matched ... checkpoint incompatible with model.")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    steps_per_epoch = len(train_loader)
    total_steps     = args.max_epochs * steps_per_epoch
    warmup_steps    = max(1, int(0.1 * total_steps))  # 10% warmup

    print(f"steps/epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}", flush=True)

    module = MAEClassifier(
        model=model,
        embed_dim=args.embed_dim,
        lr=args.lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    wandb_logger = WandbLogger(project="mae-finetune")

    checkpoint_callback = ModelCheckpoint(
        dirpath="/scratch/izar/mayila/mae_checkpoints",
        filename="best",
        monitor="val_top1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
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