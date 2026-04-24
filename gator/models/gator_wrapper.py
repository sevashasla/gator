from dataclasses import dataclass

import lightning as L
import torch

from gator.models.model_gator import Gator
from gator.models.gator_losses.base import GatorBaseLoss
from gator.utils import misc
from gator import logger
from transformers import get_cosine_schedule_with_warmup

@dataclass
class OptimizationParameters:
    weight_decay: float = 0.05
    """weight decay (default: 0.05)"""
    lr: float = None
    """learning rate (absolute lr)"""
    blr: float = 1.5e-4
    """base learning rate: absolute_lr = base_lr * total_batch_size / 256"""
    min_lr: float = 0.
    """lower lr bound for cyclic schedulers that hit 0"""
    warmup_epochs: int = 40
    """epochs to warmup LR"""
    accum_iter: int = 1
    """
    Accumulate gradient iterations (for increasing the effective batch size
    under memory constraints)
    """
    batch_size: int = 64
    """Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus"""
    epochs: int = 800
    """Maximum number of epochs for the scheduler"""
    max_epoch: int = 400
    """Stop training at this epoch"""

    dataset_size: int = 181*10000
    """
    number of shards * shard_size
    """

    steps_per_epoch: int | None = None

    def __post_init__(self):
        self.update_lr()
        self.update_steps_per_epoch()

    def update_lr(self):
        eff_batch_size = self.batch_size * self.accum_iter * misc.get_world_size()
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 256
        
        logger.info("Updated LR")
        logger.info(f"base lr: {self.lr * 256 / eff_batch_size:.2e}")
        logger.info(f"actual lr: {self.lr:.2e}")
        logger.info(f"accumulate grad iterations: {self.accum_iter}")
        logger.info(f"effective batch size: {eff_batch_size}")

    def update_steps_per_epoch(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = self.dataset_size // (self.batch_size * misc.get_world_size())
        
        logger.info(f"Updated steps per epoch: {self.steps_per_epoch}")


class GatorWrapper(L.LightningModule):
    def __init__(
            self, 
            model: Gator,
            loss_fn: GatorBaseLoss,
            optimization_config: OptimizationParameters,
        ) -> None:
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn
        self._opt_config = optimization_config

    def forward(self, images) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Images must be of shape (B, 2, C, H, W)
        """

        image_to_pred = images[:, 0, :, :, :]
        image_ref = images[:, 1, :, :, :]

        out, gt_pos, num_register_tokens = self._model(image_to_pred, image_ref)
        return out, gt_pos, num_register_tokens

    def training_step(self, batch, batch_idx):
        # batch = torch.stack(batch, dim=0) # (B, 2, C, H, W)
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)

        out, gt_pos, num_register_tokens = self.forward(batch)
        loss = self._loss_fn(
            pred=out, 
            gt_pos=gt_pos, 
            gt_image=batch[:, 0, :, :, :],
            num_register_tokens=num_register_tokens,
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)
        
        out, gt_pos, num_register_tokens = self.forward(batch)
        loss = self._loss_fn(
            pred=out, 
            gt_pos=gt_pos, 
            gt_image=batch[:, 0, :, :, :],
            num_register_tokens=num_register_tokens,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler used in optimization.
        """ 
        optimizer = torch.optim.AdamW(
            self._model.parameters(), 
            lr=self._opt_config.lr, 
            weight_decay=self._opt_config.weight_decay,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._opt_config.warmup_epochs * self._opt_config.steps_per_epoch,
            num_training_steps=self._opt_config.epochs * self._opt_config.steps_per_epoch,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }