from dataclasses import dataclass

import lightning as L
import torch

from gator.models.jigsaw_1view.model_jigsaw import Jigsaw1View
from gator.models.gator_losses.base import GatorBaseLoss
from gator.models.gator_visualizer.base import GatorBaseVis

from gator.utils import misc
from gator import logger
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy
import torchvision.transforms.v2.functional as TF

@dataclass
class OptimizationParameters:
    weight_decay: float = 0.01
    """weight decay"""
    lr: float = None
    """learning rate (absolute lr)"""
    blr: float = 3e-4
    """base learning rate: absolute_lr = base_lr * total_batch_size / 128"""
    min_lr: float = 0.
    """lower lr bound for cyclic schedulers that hit 0"""
    warmup_epochs: float = 1.0
    """epochs to warmup LR"""
    accum_iter: int = 1
    """
    Accumulate gradient iterations (for increasing the effective batch size
    under memory constraints)
    """
    batch_size: int = 128
    """Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus"""
    epochs: int = 200
    """Maximum number of epochs for the scheduler"""
    max_epoch: int = 50
    """Stop training at this epoch"""

    dataset_size: int = 181*10000
    """
    number of shards * shard_size
    """

    tt_split_ratio: float = 0.03
    """
    train-test split ratio for the dataset (used to determine number of steps
    per epoch)
    """

    steps_per_epoch: int | None = None

    def __post_init__(self):
        self.update_lr()
        self.update_steps_per_epoch()

    def update_lr(self):
        eff_batch_size = self.batch_size * self.accum_iter * misc.get_world_size()
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 128
        
        logger.info("Updated LR")
        logger.info(f"base lr: {self.lr * 128 / eff_batch_size:.2e}")
        logger.info(f"actual lr: {self.lr:.2e}")
        logger.info(f"accumulate grad iterations: {self.accum_iter}")
        logger.info(f"effective batch size: {eff_batch_size}")

    def update_steps_per_epoch(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = int(self.dataset_size * (1 - self.tt_split_ratio)) // \
                (self.batch_size * misc.get_world_size())
        
        logger.info(f"Updated steps per epoch: {self.steps_per_epoch}")


class Jigsaw1ViewWrapper(L.LightningModule):
    def __init__(
            self, 
            model: Jigsaw1View,
            loss_fn: GatorBaseLoss,
            visualizer: GatorBaseVis,
            optimization_config: OptimizationParameters,
        ) -> None:
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn
        self._opt_config = optimization_config

        self._grid_size = (
            model._config.image_size // model._config.patch_size, 
            model._config.image_size // model._config.patch_size
        )
        self._visualizer = visualizer
        self._acc = Accuracy(
            task="multiclass", 
            num_classes=self._grid_size[0] * self._grid_size[1],
        )

    def distance_based_to_idx(self, pred: torch.Tensor, num_register_tokens: int) -> torch.Tensor:
        """
        pred is of shape (B, N1, 2), where the last dimension is the normalized
        (y, x) coordinates in the range [-1, 1]. This function converts the
        normalized coordinates to patch indices and returns a tensor of shape
        (B, N1) containing the patch indices.
        """
        pred_no_reg = pred[:, num_register_tokens:, :] # (B, N1, 2)
        pred_no_reg_unnorm = torch.stack([
            (pred_no_reg[:, :, 0] + 1) / 2 * (self._grid_size[0] - 1),
            (pred_no_reg[:, :, 1] + 1) / 2 * (self._grid_size[1] - 1),
        ], dim=-1).round().to(torch.long) # (B, N1, 2)
        pred_ids = pred_no_reg_unnorm[:, :, 0] * self._grid_size[1] + pred_no_reg_unnorm[:, :, 1] # (B, N1)
        return pred_ids

    def forward(self, images, shuffle_ratio: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Images must be of shape (B, 2, C, H, W)
        """

        image_to_pred = images[:, 0, :, :, :]
        image_ref = images[:, 1, :, :, :]

        out, gt_pos, num_register_tokens = self._model.forward(
            image_to_pred, 
            image_ref, 
            shuffle_ratio=shuffle_ratio
        )
        return out, gt_pos, num_register_tokens
    
    def _my_log_images(
            self, 
            batch: torch.Tensor, 
            out: torch.Tensor, 
            gt_pos: torch.Tensor, 
            num_register_tokens: int, 
            num_random: int = 8
        ):
        # randomly select 8 images from the batch
        indices = torch.randperm(batch.size(0))[:num_random]
        images_gt_part = batch[indices, :, :, :]

        images_pred = self._visualizer.forward(
            pred=out[indices], 
            gt_pos=gt_pos[indices],
            gt_image=images_gt_part,
            num_register_tokens=num_register_tokens,
        )

        # (8, C, H, W) -> (C, H, 8*W)
        images_gt_cat = torch.cat(images_gt_part.unbind(0), dim=-1)
        images_pred_cat = torch.cat(images_pred.unbind(0), dim=-1)

        # concatenate them into one image
        # (C, H, 8*W) -> (C, 2*H, 8*W)
        images_cat = torch.cat([ 
            images_gt_cat, 
            images_pred_cat
        ], dim=1)
        images_cat = TF.resize(images_cat, (112 * 2, 112 * num_random))
        images_cat = images_cat.clamp(0, 1)

        self.logger.log_image(
            "val_images", [images_cat.cpu()], self.current_epoch
        )

    def training_step(self, batch, batch_idx):
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)

        out, gt_pos, num_register_tokens = self.forward(batch)
        
        batch = torch.cat([
            batch[:, 0, :, :, :], 
            batch[:, 1, :, :, :]
        ], dim=0) # (2B, C, H, W)

        loss = self._loss_fn.forward(
            pred=out, 
            gt_pos=gt_pos, 
            gt_image=batch,
            num_register_tokens=num_register_tokens,
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)
        
        out, gt_pos, num_register_tokens = self.forward(
            batch, 
            shuffle_ratio=1.0,
        )
        batch = torch.cat([
            batch[:, 0, :, :, :], 
            batch[:, 1, :, :, :]
        ], dim=0) # (2B, C, H, W)

        # compute and log validation loss
        loss = self._loss_fn.forward(
            pred=out, 
            gt_pos=gt_pos, 
            gt_image=batch,
            num_register_tokens=num_register_tokens,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        # compute and log accuracy
        if out.shape[-1] != self._grid_size[0] * self._grid_size[1]:
            pred_ids = self.distance_based_to_idx(out, num_register_tokens) # (2B, N1)
        else:
            pred_ids = out[:, num_register_tokens:, :].argmax(dim=-1) # (2B, N1)

        gt_ids_flat = gt_pos[:, :, 0] * self._grid_size[1] + gt_pos[:, :, 1] # (2B, N1)
        self._acc.update(pred_ids, gt_ids_flat)
        acc_value = self._acc.compute()
        self.log("val_acc", acc_value, on_step=False, on_epoch=True, sync_dist=True)
        self._acc.reset()

        if batch_idx == 0:
            self._my_log_images(
                batch=batch, 
                out=out, 
                gt_pos=gt_pos, 
                num_register_tokens=num_register_tokens,
                num_random=8,
            )

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
            num_warmup_steps=int(self._opt_config.warmup_epochs * self._opt_config.steps_per_epoch),
            num_training_steps=self._opt_config.epochs * self._opt_config.steps_per_epoch,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
