from dataclasses import dataclass

import lightning as L
import torch
from gator.models.criterion import MaskedMSE
from gator.models.croco import CroCoNet
from gator.utils import misc
from gator import logger
import torchvision.transforms.v2.functional as TF

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
            self.lr = self.blr * eff_batch_size / 256
        
        logger.info("Updated LR")
        logger.info(f"base lr: {self.lr * 256 / eff_batch_size:.2e}")
        logger.info(f"actual lr: {self.lr:.2e}")
        logger.info(f"accumulate grad iterations: {self.accum_iter}")
        logger.info(f"effective batch size: {eff_batch_size}")

    def update_steps_per_epoch(self):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = int(self.dataset_size * (1 - self.tt_split_ratio)) // \
                (self.batch_size * misc.get_world_size())
        
        logger.info(f"Updated steps per epoch: {self.steps_per_epoch}")


class CroCoWrapper(L.LightningModule):
    def __init__(
            self, 
            model: CroCoNet,
            loss_fn: MaskedMSE,
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

        out, mask1, target = self._model(image_to_pred, image_ref)
        return out, mask1, target

    def on_train_batch_start(self, batch, batch_idx) -> None | int:
        optimizer = self.optimizers()
        epoch = self.global_step / self._opt_config.steps_per_epoch
        misc.adjust_learning_rate(
            optimizer, epoch, 
            self._opt_config,
        )
        super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        # batch = torch.stack(batch, dim=0) # (B, 2, C, H, W)
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)

        out, mask1, target = self.forward(batch)
        loss = self._loss_fn(out, mask1, target)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)
        
        out, mask1, target = self.forward(batch)
        loss = self._loss_fn(out, mask1, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            # randomly select 8 images from the batch
            indices = torch.randperm(batch.size(0))[:8]
            images_gt_part = batch[indices, 0, :, :, :]
            images_ref = batch[indices, 1, :, :, :]
            images_pred = self._model.unpatchify(out[indices])

            # (8, C, H, W) -> (C, H, 8*W)
            images_ref_cat = torch.cat(images_ref.unbind(0), dim=-1)
            images_gt_cat = torch.cat(images_gt_part.unbind(0), dim=-1)
            images_pred_cat = torch.cat(images_pred.unbind(0), dim=-1)

            # concatenate them into one image
            # (C, H, 8*W)x3 -> (C, 3*H, 8*W)
            images_cat = torch.cat([
                images_ref_cat, 
                images_gt_cat, 
                images_pred_cat
            ], dim=1)
            images_cat = TF.resize(images_cat, (112 * 3, 112 * 8))
            images_cat = images_cat.clamp(0, 1)

            self.logger.log_image(
                "val_images", [images_cat.cpu()], self.current_epoch
            )

        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer and scheduler used in optimization.
        """

        param_groups = misc.get_parameter_groups(
            self._model, self._opt_config.weight_decay
        ) # following timm: set wd as 0 for bias and norm layers
        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self._opt_config.lr, 
            betas=(0.9, 0.95)
        )

        return optimizer