from dataclasses import dataclass

import lightning as L
import torch
from gator.models.criterion import MaskedMSE
from gator.models.croco import CroCoNet
from gator.utils import misc
from gator import logger

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

    def update_lr(self, batch_size):
        eff_batch_size = batch_size * self.accum_iter * misc.get_world_size()
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 256
        
        logger.info("Updated LR")
        logger.info("base lr: %.2e" % (self.lr * 256 / eff_batch_size))
        logger.info("actual lr: %.2e" % self.lr)
        logger.info("accumulate grad iterations: %d" % self.accum_iter)
        logger.info("effective batch size: %d" % eff_batch_size)


class CroCoWrapper(L.LightningModule):
    def __init__(
            self, 
            model: CroCoNet,
            loss_fn: MaskedMSE,
            optimization_params: OptimizationParameters,
        ) -> None:
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn
        self._opt_params = optimization_params

    def forward(self, images) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Images must be of shape (B, 2, C, H, W)
        """

        image_to_pred = images[:, 0, :, :, :]
        image_ref = images[:, 1, :, :, :]

        out, mask1, target = self._model(image_ref, image_to_pred)
        return out, mask1, target

    def on_train_batch_start(self, batch, batch_idx) -> None | int:
        optimizer = self.optimizers()
        # steps_per_epoch = len(self.trainer.train_dataloader)
        steps_per_epoch = 28_000 # TODO
        epoch = self.global_step / steps_per_epoch
        misc.adjust_learning_rate(
            optimizer, epoch, 
            self._opt_params,
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
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer and scheduler used in optimization.
        """

        param_groups = misc.get_parameter_groups(
            self._model, self._opt_params.weight_decay
        ) # following timm: set wd as 0 for bias and norm layers
        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self._opt_params.lr, 
            betas=(0.9, 0.95)
        )

        return optimizer