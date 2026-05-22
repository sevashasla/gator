from dataclasses import dataclass

import lightning as L
import torch
from torch import nn

from gator.relpose.loss import RelativeCameraPoseRegression
from gator.relpose.utils.device import to_numpy
from gator.relpose.utils.metric import error_auc, get_rot_err, get_transl_ang_err
from gator.utils import misc
from gator import logger
import numpy as np
from transformers import get_constant_schedule_with_warmup

@dataclass
class RelposeOptimizationParameters:
    weight_decay: float = 0.01
    """weight decay"""
    lr: float = None
    """learning rate (absolute lr)"""
    blr: float = 1e-5
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
    epochs: int = 60
    """Maximum number of epochs for the scheduler"""
    max_epoch: int = 15
    """Stop training at this epoch"""

    steps_per_epoch: int | None = None

    def __post_init__(self):
        self.update_lr()

    def update_lr(self):
        eff_batch_size = self.batch_size * self.accum_iter * misc.get_world_size()
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 128
        
        logger.info("Updated LR")
        logger.info(f"base lr: {self.lr * 128 / eff_batch_size:.2e}")
        logger.info(f"actual lr: {self.lr:.2e}")
        logger.info(f"accumulate grad iterations: {self.accum_iter}")
        logger.info(f"effective batch size: {eff_batch_size}")

    def update_steps_per_epoch(self, dataset_size: int):
        if self.steps_per_epoch is None:
            self.steps_per_epoch = dataset_size // (self.batch_size * misc.get_world_size())
        
        logger.info(f"Updated steps per epoch: {self.steps_per_epoch}")


class RelposeWrapper(L.LightningModule):
    def __init__(
            self, 
            model: nn.Module,
            loss_fn: RelativeCameraPoseRegression,
            optimization_config: RelposeOptimizationParameters,
        ) -> None:
        super().__init__()
        self._model = model

        self._loss_fn = loss_fn
        self._opt_config = optimization_config

        self._rerrs_prh = []
        self._terrs_prh = []

    def forward(
            self, 
            view1: dict[str, torch.Tensor], 
            view2: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Images must be of shape (B, C, H, W)
        """

        return self._model.forward(view1, view2)
    
    def batch_to_device(self, batch, device):
        for view in batch:
            for name in 'img camera_intrinsics camera_pose'.split(): 
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        return batch

    def training_step(self, batch, batch_idx):
        batch = self.batch_to_device(batch, self.device)
        view1, view2 = batch

        pose12, pose21 = self.forward(view1, view2)
        
        with torch.amp.autocast("cuda", enabled=False):
            loss = self._loss_fn(view1, view2, pose12, pose21)[0]

        self.log("train_loss", loss, sync_dist=True)
        return loss
    

    def on_validation_epoch_start(self):
        self._rerrs_prh.clear()
        self._terrs_prh.clear()

    def on_validation_epoch_end(self):

        rerrs = np.array(self._rerrs_prh)
        terrs = np.array(self._terrs_prh)
        
        # auc
        auc = error_auc(rerrs, terrs, thresholds=[5, 10, 20])
        for k, v in auc.items():
            self.log(f"val_{k}", v, sync_dist=True)

        self._rerrs_prh.clear()
        self._terrs_prh.clear()
    
    def validation_step(self, batch, batch_idx):
        
        batch = self.batch_to_device(batch, self.device)
        view1, view2 = batch

        pose12, pose21 = self.forward(view1, view2)

        with torch.amp.autocast("cuda", enabled=False):
            loss = self._loss_fn(view1, view2, pose12, pose21)[0]
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        # calculate metrics
        pose = pose21['pose']
        gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
        rerrs_prh = []
        terrs_prh = []

        # rotation angular err
        R_prd = pose[:,0:3,0:3]
        for sid in range(len(R_prd)):
            rerrs_prh.append(get_rot_err(
                to_numpy(R_prd[sid]), to_numpy(gt_pose2to1[sid,0:3,0:3])
            ))  # noqa: F821
        
        # translation direction angular err
        t_prd = pose[:,0:3,3]
        for sid in range(len(t_prd)): 
            transl = to_numpy(t_prd[sid])
            gt_transl = to_numpy(gt_pose2to1[sid,0:3,-1])
            transl_dir = transl / np.linalg.norm(transl)
            gt_transl_dir = gt_transl / np.linalg.norm(gt_transl)
            terrs_prh.append(get_transl_ang_err(transl_dir, gt_transl_dir)) 

        self._rerrs_prh.extend(rerrs_prh)
        self._terrs_prh.extend(terrs_prh)
        
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
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                self._opt_config.warmup_epochs * self._opt_config.steps_per_epoch
            ),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
