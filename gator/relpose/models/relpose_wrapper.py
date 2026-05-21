from dataclasses import dataclass

import lightning as L
import torch
from torch import nn

from gator.models.jigsaw_1view.model_jigsaw import Jigsaw1View
from gator.models.gator_losses.base import GatorBaseLoss
from gator.models.gator_visualizer.base import GatorBaseVis

from gator.relpose.loss import RelativeCameraPoseRegression
from gator.relpose.models.pose_head import PoseHead
from gator.relpose.utils.device import to_numpy
from gator.relpose.utils.metric import error_auc, get_rot_err, get_transl_ang_err
from gator.relpose.utils.misc import transpose_to_landscape
from gator.utils import misc
from gator import logger
from torchmetrics import Accuracy
import torchvision.transforms.v2.functional as TF
import numpy as np

@dataclass
class OptimizationParameters:
    weight_decay: float = 0.01
    """weight decay"""
    lr: float = None
    """learning rate (absolute lr)"""
    blr: float = 1e-4
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


class RelposeWrapper(L.LightningModule):
    def __init__(
            self, 
            model: nn.Module,
            loss_fn: RelativeCameraPoseRegression,
            optimization_config: OptimizationParameters,
        ) -> None:
        super().__init__()
        self._model = model

        self.pose_head = PoseHead(net=self)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

        self._loss_fn = loss_fn
        self._opt_config = optimization_config

        self._grid_size = (
            model._config.image_size // model._config.patch_size, 
            model._config.image_size // model._config.patch_size
        )

        self.rerrs_prh = []
        self.terrs_prh = []

    def _downstream_head(self, decout, img_shape):
        B, S, D = decout[-1].shape
        return self.head(decout, img_shape)

    def forward(self, images) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Images must be of shape (B, 2, C, H, W)
        """

        images1 = images[:, 0, :, :, :]
        images2 = images[:, 1, :, :, :]

        out, num_register_tokens = self._model.forward(images1, images2)

        # out is of shape [B, N1+num_register_tokens, D]
        out_noreg = out[:, num_register_tokens:, :] # [B, N1, D]
        with torch.amp.autocast("cuda", enabled=False):
            pose21 = self._downstream_head(out_noreg, img_shape=images.shape[-2:])
        return pose21, out, num_register_tokens
    
    def batch_to_device(self, batch, device):
        for view in batch:
            for name in 'img camera_intrinsics camera_pose'.split(): 
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

    def training_step(self, batch, batch_idx):
        batch = self.batch_to_device(batch, self.device)
        view1, view2 = batch

        pose12, _, _ = self.forward(batch)
        
        # relative camera pose from 2 to 1.
        # swap the two views in the batch;
        pose21, _, _ = self.forward(batch[::-1]) 
        
        with torch.amp.autocast("cuda", enabled=False):
            loss = self._loss_fn(view1, view2, pose12, pose21)

        self.log("train_loss", loss, sync_dist=True)
        return loss
    

    def on_validation_epoch_start(self):
        self.rerrs_prh = []
        self.terrs_prh = []

    def on_validation_epoch_end(self):

        rerrs = np.array(self.rerrs_prh)
        terrs = np.array(self.terrs_prh)
        

        # auc
        auc = error_auc(rerrs, terrs, thresholds=[5, 10, 20])
        for k, v in auc.items():
            self.log(f"val_{k}", v, sync_dist=True)
            
    
    def validation_step(self, batch, batch_idx):
        
        batch = self.batch_to_device(batch, self.device)
        view1, view2 = batch

        pose12, _, _ = self.forward(batch)
        
        # relative camera pose from 2 to 1.
        # swap the two views in the batch;
        pose21, _, _ = self.forward(batch[::-1]) 
        
        with torch.amp.autocast("cuda", enabled=False):
            loss = self._loss_fn(view1, view2, pose12, pose21)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)


        # calculate metrics
        pose = pose21
        gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
        rerrs_prh = []
        terrs_prh = []

        # rotation angular err
        R_prd = pose[:,0:3,0:3]
        for sid in range(len(R_prd)):
            rerrs_prh.append(get_rot_err(to_numpy(R_prd[sid]), to_numpy(gt_pose2to1[sid,0:3,0:3])))  # noqa: F821
        
        # translation direction angular err
        t_prd = pose[:,0:3,3]
        for sid in range(len(t_prd)): 
            transl = to_numpy(t_prd[sid])
            gt_transl = to_numpy(gt_pose2to1[sid,0:3,-1])
            transl_dir = transl / np.linalg.norm(transl)
            gt_transl_dir = gt_transl / np.linalg.norm(gt_transl)
            terrs_prh.append(get_transl_ang_err(transl_dir, gt_transl_dir)) 

        self.rerrs_prh.extend(rerrs_prh)
        self.terrs_prh.extend(terrs_prh)

        
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

        return {"optimizer": optimizer}
