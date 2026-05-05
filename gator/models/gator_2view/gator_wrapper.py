from dataclasses import dataclass

import lightning as L
import torch

from gator.models.gator_2view.model_gator import Gator
from gator.models.jigsaw_1view.jigsaw_wrapper import Jigsaw1ViewWrapper, OptimizationParameters
from gator.models.gator_losses.base import GatorBaseLoss
from gator.models.gator_visualizer.base import GatorBaseVis
from gator.utils import misc
from gator import logger
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy
import torchvision.transforms.v2.functional as TF

class GatorWrapper(Jigsaw1ViewWrapper):
    def __init__(
            self, 
            model: Gator,
            loss_fn: GatorBaseLoss,
            visualizer: GatorBaseVis,
            optimization_config: OptimizationParameters,
        ) -> None:

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            visualizer=visualizer,
            optimization_config=optimization_config,
        )

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
        images_gt_part = batch[indices, 0, :, :, :]
        images_gt_ref = batch[indices, 1, :, :, :]

        images_pred = self._visualizer.forward(
            pred=out[indices], 
            gt_pos=gt_pos[indices],
            gt_image=images_gt_part,
            num_register_tokens=num_register_tokens,
        )

        # (8, C, H, W) -> (C, H, 8*W)
        images_ref_cat = torch.cat(images_gt_ref.unbind(0), dim=-1)
        images_gt_cat = torch.cat(images_gt_part.unbind(0), dim=-1)
        images_pred_cat = torch.cat(images_pred.unbind(0), dim=-1)

        # concatenate them into one image
        # (C, H, 8*W) -> (C, 2*H, 8*W)
        images_cat = torch.cat([
            images_ref_cat, 
            images_gt_cat, 
            images_pred_cat
        ], dim=1)
        images_cat = TF.resize(images_cat, (112 * 3, 112 * num_random))
        images_cat = images_cat.clamp(0, 1)

        self.logger.log_image(
            "val_images", [images_cat.cpu()], self.current_epoch
        )

    def training_step(self, batch, batch_idx):
        # batch = torch.stack(batch, dim=0) # (B, 2, C, H, W)
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)

        out, gt_pos, num_register_tokens = self.forward(batch)
        loss = self._loss_fn.forward(
            pred=out, 
            gt_pos=gt_pos, 
            gt_image=batch[:, 0, :, :, :],
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

        # compute and log validation loss
        loss = self._loss_fn.forward(
            pred=out, 
            gt_pos=gt_pos, 
            gt_image=batch[:, 0, :, :, :],
            num_register_tokens=num_register_tokens,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        # compute and log accuracy
        if out.shape[-1] != self._grid_size[0] * self._grid_size[1]:
            pred_ids = self.distance_based_to_idx(out, num_register_tokens) # (B, N1)
        else:
            pred_ids = out[:, num_register_tokens:, :].argmax(dim=-1) # (B, N1)
            
        gt_ids_flat = gt_pos[:, :, 0] * self._grid_size[1] + gt_pos[:, :, 1] # (B, N1)
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
    