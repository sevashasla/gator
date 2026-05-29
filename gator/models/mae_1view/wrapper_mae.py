from gator.models.criterion import MaskedMSE
from gator.models.gator_visualizer.mae import MAEVis
from gator.models.jigsaw_1view.jigsaw_wrapper import Jigsaw1ViewWrapper, OptimizationParameters
from gator.models.gator_visualizer.denormalize import Denormalize
import lightning as L
import torch

from gator.models.mae_1view.model_mae import MAEModel
from gator.models.gator_visualizer.base import GatorBaseVis

from gator.utils import misc
from gator import logger
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy
import torchvision.transforms.v2.functional as TF


class MAE1ViewWrapper(Jigsaw1ViewWrapper):
    def __init__(
            self, 
            model: MAEModel,
            loss_fn: MaskedMSE,
            visualizer: GatorBaseVis,
            optimization_config: OptimizationParameters,
            denormalize_imagenet: bool = True,
        ) -> None:

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            visualizer=visualizer,
            optimization_config=optimization_config,
        )
        if not isinstance(model, MAEModel):
            raise ValueError("model should be an instance of MAEModel")
        if not isinstance(loss_fn, MaskedMSE):
            raise ValueError("loss_fn should be an instance of MaskedMSE")
        if not isinstance(visualizer, MAEVis):
            raise ValueError("visualizer should be an instance of MAEVis")

        # purely for better vscode type hinting
        self._model: MAEModel
        self._loss_fn: MaskedMSE

        self._denormalize_imagenet = denormalize_imagenet
        self._denormalize = Denormalize(
            denormalize_imagenet=self._denormalize_imagenet,
            denormalize_target=self._loss_fn.norm_pix_loss,
        )

    def forward(self, images, mask_ratio: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Images must be of shape (B, 2, C, H, W)
        """

        image_to_pred = images[:, 0, :, :, :]
        image_ref = images[:, 1, :, :, :]

        out, mask_taken, num_register_tokens = self._model.forward(
            image_to_pred, 
            image_ref, 
            mask_ratio=mask_ratio
        )
        return out, mask_taken, num_register_tokens
    
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

        patches_pred = self._denormalize.denormalize_pred(
            pred=self._visualizer.patchify(images_pred),
            target=self._visualizer.patchify(images_gt_part),
        )
        images_pred = self._visualizer.unpatchify(patches_pred)
        
        images_gt_part, images_pred = self._denormalize.denormalize_images(
            [images_gt_part, images_pred]
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

        out, mask_taken, num_register_tokens = self.forward(batch)
        
        batch = torch.cat([
            batch[:, 0, :, :, :], 
            batch[:, 1, :, :, :]
        ], dim=0) # (2B, C, H, W)

        patches = self._visualizer.patchify(batch) # (2B, N, patch_size**2 * 3)

        loss = self._loss_fn.forward(
            pred=out[:, num_register_tokens:, :], 
            mask=~mask_taken, 
            target=patches,
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = torch.stack(batch, dim=1) # (B, 2, C, H, W)
        
        out, mask_taken, num_register_tokens = self.forward(batch)
        batch = torch.cat([
            batch[:, 0, :, :, :], 
            batch[:, 1, :, :, :]
        ], dim=0) # (2B, C, H, W)

        patches = self._visualizer.patchify(batch) # (2B, N, patch_size**2 * 3)

        # compute and log validation loss
        loss = self._loss_fn.forward(
            pred=out[:, num_register_tokens:, :], 
            mask=~mask_taken, 
            target=patches,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._my_log_images(
                batch=batch, 
                out=out, 
                gt_pos=mask_taken, 
                num_register_tokens=num_register_tokens,
                num_random=8,
            )

        return loss
