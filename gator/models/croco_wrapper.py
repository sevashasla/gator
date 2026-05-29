import lightning as L
import torch
from gator.models.criterion import MaskedMSE
from gator.models.croco import CroCoNet
from gator.models.gator_visualizer.denormalize import Denormalize
from gator.utils import misc
from gator import logger
import torchvision.transforms.v2.functional as TF
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters


class CroCoWrapper(L.LightningModule):
    def __init__(
            self, 
            model: CroCoNet,
            loss_fn: MaskedMSE,
            optimization_config: OptimizationParameters,
            denormalize_imagenet: bool = True,
        ) -> None:
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn
        self._opt_config = optimization_config
        self._denormalize_imagenet = denormalize_imagenet

        self._denormalize = Denormalize(
            denormalize_imagenet=self._denormalize_imagenet,
            denormalize_target=self._loss_fn.norm_pix_loss,
        )

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

        if batch_idx == 0 and self.trainer.is_global_zero:
            # randomly select 8 images from the batch
            indices = torch.randperm(batch.size(0))[:8]
            images_gt_part = batch[indices, 0, :, :, :]
            images_ref = batch[indices, 1, :, :, :]
            images_pred = out[indices]

            patches_pred = self._denormalize.denormalize_pred(
                pred=images_pred,
                target=self._model.patchify(images_gt_part),
            )
            images_pred = self._model.unpatchify(patches_pred)
            images_gt_part, images_ref, images_pred = self._denormalize.denormalize_images(
                [images_gt_part, images_ref, images_pred]
            )

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