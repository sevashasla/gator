import torch
import torch.nn.functional as F
import torch.nn as nn

from gator.models.gator_losses.base import GatorBaseLoss

class GatorDistanceBasedLoss(GatorBaseLoss):
    def forward(
            self, 
            pred: torch.Tensor, 
            gt_pos: torch.Tensor, 
            gt_image: torch.Tensor,
            num_register_tokens: int,
        ):
        """
        pred: (B, n_reg+N1,2)
        gt_pos: (B, N1, 2)
        gt_image: (B, C, H, W)
        num_register_tokens: int
        """

        if pred.size(2) != 2:
            raise ValueError(f"pred.size(2) should be 2 but got {pred.size(2)}")
        pred_no_reg = pred[:, num_register_tokens:, :] # (B, N1, 2)
        # pred_no_reg has values from (-1, 1)
        gt_pos_float = gt_pos.float()
        # normalize to (-1, 1)
        gt_pos_float[:, :, 0] = gt_pos_float[:, :, 0] / (self._grid_size[0] - 1) * 2 - 1 
        gt_pos_float[:, :, 1] = gt_pos_float[:, :, 1] / (self._grid_size[1] - 1) * 2 - 1

        return F.mse_loss(pred_no_reg, gt_pos_float)
