import torch
import torch.nn.functional as F

from gator.models.gator_losses.base import GatorBaseLoss

class GatorClassificationLoss(GatorBaseLoss):
    def forward(
            self, 
            pred: torch.Tensor, 
            gt_pos: torch.Tensor, 
            gt_image: torch.Tensor,
            num_register_tokens: int,
        ):
        """
        pred: (B, num_register_tokens+N1,num_patches)
        gt_pos: (B, N1, 2)
        num_register_tokens: int
        """

        if pred.size(2) != self._num_patches:
            raise ValueError(f"pred.size(2) should be {self._num_patches} but got {pred.size(2)}")

        classes = gt_pos[:, :, 0] * self._grid_size[1] + gt_pos[:, :, 1] # (B, N1)
        loss = F.cross_entropy(
            pred[:, num_register_tokens:, :].flatten(0, 1),
            classes.flatten(0, 1),
        )
        return loss