"""
The idea is to use the similarity between patches with the position probability is proportional to that similarity.

We use the following formula: p_ij = exp(-||p_i - p_j||_2^2 / sigma^2) / sum_k exp(-|p_i - p_k||_2^2 / sigma^2)
"""

import torch
import torch.nn.functional as F

from gator.models.gator_losses.base import GatorBaseLoss
from gator.models.gator_losses.visual import GatorVisualLoss

class GatorSimilarityLoss(GatorBaseLoss):

    def __init__(
            self,
            tau: float=0.01,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tau = tau

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

        patches_images = GatorVisualLoss.patchify(gt_image, self._patch_size) # (B, N, D)
        B, N, D = patches_images.shape

        input_patches = torch.gather(
            patches_images, 
            dim=1, 
            index=classes[:, :, None].expand(-1, -1, D),
        )
        # find difference between each patch and the input patch
        # divide sqrt(D) ** 2 to make it invariant to the dimension of the patch
        diffs = torch.cdist(input_patches, patches_images).pow(2) / (self._tau ** 2.0 * D) # (B, N1, N)
        probs_target = F.softmax(-diffs, dim=-1) # (B, N1, N)

        loss = F.cross_entropy(
            pred[:, num_register_tokens:, :].flatten(0, 1),
            probs_target.flatten(0, 1),
        )
        return loss
