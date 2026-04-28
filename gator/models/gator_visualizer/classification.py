import torch

from gator.models.gator_visualizer.base import GatorBaseVis


class ClassificationVis(GatorBaseVis):
    @torch.no_grad()
    def forward(
        self, 
        pred: torch.Tensor, 
        gt_pos: torch.Tensor, 
        gt_image: torch.Tensor,
        num_register_tokens: int,
    ) -> torch.Tensor:
        """
        pred: (B, n_reg+N1,N)
        gt_pos: (B, N1, 2)
        gt_image: (B, C, H, W)
        num_register_tokens: int
        """
        pred_no_reg = pred[:, num_register_tokens:, :] # (B, N1, N)
        pred_indices = pred_no_reg.argmax(dim=-1) # (B, N1)

        patches = self.patchify(gt_image) # (B, N, D)
        if patches.size(1) != self._num_patches:
            raise ValueError(f"patches.size(1) should be {self._num_patches} but got {patches.size(1)}")
        _, _, D = patches.shape
        gt_pos_flat = gt_pos[:, :, 0] * self._grid_size[1] + gt_pos[:, :, 1] # (B, N1)
        patches_selected = torch.gather(
            patches, 
            dim=1, 
            index=gt_pos_flat[:, :, None].expand(-1, -1, D),
        ) # (B, N1, D)

        patches_pred = torch.zeros_like(patches)
        patches_pred.scatter_reduce_(
            dim=1,
            index=pred_indices[:, :, None].expand(-1, -1, D),
            src=patches_selected,
            reduce="mean",
            include_self=False,
        )

        images_pred = self.unpatchify(patches_pred, channels=gt_image.shape[1])
        return images_pred
