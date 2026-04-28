import torch

from gator.models.gator_visualizer.base import GatorBaseVis


class VisualVis(GatorBaseVis):
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
        if pred.size(2) != self._num_patches:
            raise ValueError(f"pred.size(2) should be {self._num_patches} but got {pred.size(2)}")
        pred_no_reg = pred[:, num_register_tokens:, :] # (B, N1, N)
        pred_sf = torch.softmax(pred_no_reg, dim=-1) # (B, N1, N)
        
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

        """
        s - softmax score, [N1, N]
        p - patch, [N, D]
        r - result [N, D]

        for i in range(N1):
            for j in range(N):
                for d in range(D):
                    r_jd = s_ij * p_id
        """
        pred_weighted_sum = torch.einsum(
            "bij,bid->bjd",
            pred_sf,
            patches_selected,
        )  # (B, N, D)

        return self.unpatchify(pred_weighted_sum)