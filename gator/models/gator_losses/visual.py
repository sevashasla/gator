import torch
import torch.nn.functional as F
import torch.nn as nn

from gator.models.gator_losses.base import GatorBaseLoss

class GatorVisualLoss(GatorBaseLoss):
    @staticmethod
    def patchify(imgs, patch_size):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def forward(
            self, 
            pred: torch.Tensor, 
            gt_pos: torch.Tensor, 
            gt_image: torch.Tensor,
            num_register_tokens: int,
        ):
        """
        pred: (B, n_reg+N1,num_patches)
        gt_pos: (B, N1, 2)
        gt_image: (B, C, H, W)
        num_register_tokens: int

        1. patchify image from (B, C, H, W) -> (B, N, D)
        2. select N1 patches (B, N1, D) that were chosen for shuffling
        3. expand by repeating the same patch (B, N1, N, D) along the 2nd axis
        4. Multiply by softmax weight (B, N1, N)
        5. sum (B, N1, N, D) -> (B, N, D)
        6. select only gt tokens (B, N, D) -> (B, N1, D)
        7. compare with the ground truth
        """

        if pred.size(2) != self._num_patches:
            raise ValueError(f"pred.size(2) should be {self._num_patches} but got {pred.size(2)}")
        pred_no_reg = pred[:, num_register_tokens:, :] # (B, N1, N)
        pred_sf = torch.softmax(pred_no_reg, dim=-1) # (B, N1, N)
        
        patches = self.patchify(gt_image, self._patch_size) # (B, N, D)
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

        pred_selected = torch.gather(
            pred_weighted_sum,
            dim=1,
            index=gt_pos_flat[:, :, None].expand(-1, -1, D),
        )  # (B, N1, D)

        loss = F.mse_loss(pred_selected, patches_selected)
        return loss
        
        # # find indices to take for the loss computation
        # sorted_ids = torch.argsort(pred_no_reg, dim=-1)
        # # take top best indices
        # best_match_ids = sorted_ids[:, :, -self._top_n:] # (B, N1, top_n)
        
        # # take top worst indices
        # worst_match_ids = sorted_ids[:, :, :self._top_n] # (B, N1, top_n)

        # # take indices around gt_pos
        # gt_pos_ids = gt_pos_flat[:, :, None] # (B, N1, 1)
        # around_gt_pos_ids = gt_pos_ids + torch.arange(-self._top_n//2, self._top_n//2+1, device=gt_pos_ids.device) # (B, N1, top_n)
        # around_gt_pos_ids = torch.clamp(around_gt_pos_ids, 0, self._num_patches - 1) # (B, N1, top_n)

        # indices = torch.cat([best_match_ids, worst_match_ids, around_gt_pos_ids], dim=-1) # (B, N1, top_n*3)

        # # find prediction
        # pred_sf = torch.softmax(pred_no_reg, dim=-1) # (B, N1, N)
        # weights_selected = torch.gather(pred_sf, index=indices, dim=-1) # (B, N1, top_n*3)

        # # find gt
        # patches_selected = torch.gather(
        #     patches, 
        #     dim=1, 
        #     index=gt_pos_flat[:, :, None].expand(-1, -1, patches.size(2)),
        # ) # (B, N1, patch_size**2 * 3)
        # patches_selected = patches_selected[:, :, None, :] # (B, N1, patch_size**2 * 3)
