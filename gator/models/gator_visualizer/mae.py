import torch

from gator.models.gator_visualizer.base import GatorBaseVis


class MAEVis(GatorBaseVis):
    @torch.no_grad()
    def forward(
        self, 
        pred: torch.Tensor, 
        gt_pos: torch.Tensor, 
        gt_image: torch.Tensor,
        num_register_tokens: int,
    ) -> torch.Tensor:
        """
        pred: (B, n_reg+N, D)
        gt_pos: (B, N)
        >    Yes, it is better to create three classes:
            - base visualizer
            - jigsaw-like base visualizer (child of base visualizer)
            - mae-like base visualizer (child of base visualizer)

            and then inherit from the base visualizer in the jigsaw wrapper and mae wrapper respectively,
            but I am lazy...

        gt_image: (B, C, H, W)
        num_register_tokens: int
        """

        B, C, H, W = gt_image.shape
        D = self._patch_size ** 2 * C
        
        if pred.size(-1) != D:
            raise ValueError(f"pred.size(-1) should be {D} but got {pred.size(-1)}")
        pred_no_reg = pred[:, num_register_tokens:, :] # (B, N, D)
        
        if gt_pos.ndim != 2 or gt_pos.size(1) != pred_no_reg.size(1):
            raise ValueError(f"gt_pos should be of shape (B, N) but got {gt_pos.shape}")
        
        result = torch.empty(
            (B, self._grid_size[0] * self._grid_size[1], D)
        ).to(pred_no_reg)

        num_mask = gt_pos.sum(dim=1)
        assert num_mask.min() == num_mask.max(), "All images in the batch should have the same number of masked tokens for visualization"
        num_mask = num_mask[0].item()
        result[gt_pos] = pred_no_reg[:, :num_mask].reshape(-1, D)
        result[~gt_pos] = pred_no_reg[:, num_mask:].reshape(-1, D)
        result_img = self.unpatchify(result) 
        
        return result_img
        