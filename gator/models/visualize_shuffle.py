import torch


class VisualizeShuffle:
    def __init__(
            self,
            grid_size: tuple[int, int],
            patch_size: int,
        ):
        super().__init__()

        self._grid_size = grid_size
        self._num_patches = grid_size[0] * grid_size[1]
        self._patch_size = patch_size

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self._patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x
    
    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self._patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs
    
    @torch.no_grad()
    def forward(
        self, 
        pred: torch.Tensor, 
        gt_image: torch.Tensor,
        num_register_tokens: int
    ) -> torch.Tensor:
        """
        pred: (B, n_reg+N1,N)
        gt_image: (B, C, H, W)
        num_register_tokens: int
        """
        pred_no_reg = pred[:, num_register_tokens:, :] # (B, N1, N)
        pred_indices = pred_no_reg.argmax(dim=-1) # (B, N1)

        patches = self.patchify(gt_image) # (B, N, D)
        if patches.size(1) != self._num_patches:
            raise ValueError(f"patches.size(1) should be {self._num_patches} but got {patches.size(1)}")
        _, _, D = patches.shape
        patches_pred = torch.zeros_like(patches)
        patches_pred.scatter_reduce_(
            dim=1,
            index=pred_indices[:, :, None].expand(-1, -1, D),
            src=patches,
            reduce="mean",
            include_self=False,
        )

        images_pred = self.unpatchify(patches_pred, channels=gt_image.shape[1])
        return images_pred

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)