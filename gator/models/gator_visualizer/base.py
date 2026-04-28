import torch
import torch.nn as nn
from abc import abstractmethod

class GatorBaseVis(nn.Module):
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

    @abstractmethod
    def forward(
        self, 
        pred: torch.Tensor, 
        gt_pos: torch.Tensor, 
        gt_image: torch.Tensor,
        num_register_tokens: int,
    ):
        raise NotImplementedError(
            "Forward method should be implemented by subclasses of GatorBaseVis"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

