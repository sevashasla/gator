import torch
import torch.nn as nn
from abc import abstractmethod

class GatorBaseLoss(nn.Module):
    def __init__(
            self,
            grid_size: tuple[int, int],
            patch_size: int,
        ):
        super().__init__()

        self._grid_size = grid_size
        self._num_patches = grid_size[0] * grid_size[1]
        self._patch_size = patch_size

    @abstractmethod
    def forward(
        self, 
        pred: torch.Tensor, 
        gt_pos: torch.Tensor, 
        gt_image: torch.Tensor,
        num_register_tokens: int,
    ):
        raise NotImplementedError(
            "Forward method should be implemented by subclasses of GatorBaseLoss"
        )

