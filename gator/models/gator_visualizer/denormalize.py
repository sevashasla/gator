import torch
from gator.models.jigsaw_1view.model_jigsaw import _IMAGE_MEAN, _IMAGE_STD

class Denormalize:
    def __init__(
        self, 
        denormalize_imagenet: bool,
        denormalize_target: bool,
    ):
        self._denormalize_imagenet = denormalize_imagenet
        self._denormalize_target = denormalize_target

    def denormalize_pred(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred must be of shape (B, N, D)
        target must be of shape (B, N, D)
        """
        if pred.ndim != 3:
            raise ValueError(f"`pred` should be of shape (B, N, D) but got {pred.shape}")
        if target.ndim != 3:
            raise ValueError(f"`target` should be of shape (B, N, D) but got {target.shape}")
        if pred.shape != target.shape:
            raise ValueError(f"`pred` and `target` should have the same shape but got {pred.shape} and {target.shape}")
        
        if self._denormalize_target:
            target_orig_mean = target.mean(dim=-1, keepdim=True)
            target_orig_var = target.var(dim=-1, keepdim=True)
            pred = pred * (target_orig_var + 1.e-6)**.5 + target_orig_mean

        return pred

    def denormalize_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Each image in `images` must be of shape (B, C, H, W)
        """

        for i, img in enumerate(images):
            if img.ndim != 4:
                raise ValueError(f"Each image should be of shape (B, C, H, W) but got {img.shape} at index {i}")

        if self._denormalize_imagenet:
            mean = torch.tensor(_IMAGE_MEAN).reshape(1, 3, 1, 1).to(images[0].device)
            std = torch.tensor(_IMAGE_STD).reshape(1, 3, 1, 1).to(images[0].device)
            images = [img * std + mean for img in images]
        return images
