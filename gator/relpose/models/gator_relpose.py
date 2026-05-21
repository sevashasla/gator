from dataclasses import dataclass

import torch
import torch.nn as nn
from gator.models.blocks import Block, DecoderBlock, PatchEmbed
from gator.models.gator_2view.model_gator import Gator, GatorConfig
from gator.models.jigsaw_1view.model_jigsaw import (
    _IMAGE_MEAN,
    _IMAGE_STD,
)
from gator.models.pos_embed import RoPE2D, get_2d_sincos_pos_embed


class GatorRelpose(Gator):
    def __init__(self, config: GatorConfig) -> None:
        super().__init__(config)
        self.dec_emb_dim = config.dec_embed_dim
        self.patch_size = config.patch_size

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        img1: (B, C, H, W), not normalized
        img2: (B, C, H, W), not normalized

        use RoPE for both encoders
        use absolute positional embedding for both images in decoder

        :returns:
        out: (B, N1, num_patches), the predicted positions
        img1_gt_pos: (B, num_reg_tokens+N1, 2), the ground truth positions of the shuffled tokens in img1
        num_register_tokens: int, the number of register tokens used

        **important** When calculating the loss do not forget to remove the register tokens
        """

        img1 = (img1 - self._image_mean) / self._image_std
        img2 = (img2 - self._image_mean) / self._image_std

        img1_enc, _, _ = self._forward_encoder(img1, shuffle=False)
        img2_enc, _, _ = self._forward_encoder(img2, shuffle=False)

        img1_to_dec = self._decoder_embed(img1_enc)
        img2_to_dec = self._decoder_embed(img2_enc)

        # very bad way to do so, but I am lazy to refactor...
        img1_to_dec = img1_to_dec + self._dec_pos_embed[None, ...]

        out_dec = self._forward_decoder(img1_to_dec, img2_to_dec)
        out = self._decoder_pred(out_dec)

        return out, self._config.num_register_tokens

