from typing import ClassVar

import torch
import torch.nn.functional as F
import torch.nn as nn

from gator.models.blocks import PatchEmbed, Block, DecoderBlock
from gator.models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
from gator.models.gator_2view.model_gator import GatorConfig, _IMAGE_MEAN, _IMAGE_STD

from dataclasses import dataclass

@dataclass
class MAEConfig(GatorConfig):
    """
    config for MAE 1-View Model
    """

    # turn them off
    shuffle_ratio: ClassVar[None] = None
    predict_position: ClassVar[None] = None

    mask_ratio: tuple[float, float] = (0.75, 0.75)


class MAEModel(nn.Module):
    def __init__(self, config: MAEConfig) -> None:
        super().__init__()

        self._config = config

        self._masked_token = nn.Parameter(
            torch.randn(1, 1, self._config.dec_emb_dim)
        )
        self._register_tokens = nn.Parameter(
            torch.randn(1, self._config.num_register_tokens, self._config.enc_emb_dim)
        )
        
        self._patch_embed = PatchEmbed(
            img_size=self._config.image_size,
            patch_size=self._config.patch_size,
            in_chans=3,
            embed_dim=self._config.enc_emb_dim,
            norm_layer=None,
            flatten=True,
        )
        
        # relative positional embedding for encoder
        if RoPE2D is None: 
            raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        self._rope = RoPE2D(freq=self._config.rope_freq)

        self._encoder_blocks = nn.ModuleList([
            Block(
                dim=self._config.enc_emb_dim,
                num_heads=self._config.enc_num_heads,
                mlp_ratio=self._config.mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                fused_attn=self._config.fused_attn,
                rope=self._rope,
            ) for _ in range(self._config.enc_depth)
        ])
        self._enc_norm = nn.LayerNorm(self._config.enc_emb_dim)
        self._decoder_embed = nn.Linear(self._config.enc_emb_dim, self._config.dec_emb_dim)

        self._decoder_blocks = nn.ModuleList([
            Block(
                dim=self._config.dec_emb_dim,
                num_heads=self._config.dec_num_heads,
                mlp_ratio=self._config.mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                rope=self._rope,
                fused_attn=self._config.fused_attn,
            ) for _ in range(self._config.dec_depth)
        ])

        self._decoder_norm = nn.LayerNorm(self._config.dec_emb_dim)
        self._decoder_pred = nn.Linear(
            self._config.dec_emb_dim, 
            self._config.patch_size**2 * 3,
        )

        # register normalization
        for name, value in (("_image_mean", _IMAGE_MEAN), ("_image_std", _IMAGE_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def _forward_encoder(
            self, 
            img: torch.Tensor, 
            mask: bool, 
            mask_ratio: float | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x, pos = self._patch_embed(img) # (B, N, D), (B, N, 2)

        B, N, D = x.shape
        pos_masked: torch.Tensor | None = None
        mask_taken: torch.Tensor | None = None

        if mask:
            # select random mask at each image
            rand_item = torch.rand(1).item()
            if mask_ratio is None:
                rand_ratio = self._config.mask_ratio[0] + (self._config.mask_ratio[1] - self._config.mask_ratio[0]) * rand_item
            else:
                rand_ratio = mask_ratio
            keep_ratio = 1.0 - rand_ratio

            N1 = int(N * keep_ratio)

            random_positions = torch.rand(B, N, device=x.device).argsort(dim=1) # (B, N)
            mask_taken = random_positions < N1

            pos_masked = pos[~mask_taken].view(B, N - N1, 2) # (B, N - N1, 2)

            x = x[mask_taken].view(B, N1, D) # (B, N1, D)
            pos = pos[mask_taken].view(B, N1, 2) # (B, N1, 2)

        register_tokens = self._register_tokens.expand(B, -1, -1) # (B, num_register_tokens, D)
        x = torch.cat([register_tokens, x], dim=1)
        pos = pos + 1
        pos = torch.cat([
            torch.zeros((B, self._config.num_register_tokens, 2), dtype=torch.long, device=pos.device), # no pos for register tokens
            pos,
        ], dim=1)
        if pos_masked is not None:
            pos_masked = pos_masked + 1
        
        for block in self._encoder_blocks:
            x = block(x, xpos=pos, use_rope=True)
        x = self._enc_norm(x)

        return x, pos, pos_masked, mask_taken

    def _forward_decoder(
            self, 
            x: torch.Tensor, 
            x_pos: torch.Tensor,
            masked_pos: torch.Tensor,
        ) -> torch.Tensor:
        """
        x: (B, N1, C)
        x_pos: (B, N1, 2)
        masked_pos: (B, N - N1, 2)
        """

        decoder_input = self._masked_token.expand(
            x.shape[0],
            masked_pos.shape[1],
            -1,
        )
        # register tokens, x, masked tokens
        x = torch.cat([x, decoder_input], dim=1) # (B, N, C)
        pos = torch.cat([x_pos, masked_pos], dim=1) # (B, N, 2)

        for block in self._decoder_blocks:
            x = block(x, xpos=pos, use_rope=True)
        x = self._decoder_norm(x)
        return x

    def forward(
            self, 
            img1: torch.Tensor, 
            img2: torch.Tensor | None, 
            mask_ratio: float | None = None
        ):
        """
        We would like to keep the same interfact for both 1-view and 2-view
        models, so we keep img2 as an optional input. When img2 is provided we
        simply concatenate them in a bigger batch.

        img1: (B, C, H, W), not normalized
        img2: (B, C, H, W), not normalized


        :returns:
        out: (B, N1, num_patches), the predicted patches
        mask_taken: (B, N), the mask indicating which tokens were taken
        num_register_tokens: int, the number of register tokens used

        **important** When calculating the loss do not forget to remove the register tokens
        """

        img = img1 if img2 is None else torch.cat([img1, img2], dim=0)
        img = (img - self._image_mean) / self._image_std

        img_enc, pos, pos_masked, mask_taken = self._forward_encoder(img, mask=True, mask_ratio=mask_ratio)
        img_to_dec = self._decoder_embed(img_enc)

        out_dec = self._forward_decoder(img_to_dec, pos, pos_masked)
        out = self._decoder_pred(out_dec)

        return out, mask_taken, self._config.num_register_tokens

