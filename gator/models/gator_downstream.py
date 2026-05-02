"""Gator model adapted for binocular downstream tasks (stereo/optical flow)."""

import numpy as np
import torch
import torch.nn as nn

from gator.models.model_gator import GatorConfig
from gator.models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
from gator.models.blocks import PatchEmbed, Block, DecoderBlock


class GatorDownstreamBinocular(nn.Module):
    """
    Gator encoder+decoder for binocular downstream tasks (stereo/flow).

    Compatible with CroCo's PixelwiseTaskWithDPT head via these exposed attributes:
        enc_depth, dec_depth, enc_embed_dim, dec_embed_dim, dec_blocks

    Key differences from pretraining Gator:
    - Both images encoded without shuffling (RoPE applied to both)
    - img1 is the decoder query, img2 is the decoder context
    - Register tokens are stripped from intermediate features passed to the DPT head
    - Image normalization is NOT applied here (datasets normalize before the model)
    - PatchEmbed is rebuilt for the target crop size
    """

    def __init__(self, head, config: GatorConfig, img_size=(224, 224)):
        super().__init__()

        # CroCo-compatible attribute names required by PixelwiseTaskWithDPT.setup()
        self.enc_depth = config.enc_depth
        self.dec_depth = config.dec_depth
        self.enc_embed_dim = config.enc_emb_dim
        self.dec_embed_dim = config.dec_emb_dim
        self.dec_blocks = True  # tells the head that an encoder+decoder architecture exists
        self.num_register_tokens = config.num_register_tokens
        self._config = config

        # Patch embedding rebuilt for the target crop size
        # Conv2d weights are identical to pretrained; only img_size assertion changes.
        self._patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.enc_emb_dim,
            norm_layer=None,
            flatten=True,
        )

        self._no_pos_emb = nn.Parameter(torch.randn(1, 1, config.enc_emb_dim))
        self._register_tokens = nn.Parameter(
            torch.randn(1, config.num_register_tokens, config.enc_emb_dim)
        )

        if RoPE2D is None:
            raise ImportError("RoPE2D is not available; check installation")
        self._rope = RoPE2D(freq=config.rope_freq)

        self._encoder_blocks = nn.ModuleList([
            Block(
                dim=config.enc_emb_dim,
                num_heads=config.enc_num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                fused_attn=config.fused_attn,
                rope=self._rope,
            ) for _ in range(config.enc_depth)
        ])
        self._enc_norm = nn.LayerNorm(config.enc_emb_dim)

        self._decoder_embed = nn.Linear(config.enc_emb_dim, config.dec_emb_dim)
        self._decoder_blocks = nn.ModuleList([
            DecoderBlock(
                dim=config.dec_emb_dim,
                num_heads=config.dec_num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                rope=None,
                fused_attn=config.fused_attn,
            ) for _ in range(config.dec_depth)
        ])
        self._decoder_norm = nn.LayerNorm(config.dec_emb_dim)

        # Sincos positional embedding for decoder context (img2), recomputed for target size.
        # Not loaded from checkpoint to avoid shape mismatch when crop != pretraining size.
        grid_h = img_size[0] // config.patch_size
        grid_w = img_size[1] // config.patch_size
        dec_pos_embed = get_2d_sincos_pos_embed(
            config.dec_emb_dim, (grid_h, grid_w), n_cls_token=config.num_register_tokens
        )
        self.register_buffer('_dec_pos_embed', torch.from_numpy(dec_pos_embed).float())

        # Setup and attach the downstream prediction head
        head.setup(self)
        self.head = head

    def _encode_image(self, img, return_all_blocks=False):
        """Encode an image with RoPE positional encoding (no shuffling).

        Args:
            return_all_blocks: if True, also return per-layer patch features for the DPT head.

        Returns:
            return_all_blocks=False:
                full_enc  (B, num_reg+N, enc_emb_dim)  — used as decoder input
            return_all_blocks=True:
                (patch_layers, full_enc)
                patch_layers  list[enc_depth] of (B, N, enc_emb_dim)  — registers stripped
                full_enc      (B, num_reg+N, enc_emb_dim)              — used as decoder input
        """
        x, pos = self._patch_embed(img)  # (B, N, D), (B, N, 2)
        B = x.size(0)

        # Prepend register tokens (same as pretraining)
        reg = self._register_tokens.expand(B, -1, -1)
        x = torch.cat([reg, x], dim=1)
        # Shift patch positions by 1 so register slots can use position 0
        pos = pos + 1
        pos = torch.cat([
            torch.zeros((B, self.num_register_tokens, 2), dtype=torch.long, device=pos.device),
            pos,
        ], dim=1)

        if return_all_blocks:
            patch_layers = []
            for block in self._encoder_blocks:
                x = block(x, xpos=pos, use_rope=True)
                patch_layers.append(x[:, self.num_register_tokens:, :])
            x = self._enc_norm(x)
            patch_layers[-1] = x[:, self.num_register_tokens:, :]
            return patch_layers, x
        else:
            for block in self._encoder_blocks:
                x = block(x, xpos=pos, use_rope=True)
            x = self._enc_norm(x)
            return x

    def _decode(self, x1_full, x2_full, return_all_blocks=False):
        """Decode with img1 as query and img2 as context.

        Args:
            x1_full: (B, num_reg+N, enc_emb_dim)
            x2_full: (B, num_reg+N, enc_emb_dim)
            return_all_blocks: if True, return per-layer patch features for the DPT head.

        Returns:
            return_all_blocks=False:
                (B, N, dec_emb_dim)  — registers stripped
            return_all_blocks=True:
                list[dec_depth] of (B, N, dec_emb_dim)  — registers stripped
        """
        x1 = self._decoder_embed(x1_full)  # (B, num_reg+N, dec_emb_dim)
        x2 = self._decoder_embed(x2_full)  # (B, num_reg+N, dec_emb_dim)

        # Add sincos positional embedding to the context (img2)
        x2 = x2 + self._dec_pos_embed[None, ...]

        if return_all_blocks:
            dec_layers = []
            for block in self._decoder_blocks:
                x1, x2 = block(x1, x2, xpos=None, ypos=None,
                                use_sa_rope=False, use_ca_rope=False)
                dec_layers.append(x1[:, self.num_register_tokens:, :])
            x1 = self._decoder_norm(x1)
            dec_layers[-1] = x1[:, self.num_register_tokens:, :]
            return dec_layers
        else:
            for block in self._decoder_blocks:
                x1, x2 = block(x1, x2, xpos=None, ypos=None,
                                use_sa_rope=False, use_ca_rope=False)
            x1 = self._decoder_norm(x1)
            return x1[:, self.num_register_tokens:, :]

    def forward(self, img1, img2):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks

        if return_all_blocks:
            enc1_layers, enc1 = self._encode_image(img1, return_all_blocks=True)
            enc2 = self._encode_image(img2)
            dec_layers = self._decode(enc1, enc2, return_all_blocks=True)
            # Combined list: enc_layers (enc_depth) + dec_layers (dec_depth)
            # Matches the hook-index scheme of PixelwiseTaskWithDPT
            features = enc1_layers + dec_layers
        else:
            enc1 = self._encode_image(img1)
            enc2 = self._encode_image(img2)
            features = self._decode(enc1, enc2)

        return self.head(features, img_info)


def load_gator_state_dict(ckpt):
    """Extract Gator weights from a Lightning checkpoint.

    Lightning saves GatorWrapper parameters under the '_model.' prefix.
    We strip that prefix, drop the pretraining prediction head and the
    decoder positional embedding (recomputed for the new crop size).
    """
    state_dict = ckpt['state_dict']
    # Keep only Gator backbone weights
    state_dict = {
        k[len('_model.'):]: v
        for k, v in state_dict.items()
        if k.startswith('_model.')
    }
    # Drop keys that are incompatible with the downstream model:
    #   _dec_pos_embed  — recomputed analytically for the new image size
    #   _decoder_pred.* — pretraining prediction head, not used downstream
    state_dict = {
        k: v for k, v in state_dict.items()
        if k != '_dec_pos_embed' and not k.startswith('_decoder_pred')
    }
    return state_dict
