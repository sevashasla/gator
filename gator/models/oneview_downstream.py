"""Binocular downstream wrapper for 1-view pretrained encoders (MAE, Jigsaw1View).

Both images are encoded independently through the same 1-view encoder.
The concatenated feature lists (enc1_layers + enc2_layers) are passed directly
to the DPT head — no cross-attention between the two views.

This serves as a baseline to compare 1-view vs 2-view pretraining on
binocular downstream tasks (stereo / optical flow).

Model-specific encoder behaviour
---------------------------------
MAE  (use_rope=True):
    Pretrained with RoPE enabled and actual patch positions.
    No _no_pos_emb — pass use_rope=True (default).

Jigsaw1View  (use_rope=False):
    Pretrained with RoPE *disabled* (patches are shuffled so positions are
    zeroed out) and a learnable _no_pos_emb bias added to every patch token.
    Pass use_rope=False to replicate the pretraining input distribution.
"""

import torch
import torch.nn as nn

from gator.models.blocks import PatchEmbed, Block
from gator.models.pos_embed import RoPE2D


class OneViewDownstreamBinocular(nn.Module):
    """
    Encoder-only binocular downstream for 1-view pretrained models.

    Compatible with CroCo's PixelwiseTaskWithDPT head via these exposed attributes:
        enc_depth, dec_depth, enc_embed_dim, dec_embed_dim, dec_blocks
    """

    def __init__(self, head, config, img_size=(224, 224), use_rope=True):
        super().__init__()

        self.enc_depth = config.enc_depth
        self.dec_depth = config.enc_depth  # img2 encoder layers fill the "decoder" slots
        self.enc_embed_dim = config.enc_emb_dim
        self.dec_embed_dim = config.enc_emb_dim
        self.dec_blocks = True
        self.num_register_tokens = config.num_register_tokens
        self._config = config
        self._use_rope = use_rope

        self._patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.enc_emb_dim,
            norm_layer=None,
            flatten=True,
        )

        self._register_tokens = nn.Parameter(
            torch.randn(1, config.num_register_tokens, config.enc_emb_dim)
        )

        # Jigsaw1View adds _no_pos_emb to every patch token during pretraining.
        # Recreate the parameter so its pretrained value can be loaded and the
        # input distribution seen by the encoder matches pretraining.
        if not use_rope:
            self._no_pos_emb = nn.Parameter(torch.zeros(1, 1, config.enc_emb_dim))

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

        head.setup(self)
        self.head = head

    def _encode_image(self, img):
        """Encode one image, returning per-layer patch features (registers stripped).

        Returns:
            list of enc_depth tensors, each (B, N, enc_emb_dim)
        """
        x, pos = self._patch_embed(img)  # (B, N, D), (B, N, 2)
        B = x.size(0)

        # Jigsaw1View adds _no_pos_emb to patch tokens before encoding.
        if hasattr(self, '_no_pos_emb'):
            x = x + self._no_pos_emb

        reg = self._register_tokens.expand(B, -1, -1)
        x = torch.cat([reg, x], dim=1)
        pos = pos + 1
        pos = torch.cat([
            torch.zeros((B, self.num_register_tokens, 2), dtype=torch.long, device=pos.device),
            pos,
        ], dim=1)

        layers = []
        for block in self._encoder_blocks:
            x = block(x, xpos=pos, use_rope=self._use_rope)
            layers.append(x[:, self.num_register_tokens:, :])
        x = self._enc_norm(x)
        layers[-1] = x[:, self.num_register_tokens:, :]
        return layers

    def forward(self, img1, img2):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}

        enc1_layers = self._encode_image(img1)  # enc_depth tensors from img1
        enc2_layers = self._encode_image(img2)  # enc_depth tensors from img2
        features = enc1_layers + enc2_layers    # 2*enc_depth total

        return self.head(features, img_info)


def load_oneview_state_dict(ckpt):
    """Extract 1-view encoder weights from a Lightning checkpoint.

    Strips the '_model.' prefix and drops pretraining-only components
    that are not part of the downstream encoder:
      - MAE: _masked_token, _decoder_embed, _decoder_blocks, _decoder_norm, _decoder_pred
      - Jigsaw: _final_layer
      Note: _no_pos_emb is kept — it is loaded into the downstream model for Jigsaw.
    """
    state_dict = ckpt['state_dict']
    state_dict = {
        k[len('_model.'):]: v
        for k, v in state_dict.items()
        if k.startswith('_model.')
    }
    drop_prefixes = (
        '_decoder_pred', '_decoder_embed', '_decoder_blocks', '_decoder_norm',
        '_masked_token', '_final_layer',
    )
    state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(p) for p in drop_prefixes)
    }
    return state_dict
