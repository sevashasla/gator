import torch
import torch.nn.functional as F
import torch.nn as nn

from gator.models.blocks import PatchEmbed, Block, DecoderBlock
from gator.models.pos_embed import get_2d_sincos_pos_embed, RoPE2D

from dataclasses import dataclass
@dataclass
class GatorConfig:
    """
    config for Gator model
    """

    num_register_tokens: int = 4
    """
    Number of register tokens to use
    """

    patch_size: int = 16
    image_size: int = 224
    shuffle_ratio: float = 0.5
    
    # https://huggingface.co/WinKawaks/vit-tiny-patch16-224/blob/main/config.json
    enc_emb_dim: int = 192
    enc_depth: int = 12
    enc_num_heads: int = 3

    dec_emb_dim: int = 192
    dec_depth: int = 8
    dec_num_heads: int = 3
    mlp_ratio: float = 4.0

    fused_attn: bool = True

    rope_freq: int = 100.0

class Gator(nn.Module):
    def __init__(self, config: GatorConfig) -> None:
        super().__init__()

        self._config = config

        self._no_pos_emb = nn.Parameter(
            torch.randn(1, 1, self._config.enc_emb_dim)
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

        # positional embedding of the decoder  
        dec_pos_embed = get_2d_sincos_pos_embed(
            self._config.dec_emb_dim, 
            self._patch_embed.grid_size, 
            n_cls_token=self._config.num_register_tokens,
        )
        self.register_buffer('_dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
        
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
            DecoderBlock(
                dim=self._config.dec_emb_dim,
                num_heads=self._config.dec_num_heads,
                mlp_ratio=self._config.mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                rope=None,
                fused_attn=self._config.fused_attn,
            ) for _ in range(self._config.dec_depth)
        ])

        self._decoder_norm = nn.LayerNorm(self._config.dec_emb_dim)
        self._decoder_pred = nn.Linear(self._config.dec_emb_dim, self._patch_embed.num_patches)

    def _forward_encoder(self, img: torch.Tensor, shuffle: bool) -> torch.Tensor:
        x, pos = self._patch_embed(img) # (B, N, D), (B, N, 2)
        B, N, D = x.shape
        ground_truth_pos: torch.Tensor | None = None

        if shuffle:
            # select random tokens at each image
            N1 = int(N * self._config.shuffle_ratio)

            random_positions = torch.rand(B, N).argsort(dim=1) # (B, N)
            # random_positions = random_positions[:, :N1] # (B, N1)
            # x = torch.gather(x, dim=1, index=random_positions.unsqueeze(-1).expand(-1, -1, D)) # (B, N1, D)
            # pos = torch.gather(pos, dim=0, index=random_positions.unsqueeze(-1).expand(-1, -1, D)) # (B, N1, D)
            mask = random_positions < N1

            x = x[mask].view(B, N1, D) # (B, N1, D)
            x = x + self._no_pos_emb # add a learnable no_pos_emb to the input tokens
            ground_truth_pos = pos[mask].view(B, N1, 2) # (B, N1, 2)
            # dummy pos for the encoder to behave like shuffling
            pos = torch.zeros_like(pos)

        register_tokens = self._register_tokens.expand(B, -1, -1) # (B, num_register_tokens, D)
        x = torch.cat([register_tokens, x], dim=1)
        pos = pos + 1
        pos = torch.cat([
            torch.zeros((B, self._config.num_register_tokens, 2), dtype=torch.long, device=pos.device), # no pos for register tokens
            pos,
        ], dim=1)
        
        for block in self._encoder_blocks:
            x = block(x, xpos=pos, use_rope=not shuffle)

        x = self._enc_norm(x)

        return x, pos, ground_truth_pos

    def _forward_decoder(
            self, 
            x: torch.Tensor, 
            context: torch.Tensor,
        ) -> torch.Tensor:
        """
        x: (B, N1, C)
        context: (B, N2, C)
        N2 >= N1
        """
        context = context + self._dec_pos_embed[None, ...]

        for block in self._decoder_blocks:
            x, context = block(x, context, xpos=None, ypos=None, use_sa_rope=False, use_ca_rope=False)
        x = self._decoder_norm(x)
        return x

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        img1: (B, C, H, W)
        img2: (B, C, H, W)

        1. use RoPE in encoder for img2
        2. do not use RoPE for encoder in img1

        3. use absolute positional embedding for img2 in decoder
        4. do not use absolute positional embedding for img1 in decoder

        :returns:
        out: (B, N1, num_patches), the predicted positions
        img1_gt_pos: (B, num_reg_tokens+N1, 2), the ground truth positions of the shuffled tokens in img1
        num_register_tokens: int, the number of register tokens used

        **important** When calculating the loss do not forget to remove the register tokens
        """

        img1_enc, img1_pos, img1_gt_pos = self._forward_encoder(img1, shuffle=True)
        img2_enc, img2_pos, _ = self._forward_encoder(img2, shuffle=False)

        img1_to_dec = self._decoder_embed(img1_enc)
        img2_to_dec = self._decoder_embed(img2_enc)

        out_dec = self._forward_decoder(img1_to_dec, img2_to_dec)
        out = self._decoder_pred(out_dec)
        return out, img1_gt_pos, self._config.num_register_tokens

