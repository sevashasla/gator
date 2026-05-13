from dataclasses import dataclass

import torch
import torch.nn as nn
from gator.models.blocks import Block, PatchEmbed
from gator.models.pos_embed import RoPE2D


@dataclass
class Jigsaw1ViewConfig:
    """
    config for 1-View Gator Model
    """

    num_register_tokens: int = 4
    """
    Number of register tokens to use
    """

    patch_size: int = 16
    image_size: int = 224
    shuffle_ratio: tuple[float, float] = (0.15, 0.65)

    # https://huggingface.co/WinKawaks/vit-tiny-patch16-224/blob/main/config.json
    enc_emb_dim: int = 192
    enc_depth: int = 12
    enc_num_heads: int = 3

    mlp_ratio: float = 4.0

    fused_attn: bool = True

    rope_freq: int = 100.0

    predict_position: bool = False


_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


class Jigsaw1View(nn.Module):
    def __init__(self, config: Jigsaw1ViewConfig) -> None:
        super().__init__()

        self._config = config

        self._no_pos_emb = nn.Parameter(torch.randn(1, 1, self._config.enc_emb_dim))
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
            raise ImportError(
                "Cannot find cuRoPE2D, please install it following the README instructions"
            )
        self._rope = RoPE2D(freq=self._config.rope_freq)

        self._encoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self._config.enc_emb_dim,
                    num_heads=self._config.enc_num_heads,
                    mlp_ratio=self._config.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    fused_attn=self._config.fused_attn,
                    rope=self._rope,
                )
                for _ in range(self._config.enc_depth)
            ]
        )
        self._enc_norm = nn.LayerNorm(self._config.enc_emb_dim)

        if self._config.predict_position:
            self._final_layer = nn.Sequential(
                nn.Linear(self._config.enc_emb_dim, 2),
                nn.Tanh(),
            )
        else:
            self._final_layer = nn.Linear(
                self._config.enc_emb_dim, self._patch_embed.num_patches
            )

        # register normalization
        for name, value in (("_image_mean", _IMAGE_MEAN), ("_image_std", _IMAGE_STD)):
            self.register_buffer(
                name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False
            )

    def _forward_encoder(
        self, img: torch.Tensor, shuffle: bool, shuffle_ratio: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x, pos = self._patch_embed(img)  # (B, N, D), (B, N, 2)
        B, N, D = x.shape
        ground_truth_pos: torch.Tensor | None = None

        if shuffle:
            # select random tokens at each image
            if shuffle_ratio is None:
                rand_item = torch.rand(1).item()
                rand_ratio = (
                    self._config.shuffle_ratio[0]
                    + (self._config.shuffle_ratio[1] - self._config.shuffle_ratio[0])
                    * rand_item
                )
            else:
                rand_ratio = shuffle_ratio

            N1 = int(N * rand_ratio)

            random_positions = torch.rand(B, N).argsort(dim=1)  # (B, N)
            # random_positions = random_positions[:, :N1] # (B, N1)
            # x = torch.gather(x, dim=1, index=random_positions.unsqueeze(-1).expand(-1, -1, D)) # (B, N1, D)
            # pos = torch.gather(pos, dim=0, index=random_positions.unsqueeze(-1).expand(-1, -1, D)) # (B, N1, D)
            mask = random_positions < N1

            x = x[mask].view(B, N1, D)  # (B, N1, D)
            x = x + self._no_pos_emb  # add a learnable no_pos_emb to the input tokens
            ground_truth_pos = pos[mask].view(B, N1, 2)  # (B, N1, 2)
            # dummy pos for the encoder to behave like shuffling
            pos = torch.zeros_like(pos[mask]).view(B, N1, 2)

        register_tokens = self._register_tokens.expand(
            B, -1, -1
        )  # (B, num_register_tokens, D)
        x = torch.cat([register_tokens, x], dim=1)
        pos = pos + 1
        pos = torch.cat(
            [
                torch.zeros(
                    (B, self._config.num_register_tokens, 2),
                    dtype=torch.long,
                    device=pos.device,
                ),  # no pos for register tokens
                pos,
            ],
            dim=1,
        )

        for block in self._encoder_blocks:
            x = block(x, xpos=pos, use_rope=not shuffle)

        x = self._enc_norm(x)

        return x, pos, ground_truth_pos

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor | None,
        shuffle_ratio: float | None = None,
    ):
        """
        We would like to keep the same interfact for both 1-view and 2-view
        models, so we keep img2 as an optional input. When img2 is provided we
        simply concatenate them in a bigger batch.

        - img1: (B, C, H, W), not normalized
        - img2: (B, C, H, W) or None, not normalized


        1. do not use RoPE for encoder in img1

        3. use absolute positional embedding for img2 in decoder
        4. do not use absolute positional embedding for img1 in decoder

        :returns:
        out: (B_, N1, num_patches), the predicted positions img1_gt_pos: (B,
        num_reg_tokens+N1, 2), the ground truth positions of the shuffled tokens
        in img1 num_register_tokens: int, the number of register tokens used

        **important** When calculating the loss do not forget to remove the
        register tokens
        """

        img = img1 if img2 is None else torch.cat([img1, img2], dim=0)
        img = (img - self._image_mean) / self._image_std

        img_enc, _, img_gt_pos = self._forward_encoder(
            img, shuffle=True, shuffle_ratio=shuffle_ratio
        )

        out = self._final_layer(img_enc)
        return out, img_gt_pos, self._config.num_register_tokens
