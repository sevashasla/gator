from pathlib import Path

import torch
import torch.nn as nn
from gator.models.blocks import DecoderBlock
from gator.models.mae_1view.model_mae import MAEConfig, MAEModel
from gator.models.gator_2view.model_gator import Gator
from gator.models.pos_embed import get_2d_sincos_pos_embed
from gator.relpose.models.gator_relpose import GatorRelpose
from gator.relpose.utils.misc import transpose_to_landscape
from gator.relpose.models.pose_head import PoseHead


class MAERelpose(nn.Module):
    def __init__(
            self,
            config: MAEConfig,
            mae_ckpt_path: Path,
            freeze: bool = True,
        ) -> None:
        super().__init__()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = config

        self._mae = MAEModel(config)
        self._mae.to(self._device)

        mae_state_dict = GatorRelpose._get_encdec_state_dict(
            mae_ckpt_path, self._device,
        )
        self._mae.load_state_dict(mae_state_dict, strict=True)

        if freeze:
            for param in self._mae.parameters():
                param.requires_grad = False

        # build a cross-attention decoder
        dec_pos_embed = get_2d_sincos_pos_embed(
            self._config.dec_emb_dim,
            self._mae._patch_embed.grid_size,
            n_cls_token=self._config.num_register_tokens,
        )
        self.register_buffer("_dec_pos_embed", torch.from_numpy(dec_pos_embed).float())
        self._decoder_embed = nn.Linear(
            self._config.enc_emb_dim, self._config.dec_emb_dim
        )
        self._decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self._config.dec_emb_dim,
                    num_heads=self._config.dec_num_heads,
                    mlp_ratio=self._config.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    rope=None,
                    fused_attn=self._config.fused_attn,
                )
                for _ in range(self._config.dec_depth)
            ]
        )
        self._decoder_norm = nn.LayerNorm(self._config.dec_emb_dim)

        self._mae.dec_embed_dim = config.dec_emb_dim
        self._mae.patch_embed = self._mae._patch_embed

        self.pose_head = PoseHead(net=self._mae)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

    def forward(self, view1: dict[str, torch.Tensor], view2: dict[str, torch.Tensor]):
        img1 = (view1["img"] - self._mae._image_mean) / self._mae._image_std
        img2 = (view2["img"] - self._mae._image_mean) / self._mae._image_std

        B, C, H, W = img1.shape

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        img1_enc, _, _, _ = self._mae._forward_encoder(img1, mask=False)
        img2_enc, _, _, _ = self._mae._forward_encoder(img2, mask=False)

        img1_to_dec = self._decoder_embed(img1_enc)
        img2_to_dec = self._decoder_embed(img2_enc)

        out12 = Gator._forward_decoder(
            self,
            img1_to_dec + self._dec_pos_embed[None, ...],
            img2_to_dec
        )
        out12_noreg = out12[:, self._mae._config.num_register_tokens:, :]

        out21 = Gator._forward_decoder(
            self,
            img2_to_dec + self._dec_pos_embed[None, ...],
            img1_to_dec
        )
        out21_noreg = out21[:, self._mae._config.num_register_tokens:, :]

        with torch.amp.autocast("cuda", enabled=False):
            pose12 = self.head([out12_noreg], shape1)
            pose21 = self.head([out21_noreg], shape2)

        return pose12, pose21
