from pathlib import Path
import torch
import torch.nn as nn
from gator.models.blocks import DecoderBlock
from gator.models.jigsaw_1view.model_jigsaw import Jigsaw1View
from gator.models.gator_2view.model_gator import GatorConfig, Gator
from gator.models.pos_embed import get_2d_sincos_pos_embed
from gator.relpose.models.gator_relpose import GatorRelpose
from gator.relpose.utils.misc import transpose_to_landscape
from gator.relpose.models.pose_head import PoseHead


class JigsawRelpose(nn.Module):
    def __init__(
            self, 
            config: GatorConfig, # TODO: fix later?
            jigsaw_ckpt_path: Path,
            freeze: bool = True,
        ) -> None:
        super().__init__()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = config

        # prepare inner jigsaw
        self._jigsaw = Jigsaw1View(config)
        self._jigsaw.to(self._device)

        jigsaw_state_dict = GatorRelpose._get_encdec_state_dict(
            jigsaw_ckpt_path, self._device,
        )
        self._jigsaw.load_state_dict(jigsaw_state_dict, strict=True)

        if freeze:
            for param in self._jigsaw.parameters():
                param.requires_grad = False
            self._jigsaw.eval()

        # build a decoder
        dec_pos_embed = get_2d_sincos_pos_embed(
            self._config.dec_emb_dim,
            self._jigsaw._patch_embed.grid_size,
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

        # build a head
        self._jigsaw.dec_embed_dim = config.dec_emb_dim
        self._jigsaw.patch_embed = self._jigsaw._patch_embed

        self.pose_head = PoseHead(net=self._jigsaw)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

    def forward(self, view1: dict[str, torch.Tensor], view2: dict[torch.Tensor]):
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

        img1 = (view1["img"] - self._jigsaw._image_mean) / self._jigsaw._image_std
        img2 = (view2["img"] - self._jigsaw._image_mean) / self._jigsaw._image_std

        B, C, H, W = img1.shape

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        img1_enc, _, _ = self._jigsaw._forward_encoder(img1, shuffle=False)
        img2_enc, _, _ = self._jigsaw._forward_encoder(img2, shuffle=False)

        img1_to_dec = self._decoder_embed(img1_enc)
        img2_to_dec = self._decoder_embed(img2_enc)

        # very bad way to do so, but I am lazy to refactor...
        # it is also bad to use _forward_decoder of Gator, but 
        # the fields used are literally the same
        out12 = Gator._forward_decoder(
            self,
            img1_to_dec + self._dec_pos_embed[None, ...],
            img2_to_dec
        )
        out12_noreg = out12[:, self._jigsaw._config.num_register_tokens:, :] # [B, N1, D]

        out21 = Gator._forward_decoder(
            self,
            img2_to_dec + self._dec_pos_embed[None, ...],
            img1_to_dec
        )
        out21_noreg = out21[:, self._jigsaw._config.num_register_tokens:, :] # [B, N1, D]

        with torch.amp.autocast("cuda", enabled=False):
            # I think it should be relative pose 1 -> 2, since if we look at the
            # original implementation in reloc3r/reloc3r/reloc3r_relpose.py
            # (functions `_decoder`, `forward`, and `inference_relpose`)
            pose12 = self.head([out12_noreg], shape1)
            pose21 = self.head([out21_noreg], shape2)

        return pose12, pose21

