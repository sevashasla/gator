from matplotlib.path import Path
import torch
import torch.nn as nn
from gator.models.gator_2view.model_gator import Gator, GatorConfig
from gator.relpose.utils.misc import transpose_to_landscape
from gator.relpose.models.pose_head import PoseHead


class GatorRelpose(nn.Module):
    def __init__(
            self, 
            config: GatorConfig,
            gator_ckpt_path: Path,
            freeze: bool = True,
        ) -> None:
        super().__init__()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare inner gator
        self._gator = Gator(config)
        self._gator.to(self._device)

        gator_state_dict = self._get_encdec_state_dict(
            gator_ckpt_path, self._device,
        )
        self._gator.load_state_dict(gator_state_dict, strict=True)

        if freeze:
            for param in self._gator.parameters():
                param.requires_grad = False

        # build a head
        self._gator.dec_embed_dim = config.dec_emb_dim
        self._gator.patch_embed = self._gator._patch_embed

        self.pose_head = PoseHead(net=self._gator)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

    @staticmethod
    def _get_encdec_state_dict(ckpt_path, device):
        state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
        return {
            k[len('_model.'):]: v
            for k, v in state_dict.items()
            if k.startswith('_model.')
        }

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

        img1 = (view1["img"] - self._gator._image_mean) / self._gator._image_std
        img2 = (view2["img"] - self._gator._image_mean) / self._gator._image_std

        B, C, H, W = img1.shape

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        img1_enc, _, _ = self._gator._forward_encoder(img1, shuffle=False)
        img2_enc, _, _ = self._gator._forward_encoder(img2, shuffle=False)

        img1_to_dec = self._gator._decoder_embed(img1_enc)
        img2_to_dec = self._gator._decoder_embed(img2_enc)

        # very bad way to do so, but I am lazy to refactor...
        out12 = self._gator._forward_decoder(
            img1_to_dec + self._gator._dec_pos_embed[None, ...],
            img2_to_dec
        )
        out12_noreg = out12[:, self._gator._config.num_register_tokens:, :] # [B, N1, D]

        out21 = self._gator._forward_decoder(
            img2_to_dec + self._gator._dec_pos_embed[None, ...],
            img1_to_dec
        )
        out21_noreg = out21[:, self._gator._config.num_register_tokens:, :] # [B, N1, D]

        with torch.amp.autocast("cuda", enabled=False):
            # I think it should be relative pose 1 -> 2, since if we look at the
            # original implementation in reloc3r/reloc3r/reloc3r_relpose.py
            # (functions `_decoder`, `forward`, and `inference_relpose`)
            pose12 = self.head([out12_noreg], shape1)
            pose21 = self.head([out21_noreg], shape2)

        return pose12, pose21

