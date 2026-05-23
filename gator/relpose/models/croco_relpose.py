from pathlib import Path

import torch
import torch.nn as nn

from gator.models.croco import CroCoNet
from gator.models.jigsaw_1view.model_jigsaw import _IMAGE_MEAN, _IMAGE_STD
from gator.relpose.models.gator_relpose import GatorRelpose
from gator.relpose.models.pose_head import PoseHead
from gator.relpose.utils.misc import transpose_to_landscape


class CroCoNetRelpose(nn.Module):
    def __init__(
        self, 
        croco_ckpt_path: Path,
        freeze: bool = True,
        *args, 
        **kwargs
    ):

        super().__init__()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare inner croco
        self._croco = CroCoNet(*args, **kwargs)
        self._croco.to(self._device)

        croco_state_dict = GatorRelpose._get_encdec_state_dict(
            croco_ckpt_path, self._device,
        )
        self._croco.load_state_dict(croco_state_dict, strict=True)

        if freeze:
            for param in self._croco.parameters():
                param.requires_grad = False
            self._croco.eval()

        self.pose_head = PoseHead(net=self._croco)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

        # build mean and std
        for name, value in (("_image_mean", _IMAGE_MEAN), ("_image_std", _IMAGE_STD)):
            self.register_buffer(
                name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False
            )

    def forward(self, view1: dict[str, torch.Tensor], view2: dict[torch.Tensor]):
        img1 = (view1["img"] - self._image_mean) / self._image_std
        img2 = (view2["img"] - self._image_mean) / self._image_std

        B, C, H, W = img1.shape

        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        img1_enc, pos1, _ = self._croco._encode_image(img1, do_mask=False)
        img2_enc, pos2, _ = self._croco._encode_image(img2, do_mask=False)

        out12 = self._croco._decoder(img1_enc, pos1, None, img2_enc, pos2)
        out21 = self._croco._decoder(img2_enc, pos2, None, img1_enc, pos1)

        with torch.amp.autocast("cuda", enabled=False):
            # I think it should be relative pose 1 -> 2, since if we look at the
            # original implementation in reloc3r/reloc3r/reloc3r_relpose.py
            # (functions `_decoder`, `forward`, and `inference_relpose`)
            pose12 = self.head([out12], shape1)
            pose21 = self.head([out21], shape2)

        return pose12, pose21



