"""
python3 gator/scripts/features_exploration/cross_attn_vis2.py \
    --input_folder /scratch/izar/skorokho/gator/attn-analysis/ \
    --output_dir /scratch/izar/skorokho/gator/attn-analysis/decoder/ \
    --image_index 0 --patches_positions 0 0 5 5 12 0 0 12 \
    --attention_file /scratch/izar/skorokho/gator/attn-analysis/_decoder_blocks.3.cross_attn-000.pth 
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import tyro
import numpy as np
import torch.nn.functional as F

from gator.scripts.gator_multiview_usage.visualization_helpers import draw_rectangle


@dataclass(kw_only=True)
class Args:
    input_folder: Path
    output_dir: Path
    image_index: int = 0
    patches_positions: list[tuple[int, int]]
    attention_file: str
    num_registers: int = 4

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(self.patches_positions)

def draw_connecting_line(
        image, 
        p_y1, p_x1, p_y2, p_x2, 
        patch_size=16, line_color=(0, 255, 0), line_thickness=1
    ):
    """
    p_y1, p_x1 - patch idx 1
    p_y2, p_x2 - patch idx 2
    """

    original_width = image.shape[1] / 2
    c_y1 = int(p_y1 * patch_size + patch_size / 2)
    c_x1 = int(p_x1 * patch_size + patch_size / 2)
    c_y2 = int(p_y2 * patch_size + patch_size / 2)
    c_x2 = int(p_x2 * patch_size + patch_size / 2 + original_width)

    cv2.line(image, (c_x1, c_y1), (c_x2, c_y2), line_color, line_thickness)
    return image

def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = torch.load(args.input_folder / "input_batch.pth", map_location=device)

    image_shu_idx = 2 * args.image_index
    image_ref_idx = 2 * args.image_index + 1

    image_shu = batch[image_shu_idx]\
        .permute(1, 2, 0)\
        .multiply_(255)\
        .clip(0, 255)\
        .detach()\
        .cpu()\
        .numpy()\
        .astype(np.uint8)

    image_ref = batch[image_ref_idx]\
        .permute(1, 2, 0)\
        .multiply_(255)\
        .clip(0, 255)\
        .detach()\
        .cpu()\
        .numpy()\
        .astype(np.uint8)
    
    image_shuref = np.concatenate([image_shu, image_ref], axis=1)

    # (B, num_heads, seq_len, head_dim)
    q, k, _ = torch.load(args.input_folder / args.attention_file, map_location=device)

    # (num_heads, seq_len_shu, seq_len_ref)
    qk = torch.matmul(
        q[args.image_index, :, args.num_registers:, :], 
        k[args.image_index, :, args.num_registers:, :].transpose(-2, -1)
    )
    qk = qk / q.shape[-1] ** 0.5
    qk = F.softmax(qk, dim=-1)
    qk = qk.mean(dim=0) # average over heads

    # visualize
    for patch_pos in args.patches_positions:
        image_shu_copy = image_shu.copy()
        image_ref_copy = image_ref.copy()

        image_shuref = np.concatenate([image_shu_copy, image_ref_copy], axis=1)
        image_shu_copy = draw_rectangle(
            image_shuref,
            (patch_pos[0], patch_pos[1], 1, 1),
            line_color=(0, 255, 0)
        )

        qk_curr = qk[patch_pos[0] * 14 + patch_pos[1]]
        qk_curr_argmax = qk_curr.argmax().detach().cpu().tolist()
        max_idx = (qk_curr_argmax // 14, qk_curr_argmax % 14)
        image_shuref = draw_rectangle(
            image_shuref,
            (max_idx[0], max_idx[1] + 14, 1, 1),
            line_color=(0, 255, 0)
        )

        image_shuref = draw_connecting_line(
            image_shuref, 
            patch_pos[0], patch_pos[1], 
            max_idx[0], max_idx[1]
        )

        image_shuref = cv2.cvtColor(image_shuref, cv2.COLOR_RGB2BGR)
        # save
        name = "image_"
        name += f"{args.image_index}_patchv2_{patch_pos[0]}_{patch_pos[1]}"
        cv2.imwrite(
            str(args.output_dir / f"{name}.png"), 
            image_shuref
        )

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
