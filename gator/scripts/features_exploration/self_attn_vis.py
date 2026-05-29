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
    reference: bool = True
    patches_positions: list[tuple[int, int]]
    attention_file: str
    num_registers: int = 4

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.vis_image_idx = self.image_index * 2 + (1 if self.reference else 0)
        print(self.patches_positions)


def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = torch.load(args.input_folder / "input_batch.pth", map_location=device)
    image_orig = batch[args.vis_image_idx]\
        .permute(1, 2, 0)\
        .multiply_(255)\
        .clip(0, 255)\
        .detach()\
        .cpu()\
        .numpy()\
        .astype(np.uint8)

    # (B, num_heads, seq_len, head_dim)
    q, k, _ = torch.load(args.input_folder / args.attention_file, map_location=device)

    # (num_heads, seq_len, seq_len)
    qk = torch.matmul(
        q[args.image_index, :, args.num_registers:, :], 
        k[args.image_index, :, args.num_registers:, :].transpose(-2, -1)
    )
    qk = qk / q.shape[-1] ** 0.5
    qk = F.softmax(qk, dim=-1)
    qk = qk.mean(dim=0) # average over heads

    # visualize
    for patch_pos in args.patches_positions:
        image_copy = image_orig.copy()
        image_copy = draw_rectangle(
            image_copy,
            (patch_pos[0], patch_pos[1], 1, 1),
            line_color=(255, 0, 0)
        )

        qk_curr = qk[patch_pos[0] * 14 + patch_pos[1]]
        qk_curr = qk_curr.reshape(14, 14)
        qk_curr = F.interpolate(qk_curr[None, None], size=image_copy.shape[:2], mode="nearest")[0]
        qk_curr = qk_curr.expand(3, -1, -1).permute(1, 2, 0).cpu().numpy()
        qk_curr = qk_curr / qk_curr.max() * 255
        qk_curr = qk_curr.astype(np.uint8)

        qk_curr = draw_rectangle(
            qk_curr,
            (patch_pos[0], patch_pos[1], 1, 1),
            line_color=(255, 0, 0)
        )

        image_comb = np.concatenate([
            image_copy, 
            qk_curr
        ], axis=1)
        image_comb = cv2.cvtColor(image_comb, cv2.COLOR_RGB2BGR)
        # save
        name = "ref_image_" if args.reference else "noref_image_"
        name += f"{args.image_index}_patch_{patch_pos[0]}_{patch_pos[1]}"
        cv2.imwrite(
            str(args.output_dir / f"{name}.png"), 
            image_comb
        )

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
