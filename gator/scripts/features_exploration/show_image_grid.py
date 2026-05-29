"""
python3 gator/scripts/features_exploration/show_image_grid.py \
    --input_folder /scratch/izar/skorokho/gator/attn-analysis/ \
    --output_dir /scratch/izar/skorokho/gator/attn-analysis/encoder/ \
    --image_indices 0 5 12 \
    --reference
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import tyro
import numpy as np
import torch.nn.functional as F

from gator.scripts.gator_multiview_usage.visualization_helpers import draw_patch_grid


@dataclass(kw_only=True)
class Args:
    input_folder: Path
    output_dir: Path
    image_indices: list[int] = 0
    reference: bool = True

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)

def main(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = torch.load(args.input_folder / "input_batch.pth", map_location=device)
    for image_index in args.image_indices:
        image_shu_idx = image_index * 2 + (1 if args.reference else 0)


        image_shu = batch[image_shu_idx]\
            .permute(1, 2, 0)\
            .multiply_(255)\
            .clip(0, 255)\
            .detach()\
            .cpu()\
            .numpy()\
            .astype(np.uint8)

        image_shu = cv2.cvtColor(image_shu, cv2.COLOR_RGB2BGR)
        image_shu = draw_patch_grid(
            image_shu, 
            line_color=(0, 127, 0),
            text_color=(0, 127, 0),
        )
        cv2.imwrite(
            str(
                args.output_dir /\
                f"{'ref' if args.reference else 'nref'}_grid_{image_index:03}.png"), 
            image_shu
        )

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
