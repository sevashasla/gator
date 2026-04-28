from pathlib import Path

import tyro
from dataclasses import dataclass
import webdataset as wds
from gator.datasets.pairs_dataset import dnames_to_image_pairs
from typing import Literal
from gator import logger

@dataclass(kw_only=True)
class CreateShardsArguments:
    image_pairs_file: Path = Path("/scratch/izar/skorokho/croco-dataset/habitat_release/pairs.txt")
    """Path to the text file containing the list of image pairs after running pairs_dataset."""
    output_dir: Path = Path("/scratch/izar/skorokho/croco-dataset/habitat_release_shards/")
    """Directory where the output shards will be saved."""
    shard_size: int = 10_000
    """Number of samples per shard."""
    dnames: Literal['habitat_release'] = 'habitat_release'

def main(args: CreateShardsArguments):
    pattern = f"{str(args.output_dir)}/train-%06d.tar"
    args.output_dir.mkdir(exist_ok=True, parents=True)

    image_pairs = dnames_to_image_pairs(
        args.dnames,
        data_dir=str(args.image_pairs_file.parent.parent)
    )
    logger.info(f"Found {len(image_pairs):,} image pairs for dataset {args.dnames}.")

    with wds.ShardWriter(pattern, maxcount=args.shard_size) as sink:
        for idx, (im1path, im2path) in enumerate(image_pairs):
            with open(im1path, "rb") as f:
                im1_bytes = f.read()
            with open(im2path, "rb") as f:
                im2_bytes = f.read()

            sample = {
                "__key__": f"{idx:08d}",
                "im1.jpg": im1_bytes,
                "im2.jpg": im2_bytes,
            }

            sink.write(sample)
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1:,} / {len(image_pairs):,} image pairs.")

if __name__ == "__main__":
    args = tyro.cli(CreateShardsArguments)
    main(args)
