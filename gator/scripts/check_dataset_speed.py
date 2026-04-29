import tyro
import webdataset as wds
from dataclasses import dataclass
from pathlib import Path
import time
import torch
from tqdm import tqdm

from gator.datasets.shard.transforms import get_pair_transforms_gator

@dataclass(kw_only=True)
class Args:
    data_dir: Path = Path('/scratch/izar/skorokho/croco-dataset/')
    dataset: str = 'habitat_release_shards'
    transforms: str = 'crop224+acolor'
    batch_size: int = 64
    num_workers: int = 1

def main(args: Args):

    transform = get_pair_transforms_gator(args.transforms)

    dataset = wds.WebDataset(
            urls=str(args.data_dir / args.dataset / "train-{000000..000180}.tar"),
            shardshuffle=True,
        )\
            .shuffle(512)\
            .decode("torchrgb8")\
            .rename(im1="im1.jpg", im2="im2.jpg")\
            .to_tuple("im1", "im2")\
            .map(lambda x: transform(x[0], x[1]))\
            .batched(args.batch_size, partial=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=None,
    )
    
    t = time.time()
    i = 0
    for _ in tqdm(data_loader_train):
        # print(f"Batch {i}: {batch[0].shape}, {batch[1].shape}")
        i += 1
        if i >= 100:
            break

    print(f"Time taken for 100 batches: {time.time() - t:.2f} seconds")

if __name__ == "__main__":
    tyro.cli(main)