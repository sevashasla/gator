import sys
# Remove gator package from path to avoid shadowing HuggingFace datasets
sys.path = [p for p in sys.path if 'gator' not in p]

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def save_split(
    split: str,
    output_dir: Path,
    max_samples: int | None = None,
    shuffle_buffer: int = 50_000,
    seed: int = 0,
):
    print(f"\nLoading {split} split (streaming)...")

    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split=split,
        streaming=True,
        token=True,
    )

    if max_samples is not None:
        print(f"  Shuffling with buffer_size={shuffle_buffer} for balanced classes...")
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        ds = ds.take(max_samples)
        print(f"  Taking {max_samples} samples")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    errors = 0
    counters: dict[int, int] = defaultdict(int)  # per-class image counter

    for example in tqdm(ds, desc=f"Saving {split}", total=max_samples):
        label: int = example["label"]
        image = example["image"].convert("RGB")

        class_dir = split_dir / str(label)
        class_dir.mkdir(exist_ok=True)

        img_path = class_dir / f"{counters[label]:06d}.jpg"
        counters[label] += 1

        try:
            image.save(img_path, format="JPEG", quality=95)
            saved += 1
        except Exception as e:
            errors += 1
            print(f"Error saving {img_path}: {e}")

    print(f"{split}: saved {saved} images, {errors} errors")
    print(f"Output: {split_dir}")

    if counters:
        counts = list(counters.values())
        print(f"  Classes: {len(counts)}, min/max per class: {min(counts)}/{max(counts)}, avg: {sum(counts)//len(counts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download ImageNet-1k and save as ImageFolder structure."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save ImageNet (e.g. /scratch/izar/mayila/imagenet_full)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "train"],
        help="Splits to download. Recommended: download 'validation' first to test connectivity.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "Max number of training images to download. "
            "e.g. 100000 for ~8GB. "
            "Val is always downloaded fully. "
            "If not set, downloads the full train set (~150GB)."
        ),
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=50_000,
        help=(
            "Buffer size for shuffling before taking max_train_samples. "
            "Larger = better class balance but slower start. Default: 50000."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving ImageNet to : {output_dir}")
    print(f"Splits             : {args.splits}")
    print(f"Max train samples  : {args.max_train_samples or 'all (~150GB)'}")
    print(f"Structure          : {output_dir}/<split>/<label_id>/<image>.jpg")
    print(f"Compatible with    : torchvision.datasets.ImageFolder")

    for split in args.splits:
        max_samples = args.max_train_samples if split == "train" else None
        save_split(
            split=split,
            output_dir=output_dir,
            max_samples=max_samples,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
        )

    print("\nDone. Use in your training script:")
    print(f"  ImageFolder('{output_dir}/train',      transform=transform_train)")
    print(f"  ImageFolder('{output_dir}/validation', transform=transform_val)")


if __name__ == "__main__":
    main()