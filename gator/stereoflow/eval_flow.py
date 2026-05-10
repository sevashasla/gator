"""Evaluate and visualize optical flow on MPI-Sintel.

Supports both CroCo and Gator checkpoints (auto-detected from saved args).

Usage:
    python stereoflow/eval_flow.py \\
        --model checkpoints/run/checkpoint-best.pth \\
        --split test_cleanpass \\
        [--save]               save flow images alongside metrics \\
        [--save_dir DIR]       output directory (default: <model>_sintel_<split>) \\
        [--max_pairs N]        stop after N pairs (default: all) \\
        [--tile_overlap 0.7]
        [--num_workers 4]

Available splits:
    train_cleanpass  / train_finalpass  / train_allpass
    subtrain_cleanpass / subtrain_finalpass
    subval_cleanpass / subval_finalpass
    test_cleanpass   / test_finalpass   / test_allpass

For train/subval splits (ground truth available):
    EPE, L1err, bad@{0.5,1,3,5}px and per-speed-bin EPE are printed.
    An EPE heatmap is saved next to the flow image when --save is set.

For test splits (no ground truth): only flow colour images are saved.

python stereoflow/eval_flow.py --model checkpoints/gator_ft_flow_2872508/checkpoint-best.pth --split subval_cleanpass --max_pairs 10 --save
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import gator.utils.misc as misc
from gator.models.croco_downstream import CroCoDownstreamBinocular
from gator.models.gator_downstream import GatorDownstreamBinocular
from gator.models.oneview_downstream import OneViewDownstreamBinocular
from gator.models.head_downstream import PixelwiseTaskWithDPT
from gator.stereoflow.criterion import *
from gator.stereoflow.datasets_flow import (
    MPISintelDataset, flowToColor, flowMaxNorm,
)
from gator.stereoflow.engine import tiled_pred


# ---------------------------------------------------------------------------
# Model loading (mirrors test.py logic)
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]

    assert ckpt_args.task == "flow", (
        f"Checkpoint was trained for task '{ckpt_args.task}', expected 'flow'."
    )

    with_conf = eval(ckpt_args.criterion).with_conf
    num_channels = 2 + int(with_conf)

    head = PixelwiseTaskWithDPT()
    head.num_channels = num_channels

    model_type = getattr(ckpt_args, "model", "croco")
    if model_type == "croco":
        print(f"  backbone: CroCo  |  croco_args: {ckpt_args.croco_args}")
        model = CroCoDownstreamBinocular(head, **ckpt_args.croco_args)
    elif model_type == "gator":
        print(f"  backbone: Gator  |  config: {ckpt_args.gator_config}")
        model = GatorDownstreamBinocular(
            head, ckpt_args.gator_config, img_size=tuple(ckpt_args.crop)
        )
    else:  # mae or jigsaw_1view
        print(f"  backbone: {model_type}  |  config: {ckpt_args.oneview_config}")
        model = OneViewDownstreamBinocular(
            head, ckpt_args.oneview_config, img_size=tuple(ckpt_args.crop)
        )

    msg = model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model = model.to(device)

    return model, ckpt_args, with_conf


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_flow_image(flow_hw2, path):
    """Save a colorised optical flow image. flow_hw2 is (H,W,2) numpy array."""
    img = flowToColor(flow_hw2.copy())
    Image.fromarray(img).save(path)


def _save_epe_map(epe_hw, path, vmax=None):
    """Save an EPE heatmap. epe_hw is (H,W) numpy array."""
    if vmax is None:
        vmax = np.nanpercentile(epe_hw[np.isfinite(epe_hw)], 95) if np.any(np.isfinite(epe_hw)) else 1.0
    norm = np.clip(epe_hw / (vmax + 1e-6), 0, 1)
    norm[~np.isfinite(epe_hw)] = 0.0
    rgba = (cm.inferno(norm) * 255).astype(np.uint8)
    Image.fromarray(rgba[:, :, :3]).save(path)


def _save_comparison(img1_np, pred_hw2, gt_hw2, epe_hw, out_path):
    """Save a 4-panel comparison figure (img, pred flow, gt flow, EPE map)."""
    has_gt = gt_hw2 is not None

    ncols = 4 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].imshow(img1_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(flowToColor(pred_hw2.copy()))
    axes[1].set_title("Predicted flow")
    axes[1].axis("off")

    if has_gt:
        maxflow = max(flowMaxNorm(gt_hw2.copy()), 1e-3)
        axes[2].imshow(flowToColor(gt_hw2.copy(), maxflow=maxflow))
        axes[2].set_title("GT flow")
        axes[2].axis("off")

        vmax = np.nanpercentile(epe_hw[np.isfinite(epe_hw)], 95) if np.any(np.isfinite(epe_hw)) else 1.0
        axes[3].imshow(epe_hw, cmap="inferno", vmin=0, vmax=vmax)
        axes[3].set_title(f"EPE (p95={vmax:.2f}px)")
        axes[3].axis("off")
        plt.colorbar(axes[3].images[0], ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate / visualise optical flow on MPI-Sintel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to finetuned checkpoint (.pth)")
    parser.add_argument("--split", default="test_cleanpass",
                        help="MPISintel split (e.g. test_cleanpass, subval_finalpass)")
    parser.add_argument("--save", action="store_true",
                        help="Save flow colour images (and EPE maps when GT available)")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory. Default: <model_stem>_sintel_<split>")
    parser.add_argument("--comparison", action="store_true",
                        help="Save 4-panel comparison figure instead of individual images")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs processed")
    parser.add_argument("--tile_overlap", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- load model ---
    model_path = Path(args.model)
    assert model_path.is_file(), f"Checkpoint not found: {model_path}"
    print(f"Loading model from {model_path} ...")
    model, ckpt_args, with_conf = load_model(model_path, device)
    crop = ckpt_args.crop
    tile_conf_mode = ckpt_args.tile_conf_mode
    print(f"  crop={crop}  tile_conf_mode={tile_conf_mode}  with_conf={with_conf}")

    # --- load dataset ---
    has_gt = not args.split.startswith("test")
    print(f"\nLoading MPISintel split='{args.split}'  (GT available: {has_gt})")
    dataset = MPISintelDataset(split=args.split)
    print(repr(dataset))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # --- output directory ---
    if args.save:
        save_dir = Path(args.save_dir) if args.save_dir else \
            model_path.parent / f"{model_path.stem}_sintel_{args.split}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving images to: {save_dir}")

    # --- metrics ---
    metrics = FlowDatasetMetrics().to(device)
    metrics.reset()

    # --- inference loop ---
    total = len(loader) if args.max_pairs is None else min(args.max_pairs, len(loader))
    print()
    for step, (img1, img2, gt, pairnames) in enumerate(tqdm(loader, total=total)):
        if args.max_pairs is not None and step >= args.max_pairs:
            break

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        gt_dev = gt.to(device, non_blocking=True) if has_gt and gt.numel() > 0 else None

        with torch.inference_mode():
            pred, _, _, _ = tiled_pred(
                model, None, img1, img2,
                gt_dev,
                conf_mode=tile_conf_mode,
                overlap=args.tile_overlap,
                crop=crop,
                with_conf=with_conf,
                return_time=True,
            )

        if has_gt and gt_dev is not None:
            metrics.add_batch(pred, gt_dev)

        if not args.save:
            continue

        # --- per-pair saving ---
        pairname = pairnames[0]
        pname_str = dataset.pairname_to_str(eval(pairname) if pairname.startswith("(") else pairname)
        out_base = save_dir / pname_str
        out_base.parent.mkdir(parents=True, exist_ok=True)

        pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # (H,W,2)

        if args.comparison:
            # denormalise img1 for display
            _mean = np.array([0.485, 0.456, 0.406])
            _std  = np.array([0.229, 0.224, 0.225])
            img1_np = img1[0].permute(1, 2, 0).cpu().numpy()
            img1_np = np.clip(img1_np * _std + _mean, 0, 1)

            gt_np = gt[0].permute(1, 2, 0).cpu().numpy() if has_gt else None
            epe_np = None
            if has_gt and gt_np is not None:
                err = np.sqrt(np.sum((pred_np - gt_np) ** 2, axis=-1))
                valid = np.all(np.isfinite(gt_np), axis=-1)
                epe_np = np.where(valid, err, np.nan)

            _save_comparison(img1_np, pred_np, gt_np, epe_np,
                             str(out_base) + "_comparison.png")
        else:
            # individual images
            _mean = np.array([0.485, 0.456, 0.406])
            _std  = np.array([0.229, 0.224, 0.225])
            img1_np = img1[0].permute(1, 2, 0).cpu().numpy()
            img1_np = np.clip(img1_np * _std + _mean, 0, 1)
            Image.fromarray((img1_np * 255).astype(np.uint8)).save(str(out_base) + "_img.png")

            _save_flow_image(pred_np, str(out_base) + "_flow_pred.png")

            if has_gt and gt.numel() > 0:
                gt_np = gt[0].permute(1, 2, 0).cpu().numpy()
                _save_flow_image(gt_np, str(out_base) + "_flow_gt.png")

                err = np.sqrt(np.sum((pred_np - gt_np) ** 2, axis=-1))
                valid = np.all(np.isfinite(gt_np), axis=-1)
                epe_np = np.where(valid, err, np.nan)
                _save_epe_map(epe_np, str(out_base) + "_epe.png")

    # --- print metrics ---
    if has_gt:
        print()
        results = metrics.get_results()
        col_w = max(len(k) for k in results) + 2
        print(f"{'Metric':<{col_w}} Value")
        print("-" * (col_w + 10))
        for k, v in results.items():
            print(f"{k:<{col_w}} {v:.4f}")

    if args.save:
        print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
