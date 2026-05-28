"""Plot training metrics from a flow training log file.

Accepts two formats:
  - log.txt  (JSON-lines written by train.py, one dict per epoch)
  - *.out    (SLURM stdout: parses per-epoch "Averaged stats" lines)

Usage:
    python stereoflow/plot_metrics_flow.py logs/gator_ft_2872508.out
    python stereoflow/plot_metrics_flow.py checkpoints/run/log.txt --out metrics.png

    or from gator root,
    python -m gator.stereoflow.plot_metrics_flow gator/logs/gator_ft_2872508.out
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsers (shared with plot_metrics_stereo)
# ---------------------------------------------------------------------------

def _parse_kv_line(line):
    """Parse 'key: val (avg)  key2: val2 ...' into {key: float}."""
    row = {}
    for m in re.finditer(r"(\S+):\s+([\d.e+\-]+)(?:\s+\([\d.e+\-]+\))?", line):
        try:
            row[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return row


def load_json_log(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_slurm_log(path):
    records = []
    current_epoch = None
    current = {}

    with open(path) as f:
        for raw in f:
            line = raw.strip()

            m = re.search(r"Epoch:\s+\[(\d+)\]\s+Total time", line)
            if m:
                epoch_num = int(m.group(1))
                if current_epoch != epoch_num:
                    if current_epoch is not None and current:
                        records.append({"epoch": current_epoch, **current})
                        current = {}
                    current_epoch = epoch_num
                continue

            idx = line.find("Averaged stats:")
            if idx == -1:
                continue
            payload = line[idx + len("Averaged stats:"):].strip()

            if payload.startswith("{"):
                try:
                    d = ast.literal_eval(payload)
                    for k, v in d.items():
                        current[f"val_{k}"] = v
                except Exception:
                    pass
            else:
                row = _parse_kv_line(payload)
                for k, v in row.items():
                    current[f"train_{k}"] = v

    if current_epoch is not None and current:
        records.append({"epoch": current_epoch, **current})

    return records


def load_log(path):
    path = Path(path)
    with open(path) as f:
        first = f.read(1)
    if first == "{":
        return load_json_log(path)
    return load_slurm_log(path)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _get(records, key):
    epochs, values = [], []
    for r in records:
        if key in r:
            epochs.append(r["epoch"])
            values.append(r[key])
    return np.array(epochs), np.array(values)


def _val_dataset_prefixes(records):
    """Return all per-dataset validation prefixes (excludes AVG_)."""
    seen = set()
    for r in records:
        for k in r:
            if k.startswith("val_") and k.endswith("_EPE") and not k.startswith("val_AVG_"):
                seen.add(k[len("val_"):k.rfind("_EPE")])
    return sorted(seen)


def _short_label(prefix):
    """'MPISintelDataset_subval_cleanpass' -> 'MPISintel (subval cleanpass)'."""
    parts = prefix.split("Dataset", 1)
    if len(parts) == 2:
        name = parts[0]
        split = parts[1].lstrip("_").replace("_", " ")
        return f"{name} ({split})" if split else name
    return prefix.replace("_", " ")


def plot_metrics(records, out_path):
    if not records:
        print("No records found — nothing to plot.", file=sys.stderr)
        return

    prefixes = _val_dataset_prefixes(records)
    multi = len(prefixes) > 1
    has_avg = any("val_AVG_EPE" in r for r in records)

    ds_colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Flow training metrics", fontsize=13)

    def _ax_base(ax, title, xlabel, ylabel):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # -----------------------------------------------------------------
    # (0,0)  Loss
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    te, tl = _get(records, "train_loss")
    ve, vl = _get(records, "val_loss")
    vte, vtl = _get(records, "val_loss_tiled")
    if len(te):  ax.plot(te,  tl,  "b-",  label="train",        linewidth=1.5)
    if len(ve):  ax.plot(ve,  vl,  "r-",  label="val (center)", linewidth=1.5)
    if len(vte): ax.plot(vte, vtl, "r--", label="val (tiled)",  linewidth=1.5)
    _ax_base(ax, "Loss", "epoch", "loss")

    # -----------------------------------------------------------------
    # (0,1)  EPE — train + per-dataset + AVG
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    te, ta = _get(records, "train_EPE")
    if len(te): ax.plot(te, ta, "b-", label="train", linewidth=1.5)
    for i, pfx in enumerate(prefixes):
        ve, va = _get(records, f"val_{pfx}_EPE")
        if len(ve):
            ax.plot(ve, va, color=ds_colors[i % 10],
                    linewidth=1.0 if multi else 1.5,
                    linestyle="--" if multi else "-",
                    label=_short_label(pfx), alpha=0.8)
    if has_avg:
        ve, va = _get(records, "val_AVG_EPE")
        if len(ve): ax.plot(ve, va, "r-", linewidth=2.0, label="val AVG")
    _ax_base(ax, "EPE (End-Point Error)", "epoch", "px")

    # -----------------------------------------------------------------
    # (0,2)  L1err — train + AVG or per-dataset
    # -----------------------------------------------------------------
    ax = axes[0, 2]
    te, tl1 = _get(records, "train_L1err")
    if len(te): ax.plot(te, tl1, "b-", label="train", linewidth=1.5)
    if has_avg:
        ve, vl1 = _get(records, "val_AVG_L1err")
        if len(ve): ax.plot(ve, vl1, "r-", label="val AVG", linewidth=1.5)
    else:
        for i, pfx in enumerate(prefixes):
            ve, vl1 = _get(records, f"val_{pfx}_L1err")
            if len(ve):
                ax.plot(ve, vl1, color=ds_colors[i % 10], linewidth=1.5,
                        label=_short_label(pfx))
    _ax_base(ax, "L1 error", "epoch", "px")

    # -----------------------------------------------------------------
    # (1,0)  bad@ rates — val AVG (or single dataset)
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    bad_thresholds = ["1.0", "3.0", "5.0"]
    bad_colors = ["#e41a1c", "#ff7f00", "#377eb8"]
    pfx_for_bad = None if has_avg else (prefixes[0] if prefixes else None)
    for thr, c in zip(bad_thresholds, bad_colors):
        key = f"val_AVG_bad@{thr}" if has_avg else (
              f"val_{pfx_for_bad}_bad@{thr}" if pfx_for_bad else None)
        if key:
            ve, vb = _get(records, key)
            if len(ve): ax.plot(ve, vb, color=c, label=f"bad@{thr}px", linewidth=1.5)
    te, tb3 = _get(records, "train_bad@3.0")
    if len(te): ax.plot(te, tb3, "k--", label="train bad@3.0", linewidth=1, alpha=0.6)
    title_suffix = "AVG" if has_avg else (_short_label(pfx_for_bad) if pfx_for_bad else "")
    _ax_base(ax, f"Bad-pixel rates ({title_suffix})", "epoch", "% bad pixels")

    # -----------------------------------------------------------------
    # (1,1)  Learning rate
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    te, tlr = _get(records, "train_lr")
    if len(te): ax.plot(te, tlr, "g-", linewidth=1.5)
    ax.set_title("Learning rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lr")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    # -----------------------------------------------------------------
    # (1,2)  Per-dataset EPE breakdown (epoch ≥ 1)
    # -----------------------------------------------------------------
    ax = axes[1, 2]
    te2, ta2 = _get(records, "train_EPE")
    mask2 = te2 >= 1
    if mask2.any(): ax.plot(te2[mask2], ta2[mask2], "b-", linewidth=1.5, label="train")
    for i, pfx in enumerate(prefixes):
        ve, va = _get(records, f"val_{pfx}_EPE")
        mask = ve >= 1
        if mask.any():
            ax.plot(ve[mask], va[mask], color=ds_colors[i % 10],
                    linewidth=1.5, label=_short_label(pfx))
    if has_avg:
        ve, va = _get(records, "val_AVG_EPE")
        mask = ve >= 1
        if mask.any():
            ax.plot(ve[mask], va[mask], "r-", linewidth=2.0, label="val AVG")
    _ax_base(ax, "EPE per dataset (epoch ≥ 1)", "epoch", "px")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot flow training metrics")
    parser.add_argument("log", help="Path to log.txt (JSON-lines) or SLURM .out file")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: <log_dir>/metrics_flow.png)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else log_path.parent / "metrics_flow.png"

    print(f"Loading {log_path} ...")
    records = load_log(log_path)
    print(f"  {len(records)} epoch(s) found")

    plot_metrics(records, out_path)


if __name__ == "__main__":
    main()
