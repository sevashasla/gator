"""
Plot pose recall curves for multiple relpose models.

What this does:
  - For each model, loads checkpoint and runs inference on ALL 7-Scenes test pairs
    (all 7 scenes, thousands of pairs, pre-computed via NetVLAD retrieval)
  - Collects rotation error and translation error for every pair
  - Plots recall curve: x = error threshold (degrees), y = % of pairs where
    max(rot_error, trans_error) < threshold
  - Higher curve = better model
  - AUC@5/10/20 in the legend = area under the curve up to 5°/10°/20°

Usage:
  python -m gator.scripts.relpose.recall_curve \
    --ckpts \
      gator:/scratch/izar/skorokho/gator/relpose/gator-small-000-nf/checkpoints/last.ckpt \
      croco:/scratch/izar/bosi/gator/relpose/croco-small-48hrs-lr-1e-6-nf/checkpoints/last.ckpt \
      mae:/scratch/izar/bosi/gator/relpose/mae-small-unfrozen-blr3e5/checkpoints/last.ckpt \
      jigsaw:/scratch/izar/bosi/gator/relpose/jigsaw-small-24hrs-lr-1e-5-nf/checkpoints/last.ckpt \
    --out visuals/relpose/recall_curve.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from gator.relpose.loss import L21Loss, RelativeCameraPoseRegression
from gator.relpose.models.relpose_wrapper import RelposeOptimizationParameters, RelposeWrapper
from gator.relpose.utils.metric import get_rot_err, get_transl_ang_err, error_auc
from gator.relpose.utils.device import to_numpy
from gator.scripts.relpose.load_dataset import LoadDatasetArguments


# ── model loading (same as visualize_relpose.py) ───────────────────────────────

def _load_croco(cfg, device):
    from gator.relpose.models.croco_relpose import CroCoNetRelpose  # noqa: F401
    inner_ckpt = cfg['inner_model_ckpt_path']
    model_cfg_str = cfg.get('model_configuration', 'CroCoNet()')
    freeze = cfg.get('freeze_encdec', True)
    expr = (
        "CroCoNetRelpose("
        f"croco_ckpt_path='{inner_ckpt}', freeze={freeze}, "
        + model_cfg_str[9:-1] + ")"
    )
    return eval(expr).to(device)


def _load_mae(cfg, device):
    from gator.models.mae_1view.model_mae import MAEConfig
    from gator.relpose.models.mae_relpose import MAERelpose
    model = MAERelpose(
        config=MAEConfig(**cfg['model_config']),
        mae_ckpt_path=cfg['inner_model_ckpt_path'],
        freeze=cfg.get('freeze_encdec', True),
    )
    return model.to(device)


def _load_jigsaw(cfg, device):
    from gator.models.gator_2view.model_gator import GatorConfig
    from gator.relpose.models.jigsaw_relpose import JigsawRelpose
    model = JigsawRelpose(
        config=GatorConfig(**cfg['model_config']),
        jigsaw_ckpt_path=cfg['inner_model_ckpt_path'],
        freeze=cfg.get('freeze_encdec', True),
    )
    return model.to(device)


def _load_gator(cfg, device):
    from gator.relpose.models.gator_relpose import GatorRelpose
    from gator.models.gator_2view.model_gator import GatorConfig
    model = GatorRelpose(
        config=GatorConfig(**cfg['model_config']),
        gator_ckpt_path=cfg['inner_model_ckpt_path'],
        freeze=cfg.get('freeze_encdec', True),
    )
    return model.to(device)


def build_wrapper(cfg, ckpt_path, device):
    exp_name = cfg.get('exp_name', '')
    if 'gator' in exp_name:
        inner = _load_gator(cfg, device)
    elif 'mae' in exp_name:
        inner = _load_mae(cfg, device)
    elif 'jigsaw' in exp_name or 'model_config' in cfg:
        inner = _load_jigsaw(cfg, device)
    else:
        inner = _load_croco(cfg, device)

    wrapper = RelposeWrapper(
        model=inner,
        loss_fn=RelativeCameraPoseRegression(L21Loss()),
        optimization_config=RelposeOptimizationParameters(),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    wrapper.load_state_dict(ckpt['state_dict'])
    wrapper.eval()
    return wrapper.to(device)


# ── evaluation ─────────────────────────────────────────────────────────────────

def collect_errors(wrapper, device):
    """Run inference on all 7-scenes test pairs and return (rerrs, terrs)."""
    loader_args = LoadDatasetArguments(
        resolution=(224, 224),
        db_step=1,
        topk_train=1,
        topk_test=1,
        cache_folder='./_db-q_pair_info',
        scenes=['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'],
    )
    test_ds = loader_args.get_test_dataset()
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, drop_last=False
    )

    rerrs, terrs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  evaluating', leave=False):
            view1, view2 = batch
            for v in (view1, view2):
                for k in ('img', 'camera_pose', 'camera_intrinsics'):
                    if k in v:
                        v[k] = v[k].to(device)

            _, pose21 = wrapper(view1, view2)

            gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
            pred = to_numpy(pose21['pose'])      # (B, 4, 4)
            gt   = to_numpy(gt_pose2to1)         # (B, 4, 4)

            for b in range(len(pred)):
                R_pr, t_pr = pred[b, :3, :3], pred[b, :3, 3]
                R_gt, t_gt = gt[b,   :3, :3], gt[b,   :3, 3]
                rerrs.append(get_rot_err(R_pr, R_gt))
                t_pr_n = t_pr / (np.linalg.norm(t_pr) + 1e-8)
                t_gt_n = t_gt / (np.linalg.norm(t_gt) + 1e-8)
                terrs.append(get_transl_ang_err(t_pr_n, t_gt_n))

    return np.array(rerrs), np.array(terrs)


# ── recall curve (SuperGlue / Reloc3r convention) ──────────────────────────────

def recall_curve(rerrs, terrs, max_threshold=20):
    """
    Returns (thresholds, recall) where recall[i] = fraction of pairs
    with max(rerr, terr) < thresholds[i].
    Follows SuperGlue's pose_auc convention.
    """
    max_errors = np.maximum(rerrs, terrs)
    thresholds = np.linspace(0, max_threshold, 200)
    recall = [(max_errors < t).mean() for t in thresholds]
    return thresholds, np.array(recall)


# ── plotting ───────────────────────────────────────────────────────────────────

# Color palette — distinct, presentation-friendly
PALETTE = {
    'gator':  '#E63946',   # red
    'croco':  '#457B9D',   # blue
    'mae':    '#2A9D8F',   # teal
    'jigsaw': '#E9C46A',   # yellow
}
DEFAULT_COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#8338EC', '#FB5607']


def plot_recall_curves(results: dict[str, tuple], out: Path, max_threshold=20):
    """
    results: {label: (rerrs, terrs)}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, (label, (rerrs, terrs)) in enumerate(results.items()):
        color = PALETTE.get(label.lower(), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        thresholds, recall = recall_curve(rerrs, terrs, max_threshold)

        aucs = error_auc(rerrs, terrs, thresholds=[5, 10, 20])
        auc5  = aucs['auc@5']  * 100
        auc10 = aucs['auc@10'] * 100
        auc20 = aucs['auc@20'] * 100

        display = label.upper() if label.lower() == 'mae' else label.capitalize()
        legend_label = f'{display}   AUC@5={auc5:.1f}  @10={auc10:.1f}  @20={auc20:.1f}'
        ax.plot(thresholds, recall * 100, label=legend_label, color=color, lw=2.5, zorder=3)
        ax.fill_between(thresholds, recall * 100, alpha=0.12, color=color, zorder=2)

    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Error threshold (degrees)', fontsize=12)
    ax.set_ylabel('Correctly estimated pairs (%)', fontsize=12)
    ax.set_xlim(0, max_threshold)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(colors='#555555')
    ax.xaxis.label.set_color('#333333')
    ax.yaxis.label.set_color('#333333')
    ax.set_title('Pose Recall Curve — 7-Scenes', fontsize=13, pad=12, color='#222222')

    ax.legend(loc='upper left', fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--ckpts', nargs='+', required=True,
        metavar='LABEL:PATH',
        help='Checkpoint paths with labels, e.g. gator:/path/to/last.ckpt croco:/path/to/last.ckpt'
    )
    p.add_argument('--out', type=Path, default=Path('visuals/relpose/recall_curve.png'))
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    results = {}
    for entry in args.ckpts:
        label, ckpt_path = entry.split(':', 1)
        ckpt_path = Path(ckpt_path)
        cfg_path = ckpt_path.parent.parent / 'training_config.yml'

        print(f'\n[{label}] Loading {ckpt_path}')
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        wrapper = build_wrapper(cfg, ckpt_path, device)

        print(f'[{label}] Running inference on 7-Scenes test set...')
        rerrs, terrs = collect_errors(wrapper, device)
        results[label] = (rerrs, terrs)
        print(f'[{label}] {len(rerrs)} pairs — median Rerr={np.median(rerrs):.2f}° Terr={np.median(terrs):.2f}°')

        del wrapper
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_recall_curves(results, args.out)


if __name__ == '__main__':
    main()
