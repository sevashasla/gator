"""
Pose recall curves for multiple relpose models on 7-Scenes.

Usage:
  python -m gator.scripts.relpose.recall_curve \
    --ckpts label1:<ckpt_path> label2:<ckpt_path> ... \
    --out <output.png>
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
from gator.scripts.relpose.visualize_relpose import build_wrapper

ALL_SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']

PALETTE = {
    'gator':   '#E63946',
    'croco':   '#457B9D',
    'croco48': '#457B9D',
    'croco24': '#1D3557',
    'mae':     '#2A9D8F',
    'jigsaw':  '#E9C46A',
}
_FALLBACK = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#8338EC', '#FB5607']


def collect_errors(wrapper, device):
    loader_args = LoadDatasetArguments(
        resolution=(224, 224), db_step=1, topk_train=1, topk_test=1,
        cache_folder='./_db-q_pair_info', scenes=ALL_SCENES,
    )
    loader = torch.utils.data.DataLoader(
        loader_args.get_test_dataset(),
        batch_size=32, shuffle=False, num_workers=4, drop_last=False,
    )
    rerrs, terrs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval', leave=False):
            view1, view2 = batch
            for v in (view1, view2):
                for k in ('img', 'camera_pose', 'camera_intrinsics'):
                    if k in v:
                        v[k] = v[k].to(device)
            _, pose21 = wrapper(view1, view2)
            gt = to_numpy((torch.inverse(view1['camera_pose']) @ view2['camera_pose']))
            pred = to_numpy(pose21['pose'])
            for b in range(len(pred)):
                R_pr, t_pr = pred[b, :3, :3], pred[b, :3, 3]
                R_gt, t_gt = gt[b,   :3, :3], gt[b,   :3, 3]
                rerrs.append(get_rot_err(R_pr, R_gt))
                terrs.append(get_transl_ang_err(
                    t_pr / (np.linalg.norm(t_pr) + 1e-8),
                    t_gt / (np.linalg.norm(t_gt) + 1e-8),
                ))
    return np.array(rerrs), np.array(terrs)


def recall_at_thresholds(rerrs, terrs, max_t=20, n=200):
    thresholds = np.linspace(0, max_t, n)
    recall = (np.maximum(rerrs, terrs)[:, None] < thresholds[None]).mean(axis=0)
    return thresholds, recall


def plot(results, out, max_t=20):
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in [5, 10, 20]:
        ax.axvline(t, color='#cccccc', lw=0.8, ls='--', zorder=1)

    lines, labels = [], []
    for i, (label, (rerrs, terrs)) in enumerate(results.items()):
        color = PALETTE.get(label.lower(), _FALLBACK[i % len(_FALLBACK)])
        thresholds, recall = recall_at_thresholds(rerrs, terrs, max_t)
        aucs = error_auc(rerrs, terrs, thresholds=[5, 10, 20])
        a5, a10, a20 = aucs['auc@5']*100, aucs['auc@10']*100, aucs['auc@20']*100
        display = label.upper() if label.lower() == 'mae' else label.capitalize()
        line, = ax.plot(thresholds, recall*100, color=color, lw=2.5, zorder=3)
        lines.append(line)
        labels.append(f'{display:<8s}  @5={a5:4.1f}  @10={a10:4.1f}  @20={a20:4.1f}')

    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Error threshold (degrees)', fontsize=12)
    ax.set_ylabel('Correctly estimated pairs (%)', fontsize=12)
    ax.set_xlim(0, max_t); ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc'); ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(colors='#555555')
    ax.set_title('Pose Recall — 7-Scenes', fontsize=13, pad=12)
    ax.legend(lines, labels, loc='upper left', fontsize=9.5, frameon=False,
              prop={'family': 'monospace'})
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True, metavar='LABEL:PATH')
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
        with open(ckpt_path.parent.parent / 'training_config.yml') as f:
            cfg = yaml.safe_load(f)

        print(f'\n[{label}] {ckpt_path}')
        wrapper = build_wrapper(cfg, ckpt_path, device)
        rerrs, terrs = collect_errors(wrapper, device)
        results[label] = (rerrs, terrs)

        aucs = error_auc(rerrs, terrs, thresholds=[5, 10, 20])
        print(f'  {len(rerrs)} pairs  '
              f'AUC@5={aucs["auc@5"]*100:.1f}  '
              f'AUC@10={aucs["auc@10"]*100:.1f}  '
              f'AUC@20={aucs["auc@20"]*100:.1f}')

        del wrapper
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = args.out.with_suffix('.txt')
    with open(metrics_path, 'w') as f:
        for label, (rerrs, terrs) in results.items():
            aucs = error_auc(rerrs, terrs, thresholds=[5, 10, 20])
            f.write(f'[{label}]\n'
                    f'  pairs:  {len(rerrs)}\n'
                    f'  AUC@5:  {aucs["auc@5"]*100:.1f}\n'
                    f'  AUC@10: {aucs["auc@10"]*100:.1f}\n'
                    f'  AUC@20: {aucs["auc@20"]*100:.1f}\n\n')
    print(f'Metrics → {metrics_path}')

    plot(results, args.out)


if __name__ == '__main__':
    main()
