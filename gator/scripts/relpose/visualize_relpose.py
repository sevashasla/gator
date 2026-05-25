"""
Visualize relative pose estimation: image pairs + camera frustums (pred vs GT).

What this does:
  - Loads a trained model checkpoint
  - Samples n_samples*3 pairs from the 7-Scenes test set (pairs are fixed,
    pre-computed via NetVLAD retrieval: each test query is paired with its
    most similar training image)
  - Runs model inference on those pairs
  - Shows the n_samples with lowest error (sort_by=best_terr) or highest (sort_by=terr)
  - For each pair: the two images + 3 camera frustums (green=ref, orange=GT, red=predicted)
  - Errors shown are prediction errors vs ground truth (0° = perfect)

Usage:
  python -m gator.scripts.relpose.visualize_relpose \
    --ckpt <run_dir>/checkpoints/last.ckpt \
    --scene <chess|fire|heads|office|pumpkin|redkitchen|stairs> \
    --n_samples <int>          # number of pairs to show (default: 6)
    --sort_by <best_terr|best_rerr|terr|rerr|random>
                               # best_terr/best_rerr = lowest error first (success cases, default: best_terr)
                               # terr/rerr = highest error first (failure cases)
    --out visuals/relpose/<model>_<scene>_<sort>.png

Examples:
  # MAE on chess, best (lowest) translation errors — success cases for presentation
  python -m gator.scripts.relpose.visualize_relpose \
    --ckpt /scratch/izar/bosi/gator/relpose/mae-small-unfrozen-blr3e5/checkpoints/last.ckpt \
    --scene chess --n_samples 6 --sort_by best_terr \
    --out visuals/relpose/mae_chess_best.png

  # CroCo on fire, worst rotation errors — failure case analysis
  python -m gator.scripts.relpose.visualize_relpose \
    --ckpt /scratch/izar/bosi/gator/relpose/croco-small-48hrs-lr-1e-6-nf/checkpoints/last.ckpt \
    --scene fire --n_samples 6 --sort_by rerr \
    --out visuals/relpose/croco_fire_rerr.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import torch
import yaml

from gator.relpose.loss import L21Loss, RelativeCameraPoseRegression
from gator.relpose.models.relpose_wrapper import RelposeOptimizationParameters, RelposeWrapper
from gator.relpose.utils.metric import get_rot_err, get_transl_ang_err
from gator.relpose.utils.device import to_numpy
from gator.scripts.relpose.load_dataset import LoadDatasetArguments


# ── model factories ────────────────────────────────────────────────────────────

def _load_croco(cfg: dict, device) -> torch.nn.Module:
    from gator.relpose.models.croco_relpose import CroCoNetRelpose  # noqa: F401 (used in eval)
    inner_ckpt = cfg['inner_model_ckpt_path']
    model_cfg_str = cfg.get('model_configuration', 'CroCoNet()')
    freeze = cfg.get('freeze_encdec', True)
    model_cfg_str2 = (
        "CroCoNetRelpose("
        f"croco_ckpt_path='{inner_ckpt}', freeze={freeze}, "
        + model_cfg_str[9:-1]
        + ")"
    )
    model = eval(model_cfg_str2)
    return model.to(device)


def _load_jigsaw(cfg: dict, device) -> torch.nn.Module:
    from gator.models.gator_2view.model_gator import GatorConfig
    from gator.relpose.models.jigsaw_relpose import JigsawRelpose
    inner_ckpt = cfg['inner_model_ckpt_path']
    freeze = cfg.get('freeze_encdec', True)
    model_config = GatorConfig(**cfg['model_config'])
    model = JigsawRelpose(config=model_config, jigsaw_ckpt_path=inner_ckpt, freeze=freeze)
    return model.to(device)


def _load_mae(cfg: dict, device) -> torch.nn.Module:
    from gator.models.mae_1view.model_mae import MAEConfig
    from gator.relpose.models.mae_relpose import MAERelpose
    inner_ckpt = cfg['inner_model_ckpt_path']
    freeze = cfg.get('freeze_encdec', True)
    model_config = MAEConfig(**cfg['model_config'])
    model = MAERelpose(config=model_config, mae_ckpt_path=inner_ckpt, freeze=freeze)
    return model.to(device)


def build_wrapper(cfg: dict, ckpt_path: Path, device) -> RelposeWrapper:
    exp_name: str = cfg.get('exp_name', '')
    if 'mae' in exp_name:
        inner = _load_mae(cfg, device)
    elif 'jigsaw' in exp_name or 'model_config' in cfg:
        inner = _load_jigsaw(cfg, device)
    else:
        inner = _load_croco(cfg, device)

    loss_fn = RelativeCameraPoseRegression(L21Loss())
    opt = RelposeOptimizationParameters()
    wrapper = RelposeWrapper(model=inner, loss_fn=loss_fn, optimization_config=opt)

    ckpt_data = torch.load(ckpt_path, map_location=device)
    wrapper.load_state_dict(ckpt_data['state_dict'])
    wrapper.eval()
    return wrapper.to(device)


# ── frustum drawing ─────────────────────────────────────────────────────────────

def _frustum_pts(K: np.ndarray, pose: np.ndarray, W: int, H: int, depth: float):
    """
    Returns (apex, corners) in the reference (camera-1) frame.

    pose: 4×4 matrix that transforms FROM this camera's frame TO camera-1 frame.
    """
    K_inv = np.linalg.inv(K)
    img_corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=float).T
    rays = K_inv @ img_corners                          # (3, 4)
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)

    R, t = pose[:3, :3], pose[:3, 3]
    corners = (R @ (rays * depth) + t[:, None]).T       # (4, 3)
    return t.copy(), corners


def _draw_frustum(ax, apex, corners, color, label=None, lw=1.5, alpha=0.25):
    for i, c in enumerate(corners):
        kw = dict(label=label) if i == 0 else {}
        ax.plot([apex[0], c[0]], [apex[1], c[1]], [apex[2], c[2]],
                color=color, lw=lw, **kw)
    for i in range(4):
        c1, c2 = corners[i], corners[(i + 1) % 4]
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color=color, lw=lw)
    face = Poly3DCollection([corners], alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(face)


def _draw_line(ax, p0, p1, color, lw=1.0, ls='--'):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=color, lw=lw, linestyle=ls, alpha=0.6)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, type=Path,
                   help='Lightning checkpoint (last.ckpt or epoch=N.ckpt)')
    p.add_argument('--scene', default='chess',
                   help='7-Scenes scene name (default: chess)')
    p.add_argument('--n_samples', type=int, default=6,
                   help='Number of pairs to visualize')
    p.add_argument('--sort_by', choices=['terr', 'rerr', 'best_terr', 'best_rerr', 'random'], default='best_terr',
                   help='How to pick samples: best_terr/best_rerr = lowest error first (success cases), '
                        'terr/rerr = highest error first (failure cases), random = no sorting')
    p.add_argument('--out', type=Path, default=Path('relpose_vis.png'))
    p.add_argument('--model_name', type=str, default=None,
                   help='Display name for the model (e.g. "CroCo", "MAE"). '
                        'Defaults to the run directory name.')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Read training config sitting next to the checkpoints/ dir
    cfg_path = args.ckpt.parent.parent / 'training_config.yml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Build & load model
    print(f"Loading checkpoint: {args.ckpt}")
    wrapper = build_wrapper(cfg, args.ckpt, device)

    # Dataset – single scene, test split, topk=1
    loader_args = LoadDatasetArguments(
        resolution=(224, 224),
        db_step=1,
        topk_train=1,
        topk_test=1,
        cache_folder='./_db-q_pair_info',
        scenes=[args.scene],
    )
    test_ds = loader_args.get_test_dataset()
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=True, num_workers=2, drop_last=False
    )

    def to_img(t):
        return t.cpu().float().numpy().transpose(1, 2, 0).clip(0, 1)

    samples = []
    with torch.no_grad():
        for batch in loader:
            if len(samples) >= args.n_samples * 3:  # collect extra, filter later
                break
            view1, view2 = batch
            for v in (view1, view2):
                for k in ('img', 'camera_pose', 'camera_intrinsics'):
                    if k in v:
                        v[k] = v[k].to(device)

            pose12, pose21 = wrapper(view1, view2)

            pred = to_numpy(pose21['pose'][0])   # (4,4) pred pose cam2→cam1
            gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
            gt = to_numpy(gt_pose2to1[0])        # (4,4) GT  pose cam2→cam1

            R_pr, t_pr = pred[:3, :3], pred[:3, 3]
            R_gt, t_gt = gt[:3, :3], gt[:3, 3]

            rerr = get_rot_err(R_pr, R_gt)
            t_pr_n = t_pr / (np.linalg.norm(t_pr) + 1e-8)
            t_gt_n = t_gt / (np.linalg.norm(t_gt) + 1e-8)
            terr = get_transl_ang_err(t_pr_n, t_gt_n)

            samples.append({
                'img1': to_img(view1['img'][0]),
                'img2': to_img(view2['img'][0]),
                'K':    to_numpy(view1['camera_intrinsics'][0]),
                'pred': pred,
                'gt':   gt,
                'rerr': rerr,
                'terr': terr,
            })

    if args.sort_by == 'best_terr':
        samples.sort(key=lambda x: x['terr'])           # lowest error first
    elif args.sort_by == 'best_rerr':
        samples.sort(key=lambda x: x['rerr'])
    elif args.sort_by == 'terr':
        samples.sort(key=lambda x: x['terr'], reverse=True)   # failure cases
    elif args.sort_by == 'rerr':
        samples.sort(key=lambda x: x['rerr'], reverse=True)

    samples = samples[:args.n_samples]
    n = len(samples)

    COLORS = {
        'cam1': (136/255, 176/255, 109/255),
        'gt':   (245/255, 168/255,  61/255),
        'pred': (245/255,  67/255,  62/255),
    }

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 8),
                             gridspec_kw={'height_ratios': [1, 1, 1.2]})
    if n == 1:
        axes = axes[:, None]

    for i, s in enumerate(samples):
        H, W = s['img1'].shape[:2]
        K = s['K']
        t_scale = np.linalg.norm(s['gt'][:3, 3])
        depth = max(t_scale * 0.20, 0.05)

        # images
        axes[0, i].imshow(s['img1'])
        axes[0, i].axis('off')
        axes[0, i].set_title('Reference', fontsize=10, pad=3)

        axes[1, i].imshow(s['img2'])
        axes[1, i].axis('off')
        axes[1, i].set_title('Query', fontsize=10, pad=3)

        # 3D frustums
        ax3 = axes[2, i]
        ax3.remove()
        ax3 = fig.add_subplot(3, n, 2 * n + i + 1, projection='3d')

        eye = np.eye(4)
        a1,    c1    = _frustum_pts(K, eye,       W, H, depth)
        a2_gt, c2_gt = _frustum_pts(K, s['gt'],   W, H, depth)
        a2_pr, c2_pr = _frustum_pts(K, s['pred'], W, H, depth)

        _draw_frustum(ax3, a1,    c1,    COLORS['cam1'])
        _draw_frustum(ax3, a2_gt, c2_gt, COLORS['gt'])
        _draw_frustum(ax3, a2_pr, c2_pr, COLORS['pred'])
        _draw_line(ax3, a1, a2_gt, COLORS['gt'])
        _draw_line(ax3, a1, a2_pr, COLORS['pred'])

        ax3.set_axis_off()
        ax3.set_box_aspect([1, 1, 1])
        ax3.view_init(elev=20, azim=-60)
        ax3.set_title(
            f'Rotation error: {s["rerr"]:.1f}°   Translation error: {s["terr"]:.1f}°',
            fontsize=9, pad=2
        )

    # single shared legend
    handles = [
        mpatches.Patch(color=COLORS['cam1'], label='Camera 1 (reference)'),
        mpatches.Patch(color=COLORS['gt'],   label='Camera 2 — ground truth'),
        mpatches.Patch(color=COLORS['pred'], label='Camera 2 — predicted'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    model_name = args.model_name or args.ckpt.parent.parent.name
    fig.suptitle(
        f'Model: {model_name}   —   7-Scenes / {args.scene}',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
