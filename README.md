# Gator 🐊: Cross-view Ji**g**s**a**w Objec**t**ive f**or** Self-Supervised 3D Vision

## [Project Page]() | [🤗 Demo](https://huggingface.co/spaces/sevashasla2/gator) |

> TL;DR: Explore cross-view jigsaw objective as a pretraining task for 3D computer vision tasks

<p align="center">
  <img src="assets/croco_faster.gif" alt="Method" width="70%">
</p>

## Installation ⚙️

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the repository 

```bash
git clone https://github.com/sevashasla/gator && cd gator
```

3. Install the requirements

```
uv venv && source .venv/bin/activate && uv sync --all-extras
```

**important** cuRoPE2D does not have backward implemented, so for training use the pytorch version instead.

## Pre-training 🧠

1. Generate dataset following the instructions [here](https://github.com/naver/croco/blob/master/datasets/habitat_sim/README.MD)
2. Gator-Small

  ```bash
  python3 gator/scripts/pretrainings/pretrain_gator.py \
    --model-config.enc-emb-dim 384 --model-config.enc-num-heads 6 \
    --model-config.dec-emb-dim 384 --model-config.dec-num-heads 6 \
    --opt-params.blr 1e-4 \
    --data-dir /path/to/rendered/dataset \
    --output-dir /root/to/store/the/results \
    --exp-name experiment-name
  ```

  To validate the pretraining model run

  ```bash
  python3 gator/scripts/test/test_gator.py \
      # should be /root/to/store/the/results/experiment-name/checkpoints/last.ckpt
      --ckpt-path /path/to/last.ckpt \ 
      --model-config.enc-emb-dim 384 --model-config.dec-emb-dim 384 \
      --model-config.enc-num-heads 6 --model-config.dec-num-heads 6 \
      --visualizer-config.name visual
  ```

3. CroCo-Small

  ```bash
  python3 gator/scripts/pretrainings/pretrain_croco.py \
    --exp-name croco-small-blr-1e-4-real \
    --opt-params.batch-size 128 \
    --opt-params.blr 1e-4 \
    --opt-params.max_epoch 50 \
    --opt-params.epochs 100 \
    --model_configuration "CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')" \
    --data-dir /path/to/rendered/dataset \
    --output-dir /root/to/store/the/results \
    --exp-name experiment-name
  ```

4. MAE-Small

  ```bash
  python3 gator/scripts/pretrainings/pretrain_mae.py \
    --model-config.enc-emb-dim 384 --model-config.enc-num-heads 6 \
    --model-config.dec-emb-dim 384 --model-config.dec-num-heads 6 \
    --data-dir /path/to/rendered/dataset \
    --output-dir /root/to/store/the/results \
    --exp-name experiment-name
  ```

5. Jigsaw-Small

  ```bash
  python3 gator/scripts/pretrainings/pretrain_jigsaw.py \
    --model-config.enc-emb-dim 384 --model-config.enc-num-heads 6 \
    --data-dir /path/to/rendered/dataset \
    --output-dir /root/to/store/the/results \
    --exp-name experiment-name
  ```

## Finetining on Downstream Tasks 🚀

### Classification

### Optical Flow

### Relative Pose Regression

## File Hierarchy 📚

```
project-root/
├── README.md — Project overview and setup instructions.
├── package.json — Project metadata, scripts, and dependencies.
├── docs/
│   ├── diffusion_editing/
│   ├── visual_anagrams/
│   └── two_view.md
│
├── assets/
├── gator/
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Acknowledgements

We thank the authors of [`CroCo`](https://github.com/naver/croco) and [`Reloc3r`](https://github.com/ffrivera0/reloc3r) for their amazing works!
