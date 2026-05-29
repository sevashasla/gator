# Gator 🐊: Cross-view Ji**g**s**a**w Objec**t**ive f**or** Self-Supervised 3D Vision

## [Project Page]() | [🤗 Demo](https://huggingface.co/spaces/sevashasla2/gator) |

> TL;DR: Cross-view jigsaw objective is an awersome pre-training task for 3D computer vision tasks

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
2. For Gator-Small run

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

3. For CroCo-Small run

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

4. For MAE-Small run

  ```bash
  python3 gator/scripts/pretrainings/pretrain_mae.py \
    --model-config.enc-emb-dim 384 --model-config.enc-num-heads 6 \
    --model-config.dec-emb-dim 384 --model-config.dec-num-heads 6 \
    --data-dir /path/to/rendered/dataset \
    --output-dir /root/to/store/the/results \
    --exp-name experiment-name
  ```

5. For Jigsaw-Small run

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

## Reference View Utilization

### Zero Reference

Run

```bash
python3 gator/scripts/gator_multiview_usage/by_zeroing.py \
    --ckpt-path /path/to/last.ckpt \
    --output-path /where/to/store/by_zeroing.csv \
    --model-config.enc-emb-dim 384 --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 --model-config.dec-num-heads 6 \
    --shuffle_ratios 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
```

The script creates a `by_zeroing.csv` file with results when passing the actual reference image and a zero image instead of the reference for different masking ratios.

### Tiny Dataset

Run

```bash
python3 gator/scripts/gator_multiview_usage/by_twin_images.py \
    --ckpt-path /path/to/last.ckpt \
    --exp-name exp-used-to-generate \
    --model-config.enc-emb-dim 384 --model-config.dec-emb-dim 384 \
    --model-config.enc-num-heads 6 --model-config.dec-num-heads 6
```

The script creates `twin_images_*.png` with reshuffling resuls and swapped object patches accuracy in `twin_images_accuracy.txt`. The idea of the method is discussed in [./docs/two_view.md](./docs/two_view.md)

## Cross-Attention

1. Extract q, k, v and batch images
  ```bash
  python3 gator/scripts/features_exploration/attention_maps.py \
    --ckpt-path /path/to/last.ckpt \
    --model-config.enc-emb-dim 384 --model-config.enc-num-heads 6 \
    --model-config.dec-emb-dim 384 --model-config.dec-num-heads 6 \
    --output-dir  /folder/to/store/qkv/ \
    # save only cross-attention features
    --ca-only \
    # how many eval batches to run and save
    --num-batches 10
  ```

  It produces folders /folder/to/store/qkv/batch_{chosen_idx:03}, with stored `qkv` and `input_batch.pth`

2. Create an image grid (not necessary, but easier to understand what patches to take in the next step)

  ```bash
  python3 gator/scripts/features_exploration/show_image_grid.py \
    --input_folder /folder/with/stored/qkv/batch \
    # image_indices can be any set with numbers from 0 to batch_size - 1
    --image-indices 0 5 112 3 58 \
    # whether to show shuffled image or reference image
    --no-reference \
    --output_dir /folder/to/store/the/results
  ```


3. Create an image with shown matches

  ```bash
  python3 gator/scripts/features_exploration/cross_attn_vis2.py \
    --input_folder /folder/with/stored/qkv/batch \
    --output_dir /where/to/store/the/resulting/image \
    --image_index image_index \
    # in format y x y x y x ...
    --patches_positions 5 4 7 5 7 10 1 9 7 7 6 3 \
    # use _decoder_blocks.3.cross_attn-000.pth or _decoder_blocks.4.cross_attn-000.pth
    # the later blocks (5 or 6) do not have the matching feature
    --attention_file _decoder_blocks.4.cross_attn-000.pth
  ```

<p align="center">
  <img src="assets/matching_example.png" alt="Method" width="70%">
</p>

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
