---
title: Gator Demo
emoji: 🚀
sdk: gradio
app_file: app.py
pinned: false
---

# gator
Multiview Jigsaw-Puzzle solving as a pretraining task

# Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Run

```
uv venv && uv sync --all-extras
```


**important** cuRoPE2D does not have backward implemented, so for training use the pytorch version instead.

# Dataset generation

1. Install conda (we recomment using [`miniforge`](https://github.com/conda-forge/miniforge))

2. Create an environment for habitat
```bash
conda create -n habitat python=3.9
conda activate habitat
conda install habitat-sim headless -c conda-forge -c aihabitat
```

```bash
cd gator &&
python3 gator/datasets/habitat_sim/generate_from_metadata_files.py --input_dir /scratch/izar/skorokho/multiview_habitat_metadata/ --output_dir /scratch/izar/skorokho/croco-dataset/

```

# Acknowledgements

We thank the authors of [`croco`](https://github.com/naver/croco) for their amazing work!

