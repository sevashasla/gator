#!/bin/bash
#SBATCH --job-name=download_imagenet
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err


cd /home/mayila/gator/

source ~/.bashrc
conda activate nanovlm


export HF_DATASETS_CACHE=/scratch/izar/mayila/.cache/huggingface

python3 gator/scripts/load_imagenet.py \
    --output_dir /scratch/izar/mayila/imagenet \
    --splits train \
#    --max_train_samples 700000 \
    --shuffle_buffer 10000