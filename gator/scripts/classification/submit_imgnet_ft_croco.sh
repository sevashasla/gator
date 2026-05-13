#!/bin/bash
#SBATCH --job-name=gator_imagenet_ft
#SBATCH --time=24:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/gator_ft_%j.out
#SBATCH --error=logs/gator_ft_%j.err

cd /home/mayila/gator/

source .venv/bin/activate

export PYTHONPATH=/home/mayila/gator:$PYTHONPATH
export HF_DATASETS_CACHE=/scratch/izar/mayila/.cache/huggingface

export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false

which python
python --version
nvidia-smi


python3 -m gator.scripts.imagenet_ft_croco  \
    --opt_params.batch_size 256 \
    --opt_params.lr 1e-3 \
    --opt_params.max_epoch 10 \
    --enc_embed_dim 768 \
    --enc_depth 12 \
    --enc_num_heads 12\
    --checkpoint_path /home/mayila/gator/gator/models/CroCo.pth\
    --data_dir /scratch/izar/mayila/imagenet

