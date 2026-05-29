#!/bin/bash
#SBATCH --job-name=jigsaw_imagenet_ft
#SBATCH --time=30:00:00
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

python -m gator.scripts.classification.imagenet_ft_jigsaw \
    --checkpoint_path /scratch/izar/skorokho/gator/final-ckpts/jigsaw-small-000-12hrs.ckpt\
    --data_dir /scratch/izar/mayila/imagenet_full \
    --model_name deit_small_patch16_224 \
    --batch_size 512 \
    --max_epochs 50 \
    --lr 1e-3
