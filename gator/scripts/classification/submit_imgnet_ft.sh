#!/bin/bash
#SBATCH --job-name=gator_v2_imagenet_ft
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

#python3 -m gator.scripts.classification.imagenet_ft_gator \
#    --checkpoint-path /scratch/izar/skorokho/gator/exp-gator-005/checkpoints/last.ckpt \
#    --data-dir /scratch/izar/mayila/imagenet \
#    --num-workers 4 \
#    --gator-config.enc-emb-dim 192 \
#    --gator-config.enc-depth 12 \
#    --gator-config.enc-num-heads 3 \
#    --opt-params.batch-size 128 \
#    --opt-params.max-epoch 20 \
#    --opt-params.blr 3e-4

python3 -m gator.scripts.classification.imgnet_ft_gator  \
    --gator_config.enc_emb_dim 384 \
    --gator_config.enc_num_heads 3 \
    --gator_config.enc_depth 12 \
    --gator_config.dec_emb_dim 384 \
    --gator_config.dec_num_heads 3 \
    --opt_params.batch_size 512 \
    --opt_params.lr 1e-3 \
    --opt_params.max_epoch 25 \
    --checkpoint_path /scratch/izar/skorokho/gator/final-ckpts/gator-small-disba-000-24hrs.ckpt\
    --data_dir /scratch/izar/mayila/imagenet_full