#!/bin/bash
#SBATCH --job-name=relpose-croco-lr1e4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_relpose_croco_lr1e4_%j.out

mkdir -p logs
cd /home/bosi/gator
source .venv/bin/activate

python3 gator/scripts/relpose/finetune_croco.py \
    --inner-model-ckpt-path /scratch/izar/skorokho/gator/final-ckpts/croco-small-lr-1e-4-24hrs.ckpt \
    --model_configuration "CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')" \
    --exp-name croco-small-lr-1e-4-24hrs-nf \
    --opt-params.blr 1e-6 --no-freeze-encdec \
    --output-dir /scratch/izar/bosi/gator/relpose/
