#!/bin/bash
#SBATCH --job-name=croco_pretrain
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=croco_pretrain_%j.out
#SBATCH --error=croco_pretrain_%j.err

cd /home/skorokho/coding/gator/
source .venv/bin/activate
module load gcc cudnn

# CroCoNet(enc_embed_dim=192, enc_depth=12, enc_num_heads=3, dec_embed_dim=192, dec_depth=8, dec_num_heads=3, mlp_ratio=4.0, pos_embed='RoPE100')
python3 gator/scripts/pretrainings/pretrain_croco.py "$@"


