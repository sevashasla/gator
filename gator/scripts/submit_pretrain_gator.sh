#!/bin/bash
#SBATCH --job-name=gator_pretrain
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=gator_pretrain_%j.out
#SBATCH --error=gator_pretrain_%j.err

cd /home/skorokho/coding/gator/
source .venv/bin/activate
module load gcc cudnn

python3 gator/scripts/pretrain_gator.py \
    --exp-name $1 \
    --opt_params.batch_size 128 \
    --num-workers 16 \
    --opt-params.max-epoch 50 \
    --opt-params.blr 6e-4
