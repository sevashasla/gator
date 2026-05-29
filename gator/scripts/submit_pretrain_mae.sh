#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=mae_pretrain_%j.out
#SBATCH --error=mae_pretrain_%j.err

cd /home/skorokho/coding/gator/
source .venv/bin/activate
module load gcc cudnn

python3 gator/scripts/pretrainings/pretrain_mae.py "$@"
