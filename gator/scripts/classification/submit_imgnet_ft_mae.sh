#!/bin/bash
#SBATCH --job-name=mae_imagenet_ft
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

python3 -c "
content = open('/home/mayila/gator/gator/models/pos_embed_mae.py').read()
content = content.replace('np.float)', 'np.float64)')
content = content.replace('np.float,', 'np.float64,')
open('/home/mayila/gator/gator/models/pos_embed_mae.py', 'w').write(content)
print('numpy fix applied')
"

python3 -m gator.scripts.classification.imagenet_ft_mae \
    --checkpoint_path /scratch/izar/skorokho/gator/final-ckpts/mae-small-000-24hrs.ckpt \
    --data_dir /scratch/izar/mayila/imagenet_full\
    --embed_dim 768 \
    --depth 12 \
    --num-heads 12 \
    --batch_size 512 \
    --max_epochs 10 \
    --num_workers 6 \
    --lr 1e-3

