#!/bin/bash
#SBATCH --job-name=gator-small-001-24hrs
#SBATCH --time=16:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Finetune CroCo, Gator, MAE or one-view-jigsaw on stereo or flow via torchrun on a single multi-GPU node.
#
# Usage:
#   sbatch launch_finetuning.sh
#

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# -----------------------------------------------------------------------------
# User-configurable paths / hyperparameters
# -----------------------------------------------------------------------------
MODEL="gator"                                   # "croco", "gator", "mae", or "jigsaw_1view"
# Required when MODEL="croco" and the checkpoint is a Lightning .ckpt (architecture is not stored in it).
# Ignored for official CroCo .pth files (architecture is auto-detected).
CROCO_CONFIG="CroCoNet(enc_embed_dim=384, enc_depth=12, enc_num_heads=6, dec_embed_dim=384, dec_depth=8, dec_num_heads=6, mlp_ratio=4.0, pos_embed='RoPE100')"
# Overrides for GatorConfig defaults (empty string = tiny/192-dim defaults).
# Set to match the pretrained checkpoint's architecture.
GATOR_CONFIG="enc_emb_dim=384,dec_emb_dim=384,enc_num_heads=6,dec_num_heads=6"
TASK="flow"                                     # "stereo" or "flow"
NUM_GPUS=2                                     # must match --gres=gpu:N above
CRITERION="LaplacianLossBounded()"

OUTPUT_DIR="./checkpoints/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# CroCo pretrained checkpoint (.pth):
# PRETRAINED="./pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth"
# Gator pretrained checkpoint (Lightning .ckpt) — uncomment and set MODEL="gator":
PRETRAINED="/scratch/izar/skorokho/gator/final-ckpts/gator-small-classification-000-24hrs.ckpt"

# STEREO
# DATASET="Kitti12('train')+Kitti15('train')+30*ETH3DLowRes(split='train')+50*Md14('train')+50*Md21('train')+Booster('train_balanced')"
# VAL_DATASET="ETH3DLowRes(split='subval')+Md14('subval')+Boo ter('subval_balanced')+Md21('subval')"

# FLOW
DATASET="40*MPISintel('subtrain_cleanpass')+40*MPISintel('subtrain_finalpass')+4*FlyingChairs('train')"
VAL_DATASET="MPISintel('subval_cleanpass')+MPISintel('subval_finalpass')"

BATCH_SIZE=16
EPOCHS=50
NUM_WORKERS=8
ACCUM_ITER=1

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Conda activation (same pattern as the reference script)
# -----------------------------------------------------------------------------
# shellcheck disable=SC1091
# if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
#   source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
#   source "${HOME}/anaconda3/etc/profile.d/conda.sh"
# else
#   eval "$(conda shell.bash hook 2>/dev/null)" || true
# fi

# conda activate gator    # <-- change to your env name
source ../.venv/bin/activate

# -----------------------------------------------------------------------------
# Runtime environment
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1                         # avoid thread oversubscription
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO                         # uncomment for NCCL troubleshooting

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "NUM_GPUS=${NUM_GPUS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Launch — torchrun spawns one worker per GPU on this node
# -----------------------------------------------------------------------------
torchrun --standalone --nproc_per_node=${NUM_GPUS} stereoflow/train.py "${TASK}" \
    --model        "${MODEL}" \
    --croco_config "${CROCO_CONFIG}" \
    --gator_config "${GATOR_CONFIG}" \
    --criterion   "${CRITERION}" \
    --output_dir  "${OUTPUT_DIR}" \
    --pretrained  "${PRETRAINED}" \
    --dataset     "${DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --batch_size  ${BATCH_SIZE} \
    --epochs      ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --accum_iter  ${ACCUM_ITER} \
    --img_per_epoch 30000 \
    --amp 1 \
    --wandb 1 \
    --wandb_project gator-stereoflow \
    --wandb_name gator_flow_classification
