#!/bin/bash
#SBATCH --job-name=croco_ft
#SBATCH --time=24:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Finetune CroCo or Gator on stereo or flow via torchrun on a single multi-GPU node.
#
# Usage:
#   sbatch submit_train.sh
#
# Set MODEL="croco" to finetune a CroCo checkpoint (.pth)
# Set MODEL="gator"  to finetune a Gator Lightning checkpoint (.ckpt)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# -----------------------------------------------------------------------------
# User-configurable paths / hyperparameters
# -----------------------------------------------------------------------------
MODEL="croco"                                   # "croco" or "gator"
TASK="flow"                                     # "stereo" or "flow"
NUM_GPUS=2                                      # must match --gres=gpu:N above
CRITERION="LaplacianLossBounded2()"

OUTPUT_DIR="./checkpoints/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# CroCo pretrained checkpoint (.pth):
PRETRAINED="./pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth"
# Gator pretrained checkpoint (Lightning .ckpt) — uncomment and set MODEL="gator":
# PRETRAINED="./pretrained_models/gator/checkpoints/last.ckpt"

# STEREO
# DATASET="Kitti12('train')ETH3DLowRes(split='train')"
# VAL_DATASET="ETH3DLowRes(split='subval')"

# FLOW
DATASET="40*MPISintel('subtrain_cleanpass')+40*MPISintel('subtrain_finalpass')"
VAL_DATASET="MPISintel('subval_cleanpass')+MPISintel('subval_finalpass')"

BATCH_SIZE=8
EPOCHS=32
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
    --model       "${MODEL}" \
    --criterion   "${CRITERION}" \
    --output_dir  "${OUTPUT_DIR}" \
    --pretrained  "${PRETRAINED}" \
    --dataset     "${DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --batch_size  ${BATCH_SIZE} \
    --epochs      ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --accum_iter  ${ACCUM_ITER} \
    --amp 1