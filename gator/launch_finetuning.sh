#!/bin/bash
#SBATCH --job-name=croco_ft
#SBATCH --time=48:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Finetune CroCo on stereo or flow via torchrun on a single multi-GPU node.
#
# Usage:
#   sbatch submit_train.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# -----------------------------------------------------------------------------
# User-configurable paths / hyperparameters
# -----------------------------------------------------------------------------
TASK="stereo"                                   # "stereo" or "flow"
NUM_GPUS=1                                      # must match --gres=gpu:N above

OUTPUT_DIR="./checkpoints/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
PRETRAINED="./pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth"
DATASET="ETH3DLowRes(split='train')"
VAL_DATASET="ETH3DLowRes(split='subval')"

BATCH_SIZE=1
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
    --output_dir  "${OUTPUT_DIR}" \
    --pretrained  "${PRETRAINED}" \
    --dataset     "${DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --batch_size  ${BATCH_SIZE} \
    --epochs      ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --accum_iter  ${ACCUM_ITER} \
    --amp 1