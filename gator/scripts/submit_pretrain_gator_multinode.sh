#!/bin/bash
#SBATCH --job-name=gator_pretrain_multinode
#SBATCH --time=3:00:00
#SBATCH --account=cs-503
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=gator_pretrain_multinode_%j.out
#SBATCH --error=gator_pretrain_multinode_%j.err

cd /home/skorokho/coding/gator/
source .venv/bin/activate
module load gcc cudnn

set -x
cat "$0"

export GPUS_PER_NODE=2
export NNODES="${SLURM_NNODES}"
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=25678
export NCCL_DEBUG=INFO

srun --ntasks-per-node=1 bash -c '
  echo "SLURM_PROCID=${SLURM_PROCID}"
  echo "SLURMD_NODENAME=${SLURMD_NODENAME}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "NNODES=${NNODES}"
  echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "WORLD_SIZE=${WORLD_SIZE}"

  torchrun \
    --nnodes="${NNODES}" \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --node-rank="${SLURM_PROCID}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    gator/scripts/pretrainings_ddp/pretrain_gator.py \
      --devices "${GPUS_PER_NODE}" \
      --num-nodes "${NNODES}" \
      --strategy ddp \
      --num-workers 4 \
      "$@"
' bash "$@"
