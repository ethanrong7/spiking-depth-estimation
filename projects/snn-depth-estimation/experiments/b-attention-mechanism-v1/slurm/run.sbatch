#!/bin/bash
#SBATCH --job-name=b-mechanism
#SBATCH --partition=debug
#SBATCH --nodelist=cccl3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${STEREO_SOURCE:?}" "${STEREO_RUN_ROOT:?}" "${STEREO_ENV:?}" "${STEREO_DATA:?}"
: "${MECHANISM_SEED:?}" "${MECHANISM_VARIANT:?}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
experiments="$STEREO_SOURCE/projects/snn-depth-estimation/experiments"
resume=()
if [[ "${MECHANISM_RESUME:-0}" == 1 ]]; then resume=(--resume); fi
exec "$STEREO_ENV/bin/python" \
  "$experiments/conventional-attention-placement-v1/slurm/allocation_guard.py" \
  "$STEREO_ENV/bin/python" -u "$experiments/b-attention-mechanism-v1/run_on_gpu.py" \
  --seed "$MECHANISM_SEED" --variant "$MECHANISM_VARIANT" "${resume[@]}"
