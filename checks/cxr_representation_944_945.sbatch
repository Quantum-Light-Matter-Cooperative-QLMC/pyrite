#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o scratch/repr944/job-%j.log
# #944/#945: replay saved #286 flush captures (copied to scratch/repr944/flushes/<set>/)
# through the production-resolved bin-mean reduction (Fourier on uniform axes,
# near/far elsewhere) and forced near/far, on uniform and saved adaptive axes.
# Submit from a synced checkout root with -p gpu or -p gpu-amd; arguments are
# the mode (`capture` or `replay`) and its flags for checks/cxr_representation_944_945.py.
set -euo pipefail
export PATH="/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export PYRITE_USER_CATALOG="$PWD/scratch/repr944/empty-user-catalog"
export XDG_CONFIG_HOME="$PWD/scratch/repr944/xdg"
export PYRITE_FP64=0 OPENBLAS_NUM_THREADS=1
mkdir -p "$PYRITE_USER_CATALOG" "$XDG_CONFIG_HOME"
echo "code $(cat CODE_SHA 2>/dev/null || git rev-parse HEAD)"
if [ "$PYRITE_GPU_VENDOR" = nvidia ]; then
  nvidia-smi --query-gpu=name,driver_version --format=csv
fi
pyrite_gpu_sync
if [ -n "${CHECK_SCRIPT:-}" ]; then
  # e.g. CHECK_SCRIPT=checks/cxr_bin_mean_profile_1003.py for its `wall` mode
  pixi run --as-is -e "$PYRITE_PIXI_ENV" python "$CHECK_SCRIPT" "$@"
else
  pixi run --as-is -e "$PYRITE_PIXI_ENV" python -m checks.cxr_representation_944_945 "$@"
fi
echo DONE
