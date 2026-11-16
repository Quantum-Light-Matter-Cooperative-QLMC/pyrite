#!/bin/bash
#SBATCH -p gpu-amd
#SBATCH --nodelist=qlmc-ace
#SBATCH --gres=gpu:radeon8060s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o cone203-policy-%j.log
# #203 cone half-angle scan (k/gamma): measurement only, no ledger claim.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-amd"
export OPENBLAS_NUM_THREADS=1
export CUPY_INSTALL_USE_HIP=1
source src/pyrite/remote/rocm_build_env.sh
_pyrite_rocm_env "$PWD"
export PYRITE_MC_BACKEND=rocm
echo "code $(cat CODE_SHA)"
uv sync --locked --no-default-groups --extra amd
OUT=${1:-cone203_policy_out}
mkdir -p "$OUT"
uv run --no-sync python checks/detector_cone_policy.py \
  --ne 512 --seeds 0 1 2 3 4 --repeats 1 "${@:2}" \
  --out "$OUT/policy.json"
