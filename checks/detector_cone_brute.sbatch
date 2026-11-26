#!/bin/bash
#SBATCH -p gpu-amd
#SBATCH --nodelist=qlmc-ace
#SBATCH --gres=gpu:radeon8060s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 07:00:00
#SBATCH -o cone203-brute-%j.log
# #203 brute-force analog baseline: production timing + bounded analog campaign.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export OPENBLAS_NUM_THREADS=1
export CUPY_INSTALL_USE_HIP=1
source src/pyrite/remote/rocm_build_env.sh
_pyrite_rocm_env "$PWD"
export PYRITE_MC_BACKEND=rocm
echo "code $(cat CODE_SHA)"
pixi install --locked -e amd
OUT=${1:-cone203_brute_out}
BUDGET=${2:-300}
mkdir -p "$OUT"
B=(pixi run --as-is -e amd python checks/detector_cone_variance_reduction.py)
# Production transport + monitor + full spectrum timing on one block.
"${B[@]}" baseline --ne 2048 --ne-cpu 0 --seed 1000 --out "$OUT/baseline.json"
# Bounded analog campaign: independent 2048-electron seed blocks.
"${B[@]}" brute --block 2048 --budget-minutes "$BUDGET" --seed 0 --out "$OUT/brute.json"
