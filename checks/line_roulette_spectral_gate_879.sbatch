#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o scratch/roulette879/spectral-%j.log
# Submit only after the #938 scorer phase has completed and the supervisor
# releases this gate. Run from a clean isolated checkout at the pushed SHA.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/scratch/roulette879/empty-user-catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda
export PYRITE_TEST_BACKEND=cuda
export PYRITE_FP64=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$PYRITE_USER_CATALOG"
git check-ignore scratch/roulette879/probe
git rev-parse HEAD
nvidia-smi
# pyrite-dev imports YAML through its developer tooling; the test-only group
# omits that dependency. Keep the project runner available in a fresh checkout.
uv sync -q --locked --no-default-groups --extra nvidia --group dev
uv run --no-sync pyrite-dev test tests/performance/test_line_roulette_gate_879.py -s --durations=0
uv run --no-sync python checks/line_roulette_spectral_gate_879.py \
    --out "${1:-scratch/roulette879/full-spectrum}" \
    --ne 512 --pilot 128 --seed 0 --core per-electron \
    --max-nodes 600000 --max-array-gib 3 --capture-minutes 40
