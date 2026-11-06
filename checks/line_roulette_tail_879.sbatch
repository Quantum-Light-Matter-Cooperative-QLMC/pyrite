#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o scratch/roulette879/tail-%j.log
# Run from a clean isolated checkout at the pushed SHA. First argument is the
# output root; the rest pass verbatim to checks/line_roulette_tail_879.py.
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
sha="$(git rev-parse HEAD)"
nvidia-smi
uv sync -q --locked --no-default-groups --extra nvidia --group dev
output_root="$1"; shift
uv run --no-sync python checks/line_roulette_tail_879.py --out "$output_root" --code-sha "$sha" "$@"
