#!/bin/bash
# #286 pairs-vs-tree reducer crossover replay (HOPG 1 mm 45/180, 100 e-).
#SBATCH --job-name=xover286
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=00:50:00
#SBATCH --output=xover286-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export PYTHONPATH="$PWD" UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/xover286/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv sync --extra nvidia --locked -q
py=(uv run --no-sync python -W ignore checks/line_crossover_286.py)
"${py[@]}" --energy-kev 300 --strides 1 4 16 --output "$out/x-300.json"
echo "rc=$?"
"${py[@]}" --energy-kev 29999 --strides 1 4 16 64 --output "$out/x-29999.json"
echo "rc=$?"
