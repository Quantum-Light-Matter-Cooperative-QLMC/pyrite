#!/bin/bash
# #286 wide-window split of the tree reducer on captured flushes.
# Usage: sbatch checks/line_cost_286.sbatch   (from the synced checkout root)
#SBATCH --job-name=wide286
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=00:40:00
#SBATCH --output=wide286-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/wide286/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv sync --extra nvidia --locked -q
py=(uv run --no-sync python -W ignore checks/line_tree_wide_286.py)
"${py[@]}" --energy-kev 300 --electrons 100 --reference --output "$out/wide-300.json"
echo "rc=$?"
"${py[@]}" --energy-kev 29999 --electrons 100 --flushes 2 --reference --output "$out/wide-29999.json"
echo "rc=$?"
