#!/bin/bash
# #286 light line-phase attribution: HOPG 1 mm 45/180, 300 keV ladder + 29,999 keV probe.
# Usage: sbatch checks/line_cost_286.sbatch   (from the synced checkout root)
#SBATCH --job-name=line286
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=00:15:00
#SBATCH --output=line286-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/line286/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv sync --extra nvidia --locked -q
py=(uv run --no-sync python -W ignore checks/line_cost_286.py)
"${py[@]}" --energy-kev 300 --counts 400 1600 6400 --repeat 1 --output "$out/hopg-300.json"
echo "rc=$?"
"${py[@]}" --energy-kev 29999 --counts 100 --repeat 1 --output "$out/hopg-29999.json"
echo "rc=$?"
