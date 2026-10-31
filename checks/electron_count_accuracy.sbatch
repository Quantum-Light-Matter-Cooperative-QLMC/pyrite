#!/bin/bash
# #278 plan item 5: accuracy vs electron count, Ne (lines) and Ne_brem
# (continuum), on the production default grids. HOPG 1 mm at 45/180 deg,
# 300 keV and 29,999 keV; 5 replicas at 100/400/1600 and one 6400 reference.
# Resumable: resubmit to continue from the checkpoint JSON.
#SBATCH --job-name=eca278
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=08:00:00
#SBATCH --output=eca278-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG" evidence/eca278
uv sync --extra nvidia --locked -q
cp .pyrite-sync "evidence/eca278/sync-$SLURM_JOB_ID.json"
uv run --no-sync python -W ignore checks/electron_count_accuracy.py \
    --energy-kev 300 --energy-kev 29999 --max-hours 7.5 \
    --output evidence/eca278/hopg.json > "evidence/eca278/hopg-$SLURM_JOB_ID.log" 2>&1
