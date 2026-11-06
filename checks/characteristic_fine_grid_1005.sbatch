#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o char1005fine-%j.log
# #1005 cross-call preparation reuse on a fine line grid: W and Si, 8192 electrons in
# 2048-electron blocks, C1005_LINE_BINS (default 190000) bins, cache vs C1005_NO_CACHE=1.
# usage (from an isolated synced checkout root): sbatch -J c1005-fine-<sha> checks/characteristic_fine_grid_1005.sbatch [OUTDIR]
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda C1005_LINE_BINS=${C1005_LINE_BINS:-190000}
echo "code $(cat CODE_SHA) bins $C1005_LINE_BINS"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
B=(uv run --no-sync python checks/characteristic_scoring_1005.py)
ROOT=${1:-c1005_fine}
for MAT in silicon tungsten; do
  mkdir -p "$ROOT/$MAT"
  echo "== $MAT cache"
  "${B[@]}" run streamed 8192 2048 "$ROOT/$MAT/cache.json" "$MAT" 300 cuda 1e5 1 | tail -1
  echo "== $MAT no cache"
  C1005_NO_CACHE=1 "${B[@]}" run streamed 8192 2048 "$ROOT/$MAT/nocache.json" "$MAT" 300 cuda 1e5 1 | tail -1
  "${B[@]}" compare "$ROOT/$MAT/cache.json" "$ROOT/$MAT/nocache.json"
done
