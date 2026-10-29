#!/usr/bin/env bash
# Regenerate one CUDA tutorial capture on the lab GPU (#776).
#
# From a clean checkout of the branch on the box (a git clone or `git archive`
# of the commit under test):
#   sbatch checks/tutorial_capture_cuda.sbatch 15-tuning-cuda-execution
# then copy docs/notebooks/*/<name>.ipynb back and commit it with its source.
# Sphinx only warns when a CUDA capture is stale, so this runs whenever a GPU
# is free rather than gating documentation builds.
#SBATCH --job-name=pyrite-tutorial-cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=tutorial-cuda-%j.log
set -euo pipefail
name=${1:?usage: sbatch checks/tutorial_capture_cuda.sbatch <tutorial-name>}
root=${SLURM_SUBMIT_DIR:-$PWD}
cd "$root"
export PATH=$HOME/.local/bin:$PATH  # SLURM batch environments lack uv
export PYRITE_HOME=$root PYRITE_CATALOG=$root/src/pyrite/data/catalog
PYRITE_USER_CATALOG=$(mktemp -d)/catalog
XDG_CONFIG_HOME=$(mktemp -d)  # ignore the box's stale catalog pin
export PYRITE_USER_CATALOG XDG_CONFIG_HOME
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/pyrite-uv-cache}
echo "revision=$(git rev-parse HEAD 2>/dev/null || echo archive) tutorial=$name"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv sync -q --locked --group docs --extra nvidia --extra optimization
uv run --no-sync pyrite tables fetch
uv run --no-sync python docs/_notebooks.py --regenerate "$name"
