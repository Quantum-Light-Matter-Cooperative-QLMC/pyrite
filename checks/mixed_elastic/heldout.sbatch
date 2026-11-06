#!/bin/bash
# Frozen #894 design. Resubmit until it reports PENDING 0.
# Usage: sbatch heldout.sbatch <protocol.json> <evidence dir>
#SBATCH --job-name=894-heldout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=heldout-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH" PYRITE_HOME="$HOME/pyrite" PYRITE_MC_BACKEND=cuda
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export XDG_CONFIG_HOME="$(mktemp -d)" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv sync --locked --package pyrite-mc --no-default-groups --extra nvidia -q || exit 1
.venv/bin/python checks/mixed_elastic/run_arms.py "$1" --out "$2" --stop-after-minutes 60
