#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 00:30:00
#SBATCH -o cone203-diagnostic-%j.log
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1
echo "code $(cat CODE_SHA)"
uv sync --locked --no-default-groups --extra nvidia
uv run --no-sync python checks/detector_cone_diagnostic.py --out cone203-diagnostic.json
