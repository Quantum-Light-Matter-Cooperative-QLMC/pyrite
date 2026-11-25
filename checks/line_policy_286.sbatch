#!/bin/bash
#SBATCH --job-name=policy286
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --output=policy286-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG" evidence/policy286
pixi install --locked -e nvidia
cp .pyrite-sync "evidence/policy286/sync-$SLURM_JOB_ID.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
pixi run --as-is -e nvidia python -W default -m checks.line_policy_286 \
    --output "evidence/policy286/policy-$SLURM_JOB_ID.json" "$@"
