#!/bin/bash
# Issue #344: run only on the lab's SLURM CUDA partition after `pyrite remote sync`.
# Submit on the box (the remote sync includes checks/, not tests/):
#   sbatch -p gpu --gres=gpu:1 checks/scheduled_cuda_parity.sbatch
#SBATCH --job-name=validation-344
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=validation-344-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# SLURM batch env lacks ~/.pixi/bin (pixi); the box config pins a stale catalog,
# so mirror pyrite.remote.config.remote_runtime_env() for the synced checkout.
export PATH="$HOME/.pixi/bin:$PATH"
export PYRITE_HOME="${PWD}"
export PYRITE_CATALOG="${PWD}/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="${PWD}/user-catalog"
: "${SLURM_JOB_ID:?SLURM allocation required}"
output="evidence/validation-344/${SLURM_JOB_ID}"
# Sync the selected GPU extra on the *box*, not on a GitHub runner.
pixi install --locked -e nvidia
pixi run -e nvidia pyrite tables fetch
pixi run -e nvidia python checks/scheduled_cuda_parity.py --output-dir "$output"
echo "Validation parity evidence: $PWD/$output/report.json"
