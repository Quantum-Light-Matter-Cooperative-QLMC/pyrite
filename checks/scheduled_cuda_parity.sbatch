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
export PYRITE_CATALOG="${PWD}/src/pyrite/data/catalog"
: "${SLURM_JOB_ID:?SLURM allocation required}"
output="evidence/validation-344/${SLURM_JOB_ID}"
# Sync the selected GPU extra on the *box*, not on a GitHub runner.
uv sync --extra nvidia --locked
uv run --extra nvidia pyrite tables fetch
uv run --extra nvidia python checks/scheduled_cuda_parity.py --output-dir "$output"
echo "Validation parity evidence: $PWD/$output/report.json"
