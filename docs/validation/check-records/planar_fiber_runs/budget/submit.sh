#!/bin/bash
# #1056 planar paired-arm budgets: Kephart configurations, 7 arms x 40 chunks of 100 incidents.
#SBATCH --job-name=1056-planar-budget
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --array=0-159%48
#SBATCH --output=/home/aamador/pyrite-1056-budget/scratch/budget/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-1056-budget
export PATH=$HOME/.local/bin:$PATH
export PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=80393f474c6770e7bf8e87b3bf97aad4eb386f3b
cases=(si110_17 si100_17 si100_54 si110_54)
id=$SLURM_ARRAY_TASK_ID
case=${cases[$((id / 40))]}
start=$(((id % 40) * 100))
/usr/bin/time -v uv run --no-sync python checks/planar_fiber_budget.py run "$case" "$start" \
  --out-dir scratch/budget
