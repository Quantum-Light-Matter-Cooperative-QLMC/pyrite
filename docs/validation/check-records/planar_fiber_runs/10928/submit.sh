#!/bin/bash
# #1056 planar fibre study: Kephart configurations, absorbing (300 eV) and P = R arms.
#SBATCH --job-name=1056-planar-fibre
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --array=0-159%48
#SBATCH --output=/home/aamador/pyrite-1056/scratch/runs/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-1056
export PATH=$HOME/.local/bin:$PATH
export PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=7cd7d29e7eab15101149735959ad30c56d76c5ac
cases=(si110_17 si100_17 si100_54 si110_54)
arms=(absorbing reference)
id=$SLURM_ARRAY_TASK_ID
case=${cases[$((id / 40))]}
arm=${arms[$(((id / 20) % 2))]}
start=$(((id % 20) * 100))
/usr/bin/time -v uv run --no-sync python checks/planar_fiber_study.py run "$case" "$arm" "$start" 100 \
  --out "scratch/runs/$case-$arm-$start.npz"
