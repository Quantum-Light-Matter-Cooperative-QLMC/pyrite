#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-node
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=36:00:00
set -euo pipefail
cd "$HOME/pyrite-522-scan"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=453dd745e7b7466c6051d043907e898b71674b1c
cases=(si110_17 si100_17 si100_54 si110_54)
case=${cases[$SLURM_ARRAY_TASK_ID]}
/usr/bin/time -v uv run --no-sync python checks/kephart_occupation.py \
 --cases "$case" --margins --critical-angle-multiples 4.125 \
 --orders 64 --cells 32 --dz 15.625 --out "scratch/dm/run-$case-a4.125.json"
