#!/bin/bash
# #522 production DensityMatrixSolver at critical-angle multiples a=4, 4.25 on the joint grid.
#SBATCH --job-name=522-dm-a
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-node
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=36:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/operator/pyrite-522-scan/scratch/dm/%A_%a.log
set -euo pipefail
cd /home/operator/pyrite-522-scan
export PATH=$HOME/.local/bin:$PATH
export PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=453dd745e7b7466c6051d043907e898b71674b1c
cases=(si110_17 si100_17 si100_54 si110_54)
avals=(4 4.25)
id=$SLURM_ARRAY_TASK_ID
case=${cases[$((id % 4))]}; a=${avals[$((id / 4))]}
/usr/bin/time -v uv run --no-sync python checks/kephart_occupation.py \
  --cases "$case" --margins --critical-angle-multiples "$a" \
  --orders 64 --cells 32 --dz 15.625 --out "scratch/dm/run-$case-a$a.json"
