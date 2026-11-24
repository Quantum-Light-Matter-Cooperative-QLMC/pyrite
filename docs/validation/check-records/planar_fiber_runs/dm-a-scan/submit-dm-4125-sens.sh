#!/bin/bash
# #522 a = 4.125 production sensitivity, Si(100) 17 MeV: task 0 dz halved, task 1 H = 80.
#SBATCH --job-name=522-dm-sens
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=36:00:00
#SBATCH --array=0-1
set -euo pipefail
cd "$HOME/pyrite-522-scan"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=453dd745e7b7466c6051d043907e898b71674b1c
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then tag=dz7.8125; args=(--orders 64 --cells 32 --dz 7.8125)
else tag=H80; args=(--orders 80 --cells 32 --dz 15.625); fi
/usr/bin/time -v uv run --no-sync python checks/kephart_occupation.py \
 --cases si100_17 --margins --critical-angle-multiples 4.125 "${args[@]}" \
 --out "scratch/dm/run-si100_17-a4.125-$tag.json"
