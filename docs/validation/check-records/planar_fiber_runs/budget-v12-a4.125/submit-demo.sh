#!/bin/bash
# #522 v12 (a=4.125): channeling_demo paired-arm budget at the production 4.125 psi_c margin (340.19 eV).
#SBATCH --job-name=522-demo-budget
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=1
#SBATCH --mem=600M
#SBATCH --time=4:00:00
#SBATCH --array=0-199%8
#SBATCH --output=/home/aamador/pyrite-522-v12/scratch/budget/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-522-v12
export PATH=$HOME/.local/bin:$PATH
export PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=$(cat SOURCE_SHA)
start=$((SLURM_ARRAY_TASK_ID * 100))
/usr/bin/time -v uv run --no-sync python checks/planar_fiber_budget.py run demo_17 "$start" \
  --out-dir scratch/budget
