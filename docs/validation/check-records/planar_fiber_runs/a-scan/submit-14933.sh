#!/bin/bash
# #522 critical-angle-multiple cut scan on the planar fibre engine (absorbing arm).
# margin = (a^2 - 1) D, D the well depth; seed 0, incidents paired with array 10928.
#SBATCH --job-name=522-a-scan2
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --array=0-239%8
#SBATCH --output=/home/aamador/pyrite-522-scan/scratch/scan/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-522-scan
export PATH=$HOME/.local/bin:$PATH
export PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=453dd745e7b7466c6051d043907e898b71674b1c
cases=(si110_17 si100_17 si100_54 si110_54)
depth=(21.040067102356762 11.657335114705013 11.657335114705013 21.040067102356762)
avals=(4 4.25 4.25)
id=$SLURM_ARRAY_TASK_ID
c=$((id / 60)); j=$(((id / 20) % 3)); a=${avals[$j]}; start=$(( (j == 1 ? 0 : 2000) + (id % 20) * 100 ))
case=${cases[$c]}
margin=$(python3 -c "print(($a**2 - 1) * ${depth[$c]})")
/usr/bin/time -v uv run --no-sync python checks/planar_fiber_study.py run "$case" absorbing "$start" 100 \
  --margin "$margin" --out "scratch/scan/$case-a$a-$start.npz"
