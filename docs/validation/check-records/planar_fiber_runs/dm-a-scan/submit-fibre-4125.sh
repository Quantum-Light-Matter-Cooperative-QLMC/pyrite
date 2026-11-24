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
id=$SLURM_ARRAY_TASK_ID
case=${cases[$((id / 40))]}; start=$(((id % 40) * 100))
# Same bonding-charge profile depth as DM; resolve rather than hardcode it.
margin=$(uv run --no-sync python -c "from pyrite.materials.channeling import planar_profile; h=(1,0,0) if '$case'.startswith('si100') else (1,1,0); lo,hi=planar_profile('silicon',h,293.15).extrema(); print((4.125**2-1)*(hi-lo))")
/usr/bin/time -v uv run --no-sync python checks/planar_fiber_study.py run "$case" absorbing "$start" 100 \
 --margin "$margin" --out "scratch/scan/$case-a4.125-$start.npz"
