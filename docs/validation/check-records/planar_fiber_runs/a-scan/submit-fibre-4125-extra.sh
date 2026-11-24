#!/bin/bash
# #522 a = 4.125 fibre Si(100) 54.5 MeV: incidents 4000-7900 (second 4000), same margin and seed.
#SBATCH --job-name=522-fibre-4125x
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=36:00:00
#SBATCH --array=0-39%6
set -euo pipefail
cd "$HOME/pyrite-522-scan"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=453dd745e7b7466c6051d043907e898b71674b1c
case=si100_54; start=$((4000 + SLURM_ARRAY_TASK_ID * 100))
margin=$(uv run --no-sync python -c "from pyrite.materials.channeling import planar_profile; lo,hi=planar_profile('silicon',(1,0,0),293.15).extrema(); print((4.125**2-1)*(hi-lo))")
/usr/bin/time -v uv run --no-sync python checks/planar_fiber_study.py run "$case" absorbing "$start" 100 \
 --margin "$margin" --out "scratch/scan/$case-a4.125-$start.npz"
