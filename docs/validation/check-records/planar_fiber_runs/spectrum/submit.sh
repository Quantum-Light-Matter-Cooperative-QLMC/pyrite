#!/bin/bash
# #1056 planar two-time source vs secular line model: 3 cases x (40 control + 10 fine) chunks of 100 primaries.
#SBATCH --job-name=1056-planar-2t
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --array=0-149%48
#SBATCH --output=/home/aamador/pyrite-1056-spectrum/scratch/spectrum/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-1056-spectrum
export PATH=$HOME/.local/bin:$PATH
export PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=8179040b0947be97ad57aa998f8d6e38d57a2e2a
cases=(demo_17 si100_17 si110_5)
id=$SLURM_ARRAY_TASK_ID
case=${cases[$((id / 50))]}
slot=$((id % 50))
if [ "$slot" -lt 40 ]; then arm=control; start=$((slot * 100)); else arm=fine; start=$(((slot - 40) * 100)); fi
/usr/bin/time -v uv run --no-sync python checks/planar_fiber_spectrum.py run "$case" "$start" 100 \
  --arm "$arm" --out "scratch/spectrum/$case-$arm-$start.npz"
