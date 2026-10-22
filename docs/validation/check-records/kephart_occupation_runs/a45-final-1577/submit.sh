#!/bin/bash
# Combine window, Bloch and depth refinements for all six targets.
#SBATCH --job-name=522-a45-final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=36:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/aamador/pyrite-522/scratch/boundary-a45-final/%A_%a.log
set -euo pipefail
cd /home/aamador/pyrite-522
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_SOURCE_REVISION=54b29d7c
echo '62acbe4ce96018352cdda10ea0eb0a9f7354f1c7a45855aa2ecc0e9b148c6a47  checks/kephart_occupation.py' | sha256sum --check
cases=(si110_17 si100_17 si100_54 si110_54)
id=$SLURM_ARRAY_TASK_ID
case_index=$((id % 4))
H=64; cells=32; dz=15.625
/usr/bin/time -v ~/.local/bin/uv run --no-default-groups python checks/kephart_occupation.py \
  --cases "${cases[$case_index]}" --margins --critical-angle-multiples 4.5 \
  --orders "$H" --cells "$cells" --dz "$dz" \
  --out "scratch/boundary-a45-final/run-$id.json"
