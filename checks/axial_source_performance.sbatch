#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-23%2
#SBATCH --output=runs/profile1154-%A-%a.log
# Run from the isolated synced checkout root. GPU comparison submitted separately.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/tmp/pyrite-1154-cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cpu
export PYRITE_HOME=$PWD PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$PWD/user-catalog
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
args=(--count 32 --chunk 8 --repeats 3)
case_index=$((SLURM_ARRAY_TASK_ID % 2))
cases=(A B)
case_key=${cases[$case_index]}
probe=$((SLURM_ARRAY_TASK_ID / 2))
case $probe in
  0) args+=(--profile);;
  1) args+=(--overrides '{"max_order":11}');;
  2) args+=(--overrides '{"outgoing_order_max":24}');;
  3) args+=(--overrides '{"step_ang":7.5}');;
  4) args+=(--overrides '{"step_ang":30}');;
  5) args+=(--overrides '{"harmonic_max":2}');;
  6) args+=(--overrides '{"harmonic_max":6}');;
  7) args+=(--overrides '{"axis_uvw":[1,1,1],"kinetic_energy_eV":2990000,"tilt_deg":0,"thickness_ang":3000}');;
  8) args+=(--overrides '{"axis_uvw":[1,1,1],"kinetic_energy_eV":2990000,"tilt_deg":0,"thickness_ang":11000}');;
  9) args=(--count 64 --chunk 8 --repeats 3 --workers "$((case_index + 1))");;
  10) args=(--count 64 --chunk 8 --repeats 3 --workers "$((4 + 4 * case_index))");;
  11) args=(--count 64 --chunk "$((2 + 14 * case_index))" --repeats 3 --workers 4);;
esac
# Worker/batch comparisons all use exactly B and identical IDs/seed.
if (( probe >= 9 )); then case_key=B; fi
uv run --frozen python checks/axial_source_performance.py --case "$case_key" "${args[@]}" --out "runs/profile-$SLURM_ARRAY_TASK_ID.json"
