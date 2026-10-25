#!/bin/bash
# Issue #341 emission-angle diagnostic: task i runs shard (i-1) % 10 (1,000
# primaries, batches 20*shard onward) of case (i-1) / 10. Environment as in
# run_pyrite.sbatch.
#SBATCH --job-name=issue341-angles
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=1-30
#SBATCH --output=angles/%x_%a.out
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR}
CASES=(si_1mev si_3mev w_1mev)
i=$((SLURM_ARRAY_TASK_ID - 1))
CASE=${CASES[$((i / 10))]}
SHARD=$((i % 10))
export PYRITE_HOME NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$ROOT/numba-cache}
cd "$PYRITE_CHECKOUT/checks/photon_full_track"
"$PYRITE_CHECKOUT/.venv/bin/python" emission_angles.py pyrite \
  "$CASE" 1000 $((20 * SHARD)) "$ROOT/angles/${CASE}_angles_${SHARD}.npz"
