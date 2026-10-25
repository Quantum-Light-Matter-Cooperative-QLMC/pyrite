#!/bin/bash
# Issue #341: PyRITE side of the thick-target photon check. Task i runs shard
# (i-1) % 20 (1,000 primaries, batches 20*shard onward) of case (i-1) / 20 in
# the fixed order below. Submit from the staging root; PYRITE_CHECKOUT is a
# staged checkout with its uv environment and PYRITE_HOME its data root.
#SBATCH --job-name=issue341-pyrite
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=1-160
#SBATCH --output=results/%x_%a.out
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR}
CASES=(si_0p3mev si_1mev si_3mev si_10mev w_0p3mev w_1mev w_3mev w_10mev)
i=$((SLURM_ARRAY_TASK_ID - 1))
CASE=${CASES[$((i / 20))]}
SHARD=$((i % 20))
export PYRITE_HOME NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$ROOT/numba-cache}
cd "$PYRITE_CHECKOUT"
.venv/bin/python checks/photon_full_track/pyrite_bench.py \
  "$CASE" 1000 $((20 * SHARD)) "$ROOT/results/${CASE}_pyrite_${SHARD}.npz"
