#!/bin/bash
# Issue #439: PyRITE side of the MeV full-track benchmark. Task i runs shard
# (i-1) % 10 (10,000 primaries, batches 200*shard onward) of case (i-1) / 10 in
# the fixed order below. Submit from the staging root; PYRITE_CHECKOUT is a
# synced checkout with its uv environment, PYRITE_REMOTE_UV the uv binary.
#SBATCH --job-name=issue439-pyrite
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-100
#SBATCH --output=results/%x_%a.out
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR}
CASES=(w_3mev w_10mev w_30mev w_50mev w_100mev si_3mev si_10mev si_30mev si_50mev si_100mev)
i=$((SLURM_ARRAY_TASK_ID - 1))
CASE=${CASES[$((i / 10))]}
SHARD=$((i % 10))
cd "$PYRITE_CHECKOUT"
"${PYRITE_REMOTE_UV:-$HOME/.local/bin/uv}" run --no-sync python "$ROOT/pyrite_bench.py" \
  "$CASE" 10000 $((200 * SHARD)) "$ROOT/results/${CASE}_pyrite_${SHARD}.npz"
