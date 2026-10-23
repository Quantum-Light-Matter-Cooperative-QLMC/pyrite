#!/bin/bash
# Issue #439 control: W 50/100 MeV with a point-nucleus ELSEPA table. Task 0
# writes the table; tasks 1-20 (run with --dependency) run shard (i-1) % 10 of
# case (i-1) / 10, as run_pyrite.sbatch does. Submit from the staging root.
#SBATCH --job-name=issue439-control
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-20
#SBATCH --output=results/%x_%a.out
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR}
UV=${PYRITE_REMOTE_UV:-$HOME/.local/bin/uv}
TABLE="$ROOT/results/w_point_nucleus_elsepa.npz"
cd "$PYRITE_CHECKOUT"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  "$UV" run --no-sync python "$ROOT/point_nucleus_table.py" "$TABLE"
  exit 0
fi
CASES=(w_50mev w_100mev)
i=$((SLURM_ARRAY_TASK_ID - 1))
CASE=${CASES[$((i / 10))]}
SHARD=$((i % 10))
"$UV" run --no-sync python "$ROOT/pyrite_bench.py" "$CASE" 10000 $((200 * SHARD)) \
  "$ROOT/results/${CASE}_point_${SHARD}.npz" --elastic-npz "$TABLE"
