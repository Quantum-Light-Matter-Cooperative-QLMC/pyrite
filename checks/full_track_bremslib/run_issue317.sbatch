#!/bin/bash
# Issue #317: one prod_bench.py batch per array task, rows from issue317_tasks.txt.
# Submit from the staged PyRITE checkout root:
#   sbatch --array=0-$(( $(wc -l < checks/full_track_bremslib/issue317_tasks.txt) - 1 )) \
#     checks/full_track_bremslib/run_issue317.sbatch
# PYRITE_HOME must hold the ELSEPA, SBETHE (W at 19.3 g/cm3) and BremsLib tables.
#SBATCH --job-name=i317
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --output=i317-%a-%j.log
set -euo pipefail
D=checks/full_track_bremslib
read -r CASE EL ST IN NE B AT < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$D/issue317_tasks.txt")
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$D/i317_out"
OUT="$D/i317_out/${CASE}_${EL}_${IN}_${AT}_b${B}.json"
[ -s "$OUT" ] && exit 0
"${PYRITE_REMOTE_UV:-$HOME/.local/bin/uv}" run --frozen python "$D/prod_bench.py" \
  "$CASE" "$EL" "$ST" "$IN" "$NE" "$OUT" "$B" "$AT"
