#!/bin/bash
# Issue #275: Geant4 TestEm5 pair-conversion reference, one macro per task.
#SBATCH --job-name=issue275-pairs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=1-12
#SBATCH --output=/home/aamador/.local/opt/pyrite-issue-275/results/%x_%a.log
set -euo pipefail
ROOT=/home/aamador/.local/opt/pyrite-issue-275
MAC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/cases.txt")
cd "$ROOT/build"
PAIR_OUT="$ROOT/results/${MAC%.mac}.txt" \
  "$ROOT/../pyrite-issue-182/bin/micromamba" run -p "$ROOT/../pyrite-issue-182/geant4" \
  ./TestEm5 "$ROOT/$MAC" > "$ROOT/results/${MAC%.mac}.g4log" 2>&1
gzip -f "$ROOT/results/${MAC%.mac}.txt"
