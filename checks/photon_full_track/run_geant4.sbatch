#!/bin/bash
# Issue #341: Geant4 TestEm5 thick-target photon references, one macro per task.
# Submit from the staging root holding build/TestEm5, the case macros and
# cases.txt (one macro per line); GEANT4_ENV and MICROMAMBA locate Geant4.
#SBATCH --job-name=issue341-g4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --array=1-59
#SBATCH --output=results/%x_%a.out
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
ROOT=${SLURM_SUBMIT_DIR}
MAC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/cases.txt")
test -n "$MAC"
test -f "$ROOT/$MAC"
TESTEM5=${TESTEM5:-$ROOT/build/TestEm5}
mkdir -p "$ROOT/results/task-$SLURM_ARRAY_TASK_ID"
cd "$ROOT/results/task-$SLURM_ARRAY_TASK_ID"
# Retain photon records and complete-run/configuration headers; omit unrelated
# primary-electron exits and bremsstrahlung creation records. No shared build writes.
/usr/bin/time -v -o "$ROOT/results/${MAC%.mac}.resources" \
  "$MICROMAMBA" run -p "$GEANT4_ENV" "$TESTEM5" "$ROOT/$MAC" 2>&1 \
  | awk 'index($0, "EBREM ") == 0 && index($0, "EXIT ") == 0' \
  | gzip > "$ROOT/results/${MAC%.mac}.log.gz.partial"
mv "$ROOT/results/${MAC%.mac}.log.gz.partial" "$ROOT/results/${MAC%.mac}.log.gz"
