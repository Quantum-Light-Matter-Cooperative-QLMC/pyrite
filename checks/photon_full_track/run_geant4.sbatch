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
ROOT=${SLURM_SUBMIT_DIR}
MAC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/cases.txt")
cd "$ROOT/build"
cp "$ROOT/$MAC" .
# Drop the inherited #182 per-photon EBREM creation lines; keep GAM and the summary.
"$MICROMAMBA" run -p "$GEANT4_ENV" ./TestEm5 "$MAC" 2>&1 | grep -v "EBREM " \
  | gzip > "$ROOT/results/${MAC%.mac}.log.gz"
rm "$MAC"
