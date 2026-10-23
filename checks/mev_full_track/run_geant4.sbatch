#!/bin/bash
# Issue #439: Geant4 TestEm5 MeV full-track references, one case per task.
# Submit from the staging root holding build/TestEm5, the case macros and
# cases.txt (one macro per line); GEANT4_ENV and MICROMAMBA locate Geant4.
#SBATCH --job-name=issue439-g4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --array=1-10
#SBATCH --output=results/%x_%a.out
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR}
MAC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/cases.txt")
cd "$ROOT/build"
cp "$ROOT/$MAC" .
"$MICROMAMBA" run -p "$GEANT4_ENV" ./TestEm5 "$MAC" > "$ROOT/results/${MAC%.mac}.log" 2>&1
mv "${MAC%.mac}"*.csv "$ROOT/results/"
gzip -f "$ROOT/results/${MAC%.mac}.log"
