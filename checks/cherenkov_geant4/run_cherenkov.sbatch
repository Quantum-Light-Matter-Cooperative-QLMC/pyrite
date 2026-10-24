#!/bin/bash
# Issue #587: Geant4 G4Cerenkov reference, one case per array task.
#SBATCH --job-name=issue587-cherenkov
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --array=1-4
#SBATCH --output=/home/aamador/.local/opt/pyrite-issue-587/results/%x_%a.log
set -euo pipefail
ROOT=/home/aamador/.local/opt/pyrite-issue-587
G4=/home/aamador/.local/opt/pyrite-issue-182
read -r -a ARGS < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/cases/cases.txt")
cd "$ROOT/cases"
NAME=${ARGS[10]}
ARGS[10]="$ROOT/results/$NAME"
"$G4/bin/micromamba" run -p "$G4/geant4" "$ROOT/build/cherenkov" "${ARGS[@]}" \
  > "$ROOT/results/$NAME.g4log" 2>&1
