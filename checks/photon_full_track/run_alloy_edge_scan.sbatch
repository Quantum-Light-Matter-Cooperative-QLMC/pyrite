#!/bin/bash
# Geant4-only near-edge K-line scan (issue #341; alloy_edge_scan.py).
# Submit from an isolated git-archive checkout after `uv sync --locked` and
# `alloy_edge_scan.py macros scan --events $EVENTS`; one array task per
# macro listed in scan/runs.txt (36 runs). Report with
# `alloy_edge_scan.py report scan --events $EVENTS --output scan/report.json`.
#SBATCH --job-name=issue341-edge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=1-36
#SBATCH --output=scan/%x_%A_%a.out
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR}
G4_ROOT=${PHOTON_G4_ROOT:?set PHOTON_G4_ROOT to the pinned issue-341 staging root}
ENV_ROOT=${PHOTON_G4_ENV:?set PHOTON_G4_ENV to the pinned Geant4 environment}
MICROMAMBA=${PHOTON_MICROMAMBA:?set PHOTON_MICROMAMBA}
NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/scan/runs.txt")
cd "$ROOT/scan"
"$MICROMAMBA" run -p "$ENV_ROOT" "$G4_ROOT/build/TestEm5" "$NAME.mac" > "$NAME.log" 2>&1
gzip -f "$NAME.log"
