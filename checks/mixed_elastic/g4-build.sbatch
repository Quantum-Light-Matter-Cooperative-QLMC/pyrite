#!/bin/bash
# Build the benchmark app and the Python environment for g4-mixed.sbatch (#894).
# Runs on the lab node that has cmake; the home directory is shared with queen.
#SBATCH --job-name=894-g4build
#SBATCH --partition=cpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=g4-build-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
uv sync --locked --package pyrite-mc --no-default-groups --extra nvidia -q
sha256sum "$G4BENCH"
