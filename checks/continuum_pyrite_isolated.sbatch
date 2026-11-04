#!/bin/bash
#SBATCH --time=04:00:00
set -euo pipefail
release=$1
trap 'touch "$release"' EXIT
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
scontrol show node DESKTOP-QNIHO3D > evidence/isolated278-allocation.txt
squeue -w DESKTOP-QNIHO3D > evidence/isolated278-node-jobs.txt
export PYRITE_MC_BACKEND=cuda
uv run --no-sync python checks/continuum_pyrite_precision.py --output evidence/py-precisionisolated278 --max-hours 3.5
