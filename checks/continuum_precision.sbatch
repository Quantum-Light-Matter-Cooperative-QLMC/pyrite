#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --output=precision278-%j.log
set -euo pipefail
source checks/geant4_performance/env.sh
if [[ "$1" == geant4 ]]; then
    cp evidence/g4-hopg278-spectra/spectral-statistics.json evidence/g4-precision-plan.json
    uv run --no-sync python checks/continuum_geant4_precision.py --executable "$G4BENCH" \
        --pilot evidence/g4-precision-plan.json --output evidence/g4-precision278 \
        --threads "$SLURM_CPUS_PER_TASK"
else
    export PYRITE_MC_BACKEND=cuda
    uv run --no-sync python checks/continuum_pyrite_precision.py \
        --output "evidence/py-precision${2:-278}" --max-hours "${3:-3.5}"
fi
