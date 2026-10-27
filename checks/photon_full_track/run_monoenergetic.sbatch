#!/bin/bash
# Submit from an isolated git-archive checkout after `uv sync --locked` there.
# Tasks 1..9: 12/20/30 keV x three Geant4 cascade settings; 10..12: PyRITE.
# Existing patched TestEm5 and pinned data remain read-only.
#SBATCH --job-name=issue341-mono
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=1-12
#SBATCH --output=results/%x_%A_%a.log
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
ROOT=${SLURM_SUBMIT_DIR}
export PYRITE_HOME=${PHOTON_DATA_ROOT:?set PHOTON_DATA_ROOT to the pinned issue-341 home}
export PYRITE_CATALOG="$ROOT/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$ROOT/empty-user-catalog"
export OPENBLAS_NUM_THREADS=1
EVENTS=${PHOTON_EVENTS:-100000}
G4_ROOT=${PHOTON_G4_ROOT:?set PHOTON_G4_ROOT to the pinned issue-341 staging root}
ENV_ROOT=${PHOTON_G4_ENV:?set PHOTON_G4_ENV to the pinned Geant4 environment}
MICROMAMBA=${PHOTON_MICROMAMBA:?set PHOTON_MICROMAMBA}
CHECK="$ROOT/checks/photon_full_track/monoenergetic.py"
ENERGIES=(12 20 30)
SETTINGS=(auger-off auger-cut cascade)
i=$((SLURM_ARRAY_TASK_ID - 1))
mkdir -p "$ROOT/results/task-$SLURM_ARRAY_TASK_ID"
cd "$ROOT/results/task-$SLURM_ARRAY_TASK_ID"
if ((i < 9)); then
    ENERGY=${ENERGIES[$((i / 3))]}
    SETTING=${SETTINGS[$((i % 3))]}
    NAME=w_${ENERGY}kev_$SETTING
    "$MICROMAMBA" run -p "$ENV_ROOT" "$G4_ROOT/build/TestEm5" "$ROOT/macros/$NAME.mac" \
        > "$NAME.log" 2>&1
    gzip "$NAME.log"
    uv run --project "$ROOT" --locked python "$CHECK" reduce-geant4 "$ROOT/results/$NAME.json" \
        --energy-keV "$ENERGY" --events "$EVENTS" --log "$NAME.log.gz"
else
    ENERGY=${ENERGIES[$((i - 9))]}
    uv run --project "$ROOT" --locked python "$CHECK" pyrite "$ROOT/results/w_${ENERGY}kev_pyrite.json" \
        --energy-keV "$ENERGY" --events "$EVENTS"
fi
