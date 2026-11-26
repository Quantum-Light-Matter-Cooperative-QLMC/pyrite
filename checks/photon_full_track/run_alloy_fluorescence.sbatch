#!/bin/bash
# GaAs secondary-fluorescence comparison (issue #341; alloy_fluorescence.py).
# Submit from an isolated git-archive checkout after `pixi install --locked` there
# and `alloy_fluorescence.py macros macros --events $EVENTS --shards 10`.
# Tasks 1..30: Geant4 (energy-major, 10 shards each); 31..60: PyRITE shards.
# PHOTON_G4_PHYSICS=emlivermore (macros written with --physics, --array=1-30)
# runs the report-only Geant4 Livermore diagnostic arm.
# The patched TestEm5, its Geant4 environment and pinned data stay read-only.
#SBATCH --job-name=issue341-alloy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --array=1-60
#SBATCH --output=results/%x_%A_%a.log
set -euo pipefail
export PATH=$HOME/.pixi/bin:$PATH
ROOT=${SLURM_SUBMIT_DIR}
export PYRITE_HOME=${PHOTON_DATA_ROOT:?set PHOTON_DATA_ROOT to the pinned issue-341 home}
export PYRITE_CATALOG="$ROOT/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$ROOT/empty-user-catalog"
export OPENBLAS_NUM_THREADS=1
EVENTS=${PHOTON_EVENTS:-100000}
SHARDS=10
G4_ROOT=${PHOTON_G4_ROOT:?set PHOTON_G4_ROOT to the pinned issue-341 staging root}
ENV_ROOT=${PHOTON_G4_ENV:?set PHOTON_G4_ENV to the pinned Geant4 environment}
MICROMAMBA=${PHOTON_MICROMAMBA:?set PHOTON_MICROMAMBA}
CHECK="$ROOT/checks/photon_full_track/alloy_fluorescence.py"
PHYSICS=${PHOTON_G4_PHYSICS:-empenelope}
SUFFIX=""
[[ $PHYSICS == empenelope ]] || SUFFIX="_$PHYSICS"
ENERGIES=(11.5 15 30)
NAMES=(gaas_11p5kev gaas_15kev gaas_30kev)
i=$((SLURM_ARRAY_TASK_ID - 1))
engine=$((i / 30))
k=$((i % 30))
ENERGY=${ENERGIES[$((k / SHARDS))]}
SHARD=$(printf '%02d' $((k % SHARDS)))
NAME=${NAMES[$((k / SHARDS))]}_s$SHARD
WORK="$ROOT/results/task-$SLURM_ARRAY_TASK_ID$SUFFIX"
mkdir -p "$WORK"
cd "$WORK"
if ((engine == 0)); then
    /usr/bin/time -v "$MICROMAMBA" run -p "$ENV_ROOT" "$G4_ROOT/build/TestEm5" \
        "$ROOT/macros/$NAME$SUFFIX.mac" > "$NAME.log" 2> resources.txt
    gzip -f "$NAME.log"
    pixi run --locked --manifest-path "$ROOT" python "$CHECK" reduce-geant4 \
        "$ROOT/results/${NAME}_geant4$SUFFIX.json.gz" --energy-keV "$ENERGY" --events "$EVENTS" \
        --log "$NAME.log.gz" --physics "$PHYSICS"
elif [[ -z $SUFFIX ]]; then
    /usr/bin/time -v pixi run --locked --manifest-path "$ROOT" python "$CHECK" pyrite \
        "$ROOT/results/${NAME}_pyrite.json.gz" --energy-keV "$ENERGY" --events "$EVENTS" \
        --shard $((10#$SHARD)) 2> resources.txt
else
    echo "PyRITE tasks run only with the empenelope reference arm" >&2
    exit 2
fi
