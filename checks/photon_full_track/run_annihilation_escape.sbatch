#!/bin/bash
# 511 keV escape comparison (issue #341; annihilation_escape.py).
# Submit from an isolated git-archive checkout after `uv sync --locked` there
# and `annihilation_escape.py macros macros --events $EVENTS --shards 10`.
# Tasks 1..40: Geant4 (config-major, 10 shards each); 41..80: PyRITE shards.
# PHOTON_G4_PHYSICS=emlivermore (macros written with --physics, --array=1-40)
# runs the report-only Geant4 Livermore arm.
#SBATCH --job-name=issue341-511
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --array=1-80
#SBATCH --output=results/%x_%A_%a.log
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
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
CHECK="$ROOT/checks/photon_full_track/annihilation_escape.py"
PHYSICS=${PHOTON_G4_PHYSICS:-empenelope}
SUFFIX=""
[[ $PHYSICS == empenelope ]] || SUFFIX="_$PHYSICS"
CONFIGS=(w_ext w_int si_ext si_int)
i=$((SLURM_ARRAY_TASK_ID - 1))
engine=$((i / 40))
k=$((i % 40))
CONFIG=${CONFIGS[$((k / SHARDS))]}
SHARD=$(printf '%02d' $((k % SHARDS)))
NAME=${CONFIG}_s$SHARD
WORK="$ROOT/results/task-$SLURM_ARRAY_TASK_ID$SUFFIX"
mkdir -p "$WORK"
cd "$WORK"
if ((engine == 0)); then
    /usr/bin/time -v "$MICROMAMBA" run -p "$ENV_ROOT" "$G4_ROOT/build/TestEm5" \
        "$ROOT/macros/$NAME$SUFFIX.mac" > "$NAME.log" 2> resources.txt
    gzip -f "$NAME.log"
    uv run --project "$ROOT" --locked python "$CHECK" reduce-geant4 \
        "$ROOT/results/${NAME}_geant4$SUFFIX.json.gz" --config "$CONFIG" --events "$EVENTS" \
        --log "$NAME.log.gz" --physics "$PHYSICS"
elif [[ -z $SUFFIX ]]; then
    /usr/bin/time -v uv run --project "$ROOT" --locked python "$CHECK" pyrite \
        "$ROOT/results/${NAME}_pyrite.json.gz" --config "$CONFIG" --events "$EVENTS" \
        --shard $((10#$SHARD)) 2> resources.txt
else
    echo "PyRITE tasks run only with the empenelope reference arm" >&2
    exit 2
fi
