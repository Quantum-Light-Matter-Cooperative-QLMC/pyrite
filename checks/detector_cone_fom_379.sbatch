#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=1400M
#SBATCH -t 08:00:00
#SBATCH -o cone379-fom-%j.log
# #379 monitor FOM campaign: analog vs split through the public option; measurement only.
# Site flags at submit: qlmc `-p cpu -w qlmc-queen`; Hoffman2 `-p campus -A pi_qlmc`
# with PYRITE_DATA_DIR=$HOME/pyrite exported (tables).
# SCENES: lines "material energy_keV thickness_ang observables ne seeds..." (observables comma-separated).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu"
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "code $(cat CODE_SHA 2>/dev/null || echo unknown)"
uv sync --locked --no-default-groups
OUT=${OUT:-cone379_fom_out}
mkdir -p "$OUT"
printf '%s\n' "${SCENES:?set SCENES}" | while IFS= read -r line; do
  [ -z "$line" ] && continue
  read -r material energy thickness observables ne seeds <<<"$line"
  tag="$material-$energy-$thickness-${observables//,/+}-n$ne"
  uv run --no-sync python checks/detector_cone_fom_379.py --material "$material" \
    --energy-keV "$energy" --thickness-ang "$thickness" --observables ${observables//,/ } \
    --ne "$ne" --seeds $seeds ${EXTRA:-} --out "$OUT/$tag${TAG:-}.json" || echo "scene failed: $line"
done
