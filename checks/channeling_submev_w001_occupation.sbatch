#!/bin/bash
# W <001> sub-MeV occupation array (#585): one checks/channeling_submev_w001_occupation.py
# case per task on qlmc-queen. Stage an isolated, clean copy of the commit under test
# with SYNCED_SHA, then from that copy: sbatch --array=0-27%8 checks/channeling_submev_w001_occupation.sbatch
# Logs land in runs/, records in runs/out/ of the submit directory.
#SBATCH --partition=cpu
#SBATCH --nodelist=qlmc-queen
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=16:00:00
#SBATCH --job-name=w001-occ-585
#SBATCH --output=runs/%x-%A-%a.log
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
ROOT=$SLURM_SUBMIT_DIR
export PYRITE_HOME=$ROOT PYRITE_CATALOG=$ROOT/src/pyrite/data/catalog PYRITE_USER_CATALOG=$ROOT/user-catalog
export PYRITE_MC_BACKEND=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
export PYRITE_SYNCED_SHA=$(cat SYNCED_SHA)

# energy fraction arm extra-args (main grid 0-17, convergence 18-27)
CASES=()
for e in 30 100 300; do
    for f in 0.05 0.1 0.25; do
        for arm in tds+electronic tds; do
            CASES+=("$e $f $arm")
        done
    done
done
for e in 30 100 300; do
    CASES+=("$e 0.1 tds+electronic --step-ang 0.25")
    CASES+=("$e 0.1 tds+electronic --step-ang 1.0")
    CASES+=("$e 0.1 tds+electronic --points 32")
done
CASES+=("30 0.1 tds+electronic --cells 6")

read -r ENERGY FRACTION ARM EXTRA <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
mkdir -p runs/out
OUT=runs/out/case-${SLURM_ARRAY_TASK_ID}.json
echo "sha $(cat SYNCED_SHA) task $SLURM_ARRAY_TASK_ID: $ENERGY keV f=$FRACTION $ARM ${EXTRA:-}"
# shellcheck disable=SC2086  # EXTRA holds option words
uv run --frozen --no-sync python checks/channeling_submev_w001_occupation.py \
    --energy-kev "$ENERGY" --fraction "$FRACTION" --arm "$ARM" \
    --max-depth-ang 1000 --record-ang 10 --tilts 128 --histories 16 ${EXTRA:-} > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
