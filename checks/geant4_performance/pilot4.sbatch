#!/bin/bash
# Pilot 4 (#821): variance pilot for the 3 MeV scenes, which run 2835 never
# reached. Feeds precision_plan.py (per-primary CV and warm seconds per primary
# for every configuration); no timing claim.
#SBATCH --job-name=g4bench821-pilot4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --output=pilot4-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
out="evidence/pilot4/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
for s in si-3MeV-1mm w-3MeV-100um; do
    for spec in 32:20000 16:20000 1:2000; do
        t=${spec%%:*} n=${spec##*:}
        uv run --no-sync python checks/geant4_performance/geant4_bench.py "$s" --histories "$n" \
            --warmup 200 --seed 1 --threads "$t" --executable "$G4BENCH" \
            --output "$out/g4-$t-$s.json" > "$out/g4-$t-$s.log" 2>&1 || echo "FAIL g4-$t $s"
        echo "DONE g4-$t $s"
    done
    for arm in py pyc; do
        extra=()
        [[ $arm == pyc ]] && extra=(--estimator continuous --cone-mode point --chunk 1000000)
        PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py \
            "$s" --histories 20000 --seed 1 --rows-per-round 1024 --device-staging "${extra[@]}" \
            --output "$out/$arm-$s.json" > "$out/$arm-$s.log" 2>&1 || echo "FAIL $arm $s"
        echo "DONE $arm $s"
    done
done
