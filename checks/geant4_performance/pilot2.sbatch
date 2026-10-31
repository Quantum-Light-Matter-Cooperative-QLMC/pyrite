#!/bin/bash
# Pilot 2 (#821): PyRITE transport-schedule and continuous-estimator path sweep,
# plus Geant4 cone scoring smoke. Sizes only; timing claims come from run.sbatch.
#SBATCH --job-name=g4bench821-pilot2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --output=pilot2-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
out="evidence/pilot2/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
py() { local tag=$1; shift
    PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py "$@" \
        --output "$out/$tag.json" > "$out/$tag.log" 2>&1 || echo "FAIL $tag"; echo "DONE $tag"; }
for s in si-300keV-100um w-300keV-10um w-1MeV-100um; do
    uv run --no-sync python checks/geant4_performance/geant4_bench.py "$s" --histories 20000 \
        --warmup 200 --seed 7 --threads 8 --executable "$G4BENCH" --output "$out/g4-$s.json" \
        > "$out/g4-$s.log" 2>&1 || echo "FAIL g4 $s"
    for q in 256 1024; do
        py "sched-$s-q$q-host" "$s" --histories 20000 --seed 7 --rows-per-round $q
        py "sched-$s-q$q-dev" "$s" --histories 20000 --seed 7 --rows-per-round $q --device-staging
    done
done
for s in si-300keV-100um w-300keV-10um; do
    for c in 20000 1000000; do
        py "cont-$s-adaptive-c$c" "$s" --histories 10000 --seed 7 --estimator continuous \
            --cone-mode adaptive --chunk $c
    done
    order=$(python3 -c "import json;print(max(json.load(open('$out/cont-$s-adaptive-c20000.json'))['cone_scoring']['adaptive_orders']))" || echo 8)
    for c in 20000 1000000; do
        py "cont-$s-fixed-c$c" "$s" --histories 10000 --seed 7 --estimator continuous \
            --cone-mode fixed --cone-order "$order" --chunk $c
        py "cont-$s-point-c$c" "$s" --histories 10000 --seed 7 --estimator continuous \
            --cone-mode point --cone-order "$order" --chunk $c
    done
done
