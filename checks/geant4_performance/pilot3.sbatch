#!/bin/bash
# Pilot 3 (#821): after #827/#832/#796 and row-bounded cone groups, check that
# continuous W scoring scales linearly (run 2835: 197 s at 100k vs 3.2 s at 10k)
# before the full run. PyRITE arms only, same flags as run.sbatch.
#SBATCH --job-name=g4bench821-pilot3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=02:00:00
#SBATCH --output=pilot3-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
out="evidence/pilot3/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
for s in w-300keV-10um w-1MeV-100um; do
    for h in 10000 100000; do
        for arm in py pyc; do
            extra=()
            [[ $arm == pyc ]] && extra=(--estimator continuous --cone-mode point --chunk 1000000)
            stem="$out/$arm-$s-$h"
            PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py \
                "$s" --histories "$h" --seed 1 --rows-per-round 1024 --device-staging \
                "${extra[@]}" --output "$stem.json" > "$stem.log" 2>&1 || echo "FAIL $arm $s $h"
            echo "DONE $arm $s $h"
        done
    done
done
