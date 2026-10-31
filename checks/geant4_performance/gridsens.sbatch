#!/bin/bash
# Spectral-grid sensitivity (#278; #821 plan item).
# PyRITE continuous arm on W 1 MeV / 100 um with pilot-3 flags (10k histories,
# seed 1). The same transported blocks also score the production continuum
# (`mc_coupled_brem_spectrum` with the runner chunk policy) on uniform 25, 100
# and 400 eV grids and on the refined geometric grid (`refined_continuum_grid`)
# at 250/500/1000/2000 baseline nodes (~4/2/1/0.5% dE/E at 1 MeV), each timed
# separately; the 25 eV grid is the accuracy reference.
#SBATCH --job-name=g4bench821-gridsens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --output=gridsens-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
out="evidence/gridsens/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
s=w-1MeV-100um
for h in 2000 10000; do
    stem="$out/pyc-$s-$h"
    PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py \
        "$s" --histories "$h" --seed 1 --rows-per-round 1024 --device-staging \
        --estimator continuous --cone-mode point --chunk 1000000 --warmup-histories 1000 \
        --spectrum-step-ev 25 --spectrum-step-ev 100 --spectrum-step-ev 400 \
        --spectrum-geometric-nodes 250 --spectrum-geometric-nodes 500 \
        --spectrum-geometric-nodes 1000 --spectrum-geometric-nodes 2000 \
        --output "$stem.json" > "$stem.log" 2>&1 || echo "FAIL $s $h"
    echo "DONE $s $h"
done
