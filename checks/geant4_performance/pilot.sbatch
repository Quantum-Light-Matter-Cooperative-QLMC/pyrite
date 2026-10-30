#!/bin/bash
#SBATCH --job-name=g4bench821-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pilot-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
out="evidence/pilot/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"; lscpu > "$out/cpu.txt"; nvidia-smi > "$out/gpu.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
scenes=(si-300keV-100um w-300keV-10um si-1MeV-1mm w-1MeV-100um si-3MeV-1mm w-3MeV-100um)
for s in "${scenes[@]}"; do
    uv run --no-sync python checks/geant4_performance/geant4_bench.py "$s" --histories 4000 \
        --warmup 500 --seed 1 --threads 8 --executable "$G4BENCH" --output "$out/g4-$s.json" \
        > "$out/g4-$s.log" 2>&1 || echo "FAIL g4 $s"
    PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py "$s" \
        --histories 10000 --block 10000 --seed 1 --output "$out/py-$s.json" \
        > "$out/py-$s.log" 2>&1 || echo "FAIL py $s"
    echo "DONE $s"
done
