#!/bin/bash
# Usage: sbatch --dependency=after:<hold> run.sbatch <hold-job-id> <histories> <seeds...>
#SBATCH --job-name=g4bench821-run
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=12:00:00
#SBATCH --output=run-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
hold=$1 histories=$2; shift 2; seeds=("$@")
trap 'touch "evidence/hold-release-$hold"' EXIT
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
[[ "$(squeue -h -j "$hold" -o %T)" == RUNNING ]] || { echo "hold job $hold not running"; exit 1; }
out="evidence/run/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"; lscpu > "$out/cpu.txt"; nvidia-smi > "$out/gpu.txt"
squeue -w DESKTOP-QNIHO3D > "$out/node-jobs.txt"; free -g > "$out/memory.txt"
sha256sum "$G4BENCH" checks/geant4_performance/*.py checks/geant4_performance/g4bench/* > "$out/code.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
scenes=(si-300keV-100um w-300keV-10um si-1MeV-1mm w-1MeV-100um si-3MeV-1mm w-3MeV-100um)
warm=$((histories / 100))
for seed in "${seeds[@]}"; do
    for s in "${scenes[@]}"; do
        # Engine order alternates with the seed to spread any thermal drift.
        arms=(py g4-32 g4-16 g4-1)
        (( seed % 2 )) && arms=(g4-1 g4-16 g4-32 py)
        for arm in "${arms[@]}"; do
            stem="$out/$arm-$s-$seed"
            if [[ $arm == py ]]; then
                PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py \
                    "$s" --histories "$histories" --seed "$seed" --output "$stem.json" > "$stem.log" 2>&1 \
                    || echo "FAIL $arm $s $seed"
            else
                uv run --no-sync python checks/geant4_performance/geant4_bench.py "$s" \
                    --histories "$histories" --warmup "$warm" --seed "$seed" --threads "${arm#g4-}" \
                    --executable "$G4BENCH" --output "$stem.json" > "$stem.log" 2>&1 \
                    || echo "FAIL $arm $s $seed"
            fi
            echo "DONE $arm $s $seed"
        done
    done
done
uv run --no-sync python checks/geant4_performance/analyze.py "$out" --output "$out/summary.json"
