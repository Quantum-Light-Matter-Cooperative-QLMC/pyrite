#!/bin/bash
#SBATCH --job-name=elsepa895-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=elsepa895-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
pyrite_gpu_sync -q
out="evidence/elsepa895/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.txt"
pyrite_gpu_info > "$out/gpu.txt" 2>&1
pixi run --as-is -e "$PYRITE_PIXI_ENV" python checks/equal_precision/prepare_stopping.py
pixi run --as-is -e "$PYRITE_PIXI_ENV" python checks/elsepa_statistical/parity.py > "$out/parity.log" 2>&1
mv parity895.json "$out/parity.json"
for precision in mixed fp64; do
    for seed in 89501 89502 89503; do
        laws=(quantile mixture)
        if (( seed % 2 )); then laws=(mixture quantile); fi
        for law in "${laws[@]}"; do
            stem="$out/$precision-$law-w-1MeV-100um-$seed"
            pixi run --as-is -e "$PYRITE_PIXI_ENV" python checks/elsepa_statistical/run.py --law "$law" \
                w-1MeV-100um --histories 4000 --seed "$seed" --rows-per-round 1024 \
                --device-staging --transport-precision "$precision" --no-audit \
                --estimator continuous --cone-mode point --output "$stem.json" \
                > "$stem.log" 2>&1
            echo "DONE $precision $law $seed"
        done
    done
done
