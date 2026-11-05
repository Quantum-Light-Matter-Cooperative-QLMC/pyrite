#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=matrix278-%j.log
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
mkdir -p evidence/matrix278
nvidia-smi > evidence/matrix278/hardware.txt
for energy in 300 3000 30000 100000; do
    for thickness in 1000000 10000000; do
        for repeat in 0 1; do
            arms=(automatic uniform25)
            if [ "$repeat" = 1 ]; then arms=(uniform25 automatic); fi
            for grid in "${arms[@]}"; do
                uv run --no-sync python checks/continuum_grid_end_to_end.py \
                    --grid "$grid" --energy-kev "$energy" --electrons 20 \
                    --thickness-ang "$thickness" --seed 1 --warmup \
                    --output "evidence/matrix278/${energy}-${thickness}-${repeat}-${grid}.json"
            done
        done
    done
done
