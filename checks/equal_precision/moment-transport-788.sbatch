#!/bin/bash
#SBATCH --job-name=moment-transport788
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=moment-transport788-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec 9>.pyrite-sync.lock
flock -s -w 30 9
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export PYRITE_MC_BACKEND=cuda PYRITE_TEST_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
out="evidence/moment-transport788/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
deadline=$((SECONDS + 510))
for scene in si-300-100um-0deg w-300-10um-0deg; do
    for variant in exact table refined; do
        stem="$out/preflight-$scene-$variant"
        timeout --signal=TERM --kill-after=5s 60s uv run --no-sync python \
            checks/equal_precision/benchmark_moment_transport.py "$variant" "$stem-audit.json" \
            --replay "$scene" --histories 256 --seed 78811 --core cuda \
            --batch-histories 256 --scratch-mib 4096 --probe-electrons 0 \
            --output "$stem.json" > "$stem.log" 2>&1
    done
done
for seed in 78811 78812 78813; do
    variants=(exact table refined)
    if (( seed % 2 == 0 )); then variants=(refined table exact); fi
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        for variant in "${variants[@]}"; do
            if (( SECONDS > deadline - 60 )); then
                printf 'BOUNDED_STOP %s %s %s\n' "$scene" "$seed" "$variant"
                exit 0
            fi
            stem="$out/$scene-$seed-$variant"
            timeout --signal=TERM --kill-after=5s 60s uv run --no-sync python \
                checks/equal_precision/benchmark_moment_transport.py "$variant" "$stem-audit.json" \
                "$scene" --histories 5000 --seed "$seed" --core cuda \
                --batch-histories 5000 --scratch-mib 4096 --probe-electrons 0 \
                --output "$stem.json" > "$stem.log" 2>&1
        done
    done
done
