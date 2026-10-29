#!/bin/bash
#SBATCH --job-name=moment-anchors788
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=moment-anchors788-%j.log
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
out="evidence/moment-anchors788/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
deadline=$((SECONDS + 1380))
# Wider #788 anchors: energy extremes, oblique backscatter and a 10 keV cutoff.
# Each case is "scene[:cutoff_eV]"; only exact versus default-budget tables.
cases=(si-800-100um-0deg w-800-10um-0deg si-100-1um-0deg w-100-100um-75deg si-300-100um-0deg:10000)
run() {
    local case=$1 variant=$2 seed=$3 histories=$4 stem=$5; shift 5
    local scene=${case%%:*} cutoff=()
    if [[ $case == *:* ]]; then cutoff=(--cutoff-ev "${case#*:}"); fi
    timeout --signal=TERM --kill-after=5s 120s uv run --no-sync python \
        checks/equal_precision/benchmark_moment_transport.py "$variant" "$stem-audit.json" \
        "$@" "$scene" "${cutoff[@]}" --histories "$histories" --seed "$seed" --core cuda \
        --batch-histories "$histories" --scratch-mib 4096 --probe-electrons 0 \
        --output "$stem.json" > "$stem.log" 2>&1
}
for case in "${cases[@]}"; do
    for variant in exact table; do
        run "$case" "$variant" 78811 256 "$out/preflight-${case/:/-cut}-$variant" --replay
    done
done
for seed in 78811 78812 78813; do
    variants=(exact table)
    if (( seed % 2 == 0 )); then variants=(table exact); fi
    for case in "${cases[@]}"; do
        for variant in "${variants[@]}"; do
            if (( SECONDS > deadline - 90 )); then
                printf 'BOUNDED_STOP %s %s %s\n' "$case" "$seed" "$variant"
                exit 0
            fi
            run "$case" "$variant" "$seed" 5000 "$out/${case/:/-cut}-$seed-$variant"
        done
    done
done
