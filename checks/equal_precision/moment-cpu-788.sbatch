#!/bin/bash
#SBATCH --job-name=moment-cpu788
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=moment-cpu788-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Use a read/write descriptor: shared flock on some filesystems uses
# POSIX locks, which reject a write-only FD with EBADF.
exec {sync_lock_fd}<>.pyrite-sync.lock
flock -s -w 30 "$sync_lock_fd"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export PYRITE_MC_BACKEND=cpu PYRITE_TEST_BACKEND=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
out="evidence/moment-cpu788/$SLURM_JOB_ID"
mkdir -p "$out"
[[ ! -f .pyrite-sync ]] || cp .pyrite-sync "$out/sync.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
uv run --no-sync python -c 'import platform, sys; print(platform.platform()); print(sys.version)' > "$out/host.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
# Preflight only: these runs must hash identically within each numerical policy.
for scene in si-300-100um-0deg w-300-10um-0deg; do
    for variant in exact table refined; do
        stem="$out/preflight-$scene-$variant"
        timeout --signal=TERM --kill-after=5s 240s uv run --no-sync python \
            checks/equal_precision/benchmark_moment_transport_cpu.py "$variant" "$stem-audit.json" \
            --replay "$scene" --histories 256 --seed 78811 --core per-electron \
            --batch-histories 256 --scratch-mib 4096 --probe-electrons 0 \
            --output "$stem.json" > "$stem.log" 2>&1
    done
done
# Frozen three seeds. Short CPU screens cannot qualify unchanged rare outcomes.
for seed in 78811 78812 78813; do
    variants=(exact table refined)
    if (( seed % 2 == 0 )); then variants=(refined table exact); fi
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        for variant in "${variants[@]}"; do
            stem="$out/$scene-$seed-$variant"
            timeout --signal=TERM --kill-after=5s 240s uv run --no-sync python \
                checks/equal_precision/benchmark_moment_transport_cpu.py "$variant" "$stem-audit.json" \
                "$scene" --histories 2048 --seed "$seed" --core per-electron \
                --batch-histories 2048 --scratch-mib 4096 --probe-electrons 0 \
                --output "$stem.json" > "$stem.log" 2>&1
        done
    done
done
uv run --no-sync python checks/equal_precision/analyze_moment_transport.py \
    "$out" "$out/analysis.json"
