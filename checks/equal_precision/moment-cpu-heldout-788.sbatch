#!/bin/bash
#SBATCH --job-name=moment-heldout788
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=moment-heldout788-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec {sync_lock_fd}<>.pyrite-sync.lock
flock -s -w 30 "$sync_lock_fd"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export PYRITE_MC_BACKEND=cpu PYRITE_TEST_BACKEND=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
out="evidence/moment-heldout788/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
uv run --no-sync python -c 'import platform, sys; print(platform.platform()); print(sys.version)' > "$out/host.txt"
# Freeze the exact protocol bytes alongside results before running any history.
cp checks/equal_precision/moment-heldout-788.json "$out/policy.json"
mapfile -t cases < <(uv run --no-sync python -c 'import json; print("\n".join(json.load(open("checks/equal_precision/moment-heldout-788.json"))["cases"]))')
run() {
    local case=$1 variant=$2 seed=$3 histories=$4 stem=$5; shift 5
    local scene=${case%%:*} cutoff=()
    if [[ $case == *:* ]]; then cutoff=(--cutoff-ev "${case#*:}"); fi
    timeout --signal=TERM --kill-after=5s 240s uv run --no-sync python \
        checks/equal_precision/benchmark_moment_transport_cpu.py "$variant" "$stem-audit.json" \
        "$@" "$scene" "${cutoff[@]}" --histories "$histories" --seed "$seed" --core per-electron \
        --batch-histories "$histories" --scratch-mib 4096 --probe-electrons 0 \
        --output "$stem.json" > "$stem.log" 2>&1
}
# Two fresh processes for each numerical policy: retain both replay/score records.
for variant in exact table refined; do
    for repeat in 1 2; do
        run si-300-100um-0deg "$variant" 78820 256 "$out/preflight-replay-$variant-$repeat" --replay
    done
done
for seed in 78821 78822 78823; do
    variants=(exact table refined)
    if (( seed % 2 == 0 )); then variants=(refined table exact); fi
    for case in "${cases[@]}"; do
        for variant in "${variants[@]}"; do
            run "$case" "$variant" "$seed" 2048 "$out/${case/:/-cut}-$seed-$variant"
        done
    done
done
uv run --no-sync python checks/equal_precision/qualify_moment_fractions.py "$out" "$out/fractions.json"
