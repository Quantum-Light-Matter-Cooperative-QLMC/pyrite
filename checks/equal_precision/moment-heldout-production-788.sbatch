#!/bin/bash
#SBATCH --job-name=moment-heldprod788
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=moment-heldprod788-%j.log
# Full rerun of the frozen moment-heldout-788.json CPU protocol on the shipped
# exact/table policies. SEEDS selects this job's seed subset; OUT_TAG names the
# shared evidence directory; PREFLIGHT=1 adds cross-process replay checks.
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
policy=checks/equal_precision/moment-heldout-788.json
out="evidence/moment-heldprod788/${OUT_TAG:?}"
mkdir -p "$out"
cp .pyrite-sync "$out/sync-$SLURM_JOB_ID.txt"
# Seed jobs share $out over NFS; concurrent cp onto one path can fail EEXIST.
cp "$policy" "$out/policy-$SLURM_JOB_ID.json"
scontrol show job "$SLURM_JOB_ID" > "$out/job-$SLURM_JOB_ID.txt"
# A fresh checkout lacks the W SBETHE table; generate once, serialized.
(
    flock -x -w 1800 8
    uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables-$SLURM_JOB_ID.json"
) 8>"$out/.prepare.lock"
mapfile -t cases < <(uv run --no-sync python -c "import json; print('\n'.join(json.load(open('$policy'))['cases']))")
read -ra seeds <<< "${SEEDS:?}"
run() {
    local case=$1 variant=$2 seed=$3 histories=$4 stem=$5; shift 5
    local scene=${case%%:*} cutoff=()
    if [[ $case == *:* ]]; then cutoff=(--cutoff-ev "${case#*:}"); fi
    timeout --signal=TERM --kill-after=5s 600s uv run --no-sync python \
        checks/equal_precision/benchmark_moment_production.py "$variant" "$stem-audit.json" \
        "$@" "$scene" "${cutoff[@]}" --histories "$histories" --seed "$seed" --core per-electron \
        --batch-histories "$histories" --scratch-mib 4096 --probe-electrons 0 \
        --output "$stem.json" > "$stem.log" 2>&1
}
if [[ ${PREFLIGHT:-0} == 1 ]]; then
    # Two fresh processes per policy: compare their replay/score records.
    for variant in exact table refined; do
        for repeat in 1 2; do
            run si-300-100um-0deg "$variant" 78820 256 "$out/preflight-replay-$variant-$repeat" --replay
        done
    done
fi
for seed in "${seeds[@]}"; do
    variants=(exact table refined)
    if (( seed % 2 == 0 )); then variants=(refined table exact); fi
    for case in "${cases[@]}"; do
        for variant in "${variants[@]}"; do
            run "$case" "$variant" "$seed" 2048 "$out/${case/:/-cut}-$seed-$variant"
        done
    done
done
echo COMPLETE
