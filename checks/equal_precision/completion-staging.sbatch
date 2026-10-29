#!/bin/bash
#SBATCH --job-name=transport786-staging
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --output=completion-staging-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export PYRITE_MC_BACKEND=cuda PYRITE_TEST_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
baseline_source="${1:?pass the retained pre-change hard_radiative.py path}"
out="evidence/completion-staging/$SLURM_JOB_ID"
mkdir -p "$out/source"
cp .pyrite-sync "$out/sync.txt"
cp "$baseline_source" "$out/source/baseline.py"
cp src/pyrite/montecarlo/transport/hard_radiative.py "$out/source/candidate.py"
sha256sum "$out/source/"*.py > "$out/source-sha256.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
uv run --no-sync pyrite-dev test \
    tests/montecarlo/test_hard_radiative.py -k completion_stages \
    --durations=5 > "$out/cache-tests.log" 2>&1
uv run --no-sync pyrite-dev test \
    tests/montecarlo/test_hard_radiative_cuda.py --durations=5 > "$out/cuda-tests.log" 2>&1
deadline=$((SECONDS + 420))
for seed in 78611 78612 78613; do
    variants=(baseline candidate)
    if (( seed % 2 == 0 )); then variants=(candidate baseline); fi
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        for variant in "${variants[@]}"; do
            if (( SECONDS > deadline - 60 )); then
                printf 'BOUNDED_STOP %s %s %s\n' "$scene" "$seed" "$variant"
                exit 0
            fi
            stem="$out/$scene-$seed-$variant"
            printf 'RUN %s\n' "$stem"
            if timeout --signal=TERM --kill-after=5s 60s uv run --no-sync python \
                checks/equal_precision/benchmark_completion.py "$variant" "$stem-audit.json" \
                --baseline-source "$out/source/baseline.py" \
                "$scene" --histories 5000 --seed "$seed" --core cuda \
                --batch-histories 5000 --scratch-mib 4096 --probe-electrons 0 \
                --output "$stem.json" > "$stem.log" 2>&1; then
                printf 'DONE %s\n' "$stem"
            else
                status=$?
                printf '{"status":"failed","exit_code":%d}\n' "$status" > "$stem.failed.json"
                printf 'FAILED %s %s\n' "$stem" "$status"
            fi
        done
    done
done
