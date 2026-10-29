#!/bin/bash
#SBATCH --job-name=transport786
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=transport-cost-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export PYRITE_MC_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
out="evidence/transport-cost/$SLURM_JOB_ID"
mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"
sha256sum checks/equal_precision/{diagnose_kernel.py,pyrite_adapter.py,manifest.json,transport-cost.sbatch} \
    src/pyrite/montecarlo/transport/{_jit_kernel.py,_jit_radiative_device.py,_jit_launch.py,batching.py} \
    > "$out/code.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
deadline=$((SECONDS + 540))
for seed in 78601 78602 78603; do
    variants=(baseline threads32 capacity capacity32)
    if (( seed % 2 == 0 )); then
        variants=(capacity32 capacity threads32 baseline)
    fi
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        for variant in "${variants[@]}"; do
            if (( SECONDS > deadline - 60 )); then
                printf 'BOUNDED_STOP %s %s %s\n' "$scene" "$seed" "$variant"
                exit 0
            fi
            stem="$out/$scene-$seed-$variant"
            printf 'RUN %s\n' "$stem"
            if timeout --signal=TERM --kill-after=5s 60s uv run --no-sync python \
                checks/equal_precision/diagnose_kernel.py "$variant" "$stem-audit.json" \
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
