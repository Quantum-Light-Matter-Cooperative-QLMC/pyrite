#!/bin/bash
#SBATCH --job-name=precision510kernel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --output=kernel-diagnostic-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_MC_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
OUT="evidence/kernel-diagnostic/$SLURM_JOB_ID"
mkdir -p "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/job.txt"
nvidia-smi -q > "$OUT/gpu.txt"
sha256sum checks/equal_precision/{diagnose_kernel.py,kernel-diagnostic.sbatch,pyrite_adapter.py,manifest.json} src/pyrite/montecarlo/transport/{_jit_kernel.py,_jit_radiative_device.py,batching.py} > "$OUT/code.txt"
for seed in 55001 55002 55003; do
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        for variant in baseline nodes both capacity both-capacity; do
            stem="$OUT/$scene-$seed-$variant"
            timeout --signal=TERM --kill-after=5s 35s "$HOME/.local/bin/uv" run --no-sync python \
                checks/equal_precision/diagnose_kernel.py "$variant" "$stem-audit.json" \
                "$scene" --histories 1000 --seed "$seed" --core cuda \
                --batch-histories 1000 --scratch-mib 4096 --probe-electrons 0 \
                --output "$stem.json" > "$stem.log" 2>&1 || printf 'failed %s\n' "$stem"
        done
    done
done
