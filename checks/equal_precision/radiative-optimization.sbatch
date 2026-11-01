#!/bin/bash
#SBATCH --job-name=precision510rad
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --output=radiative-optimization-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_MC_BACKEND=cuda PYRITE_TEST_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
OUT="evidence/radiative-optimization/$SLURM_JOB_ID"
mkdir -p "$OUT/source"
scontrol show job "$SLURM_JOB_ID" > "$OUT/job.txt"
nvidia-smi -q > "$OUT/gpu.txt"
cp checks/equal_precision/{benchmark_radiative.py,diagnose_kernel.py,pyrite_adapter.py,manifest.json,radiative-optimization.sbatch,radiative_baseline.py} "$OUT/source/"
cp src/pyrite/montecarlo/transport/_jit_radiative_device.py "$OUT/source/candidate.py"
cp tests/montecarlo/cascades/test_hard_radiative_cuda.py "$OUT/source/"
sha256sum "$OUT"/source/* > "$OUT/code.txt"
"$HOME/.local/bin/uv" run --no-sync pyrite-dev test \
    tests/montecarlo/cascades/test_hard_radiative_cuda.py --durations=0 > "$OUT/tests.log" 2>&1
for seed in 56001 56002 56003; do
    for scene in si-300-100um-0deg w-300-10um-0deg; do
        for variant in baseline candidate; do
            stem="$OUT/$scene-$seed-$variant"
            timeout --signal=TERM --kill-after=5s 40s "$HOME/.local/bin/uv" run --no-sync python \
                checks/equal_precision/benchmark_radiative.py "$variant" "$stem-audit.json" \
                --baseline-source checks/equal_precision/radiative_baseline.py \
                "$scene" --histories 5000 --seed "$seed" --core cuda \
                --batch-histories 5000 --scratch-mib 4096 --probe-electrons 0 \
                --output "$stem.json" > "$stem.log" 2>&1 || printf 'failed %s\n' "$stem"
        done
    done
done
