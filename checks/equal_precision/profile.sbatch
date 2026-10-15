#!/bin/bash
#SBATCH --job-name=precision510-profile
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --output=profile-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_MC_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
UV="$HOME/.local/bin/uv"
PROFILE_DIR="$PWD/evidence/profiles/$SLURM_JOB_ID"
mkdir -p "$PROFILE_DIR"
cp .pyrite-sync "$PROFILE_DIR/code.txt"
scontrol show job "$SLURM_JOB_ID" > "$PROFILE_DIR/job.txt"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --stats=true \
    --output="$PROFILE_DIR/gpu" \
    "$UV" run --no-sync python -m cProfile -o "$PROFILE_DIR/host.pstats" \
    checks/equal_precision/pyrite_adapter.py w-300-10um-0deg \
    --histories 1000 --seed 54001 --core cuda --batch-histories 1000 \
    --scratch-mib 4096 --probe-electrons 0 --output "$PROFILE_DIR/sample.json"
"$UV" run --no-sync python -c \
    'import pstats,sys; pstats.Stats(sys.argv[1]).strip_dirs().sort_stats("cumulative").print_stats(30)' \
    "$PROFILE_DIR/host.pstats" > "$PROFILE_DIR/host-summary.txt"
