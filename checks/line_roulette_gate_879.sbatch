#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o roulette879-%j.log
# Run from a clean, synced checkout; CODE_SHA records its checkpoint commit.
# h-BN reproduces #203's energy/thickness/orientation, using #872's scoring
# settings and explicit grid. This is a timing gate, not a yield reproduction.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1
echo "code $(cat CODE_SHA)"
nvidia-smi
uv sync -q --locked --no-default-groups --extra nvidia
B=(uv run --no-sync python checks/streamed_scoring_memory.py)
ROOT=${1:-roulette879_out}
mkdir -p "$ROOT/hbn" "$ROOT/tungsten"
# Each arm is a fresh process, as in #872. Warmups populate on-disk JIT caches;
# process initialization and first-call costs remain included in measured wall.
"${B[@]}" run streamed 256 256 "$ROOT/hbn/warmup.json" hbn 5000 cuda 1e7 0 10 100
for REP in 1 2; do
  "${B[@]}" run streamed 8000 512 "$ROOT/hbn/rep-$REP.json" hbn 5000 cuda 1e7 0 10 100
done
"${B[@]}" compare "$ROOT/hbn/rep-1.json" "$ROOT/hbn/rep-2.json"
"${B[@]}" run streamed 1000 1000 "$ROOT/tungsten/warmup.json" tungsten 300 cuda 1e5 1
for REP in 1 2; do
  "${B[@]}" run streamed 20000 4096 "$ROOT/tungsten/rep-$REP.json" tungsten 300 cuda 1e5 1
done
"${B[@]}" compare "$ROOT/tungsten/rep-1.json" "$ROOT/tungsten/rep-2.json"
