#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH -t 03:00:00
#SBATCH -o grid1141-%j.log
# #1141 spilled streamed line-grid evidence: h-BN 5 MeV grid pass against N with
# the evidence spilled to disk (host anon peak, spill disk peak, wall) and held
# in memory at the same N (bitwise grid, wall overhead).
# usage (from an isolated rsync of the worktree, with CODE_SHA): sbatch -J grid1141-<sha> checks/streamed_line_grid_spill_1141.sbatch [OUT]
set -uo pipefail
ROOT=${1:-grid1141_out}
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$HOME/pyrite"  # shared tables; code runs from this isolated copy
export XDG_CONFIG_HOME="$(mktemp -d)"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1
export STREAM_GRID=measured
echo "code $(cat CODE_SHA)"
nvidia-smi --query-gpu=name,memory.total --format=csv
uv sync -q --locked --no-default-groups --extra nvidia || exit 1
B=(uv run --no-sync python checks/streamed_scoring_memory.py)
OUT=$ROOT/hbn; mkdir -p "$OUT/spill"
ARGS=(hbn 5000 cuda 1e7 0 10 100)
arm() {  # NE NAME [ENV...]
  echo "== $2"
  env "${@:3}" "${B[@]}" run grid-pass "$1" 512 "$OUT/$2.json" "${ARGS[@]}" || echo "ARM FAILED $2 rc=$?"
  echo "spill dir after $2: $(ls -A "$OUT/spill" | wc -l) entries"
}
arm 64 warmup
for N in 2048 8192 32768; do
  arm "$N" "spilled-$N" STREAM_SPILL_MIB=0 STREAM_SPILL_DIR="$OUT/spill"
  arm "$N" "held-$N" STREAM_SPILL_MIB=none
done
for N in 2048 8192 32768; do
  "${B[@]}" compare "$OUT/held-$N.json" "$OUT/spilled-$N.json"
done
df -h "$OUT/spill"
