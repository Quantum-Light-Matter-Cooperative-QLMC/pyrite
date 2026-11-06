#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o char1005-%j.log
# #1005 characteristic-scoring split on the #872 harness: W and Si 10 um slabs plus a
# layered 1 um W on 10 um Si, 20000 electrons / 4096 blocks, two reps each; Nsight
# launches/copies per char.* range for W and Si.
# usage (from an isolated synced checkout root): sbatch -J c1005-<arm> checks/characteristic_scoring_1005.sbatch [OUTDIR] [NSYS]
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA)"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
B=(uv run --no-sync python checks/characteristic_scoring_1005.py)
ROOT=${1:-c1005_out}
NSYS=${2:-1}
for MAT in silicon tungsten tungsten/silicon; do
  OUT=$ROOT/${MAT//\//-}; mkdir -p "$OUT"
  "${B[@]}" run streamed 1000 1000 "$OUT/warmup.json" "$MAT" 300 cuda 1e5 1 > /dev/null
  for REP in 1 2; do
    echo "== $MAT streamed rep $REP"
    "${B[@]}" run streamed 20000 4096 "$OUT/streamed-$REP.json" "$MAT" 300 cuda 1e5 1 | tail -1
  done
  "${B[@]}" compare "$OUT/streamed-1.json" "$OUT/streamed-2.json"
done
echo "== tungsten joined 4000"
"${B[@]}" run joined 4000 0 "$ROOT/tungsten/joined-4000.json" tungsten 300 cuda 1e5 1 | tail -1
[ "$NSYS" = 1 ] || exit 0
for MAT in silicon tungsten; do
  echo "== nsys $MAT streamed"
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o "$ROOT/$MAT/nsys" -f true \
    "${B[@]}" run streamed 20000 4096 "$ROOT/$MAT/nsys.json" "$MAT" 300 cuda 1e5 1 | tail -1
  nsys export --type sqlite -o "$ROOT/$MAT/nsys.sqlite" -f true "$ROOT/$MAT/nsys.nsys-rep" > /dev/null
  uv run --no-sync python checks/characteristic_nsys_1005.py "$ROOT/$MAT/nsys.sqlite" | tee "$ROOT/$MAT/nsys_char.json"
done
