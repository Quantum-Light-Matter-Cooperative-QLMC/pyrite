#!/bin/bash
# #916 A/B of a transport-kernel change: base checkout ($1) against this one.
# Per checkout: #907-scene segment digests and transport observables at
# 2,048/4,096/16,384 histories (fp64_pipe_916.py scene-digest; no scoring) and,
# in alternating order, one #907 transport-only arm per process
# (streamed_block_sizing.py transport 16384 ...: CUDA-event kernel time).
# Then this checkout's shell soft/hard and positron CUDA tests.
# usage (from this synced checkout root): sbatch checks/fp64_pipe_916_ab.sbatch <base-dir> [rounds]
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o fp64pipe916ab-%j.log
set -uo pipefail
export PATH=$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH
NEW=$SLURM_SUBMIT_DIR; BASE=$(cd "${1:?base checkout}" && pwd); ROUNDS=${2:-3}
export PYRITE_HOME=$HOME/pyrite PYRITE_USER_CATALOG=$(mktemp -d)/catalog XDG_CONFIG_HOME=$(mktemp -d)
export PYRITE_MC_BACKEND=cuda
OUT=$NEW/evidence/fp64pipe916/$SLURM_JOB_ID; mkdir -p "$OUT"
for d in "$BASE" "$NEW"; do (cd "$d" && pixi install --locked -e nvidia); done
echo "base $(cat "$BASE/CODE_SHA") new $(cat "$NEW/CODE_SHA")" | tee "$OUT/arms.txt"
arm() {  # arm <name> <dir> <cmd...>
  local name=$1 dir=$2; shift 2
  (cd "$dir" && PYRITE_CATALOG=$dir/src/pyrite/data/catalog CUPY_CACHE_DIR=$OUT/cupy-$name \
     PROBE_SHA=$(cat CODE_SHA) pixi run --as-is -e nvidia python -u "$@")
}
for name in base new; do
  dir=$BASE; [[ $name == new ]] && dir=$NEW
  arm $name "$dir" "$NEW/checks/fp64_pipe_916.py" scene-digest 2048 4096 16384 \
    --observables "$OUT/observables-$name.json" > "$OUT/digest-$name.jsonl" 2>"$OUT/digest-$name.err"
  echo "digest $name rc=$?"; cat "$OUT/digest-$name.jsonl"
done
(cd "$NEW" && pixi run --as-is -e nvidia python checks/fp64_pipe_916.py compare-observables \
  "$OUT/observables-base.json" "$OUT/observables-new.json" > "$OUT/observables-compare.json")
echo "observables rc=$?"
for r in $(seq 1 "$ROUNDS"); do
  order="base new"; (( r % 2 == 0 )) && order="new base"
  for name in $order; do
    dir=$BASE; [[ $name == new ]] && dir=$NEW
    arm $name "$dir" "$NEW/checks/streamed_block_sizing.py" transport 16384 16384 2048 32 0 "$OUT/t-$name-$r" \
      > "$OUT/t-$name-$r.log" 2>&1
    echo "round $r $name rc=$? $(grep -o '"kernel_s": [0-9.]*' "$OUT/t-$name-$r.log")"
  done
done
cd "$NEW" && PYRITE_CATALOG=$NEW/src/pyrite/data/catalog PYRITE_TEST_BACKEND=cuda \
  pixi run --as-is -e nvidia pyrite-dev test tests/montecarlo/collisions/test_shell_soft_hard_cuda.py \
  tests/montecarlo/cascades/test_shell_secondary_cuda.py tests/montecarlo/cascades/test_positron_cuda.py \
  tests/montecarlo/transport/test_transport_kernel_variants_cuda.py \
  tests/montecarlo/collisions/test_hard_loss_newton.py > "$OUT/tests.log" 2>&1
echo "tests rc=$?"; tail -3 "$OUT/tests.log"
echo done
