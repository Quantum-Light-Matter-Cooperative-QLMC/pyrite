#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o trim862-%j.log
# #862 bitwise-neutral line-scoring trims: GPU parity tests of the checkout under test, then
# interleaved base/new production reps of serial live streamed scoring on the #872 harness
# (Si 20000 / 4096 and W 8192 / 4096), with `compare` of every new run against the base run.
# usage (new checkout root, base checkout synced beside it): sbatch -J trim862-<sha> checks/stream_trim_862.sbatch BASE_DIR [OUTDIR]
set -uo pipefail
export PATH=$HOME/.pixi/bin:$PATH
NEW=$SLURM_SUBMIT_DIR; BASE=$(readlink -f "${1:?BASE_DIR}"); OUT=$(readlink -f "${2:-$NEW/trim862_out}"); mkdir -p "$OUT"
export PYRITE_HOME=$HOME/pyrite PYRITE_USER_CATALOG=$(mktemp -d)/catalog PYRITE_MC_BACKEND=cuda
echo "new $(cat "$NEW/CODE_SHA")  base $(cat "$BASE/CODE_SHA")"
for D in "$NEW" "$BASE"; do (cd "$D" && pixi install --locked -e nvidia); done
cd "$NEW"
[ "${TRIM_NOTEST:-0}" = 1 ] || PYRITE_CATALOG=$NEW/src/pyrite/data/catalog PYRITE_TEST_BACKEND=cuda pixi run --as-is -e nvidia pyrite-dev test \
  tests/montecarlo/spectrum/test_cxr_fused_cuda_parity.py tests/montecarlo/spectrum/test_cxr_reduction_fp64.py \
  tests/montecarlo/spectrum/test_layered_batched_route.py tests/montecarlo/spectrum/test_line_recoil_exact.py \
  -p no:cacheprovider > "$OUT/pytest.log" 2>&1
[ "${TRIM_NOTEST:-0}" = 1 ] || { echo "pytest rc=$?"; grep -E "passed|failed|FAILED" "$OUT/pytest.log" | tail -8; }
run() { # run TREE TAG MAT NE BLOCK
  (cd "$1" && PYRITE_CATALOG=$1/src/pyrite/data/catalog STREAM_PHASES=0 pixi run --as-is -e nvidia python checks/streamed_scoring_memory.py \
    run streamed "$4" "$5" "$OUT/$3-$4-$2.json" "$3" 300 cuda 1e5 1 > /dev/null)
  pixi run --as-is -e nvidia python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], json.dumps({k: r[k] for k in ("wall_s", "host_peak_MiB", "device_pool_peak_above_base_MiB")}))' "$OUT/$3-$4-$2.json" "$2 $3 $4"
}
for T in "$BASE" "$NEW"; do run "$T" warm silicon 1000 1000; done
# TRIM_ARMS ("MAT NE BLOCK;..."), TRIM_REPS and TRIM_NOTEST=1 (skip the pytest step) override the defaults.
IFS=';' read -r -a ARMS <<< "${TRIM_ARMS:-silicon 20000 4096;tungsten 8192 4096}"
for ARM in "${ARMS[@]}"; do
  read -r MAT NE BLOCK <<< "$ARM"
  FILES=()
  for REP in $(seq "${TRIM_REPS:-2}"); do
    run "$BASE" "base$REP" "$MAT" "$NE" "$BLOCK"
    run "$NEW" "new$REP" "$MAT" "$NE" "$BLOCK"
    FILES+=("$OUT/$MAT-$NE-base$REP.json" "$OUT/$MAT-$NE-new$REP.json")
  done
  # Reference is base1; every other run, base repeats included, is compared against it.
  pixi run --as-is -e nvidia python "$NEW/checks/streamed_scoring_memory.py" compare "${FILES[@]}" | grep -E "spec: |worst"
done
