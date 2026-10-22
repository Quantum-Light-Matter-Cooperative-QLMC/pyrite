#!/bin/bash
#SBATCH --job-name=penred510strag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=penred-straggling-%j.log
# Straggling-on PenRed/PyRITE plumbing comparison for #510: four scenes, one
# seed per engine. Usage: sbatch penred-straggling.sbatch ENGINE (penred|pyrite).
# Each run has a timeout; failures stay retained records, never overwritten.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENGINE="$1"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_MC_BACKEND=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
PIN=145a8a357f08affa551f592300fa6dfcab98a762
EXE="$HOME/.local/opt/penred-510/$PIN/source/src/compiled/mains/penmain/pen_main"
OUT="evidence/penred-straggling/$ENGINE-$SLURM_JOB_ID"
mkdir -p "$OUT/source"
scontrol show job "$SLURM_JOB_ID" > "$OUT/job.txt"
cp checks/equal_precision/{penred_adapter.py,pyrite_adapter.py,penred-straggling.sbatch,manifest.json,matrix.json} \
  src/pyrite/perf/penred.py "$OUT/source/"
sha256sum "$OUT"/source/* > "$OUT/code.txt"
SCENES=(
  silicon-300kev-100um-0deg-anchor
  tungsten-300kev-10um-0deg-anchor
  silicon-800kev-tr0.3-0deg-straggling
  tungsten-800kev-tr0.3-0deg-straggling
)
run() {
  local scene="$1"; shift
  local rc=0
  timeout 140 "$HOME/.local/bin/uv" run --no-sync python "$@" \
    > "$OUT/$scene.stdout" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$scene exit $rc (124 = timeout)" | tee -a "$OUT/failures.txt"
  fi
}
for scene in "${SCENES[@]}"; do
  if [ "$ENGINE" = penred ]; then
    sha256sum "$EXE" > "$OUT/executable.txt"
    run "$scene" checks/equal_precision/penred_adapter.py "$scene" --arm source \
      --histories 100000 --seed 101 --executable "$EXE" --timeout-s 130 \
      --output "$OUT/$scene.json"
  else
    nvidia-smi -q > "$OUT/gpu.txt"
    run "$scene" checks/equal_precision/pyrite_adapter.py "$scene" --arm source \
      --straggling on --histories 10000 --batch-histories 5000 --seed 81001 \
      --core cuda --scratch-mib 4096 --max-steps 80000 --output "$OUT/$scene.json"
  fi
done
