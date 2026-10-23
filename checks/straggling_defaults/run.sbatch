#!/bin/bash
#SBATCH --job-name=straggling642
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=straggling-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_MC_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
material="${1:?material or anchors or refinement}"
out="evidence/straggling/$SLURM_JOB_ID-$material"
mkdir -p "$out/source"
scontrol show job "$SLURM_JOB_ID" > "$out/allocation.txt"
nvidia-smi -q > "$out/gpu.txt"
lscpu > "$out/cpu.txt"
cp .pyrite-sync "$out/sync.json"
cp checks/straggling_defaults/{study.py,protocol.json,run.sbatch} "$out/source/"
sha256sum src/pyrite/data/conduction_band.toml > "$out/conduction-band.txt"
run() {
  local name="$1"; shift
  local rc=0
  timeout 120 "$HOME/.local/bin/uv" run --no-sync python checks/straggling_defaults/study.py \
    "$@" "$out/$name" > "$out/$name.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$name exit $rc (124 = timeout)" | tee -a "$out/failures.txt"
  fi
}
if [ "$material" = anchors ]; then
  export PYRITE_MC_BACKEND=cpu
  # conftest clears PYRITE_HOME. Keep its platform-data fallback inside this
  # checkout as well, so it finds the synced tables without shared user state.
  export XDG_DATA_HOME="$PWD/test-data"
  mkdir -p "$XDG_DATA_HOME"
  if [ ! -e "$XDG_DATA_HOME/pyrite" ]; then
    ln -s "$PWD" "$XDG_DATA_HOME/pyrite"
  fi
  timeout 500 "$HOME/.local/bin/uv" run --no-sync pyrite-dev test \
    tests/montecarlo/test_straggling_cuda.py \
    tests/montecarlo/test_shell_soft_hard_cuda.py --durations=0 > "$out/tests.log" 2>&1
elif [ "$material" = cpu ]; then
  export PYRITE_MC_BACKEND=cpu
  for switch in off on; do
    run "silicon-0.8-$switch" silicon 0.8 "$switch" --core per-electron
  done
elif [ "$material" = refinement ]; then
  for cap in 0.00025 0.000125; do
    run "cap-$cap" silicon 0.8 on --cap "$cap"
  done
  for cutoff in 100 200; do
    run "cutoff-$cutoff" silicon 0.8 on --cutoff "$cutoff"
  done
else
  for fraction in 0.05 0.8; do
    for switch in off on; do
      run "$fraction-$switch" "$material" "$fraction" "$switch"
    done
  done
fi
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
