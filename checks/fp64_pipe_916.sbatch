#!/bin/bash
# #916 hot FP64 sources: Nsight Compute source counters on the #907 scene's
# full-population (16,384-history) transport launch, line-info build.
# usage (from an isolated synced checkout root): sbatch checks/fp64_pipe_916.sbatch
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o fp64pipe916-%j.log
set -uo pipefail
export PATH=$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$HOME/tmp-nsys"; mkdir -p "$TMPDIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$(mktemp -d)/catalog XDG_CONFIG_HOME=$(mktemp -d)
export PYRITE_MC_BACKEND=cuda PROBE_SHA=$(cat CODE_SHA)
pixi install --locked -e nvidia
OUT=evidence/fp64pipe916/$SLURM_JOB_ID; mkdir -p "$OUT"
cp CODE_SHA "$OUT/"; nvidia-smi > "$OUT/gpu.txt"; ncu --version > "$OUT/ncu-version.txt"
export CUPY_CACHE_DIR=$PWD/$OUT/cupy-cache CUPY_CACHE_SAVE_CUDA_SOURCE=1
py() { pixi run --as-is -e nvidia python -u "$@"; }

echo "== identity"; py checks/fp64_pipe_916.py identity 1000 > "$OUT/identity.jsonl" 2>"$OUT/identity.err"
echo "identity rc=$?"; tail -1 "$OUT/identity.jsonl"

echo "== ncu"
ncu --set full --import-source yes --kernel-name regex:transport --launch-skip 24 --launch-count 12 \
  -o "$OUT/ncu" -f \
  pixi run --as-is -e nvidia python -u checks/fp64_pipe_916.py run "$OUT/kernels" \
  transport 16384 16384 2048 32 0 "$OUT/run" > "$OUT/run.log" 2>&1
echo "ncu rc=$?"
ncu --import "$OUT/ncu.ncu-rep" --page raw --csv \
  --metrics launch__grid_size,launch__registers_per_thread,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active \
  > "$OUT/raw.csv"
cat "$OUT/raw.csv"
for i in $(seq 0 11); do
  ncu --import "$OUT/ncu.ncu-rep" --launch-skip "$i" --launch-count 1 --page source --csv \
    --print-source cuda,sass --source-folders "$CUPY_CACHE_DIR" > "$OUT/source-$i.csv" 2>"$OUT/source-$i.err"
  ncu --import "$OUT/ncu.ncu-rep" --launch-skip "$i" --launch-count 1 --page source --csv \
    --print-source sass > "$OUT/sass-$i.csv" 2>>"$OUT/source-$i.err"
done
ncu --import "$OUT/ncu.ncu-rep" --page details --csv > "$OUT/details.csv"
ls -la "$OUT" "$OUT/kernels" "$CUPY_CACHE_DIR" | head -60
echo "done"
