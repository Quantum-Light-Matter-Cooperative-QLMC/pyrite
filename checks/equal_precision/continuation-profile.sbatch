#!/bin/bash
#SBATCH --job-name=cont789-profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=continuation-profile-%j.log
# Transfer, synchronization and SM-activity attribution for #789. Each policy
# runs in its own process with a full warmup; nsys captures only the NVTX range
# "bench.measured" (the measured request), and ncu profiles only transport
# kernels inside it. Profiled timings are excluded from benchmark summaries.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BR="$PWD"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYRITE_MC_BACKEND=cuda
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog"
out="$BR/evidence/continuation/profile/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.txt"; scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
nvidia-smi -q > "$out/gpu.txt"; nsys --version > "$out/nsys-version.txt"; ncu --version > "$out/ncu-version.txt" || true
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
bench=(uv run --no-sync python checks/equal_precision/benchmark_continuation.py)
reports=cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum
profile() {  # stem scene histories policy
  local stem=$1 scene=$2 n=$3 policy=$4
  if PYRITE_MC_NSYS=1 NSYS_NVTX_PROFILER_REGISTER_ONLY=0 timeout --signal=TERM --kill-after=5s 600s \
      nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=nvtx --nvtx-capture=bench.measured --capture-range-end=stop \
      --output="$out/$stem" --force-overwrite=true \
      "${bench[@]}" "$scene" --histories "$n" --seed 78601 --policies "$policy" \
      --output "$out/$stem.json" > "$out/$stem.log" 2>&1; then
    nsys stats --report "$reports" --format csv --output "$out/$stem" "$out/$stem.nsys-rep" \
      > "$out/$stem.stats.log" 2>&1 && echo "DONE nsys $stem" || echo "FAILED nsys-stats $stem"
  else
    echo "{\"status\":\"failed\",\"exit_code\":$?}" > "$out/$stem.failed.json"; echo "FAILED nsys $stem"
  fi
}
profile w5000-learned32 w-300-10um-0deg 5000 learned32
profile w5000-cont256 w-300-10um-0deg 5000 cont-Q256-Aall-T32
profile w5000-cont256D w-300-10um-0deg 5000 cont-Q256-Aall-T32-D
profile si5000-learned32 si-300-100um-0deg 5000 learned32
profile si5000-cont256 si-300-100um-0deg 5000 cont-Q256-Aall-T32
profile w20000-learnedB5000 w-300-10um-0deg 20000 learned32-B5000
profile w20000-cont256A5000 w-300-10um-0deg 20000 cont-Q256-A5000-T32
profile w20000-cont256A5000D w-300-10um-0deg 20000 cont-Q256-A5000-T32-D
# Hardware counters: SM activity over elapsed time, achieved warps, SIMT efficiency.
metrics=gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__registers_per_thread
metrics+=,launch__occupancy_limit_registers,sm__maximum_warps_per_active_cycle_pct
metrics+=,sm__cycles_active.avg.pct_of_peak_sustained_elapsed
metrics+=,sm__warps_active.avg.pct_of_peak_sustained_active
metrics+=,smsp__thread_inst_executed_per_inst_executed.ratio
counters() {  # stem policy
  local stem=$1 policy=$2
  if timeout --signal=TERM --kill-after=5s 900s ncu --nvtx --nvtx-include "bench.measured/" \
      --kernel-name regex:_transport_kernel --metrics "$metrics" --export "$out/$stem" --force-overwrite \
      "${bench[@]}" w-300-10um-0deg --histories 5000 --seed 78601 --policies "$policy" \
      --output "$out/$stem.json" > "$out/$stem.log" 2>&1; then
    ncu --import "$out/$stem.ncu-rep" --csv --page raw > "$out/$stem.csv" 2>&1 \
      && echo "DONE ncu $stem" || echo "FAILED ncu-import $stem"
  else
    echo "{\"status\":\"failed\",\"exit_code\":$?}" > "$out/$stem.failed.json"; echo "FAILED ncu $stem"
    grep -m3 -iE "ERR_NVGPUCTRPERM|permission|error" "$out/$stem.log" || true
  fi
}
counters ncu-w5000-learned32 learned32
counters ncu-w5000-cont1024 cont-Q1024-Aall-T32
counters ncu-w5000-cont256 cont-Q256-Aall-T32
