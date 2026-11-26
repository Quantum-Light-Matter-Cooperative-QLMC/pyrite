#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 04:00:00
#SBATCH -o gfm1463-%j.log
# #1463 device fast-math: (1) fresh nsys device-time shares on the #1432
# GPU-host case (gpu_host.py + nsys_gaps.py), (2) capture Fourier flushes of
# one case, (3) replay them through the `rectangles` compile-option arms,
# (4) on NVIDIA with `ncu` on PATH, a short Nsight Compute pass of three arms.
# usage (isolated synced checkout root holding CODE_SHA and .xdg-config/pyrite):
#   sbatch [-p gpu-amd] -J gfm1463-<sha> checks/gpu_fast_math_1463.sbatch OUT_DIR \
#       [PROFILE_CASES="hopg:300:5 hopg:300:45"] [CAPTURE_CASE="hopg 300 5 31100000"]
# The capture case's population is the converged one of the long #1432 tilt
# (job 16068: 31.1M electrons at 5 deg); smaller populations flush too few lines.
# STEPS (default "profile capture arms ncu") selects steps.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export PATH="$PATH:/usr/local/cuda/bin"  # nsys/ncu on the NVIDIA node
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite/catalog"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROFILE="${PROFILE:-cleo_test}" PROBE_SHA=$(cat CODE_SHA)
OUT=$1; PCASES=${2:-"hopg:300:5 hopg:300:45"}; CCASE=${3:-"hopg 300 5 31100000"}
STEPS=${STEPS:-"profile capture arms ncu"}
export PYRITE_CACHE_DIR="$PWD/$OUT/cache"
mkdir -p "$OUT/nsys-tmp" "$OUT/flushes" "$PYRITE_CACHE_DIR"
echo "code $PROBE_SHA host $(hostname) backend $PYRITE_MC_BACKEND env $PYRITE_PIXI_ENV steps $STEPS"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
py=(pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u)
has() { [[ " $STEPS " == *" $1 "* ]]; }
if has profile && command -v nsys > /dev/null; then
    read -ra cases <<< "$PCASES"
    first=1
    for spec in "${cases[@]}"; do
        IFS=: read -r m e t <<< "$spec"; stem="$OUT/$m-$e-$t"
        rm -rf "$PYRITE_CACHE_DIR/line-grids"
        if [ $first = 1 ]; then  # Numba/CuPy cache warm-up, not traced
            timeout 1800 "${py[@]}" checks/native_cpu_audit/gpu_host.py "$m" "$e" "$t" "$stem-warm.json" \
                > "$stem-warm.log" 2>&1; echo "warm rc=$?"; first=0
            rm -rf "$PYRITE_CACHE_DIR/line-grids"
        fi
        PYRITE_MC_NSYS=1 TMPDIR="$PWD/$OUT/nsys-tmp" timeout 1800 \
            nsys profile --trace=cuda,nvtx --wait=all --force-overwrite=true --output="$stem" \
            "${py[@]}" checks/native_cpu_audit/gpu_host.py "$m" "$e" "$t" "$stem-nsys.json" --nsys \
            > "$stem-nsys.log" 2>&1
        echo "nsys $spec rc=$?"
        if [ -f "$stem.nsys-rep" ]; then
            nsys export --type sqlite --force-overwrite true --output "$stem.sqlite" "$stem.nsys-rep" > /dev/null 2>&1
            "${py[@]}" checks/native_cpu_audit/nsys_gaps.py "$stem.sqlite" "$stem-gaps.json" > "$stem-gaps.log" 2>&1
            echo "gaps rc=$?"; rm -f "$stem.sqlite"
        fi
    done
fi
if has capture && ! ls "$OUT"/flushes/flush-*.npz > /dev/null 2>&1; then
    read -ra c <<< "$CCASE"
    CAPTURE=${CAPTURE:-4} CAPTURE_MIN_LINES=${CAPTURE_MIN_LINES:-200000} CAPTURE_EVERY=${CAPTURE_EVERY:-6} \
        timeout "${CAPTURE_TIMEOUT:-3000}" "${py[@]}" checks/fourier_reduction_phases_1296.py capture "${c[@]}" "$OUT/flushes" \
        > "$OUT/capture.log" 2>&1
    echo "capture rc=$? $(ls "$OUT"/flushes/flush-*.npz 2>/dev/null | wc -l) flushes"
fi
if has arms; then
    timeout 3600 "${py[@]}" checks/gpu_fast_math_1463.py arms "$OUT/flushes" "$OUT/arms.json" \
        --repeats "${REPEATS:-7}" > "$OUT/arms.log" 2>&1
    echo "arms rc=$?"
fi
if has ncu && [ "$PYRITE_PIXI_ENV" = nvidia ] && command -v ncu > /dev/null; then
    timeout 1800 ncu --target-processes all --kernel-name 'regex:^rectangles_' --launch-skip 0 --launch-count 40 \
        --metrics gpu__time_duration.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_xu.sum,sm__inst_executed_pipe_fma.sum,smsp__inst_executed.sum \
        --csv --log-file "$OUT/ncu.csv" \
        "${py[@]}" checks/gpu_fast_math_1463.py arms "$OUT/flushes" "$OUT/arms-ncu.json" \
        --repeats 1 --modes 2 --only fp32-prod,fp32-fastmath,fp32-intrinsic > "$OUT/ncu.log" 2>&1
    echo "ncu rc=$?"
fi
rm -rf "$PYRITE_CACHE_DIR" "$OUT/nsys-tmp"
(cd "$OUT" && find . -type f ! -name '*.npz' ! -name '*.nsys-rep' ! -name SHA256SUMS -print0 \
    | xargs -0 sha256sum > SHA256SUMS)
