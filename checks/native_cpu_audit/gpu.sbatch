#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 03:00:00
#SBATCH -o gpuhost1432-%j.log
# #1432 GPU-host attribution over MATERIAL:E0_KEV:TILT cases of $PROFILE, one
# process per arm on the production serial device path (production thread
# defaults, recorded). Arms ($ARMS, default "warm sampler nsys"):
#   warm     one sampler arm on the first case only (Numba/CuPy cache warm-up)
#   sampler  10 ms main-thread stack sampler joined with NVML utilization
#   prof     cProfile instead of the sampler
#   nsys     `nsys profile` (production --nsys flags, PYRITE_MC_NSYS=1), exported
#            to SQLite and reduced by nsys_gaps.py; the .nsys-rep stays on the box
# The policy-grid speed cache is cleared before every arm.
# usage (isolated synced checkout root holding CODE_SHA and .xdg-config/pyrite):
#   sbatch -J gpuhost1432-<sha> checks/native_cpu_audit/gpu.sbatch OUT_DIR "hopg:300:5 hopg:300:85" [ARM_TIMEOUT_S]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source checks/native_cpu_audit/bind.sh
. checks/rocm/env.sh
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite/catalog"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROFILE="${PROFILE:-cleo_test}" PROBE_SHA=$(cat CODE_SHA)
OUT=$1; CASES=$2; ARM_TIMEOUT=${3:-1800}; ARMS=${ARMS:-"warm sampler nsys"}
export PYRITE_CACHE_DIR="$PWD/$OUT/cache"
mkdir -p "$PYRITE_CACHE_DIR" "$OUT/nsys-tmp"
echo "code $PROBE_SHA host $(hostname) backend $PYRITE_MC_BACKEND profile $PROFILE arms $ARMS"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
H=checks/native_cpu_audit/gpu_host.py
arm() {  # stem, command...
    local stem=$1; shift
    rm -rf "$PYRITE_CACHE_DIR/line-grids"
    echo "== $(date -Is) $stem"
    timeout "$ARM_TIMEOUT" "$@" > "$stem.log" 2>&1
    echo "rc=$? $(tail -n 1 "$stem.log" | cut -c1-300)"
}
read -ra case_list <<< "$CASES"
first=1
for spec in "${case_list[@]}"; do
    IFS=: read -r material e0 tilt <<< "$spec"
    stem="$OUT/$material-$e0-$tilt"
    run=(pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H "$material" "$e0" "$tilt")
    for name in $ARMS; do
        case $name in
        warm) [ $first = 1 ] && arm "$stem-warm" "${run[@]}" "$stem-warm.json" ;;
        sampler) arm "$stem-sampler" "${run[@]}" "$stem-sampler.json" ;;
        prof) arm "$stem-prof" "${run[@]}" "$stem-prof.json" --cprofile ;;
        nsys)
            PYRITE_MC_NSYS=1 TMPDIR="$PWD/$OUT/nsys-tmp" arm "$stem-nsys" \
                nsys profile --trace=cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree \
                --wait=all --force-overwrite=true --output="$stem" \
                "${run[@]}" "$stem-nsys.json" --nsys
            if [ -f "$stem.nsys-rep" ]; then
                nsys export --type sqlite --force-overwrite true --output "$stem.sqlite" "$stem.nsys-rep" \
                    > "$stem-export.log" 2>&1
                pixi run --as-is -e "$PYRITE_PIXI_ENV" python checks/native_cpu_audit/nsys_gaps.py "$stem.sqlite" "$stem-gaps.json" \
                    > "$stem-gaps.log" 2>&1
                echo "gaps rc=$? $(tail -n 1 "$stem-gaps.log" | cut -c1-400)"
                rm -f "$stem.sqlite"
            fi ;;
        *) echo "unknown arm $name" >&2 ;;
        esac
    done
    first=0
done
sha256sum "$OUT"/*.json "$OUT"/*.log "$OUT"/gpu-info.txt > "$OUT/SHA256SUMS" 2>/dev/null
