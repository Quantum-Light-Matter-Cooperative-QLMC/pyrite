#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 03:00:00
#SBATCH -o charmem1455-%j.log
# #1455 acceptance "no device-memory regression": device-memory high-water of
# the $PROFILE (default cleo_test) MATERIAL E0 case at each tilt for three
# characteristic arms, one process each: legacy (grouped, chunk 615, approx.
# pre-#1455), grouped (automatic chunk, #1455 A), default (fused from 64k rows,
# #1455 B). Production serial device path; the policy-grid speed cache is
# cleared before every arm, Numba/CuPy caches stay warm. One summary line per
# arm in summary.jsonl.
# usage (isolated synced checkout root holding CODE_SHA and .xdg-config/pyrite):
#   sbatch -J charmem1455-<sha> checks/characteristic_device_memory_1455.sbatch \
#       [OUT_DIR=charmem1455_out] [MATERIAL=hopg] [E0_KEV=300] ["TILT ..."="85 5"] [ARM_TIMEOUT_S=2400]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite/catalog"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROFILE="${PROFILE:-cleo_test}" PROBE_SHA=$(cat CODE_SHA 2>/dev/null || echo unknown)
OUT=${1:-charmem1455_out}; MATERIAL=${2:-hopg}; E0=${3:-300}; TILTS=${4:-"85 5"}; ARM_TIMEOUT=${5:-2400}
export PYRITE_CACHE_DIR="$PWD/$OUT/cache"
mkdir -p "$PYRITE_CACHE_DIR"
echo "code $PROBE_SHA host $(hostname) backend $PYRITE_MC_BACKEND profile $PROFILE"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
read -ra tilt_list <<< "$TILTS"
STATUS=0
for tilt in "${tilt_list[@]}"; do
    for arm in legacy grouped default; do
        rm -rf "$PYRITE_CACHE_DIR/line-grids"
        stem="$OUT/$MATERIAL-$E0-$tilt-$arm"
        echo "== $(date -Is) $stem"
        timeout "$ARM_TIMEOUT" pixi run --as-is -e amd python -u checks/characteristic_device_memory_1455.py \
            "$MATERIAL" "$E0" "$tilt" "$arm" "$stem.json" > "$stem.log" 2>&1
        rc=$?
        [ "$rc" -ne 0 ] && STATUS=1
        line=$(tail -n 1 "$stem.log")
        echo "rc=$rc $line"
        [ "$rc" -eq 0 ] && echo "$line" >> "$OUT/summary.jsonl"
    done
done
sha256sum "$OUT"/*.json "$OUT"/*.log "$OUT"/gpu-info.txt > "$OUT/SHA256SUMS" 2>/dev/null
exit $STATUS
