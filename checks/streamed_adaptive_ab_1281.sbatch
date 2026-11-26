#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH -t 24:00:00
#SBATCH -o ab1281-%j.log
# #1281 streamed vs retained adaptive A/B: one process per arm.
# usage (from an isolated synced checkout root holding CODE_SHA and the shipped
# user config under .xdg-config/pyrite):
#   sbatch -J ab1281-<sha> checks/streamed_adaptive_ab_1281.sbatch OUT_DIR \
#       "ROUTE:MATERIAL:E0_KEV:TILT:MAX_HISTORIES ..." [ARM_TIMEOUT_S] [COMPARE_PAIRS]
# COMPARE_PAIRS: "REF_STEM,OTHER_STEM ..." (stems under OUT_DIR).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROBE_SHA=$(cat CODE_SHA)
OUT=$1; ARMS=$2; ARM_TIMEOUT=${3:-36000}; PAIRS=${4:-}
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) backend ${PYRITE_MC_BACKEND:-}"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
H=checks/streamed_adaptive_ab_1281.py
read -ra arm_list <<< "$ARMS"
for spec in "${arm_list[@]}"; do
    IFS=: read -r route material e0 tilt nmax <<< "$spec"
    stem="$OUT/$material-$e0-$tilt-n$nmax-$route"
    echo "== $(date -Is) $stem"
    timeout "$ARM_TIMEOUT" pixi run --as-is -e amd python -u $H run "$route" "$material" "$e0" "$tilt" \
        "$nmax" "$stem" > "$stem.log" 2>&1
    echo "rc=$? $(tail -c 400 "$stem.log" | tr '\n' ' ' | cut -c1-400)"
    pixi run --as-is -e amd python $H table "$OUT" > "$OUT/table.txt" 2>&1
done
read -ra pair_list <<< "$PAIRS"
for pair in "${pair_list[@]}"; do
    IFS=, read -r ref other <<< "$pair"
    pixi run --as-is -e amd python $H compare "$OUT/$ref" "$OUT/$other" > "$OUT/compare-$other.json" 2>&1
    echo "compare $pair rc=$?"
done
cat "$OUT/table.txt"
sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null
