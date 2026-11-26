#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 12:00:00
#SBATCH -o rs1281-%j.log
# #1281 electron-prefix selection matrix: one process per case and seed.
# usage (from an isolated synced checkout root holding CODE_SHA and the shipped
# user config under .xdg-config/pyrite):
#   sbatch [-p gpu-amd] -J rs1281-<sha> checks/reflection_selection_subsample_1281.sbatch \
#       OUT_DIR "MATERIAL:E0_KEV:TILT ..." HISTORIES "SEED ..." BUDGETS [ARM_TIMEOUT_S]
# BUDGETS is comma-separated; the first arm also runs the production pass; 0 = full.
set -uo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
. checks/rocm/env.sh
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROBE_SHA=$(cat CODE_SHA)
OUT=$1; CASES=$2; HISTORIES=$3; SEEDS=$4; BUDGETS=$5; ARM_TIMEOUT=${6:-10800}
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) backend ${PYRITE_MC_BACKEND:-}"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
H=checks/reflection_selection_subsample_1281.py
read -ra case_list <<< "$CASES"
read -ra seed_list <<< "$SEEDS"
for spec in "${case_list[@]}"; do
    IFS=: read -r material e0 tilt <<< "$spec"
    for seed in "${seed_list[@]}"; do
        stem="$OUT/$material-$e0-$tilt-n$HISTORIES-s$seed"
        echo "== $(date -Is) $stem"
        timeout "$ARM_TIMEOUT" pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H run "$material" "$e0" "$tilt" \
            "$HISTORIES" "$seed" "$BUDGETS" "$stem.json" > "$stem.log" 2>&1
        echo "rc=$? $(tail -c 600 "$stem.log" | tr '\n' ' ' | cut -c1-600)"
    done
done
pixi run --as-is -e "$PYRITE_PIXI_ENV" python $H table "$OUT" | tee "$OUT/table.txt"
sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null
