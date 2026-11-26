#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 02:00:00
#SBATCH -o lr1295-%j.log
# #1295 selected-set line-mass concentration: one process per case.
# usage (from an isolated synced checkout root holding CODE_SHA and the shipped
# user config under .xdg-config/pyrite):
#   sbatch [-p cpu] -J lr1295-<sha> checks/line_roulette_selection_1295.sbatch \
#       OUT_DIR "MATERIAL:E0_KEV:TILT ..." HISTORIES SEED [CASE_TIMEOUT_S]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
if [ "${SLURM_JOB_PARTITION:-gpu}" = cpu ]; then
    export PYRITE_MC_BACKEND=cpu PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
    export PYRITE_PIXI_ENV=default
else
    . checks/rocm/env.sh
fi
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROBE_SHA=$(cat CODE_SHA)
OUT=$1; CASES=$2; HISTORIES=$3; SEED=$4; CASE_TIMEOUT=${5:-5400}
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) partition ${SLURM_JOB_PARTITION:-} backend ${PYRITE_MC_BACKEND:-}"
if [ "${SLURM_JOB_PARTITION:-gpu}" != cpu ]; then
    pyrite_gpu_sync -q || exit 1
    pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
else
    pixi install --locked -e "$PYRITE_PIXI_ENV" || exit 1
fi
H=checks/line_roulette_selection_1295.py
read -ra case_list <<< "$CASES"
for spec in "${case_list[@]}"; do
    IFS=: read -r material e0 tilt <<< "$spec"
    stem="$OUT/$material-$e0-$tilt-n$HISTORIES-s$SEED"
    echo "== $(date -Is) $stem"
    timeout "$CASE_TIMEOUT" pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H run "$material" "$e0" "$tilt" \
        "$HISTORIES" "$SEED" "$stem" > "$stem.log" 2>&1
    echo "rc=$? $(tail -c 600 "$stem.log" | tr '\n' ' ' | cut -c1-600)"
done
pixi run --as-is -e "$PYRITE_PIXI_ENV" python $H table "$OUT" | tee "$OUT/table.txt"
sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null
