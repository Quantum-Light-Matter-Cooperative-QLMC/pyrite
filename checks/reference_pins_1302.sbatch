#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 04:00:00
#SBATCH -o rp1302-%j.log
# #1302 formerly pinned family share of the automatic selection.
# usage (from an isolated synced checkout root holding CODE_SHA and the shipped
# user config under .xdg-config/pyrite):
#   sbatch -J rp1302-<sha> checks/reference_pins_1302.sbatch \
#       OUT_DIR "MATERIAL:E0_KEV:TILT:H,K,L ..." HISTORIES [CASE_TIMEOUT_S]
# CPU node (qlmc-queen): sbatch -p cpu -w qlmc-queen --gres=none --export=ALL,PROBE_CPU=1 ...
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
if [ "${PROBE_CPU:-0}" = 1 ]; then
    export PYRITE_MC_BACKEND=cpu UV_CACHE_DIR="$HOME/.cache/uv-cpu"
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cpu" NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    pyrite_gpu_sync() { uv sync --locked "$@"; }
    pyrite_gpu_info() { nproc; free -g; }
else
    . checks/rocm/env.sh
fi
export XDG_CONFIG_HOME="$PWD/.xdg-config" PYRITE_USER_CATALOG="$PWD/.xdg-config/pyrite"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
export PROBE_SHA=$(cat CODE_SHA)
OUT=$1; CASES=$2; HISTORIES=$3; CASE_TIMEOUT=${4:-5400}
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) backend ${PYRITE_MC_BACKEND:-}"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
H=checks/reference_pins_1302.py
read -ra case_list <<< "$CASES"
for spec in "${case_list[@]}"; do
    IFS=: read -r material e0 tilt family <<< "$spec"
    stem="$OUT/$material-$e0-$tilt-n$HISTORIES"
    echo "== $(date -Is) $stem"
    timeout "$CASE_TIMEOUT" uv run --no-sync python -u $H run "$material" "$e0" "$tilt" \
        "$HISTORIES" "$family" "$stem.json" > "$stem.log" 2>&1
    echo "rc=$? $(tail -c 600 "$stem.log" | tr '\n' ' ' | cut -c1-600)"
done
uv run --no-sync python $H table "$OUT" | tee "$OUT/table.txt"
sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null
