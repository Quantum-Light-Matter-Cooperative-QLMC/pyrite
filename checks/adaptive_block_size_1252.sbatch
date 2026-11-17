#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 12:00:00
#SBATCH -o ab1252-%j.log
# #1252 adaptive block-size matrix: one arm per process for each listed case and
# block, then stopping-rule replays from the longest capture of each case.
# usage (from an isolated synced checkout root holding CODE_SHA and the shipped
# user config under .xdg-config/pyrite):
#   sbatch [-p gpu-amd --gres=gpu:1] -J ab1252-<sha> checks/adaptive_block_size_1252.sbatch \
#       OUT_DIR "MATERIAL:E0_KEV:TILT ..." "BLOCK ..." [ARM_TIMEOUT_S]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
SHIPPED_XDG="$PWD/.xdg-config"
. checks/rocm/env.sh
export XDG_CONFIG_HOME="$SHIPPED_XDG"
export PROBE_SHA=$(cat CODE_SHA)
OUT=$1; CASES=$2; BLOCKS=$3; ARM_TIMEOUT=${4:-2700}
mkdir -p "$OUT"
echo "code $PROBE_SHA host $(hostname) backend $PYRITE_MC_BACKEND"
pyrite_gpu_sync -q || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
H=checks/adaptive_block_size_1252.py
read -ra case_list <<< "$CASES"
read -ra block_list <<< "$BLOCKS"
for spec in "${case_list[@]}"; do
    IFS=: read -r material e0 tilt <<< "$spec"
    for block in "${block_list[@]}"; do
        stem="$OUT/$material-$e0-$tilt-b$block"
        echo "== $(date -Is) $stem"
        timeout "$ARM_TIMEOUT" uv run --no-sync python -u $H run "$material" "$e0" "$tilt" "$block" "$stem" \
            > "$stem.log" 2>&1
        echo "rc=$? $(tail -c 400 "$stem.log" | tr '\n' ' ' | cut -c1-400)"
    done
    best=$(ls -S "$OUT/$material-$e0-$tilt"-b*.values.npz 2>/dev/null | head -1)
    if [ -n "$best" ]; then
        uv run --no-sync python $H replay "$best" "$OUT/$material-$e0-$tilt.replay.json" \
            1000 2000 3000 6000 12000 24000 48000 > /dev/null 2>&1
        echo "replay rc=$? from $best"
    fi
done
uv run --no-sync python $H table "$OUT" | tee "$OUT/table.txt"
sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null
