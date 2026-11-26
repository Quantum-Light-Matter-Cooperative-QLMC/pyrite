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
# CPU stopping capture (any CPU node; PROBE_CAPTURE_EVERY keeps a timed-out prefix):
#   sbatch -p cpu -w qlmc-queen --gres=none -c 12 --export=ALL,PROBE_CPU=1,PROBE_CAPTURE_EVERY=1 ...
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
SHIPPED_XDG="$PWD/.xdg-config"
if [ "${PROBE_CPU:-0}" = 1 ]; then
    # CPU stopping captures (no GPU): per-electron core, numba threads = CPUs.
    export PATH="$HOME/.pixi/bin:$PATH" PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
    export PYRITE_USER_CATALOG="$PWD/user-catalog" PYRITE_MC_BACKEND=cpu PYRITE_PIXI_ENV=default
    export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    mkdir -p "$PYRITE_USER_CATALOG"
    pyrite_gpu_sync() { pixi install --locked -e "$PYRITE_PIXI_ENV" "$@"; }
    pyrite_gpu_info() { lscpu | head -20; }
else
    . checks/rocm/env.sh
fi
# The profile, beam and detector are user-catalog overlays (default: the config dir).
export XDG_CONFIG_HOME="$SHIPPED_XDG" PYRITE_USER_CATALOG="$SHIPPED_XDG/pyrite"
# Generated xsgen tables are content-addressed in the box's default workspace.
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
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
        timeout "$ARM_TIMEOUT" pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u $H run "$material" "$e0" "$tilt" "$block" "$stem" \
            > "$stem.log" 2>&1
        echo "rc=$? $(tail -c 400 "$stem.log" | tr '\n' ' ' | cut -c1-400)"
    done
    best=$(ls -S "$OUT/$material-$e0-$tilt"-b*.values.npz 2>/dev/null | head -1)
    if [ -n "$best" ]; then
        pixi run --as-is -e "$PYRITE_PIXI_ENV" python $H replay "$best" "$OUT/$material-$e0-$tilt.replay.json" \
            1000 2000 3000 6000 12000 24000 48000 > /dev/null 2>&1
        echo "replay rc=$? from $best"
    fi
done
pixi run --as-is -e "$PYRITE_PIXI_ENV" python $H table "$OUT" | tee "$OUT/table.txt"
sha256sum "$OUT"/* > "$OUT/SHA256SUMS" 2>/dev/null
