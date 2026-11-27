#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 01:30:00
#SBATCH -o fd1481-%j.log
# #1481 Fourier driver host cost: capture flushes once (base code), replay them
# on the base and candidate checkouts (timing, outputs, then under nsys), time
# the planning threshold on the candidate, compare.
# usage (from the candidate's isolated synced checkout root holding CODE_SHA
# and the shipped user config under .xdg-config/pyrite; BASE_DIR likewise):
#   sbatch -J fd1481-<sha> checks/fourier_driver_1481.sbatch \
#       BASE_DIR OUT_DIR FLUSH_DIR MATERIAL E0_KEV TILT HISTORIES [REPEATS]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
CAND=$PWD
. checks/rocm/env.sh
export PATH="$HOME/.pixi/bin:$PATH"
export XDG_CONFIG_HOME="$CAND/.xdg-config" PYRITE_USER_CATALOG="$CAND/.xdg-config/pyrite/catalog"
export PYRITE_HOME="${PROBE_WORKSPACE:-$HOME/.pyrite/workspace}"
BASE=$(cd "$1" && pwd); OUT=$(mkdir -p "$2" && cd "$2" && pwd); FLUSH=$(mkdir -p "$3" && cd "$3" && pwd)
MATERIAL=$4; E0=$5; TILT=$6; HISTORIES=$7; REPEATS=${8:-5}
H="$CAND/checks/fourier_driver_1481.py"
echo "cand $(cat CODE_SHA) base $(cat "$BASE/CODE_SHA") host $(hostname) backend ${PYRITE_MC_BACKEND:-}"
pyrite_gpu_sync -q || exit 1
(cd "$BASE" && pyrite_gpu_sync -q) || exit 1
pyrite_gpu_info > "$OUT/gpu-info.txt" 2>&1
py() {  # dir, args...: that checkout's pixi env, pyrite and packaged catalog
    local dir=$1; shift
    (cd "$dir" && PYRITE_CATALOG="$dir/src/pyrite/data/catalog" PROBE_SHA=$(cat CODE_SHA) pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u "$@")
}
if ! ls "$FLUSH"/flush-*.npz > /dev/null 2>&1; then
    py "$BASE" "$H" capture "$MATERIAL" "$E0" "$TILT" "$HISTORIES" "$FLUSH" > "$OUT/capture.log" 2>&1
    echo "capture rc=$? $(tail -n 1 "$OUT/capture.log" | cut -c1-400)"
    cp "$FLUSH/capture.json" "$OUT/capture.json"
fi
for side in base cand; do
    dir=$CAND; [ $side = base ] && dir=$BASE
    py "$dir" "$H" replay "$FLUSH" "$OUT/replay-$side.json" "$REPEATS" > "$OUT/replay-$side.log" 2>&1
    echo "replay $side rc=$?"
    (cd "$dir" && PYRITE_CATALOG="$dir/src/pyrite/data/catalog" PROBE_SHA=$(cat CODE_SHA) TMPDIR="$OUT" nsys profile --trace=cuda,nvtx,osrt \
        --wait=all --force-overwrite=true --output="$OUT/nsys-$side" \
        pixi run --as-is -e "$PYRITE_PIXI_ENV" python -u "$H" replay "$FLUSH" "$OUT/nsys-replay-$side.json" 2) \
        > "$OUT/nsys-$side.log" 2>&1
    echo "nsys $side rc=$?"
    if [ -f "$OUT/nsys-$side.nsys-rep" ]; then
        nsys export --type sqlite --force-overwrite true --output "$OUT/nsys-$side.sqlite" \
            "$OUT/nsys-$side.nsys-rep" > "$OUT/nsys-$side-export.log" 2>&1
        py "$CAND" checks/native_cpu_audit/nsys_gaps.py "$OUT/nsys-$side.sqlite" "$OUT/nsys-$side-gaps.json" \
            > "$OUT/nsys-$side-gaps.log" 2>&1
        echo "gaps $side rc=$? $(tail -n 1 "$OUT/nsys-$side-gaps.log" | cut -c1-300)"
        rm -f "$OUT/nsys-$side.sqlite"
    fi
done
py "$CAND" "$H" plan "$FLUSH" "$OUT/plan.json" "$REPEATS" > "$OUT/plan.log" 2>&1
echo "plan rc=$?"
py "$CAND" "$H" compare "$OUT/compare.json" "$OUT/replay-base.json" "$OUT/replay-cand.json" \
    > "$OUT/compare.log" 2>&1
echo "compare rc=$? $(head -n 1 "$OUT/compare.log" | cut -c1-400)"
(cd "$OUT" && sha256sum ./*.json ./*.log gpu-info.txt > SHA256SUMS 2>/dev/null)
