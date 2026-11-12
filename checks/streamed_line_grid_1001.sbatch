#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o grid1001-%j.log
# #1001 streamed line_grid_policy: grid-pass memory/wall against N, and joined
# vs two-pass streamed on the same case (grid bitwise, spectra, peaks, wall).
# usage (from a clean, synced checkout root): sbatch -J grid1001-<sha> checks/streamed_line_grid_1001.sbatch si|hbn [OUT]
#   si:  silicon 300 keV, 10 um, tilt 30, automatic policy (sinc-nyquist)
#   hbn: h-BN 5 MeV, 1 mm, tilt 10 / azim 100 (#879), measured resonance-local policy
set -uo pipefail
CASE=$1
ROOT=${2:-grid1001_out}
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$(mktemp -d)/catalog"
export PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1
echo "code $(cat CODE_SHA) case $CASE"
nvidia-smi --query-gpu=name,memory.total --format=csv
uv sync -q --locked --no-default-groups --extra nvidia || exit 1
B=(uv run --no-sync python checks/streamed_scoring_memory.py)
OUT=$ROOT/$CASE; mkdir -p "$OUT"
arm() {  # MODE NE BLOCK NAME
  echo "== $4"
  "${B[@]}" run "$1" "$2" "$3" "$OUT/$4.json" "${ARGS[@]}" || echo "ARM FAILED $4 rc=$?"
}
case $CASE in
  si)
    export STREAM_GRID=policy
    ARGS=(silicon 300 cuda 1e5 1)
    arm streamed 1000 1000 warmup
    for N in 16000 64000 256000; do arm grid-pass "$N" 4000 "grid-pass-$N"; done
    arm joined 64000 0 joined-64000
    arm streamed 64000 4000 streamed-64000-b4000
    arm streamed 64000 16000 streamed-64000-b16000
    "${B[@]}" compare "$OUT/joined-64000.json" "$OUT/streamed-64000-b4000.json" "$OUT/streamed-64000-b16000.json"
    ;;
  hbn)
    export STREAM_GRID=measured
    ARGS=(hbn 5000 cuda 1e7 0 10 100)
    arm grid-pass 64 64 warmup
    for N in 512 2048 8192; do arm grid-pass "$N" 512 "grid-pass-$N"; done
    arm joined 2048 0 joined-2048
    arm streamed 2048 512 streamed-2048-b512
    "${B[@]}" compare "$OUT/joined-2048.json" "$OUT/streamed-2048-b512.json"
    ;;
  *) echo "unknown case $CASE"; exit 2 ;;
esac
