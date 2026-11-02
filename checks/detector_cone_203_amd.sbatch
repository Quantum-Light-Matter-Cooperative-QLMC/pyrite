#!/bin/bash
#SBATCH -p gpu-amd
#SBATCH --nodelist=qlmc-ace
#SBATCH --gres=gpu:radeon8060s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 02:00:00
#SBATCH -o cone203-amd-%j.log
# Experimental pilot only; no independent validation or characteristic tally.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD"
export PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-amd"
export OPENBLAS_NUM_THREADS=1
source src/pyrite/remote/rocm_build_env.sh
_pyrite_rocm_env "$PWD"
export PYRITE_MC_BACKEND=rocm
echo "code $(cat CODE_SHA)"
uv sync --locked --no-default-groups --extra amd
B=(uv run --no-sync python checks/detector_cone_variance_reduction.py)
OUT=${1:-cone203_amd_out}
mkdir -p "$OUT"
for SCENE in hbn5 hopg15 hopg30; do
  if [[ $SCENE == hbn5 ]]; then
    CASE=(--material hbn --energy 5000 --tilt 10 --azimuth 100)
  elif [[ $SCENE == hopg15 ]]; then
    CASE=(--material hopg --energy 15000 --tilt 45 --azimuth 180)
  else
    CASE=(--material hopg --energy 29999 --tilt 45 --azimuth 180)
  fi
  # Warm each arm in its own process; recorded timings still include per-process
  # first-call overhead. Full-spectrum baseline uses the production grid policy.
  "${B[@]}" compare "${CASE[@]}" --ne 8 --seed 999 --out "$OUT/$SCENE-warmup.json"
  "${B[@]}" baseline "${CASE[@]}" --ne 100 --ne-cpu 0 --seed 0 --out "$OUT/$SCENE-baseline.json"
  for SEED in 0 1 2 3 4; do
    "${B[@]}" compare "${CASE[@]}" --ne 512 --seed "$SEED" --out "$OUT/$SCENE-seed$SEED.json"
  done
done
