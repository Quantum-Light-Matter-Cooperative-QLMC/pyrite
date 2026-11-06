#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 05:00:00
#SBATCH -o char1006-%j.log
# #1006 characteristic vs continuum convergence: W and Si, thin to thick, 65536 electrons in
# 64-electron blocks (1024 block means); two seed replicates; 4096-electron timing arms; CUDA
# nested-prefix probes.
# usage (from an isolated synced checkout root): sbatch -J c1006 checks/characteristic_budget_1006.sbatch [OUTDIR]
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA)"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
B=(uv run --no-sync python checks/characteristic_budget_1006.py)
ROOT=${1:-c1006_out}
mkdir -p "$ROOT"
# material E0_keV thickness_ang
CASES=(
  "silicon 30 1e3" "silicon 30 1e5" "silicon 300 1e3" "silicon 300 1e4" "silicon 300 1e5"
  "silicon 300 5e6" "silicon 1000 1e6"
  "tungsten 30 1e3" "tungsten 30 5e4" "tungsten 300 1e3" "tungsten 300 1e4" "tungsten 300 1e5"
  "tungsten 300 1e6" "tungsten 1000 1e6"
)
for C in "${CASES[@]}"; do
  read -r MAT E0 T <<< "$C"
  echo "== $C"
  "${B[@]}" run "$MAT" "$E0" "$T" 65536 64 "$ROOT/${MAT}_${E0}_${T}_s1.npz" 1 cuda | tail -1
  echo "rc=$?"
done
for C in "silicon 300 1e4" "tungsten 300 1e5"; do
  read -r MAT E0 T <<< "$C"
  echo "== replicate $C"
  "${B[@]}" run "$MAT" "$E0" "$T" 65536 64 "$ROOT/${MAT}_${E0}_${T}_s2.npz" 2 cuda | tail -1
done
mkdir -p "$ROOT/timing"
for C in "${CASES[@]}"; do
  read -r MAT E0 T <<< "$C"
  echo "== timing $C"
  "${B[@]}" run "$MAT" "$E0" "$T" 16384 4096 "$ROOT/timing/${MAT}_${E0}_${T}.npz" 1 cuda | tail -1
done
for C in "silicon 30 1e4" "tungsten 300 1e3"; do
  read -r MAT E0 T <<< "$C"
  echo "== prefix cuda $C"
  "${B[@]}" prefix "$MAT" "$E0" "$T" "$ROOT/prefix_${MAT}_${E0}.json" cuda | tail -30
done
"${B[@]}" report "$ROOT/report.json" "$ROOT"/*.npz
