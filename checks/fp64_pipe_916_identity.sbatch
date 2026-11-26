#!/bin/bash
# #916 follow-up: line-info build identity (#907 scene and #831 anchor),
# compiled registers, and the FP64-pipe active-lane microbenchmark.
# usage (from an isolated synced checkout root): sbatch checks/fp64_pipe_916_identity.sbatch
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o fp64pipe916id-%j.log
set -uo pipefail
export PATH=$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME=$HOME/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$(mktemp -d)/catalog XDG_CONFIG_HOME=$(mktemp -d)
export PYRITE_MC_BACKEND=cuda
pixi install --locked -e nvidia
OUT=evidence/fp64pipe916/$SLURM_JOB_ID; mkdir -p "$OUT"; cp CODE_SHA "$OUT/"
export CUPY_CACHE_DIR=$PWD/$OUT/cupy-cache
py() { pixi run --as-is -e nvidia python -u checks/fp64_pipe_916.py "$@"; }
py pipe > "$OUT/pipe.jsonl" 2>"$OUT/pipe.err"; echo "pipe rc=$?"; cat "$OUT/pipe.jsonl"
py scene-identity 2048 > "$OUT/scene-identity.jsonl" 2>"$OUT/scene-identity.err"; echo "scene rc=$?"; cat "$OUT/scene-identity.jsonl"
py identity 1000 > "$OUT/identity.jsonl" 2>"$OUT/identity.err"; echo "identity rc=$?"; tail -1 "$OUT/identity.jsonl"
echo done
