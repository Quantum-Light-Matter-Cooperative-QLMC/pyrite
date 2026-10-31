#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o streamed864-%j.log
# #864 streamed wall time with vs without per-block hard-photon completion, 4k blocks.
# usage (from the synced checkout root): sbatch -J s864-<sha> checks/streamed_completion_864.sbatch
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME=/home/aamador/pyrite PYRITE_CATALOG=$PWD/src/pyrite/data/catalog PYRITE_USER_CATALOG=$(mktemp -d)/catalog
export PYRITE_MC_BACKEND=cuda
echo "code $(cat CODE_SHA)"
uv sync -q --locked --package pyrite-mc --no-default-groups --extra nvidia
B=(uv run --no-sync python checks/streamed_scoring_memory.py)
for MAT in silicon tungsten; do
  OUT=s864_out/$MAT; mkdir -p $OUT
  "${B[@]}" run streamed 1000 1000 $OUT/warmup.json $MAT 300 cuda 1e5 1
  # Alternate arms so drift hits both equally.
  for REP in 1 2; do
    for MODE in streamed-complete streamed; do
      echo "== $MAT $MODE rep $REP"
      "${B[@]}" run $MODE 20000 4096 $OUT/$MODE-$REP.json $MAT 300 cuda 1e5 1
    done
  done
  "${B[@]}" compare $OUT/streamed-complete-1.json $OUT/streamed-1.json $OUT/streamed-2.json $OUT/streamed-complete-2.json
done
