#!/bin/bash
#SBATCH --job-name=cont799-qualify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=05:00:00
#SBATCH --output=qualify-schedule-799-%j.log
# #799: production-default runner cases under full-history vs auto schedules.
# One process per sample; schedules interleaved, order reversed on even seeds.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BR="$PWD"
export PATH="$HOME/.local/bin:$PATH" PYRITE_MC_BACKEND=cuda UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog"
export XDG_CONFIG_HOME="$BR/config-home" XDG_DATA_HOME="$BR/data-home"
mkdir -p "$XDG_CONFIG_HOME"
out="$BR/evidence/continuation/qualify799/$SLURM_JOB_ID"; mkdir -p "$out"
git rev-parse HEAD > "$out/head.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"; nvidia-smi -q > "$out/gpu.txt"
uv sync --package pyrite-mc --no-default-groups --group dev --extra nvidia -q
for code in elsepa bremslib sbethe sbethe-tables eedl eadl; do uv run --no-sync pyrite tables fetch "$code" > /dev/null; done
run() {  # schedule scene histories seed
  local stem="$2-$3-$4-$1"
  PYRITE_MC_TRANSPORT_SCHEDULE=$1 timeout --signal=TERM --kill-after=5s 2400s uv run --no-sync \
    python "$BR/checks/equal_precision/qualify_schedule.py" "$2" --histories "$3" --seed "$4" \
    --output "$out/$stem.json" > "$out/$stem.log" 2>&1 \
    && echo "DONE $stem" || echo "FAILED $stem rc=$?"
}
for histories in 5000 20000 100000; do
  seeds="79901 79902"; (( histories == 100000 )) && seeds="79901"
  for seed in $seeds; do
    order="full-history auto"; (( seed % 2 == 0 )) && order="auto full-history"
    for scene in silicon tungsten pbwo4; do
      for schedule in $order; do run "$schedule" "$scene" "$histories" "$seed"; done
    done
  done
done
