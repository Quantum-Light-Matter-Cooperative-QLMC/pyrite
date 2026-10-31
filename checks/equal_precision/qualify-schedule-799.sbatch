#!/bin/bash
#SBATCH --job-name=cont799-qualify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=05:00:00
#SBATCH --output=qualify-schedule-799-%j.log
# #799: production-default runner transport, without spectral scoring.
# Full-history vs auto, including coupled photon completion in both arms.
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
# A Slurm --mem reservation is NOT memory enforcement. Refuse this job unless
# a real cgroup v2 memory.max <= the reserved 28 GiB protects the node.
MEM_MAX_BYTES=$((28 * 1024 * 1024 * 1024))
python "$BR/checks/equal_precision/contained_sample.py" --max-bytes "$MEM_MAX_BYTES" -- /usr/bin/true
uv sync --package pyrite-mc --no-default-groups --group dev --extra nvidia -q
for code in elsepa bremslib sbethe sbethe-tables eedl eadl epdl; do uv run --no-sync pyrite tables fetch "$code" > /dev/null; done
run() {  # schedule scene histories seed
  local stem="$2-$3-$4-$1" rc state
  # A sidecar is created *before* each sample. SIGKILL leaves it "running"
  # rather than silently omitting a measurement from qualification.
  printf '{"status":"running","exit_code":null}\n' > "$out/$stem.status.json"
  if PYRITE_MC_TRANSPORT_SCHEDULE=$1 timeout --signal=TERM --kill-after=5s 2400s \
      uv run --no-sync python "$BR/checks/equal_precision/contained_sample.py" \
      --max-bytes "$MEM_MAX_BYTES" -- \
      python "$BR/checks/equal_precision/qualify_schedule.py" "$2" \
      --histories "$3" --seed "$4" --output "$out/$stem.json" \
      > "$out/$stem.log" 2>&1; then
    rc=0; state=ok
    echo "DONE $stem"
  else
    rc=$?; state=failed
    echo "FAILED $stem rc=$rc"
  fi
  printf '{"status":"%s","exit_code":%d}\n' "$state" "$rc" \
    > "$out/$stem.status.json.tmp"
  mv "$out/$stem.status.json.tmp" "$out/$stem.status.json"
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
