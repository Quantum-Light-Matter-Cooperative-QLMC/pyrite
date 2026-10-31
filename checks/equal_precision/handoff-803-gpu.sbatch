#!/bin/bash
#SBATCH --job-name=handoff803-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=handoff803-gpu-%j.log
# #803 CUDA continuation/handoff sweep: rows_per_round x device_staging x
# keep_segments_on_device under `auto`, all result-neutral (digests must
# match). Fresh process per sample; skips existing records (resumable).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
BR="$PWD"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_MC_BACKEND=cuda OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYRITE_HOME="$BR" PYRITE_CATALOG="$BR/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$BR/user-catalog"
export XDG_CONFIG_HOME="$BR/config-home" XDG_DATA_HOME="$BR/data-home"; mkdir -p "$XDG_CONFIG_HOME"
unset PYRITE_MC_TRANSPORT_SCHEDULE PYRITE_MC_ELECTRON_BATCH_MAX
out="${HANDOFF803_OUT:-$BR/evidence/handoff803/gpu-$SLURM_JOB_ID}"; mkdir -p "$out"
{ git rev-parse HEAD 2>/dev/null || uv run --no-sync python -c "import pathlib, runpy; print(runpy.run_path('checks/equal_precision/baseline_803.py')['head_from_git_dir'](pathlib.Path.cwd()))"; } > "$out/head.txt"
nvidia-smi -q > "$out/gpu.txt"
# rows_per_round:staging:keep
arms=(256:0:0 1024:0:0 4096:0:0 256:1:0 1024:1:0 4096:1:0 256:0:1 256:1:1 1024:1:1)
for seed in 80301 80302; do
  for scene in si-300-100um w-300-10um; do
    for inelastic in continuous shell-soft-hard; do
      for arm in "${arms[@]}"; do
        IFS=: read -r q st kp <<<"$arm"
        stem="$scene-$seed-$inelastic-q$q-st$st-kp$kp"
        [ -e "$out/$stem.json" ] && continue
        extra=(--rows-per-round "$q")
        [ "$st" = 1 ] && extra+=(--device-staging)
        [ "$kp" = 1 ] && extra+=(--keep-on-device)
        timeout 1800 uv run --no-sync python checks/equal_precision/baseline_803.py "$scene" \
          --core cuda --elastic elsepa --radiative bremslib-soft-hard --inelastic "$inelastic" \
          --lut off --moments table --schedule auto --histories 8192 --seed "$seed" --repeats 3 \
          "${extra[@]}" --output "$out/$stem.json" > "$out/$stem.log" 2>&1
        echo "$stem rc=$?"
      done
    done
  done
done
