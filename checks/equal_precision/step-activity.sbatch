#!/bin/bash
#SBATCH --job-name=step-activity510
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --output=step-activity-%j.log
set -euo pipefail
cd /home/aamador/pyrite-510-precision-20261015
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" UV_CACHE_DIR=/tmp/pyrite-uv-cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg PYRITE_MC_BACKEND=cpu
out=scratch/cheap-validation/activity-$SLURM_JOB_ID
mkdir -p "$out/source"
scontrol show job "$SLURM_JOB_ID" > "$out/allocation.txt"
cp "$0" "$out/source/submitted.sbatch"
cp checks/equal_precision/{probe_step_activity.py,pyrite_adapter.py,precision_checks.py,manifest.json} "$out/source/"
status=0
for scene in si-300-100um-0deg w-300-10um-0deg si-800-100um-0deg w-800-10um-0deg si-100-1um-0deg w-100-100um-75deg; do
  for cap in 0.05 0.025 0.001; do
    name="$scene-$cap"
    if [ "$SECONDS" -gt 95 ]; then
      printf '{"reason":"allocation budget"}\n' > "$out/$name.deferred.json"
      status=1
      continue
    fi
    if timeout --signal=KILL 15s "$HOME/.local/bin/uv" run --no-sync python checks/equal_precision/probe_step_activity.py "$scene" "$cap" "$out/$name.json" > "$out/$name.log" 2>&1; then
      echo "OK $name"
    else
      printf '{"reason":"activity subprocess failed or exceeded 15s"}\n' > "$out/$name.failed.json"
      status=1
    fi
  done
done
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
tar -czf scratch/step-activity-evidence.tar.gz "$out"
sha256sum scratch/step-activity-evidence.tar.gz > scratch/step-activity-evidence.sha256
exit "$status"
