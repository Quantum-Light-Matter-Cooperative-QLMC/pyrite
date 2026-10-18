#!/bin/bash
#SBATCH --job-name=activity-ladder510
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=activity-ladder-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg
out="scratch/activity-ladder/$SLURM_JOB_ID"
mkdir -p "$out/source"
scontrol show job "$SLURM_JOB_ID" > "$out/allocation.txt"
lscpu > "$out/cpu.txt"
cp .pyrite-sync "$out/code.txt"
cp "$0" "$out/source/submitted.sbatch"
cp checks/equal_precision/{activity-ladder.json,run_activity_ladder.py,probe_step_activity.py,pyrite_adapter.py,precision_checks.py,run_pilot.py,manifest.json} "$out/source/"
protocol="${1:-checks/equal_precision/activity-ladder.json}"
cp "$protocol" "$out/source/protocol.json"
cp -r src/pyrite/montecarlo/transport "$out/source/transport"
find "$out/source" -type d -name __pycache__ -prune -exec rm -rf {} +
set +e
"$HOME/.local/bin/uv" run --no-sync python checks/equal_precision/run_activity_ladder.py "$out/samples" --protocol "$protocol"
status=$?
set -e
find "$out" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$out/SHA256SUMS"
tar -czf "scratch/activity-ladder-$SLURM_JOB_ID.tar.gz" "$out"
sha256sum "scratch/activity-ladder-$SLURM_JOB_ID.tar.gz" > "scratch/activity-ladder-$SLURM_JOB_ID.sha256"
exit "$status"
