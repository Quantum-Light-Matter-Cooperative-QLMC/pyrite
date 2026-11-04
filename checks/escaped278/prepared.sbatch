#!/bin/bash
#SBATCH --job-name=prepared278
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH --time=03:45:00
#SBATCH --output=prepared278-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
trap 'touch evidence/release-prepared278' EXIT
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p evidence/prepared278-validation evidence/prepared278-independent
scontrol show node DESKTOP-QNIHO3D > evidence/prepared278-validation/allocation.txt
squeue -w DESKTOP-QNIHO3D > evidence/prepared278-validation/jobs.txt
nvidia-smi > evidence/prepared278-validation/gpu.txt
sha256sum checks/escaped278/*.py src/pyrite/montecarlo/spectrum/brem*py > evidence/prepared278-validation/source-sha256.txt
PYRITE_TEST_BACKEND=cuda uv run --no-sync pyrite-dev test tests/montecarlo/brem/test_brem_prepared_bands.py tests/montecarlo/brem/test_brem_primary_band.py --durations=0
uv run --no-sync python -m checks.escaped278.validate_prepared --output evidence/prepared278-validation
PYRITE_MC_BACKEND=cpu uv run --no-sync python -m checks.escaped278.validate_prepared --cpu --output evidence/prepared278-validation
failed=0
for energy in 3000 30000 100000; do
    for center in 0 15; do
        if uv run --no-sync python -m checks.escaped278.prepared --energy "$energy" --center "$center" --max-minutes 20 --output evidence/prepared278-independent; then
            printf '%s %s rc=0\n' "$energy" "$center" >> evidence/prepared278-independent/process-status.txt
        else
            rc=$?
            printf '%s %s rc=%s\n' "$energy" "$center" "$rc" >> evidence/prepared278-independent/process-status.txt
            failed=1
        fi
    done
done
test "$failed" -eq 0
