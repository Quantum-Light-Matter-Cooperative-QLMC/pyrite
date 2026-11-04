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
run=${PYRITE_278_RUN:-prepared278}
validation="evidence/$run-validation"
independent="evidence/$run-independent"
trap 'touch "evidence/release-$run"' EXIT
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog" PYRITE_USER_CATALOG="$PWD/user-catalog"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "$validation" "$independent"
scontrol show node DESKTOP-QNIHO3D > "$validation/allocation.txt"
squeue -w DESKTOP-QNIHO3D > "$validation/jobs.txt"
nvidia-smi > "$validation/gpu.txt"
sha256sum checks/escaped278/*.py src/pyrite/montecarlo/spectrum/brem*py src/pyrite/_bremslib_table.py > "$validation/source-sha256.txt"
validation_args=()
if [[ -n "${PYRITE_278_TRACKS:-}" ]]; then
    cp "$PYRITE_278_TRACKS"/tracks-*.json "$PYRITE_278_TRACKS"/tracks-*.npz "$validation/"
    sha256sum "$validation"/tracks-* > "$validation/track-sha256.txt"
    validation_args+=(--reuse-tracks)
fi
PYRITE_TEST_BACKEND=cuda uv run --no-sync pyrite-dev test tests/montecarlo/brem/test_brem_prepared_bands.py tests/montecarlo/brem/test_brem_primary_band.py tests/montecarlo/brem/test_bremsstrahlung_bremslib.py --durations=0
uv run --no-sync python -m checks.escaped278.validate_prepared --output "$validation" "${validation_args[@]}"
PYRITE_MC_BACKEND=cpu uv run --no-sync python -m checks.escaped278.validate_prepared --cpu --output "$validation"
failed=0
for energy in 3000 30000 100000; do
    for center in 0 15; do
        if uv run --no-sync python -m checks.escaped278.prepared --energy "$energy" --center "$center" --max-minutes 20 --output "$independent"; then
            printf '%s %s rc=0\n' "$energy" "$center" >> "$independent/process-status.txt"
        else
            rc=$?
            printf '%s %s rc=%s\n' "$energy" "$center" "$rc" >> "$independent/process-status.txt"
            failed=1
        fi
    done
done
test "$failed" -eq 0
