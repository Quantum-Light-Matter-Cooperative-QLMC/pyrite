#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=runs/audit1432-%j.log
# #1432 CPU audit on four OS-bound CPUs (bind.sh), one Numba thread unless
# AUDIT1432_NUMBA_THREADS is set (it overrides Numba only; BLAS/OpenMP stay 1).
#   sbatch [--nodelist=N --mem=M --time=T] checks/native_cpu_audit/cpu.sbatch MODE [ARGS]
# MODE:
#   cases [NAME ...]   complete-case arms of checks/native_cpu_audit.py (default
#                      thin30 thin100 crystal300 photon-case layered mos2_5mev);
#                      REPEATS (default 2), ARM_TIMEOUT_S (default 600); the
#                      axial-prod arm needs ARM_TIMEOUT_S=1500 and --mem=3G
#   startup            `pyrite-dev startup` cold-JIT vs warm-disk, 3 fresh processes each
#   topology           identical four-case rounds under 1x1, 4x1, 1x4, 2x2 (cpu_budget.py)
#   pools              Numba mask versus threadpool_limits(limits=1) (cpu_budget.py; sets 4 threads)
#   axial-proto        solver-cache / OAM-basis prototypes (axial_prototypes.py)
#   gaps RUN_DIR       offline: re-export every RUN_DIR/*.nsys-rep of a finished GPU run to SQLite
#                      and rerun nsys_gaps.py (CPU only; uses the checkout's existing .venv)
set -euo pipefail
mode=${1:?mode}; shift
source checks/native_cpu_audit/bind.sh
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/tmp/pyrite-uv-cache
export PYRITE_HOME=$PWD PYRITE_CATALOG=$PWD/src/pyrite/data/catalog
export PYRITE_USER_CATALOG=$PWD/user-catalog PYRITE_MC_BACKEND=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
threads=${AUDIT1432_NUMBA_THREADS:-1}
export NUMBA_NUM_THREADS=$threads NUMBA_CACHE_DIR=$PWD/runs/cache-1
mkdir -p runs
audit=checks/native_cpu_audit
if [ "$mode" = gaps ]; then
    for rep in "${1:?run dir}"/*.nsys-rep; do
        stem=${rep%.nsys-rep}
        nsys export --type sqlite --force-overwrite true --output "$stem.sqlite" "$rep" > "$stem-export.log" 2>&1
        uv run --no-sync python $audit/nsys_gaps.py "$stem.sqlite" "$stem-gaps.json" > "$stem-gaps.log" 2>&1
        echo "gaps rc=$? $stem"
        rm -f "$stem.sqlite"
    done
    exit 0
fi
uv run --frozen python -c 'from pyrite.validation.check_records import current_revision; print(current_revision())'
case $mode in
cases)
    names=(thin30 thin100 crystal300 photon-case layered mos2_5mev)
    if (($#)); then names=("$@"); fi
    for name in "${names[@]}"; do
        timeout "${ARM_TIMEOUT_S:-600}" uv run --frozen python checks/native_cpu_audit.py --case "$name" \
            --repeats "${REPEATS:-2}" --out "runs/$name-t$threads.json" || echo "ARM_FAILED $name $?"
    done ;;
startup)
    for cache_mode in cold warm; do
        timeout 300 uv run --frozen pyrite-dev startup --cache "$cache_mode" --repeats 3 \
            --backend cpu --json > "runs/startup-$cache_mode.json"
    done ;;
topology)
    for name in crystal300 char-multi char-layered; do
        timeout 90 uv run --frozen python checks/native_cpu_audit.py --case "$name" --repeats 3 --out "runs/$name-t1.json"
    done
    NUMBA_NUM_THREADS=4 timeout 90 uv run --frozen python checks/native_cpu_audit.py --case crystal300 \
        --repeats 3 --out runs/crystal300-t4.json
    timeout 240 uv run --frozen python $audit/cpu_budget.py topology --case crystal300 --out runs/topology.json ;;
pools)
    NUMBA_NUM_THREADS=4 uv run --frozen python $audit/cpu_budget.py pools --out runs/pool-budget.json ;;
axial-proto)
    timeout 1500 uv run --frozen python $audit/axial_prototypes.py --histories 1000 --rounds 2 \
        --out runs/axial-prototypes.json || echo "ARM_FAILED axial-prototypes $?" ;;
*) echo "unknown mode $mode" >&2; exit 2 ;;
esac
