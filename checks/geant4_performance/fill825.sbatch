#!/bin/bash
# #825/#838 GPU fill and occupancy sweep on current main: schedule x scratch budget x
# threads/block x {exact, mixed_math}, kernel-event and request time in the same run.
# Contained (~30 min). Usage: sbatch checks/geant4_performance/fill825.sbatch
#SBATCH --job-name=fill825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=fill825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/fill825/$SLURM_JOB_ID"; mkdir -p "$out"; cp .pyrite-sync "$out/sync.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/gpu.csv"
uv sync --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
N=${FILL_HISTORIES:-8000}
# scene:schedule:scratch_mib:seg_capacity:probe:threads:variant
# prod = production defaults (512 MiB scratch, seg_capacity 512, probe 1024, 128 threads).
arms=(
  w-1MeV-100um:full-history:512:512:1024:128:exact
  w-1MeV-100um:full-history:512:512:1024:128:mixed_math
  w-1MeV-100um:full-history:8192:512:1024:32:exact
  w-1MeV-100um:full-history:8192:512:1024:32:mixed_math
  w-1MeV-100um:continuation:512:512:1024:128:exact
  w-1MeV-100um:continuation:512:512:1024:128:mixed_math
  w-1MeV-100um:continuation:512:512:1024:64:exact
  w-1MeV-100um:continuation:512:512:1024:32:exact
  w-1MeV-100um:continuation:512:512:1024:32:mixed_math
  w-1MeV-100um:continuation:4096:512:1024:32:exact
  w-1MeV-100um:continuation:4096:512:1024:32:mixed_math
  w-1MeV-100um:continuation:4096:512:1024:128:mixed_math
  si-300keV-100um:full-history:512:512:1024:128:exact
  si-300keV-100um:continuation:512:512:1024:128:exact
  si-300keV-100um:continuation:512:512:1024:32:exact
  si-300keV-100um:continuation:4096:512:1024:32:exact
  si-300keV-100um:continuation:4096:512:1024:32:mixed_math
  si-300keV-100um:full-history:8192:512:1024:32:exact
)
if [[ ${FILL_SET:-grid} == threshold ]]; then
    # `auto` threshold (CONTINUATION_AUTO_MIN_HISTORIES): production defaults at small N.
    arms=()
    for n in 1000 2000 4000; do
        for s in si-300keV-100um w-1MeV-100um; do
            for sch in full-history continuation; do
                arms+=("$s:$sch:512:512:1024:128:exact:$n")
            done
        done
    done
fi
for a in "${arms[@]}"; do
    IFS=: read -r s sch mib cap probe nt v n <<< "$a"
    N=${n:-$N}
    stem="$out/$s-$sch-m$mib-t$nt-$v-n$N"
    PYRITE_MC_TRANSPORT_SCHEDULE=$sch PV_NTHREADS=$nt timeout 420 uv run --no-sync python \
        checks/fp32_speed_of_light/precision_variants.py \
        --variant "$v" --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
        "$s" --histories "$N" --block "$N" --seed 1 --device-staging --no-audit \
        --warmup-histories 200 --scratch-mib "$mib" --seg-capacity "$cap" \
        --probe-electrons "$probe" --output "$stem.json" > "$stem.log" 2>&1
    echo "rc=$? $a"
done
python3 checks/geant4_performance/fill825_summary.py "$out" | tee "$out/summary.md"
