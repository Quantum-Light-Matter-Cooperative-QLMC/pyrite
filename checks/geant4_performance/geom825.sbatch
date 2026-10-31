#!/bin/bash
# #825 quick launch-geometry test: threads/block x histories per launch, exact kernel
# (kernel-event time via the #831 harness). Usage: sbatch checks/geant4_performance/geom825.sbatch
#SBATCH --job-name=geom825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=00:25:00
#SBATCH --output=geom825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/geom825/$SLURM_JOB_ID"; mkdir -p "$out"; cp .pyrite-sync "$out/sync.json"
uv sync --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
# scene:histories:block:nthreads:variant
arms=(w-1MeV-100um:5000:1250:128:exact w-1MeV-100um:5000:1250:32:exact
      w-1MeV-100um:5000:5000:128:exact w-1MeV-100um:5000:5000:32:exact
      w-1MeV-100um:5000:5000:32:mixed_math
      si-3MeV-1mm:7000:0:128:exact si-3MeV-1mm:7000:0:32:exact
      si-3MeV-1mm:7000:7000:32:exact)
for a in "${arms[@]}"; do
    IFS=: read -r s n b nt v <<< "$a"
    stem="$out/$v-$s-b$b-t$nt"
    PV_NTHREADS=$nt uv run --no-sync python checks/fp32_speed_of_light/precision_variants.py \
        --variant "$v" --bench checks/geant4_performance/pyrite_bench.py --stats "$stem.stats.json" \
        "$s" --histories "$n" --block "$b" --seed 1 --rows-per-round 1024 --device-staging \
        --warmup-histories 200 --scratch-mib 12288 --output "$stem.json" > "$stem.log" 2>&1
    echo "rc=$? $a"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:v for k,v in d['per_seed'].items()}, d['launch_geometry'][-3:])" "$stem.stats.json" 2>&1 | tail -1
done
