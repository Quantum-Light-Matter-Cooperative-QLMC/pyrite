#!/bin/bash
# #825/#796/#838 hardware-counter baseline and register-matched mixed-precision timing.
# Step 1: one-kernel ncu probe (fails fast on ERR_NVGPUCTRPERM).
# Step 2: `ncu --set full` on steady-state transport launches of the exact, #831
#         mixed_math and fp32 kernels (checks/fp32_speed_of_light harness;
#         `exact` is the production specialized kernel byte for byte).
# Step 3 (#838): kernel-event timing of exact / mixed_math / fp32 with matched
#         register caps (PV_MAXRREG), 3 seeds, variant order rotated per seed.
# Usage: sbatch checks/geant4_performance/ncu825.sbatch   (from the synced checkout root)
#SBATCH --job-name=ncu825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=01:45:00
#SBATCH --output=ncu825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/ncu825/$SLURM_JOB_ID"; mkdir -p "$out"
cp .pyrite-sync "$out/sync.json"; nvidia-smi > "$out/gpu.txt"; ncu --version > "$out/ncu-version.txt"
uv sync --extra nvidia --locked -q
uv run --no-sync python checks/equal_precision/prepare_stopping.py > /dev/null
py=(uv run --no-sync python)
pv=("${py[@]}" checks/fp32_speed_of_light/precision_variants.py
    --bench checks/geant4_performance/pyrite_bench.py)
flags=(--seed 1 --rows-per-round 1024 --device-staging --warmup-histories 500)
scenes=(w-1MeV-100um:5000 si-3MeV-1mm:7000)

echo "== probe"
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_active \
    "${py[@]}" -c "import cupy as cp; x = cp.arange(1 << 20); print(int((x + 1).sum()))" \
    > "$out/probe.txt" 2>&1
cat "$out/probe.txt"
if grep -q -E "ERR_NVGPUCTRPERM|does not have permission" "$out/probe.txt"; then
    echo "counters refused"; exit 3
fi

echo "== ncu"
for spec in "${scenes[@]}"; do
    s=${spec%%:*} n=${spec#*:}
    for v in exact mixed_math fp32; do
        stem="$out/ncu-$v-$s"
        # Launch 0 is the cold warmup request; profile three launches after it.
        ncu --target-processes all --kernel-name "regex:transport_kernel" \
            --launch-skip 1 --launch-count 3 --set full --import-source yes \
            --replay-mode kernel -f -o "$stem" \
            "${pv[@]}" --variant "$v" --stats "$stem.stats.json" "$s" --histories "$n" \
            "${flags[@]}" --output "$stem.json" > "$stem.log" 2>&1
        echo "rc=$? $v $s"
        ncu --import "$stem.ncu-rep" --page details --csv > "$stem-details.csv" 2>&1
        ncu --import "$stem.ncu-rep" --page raw --csv > "$stem-raw.csv" 2>&1
        ncu --import "$stem.ncu-rep" --page details --print-summary per-kernel | head -150
    done
done

echo "== timing"
# label:variant:maxrreg (0 = uncapped). exact r112 and mixed_math r112 match fp32's
# ~106-112 registers (4 blocks/SM at 128 threads); mixed_math r128 sits between.
arms=(exact:exact:0 mixed_math:mixed_math:0 mixed_math-r128:mixed_math:128
      mixed_math-r112:mixed_math:112 exact-r112:exact:112 fp32:fp32:0)
for spec in "${scenes[@]}"; do
    s=${spec%%:*} n=${spec#*:}
    for seed in 1 2 3; do
        r=$(( seed % ${#arms[@]} )); as=("${arms[@]:r}" "${arms[@]:0:r}")
        for a in "${as[@]}"; do
            IFS=: read -r label v cap <<< "$a"
            stem="$out/t-$label-$s-$seed"
            env PV_MAXRREG="${cap/#0/}" "${pv[@]}" --variant "$v" --stats "$stem.stats.json" \
                "$s" --histories "$n" --seed "$seed" --rows-per-round 1024 --device-staging \
                --output "$stem.json" > "$stem.log" 2>&1 \
                && echo "DONE $label $s $seed" || echo "FAIL $label $s $seed"
        done
    done
done
