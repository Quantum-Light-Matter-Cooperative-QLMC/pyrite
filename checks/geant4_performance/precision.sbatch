#!/bin/bash
# #821 held-out time-to-precision runs (#510 T(eps)). Each arm runs the history
# count precision_plan.py froze for it, on the plan's held-out seeds; analyze.py
# --plan then verifies achieved RSE and reports T(eps).
# Usage: sbatch --dependency=after:<hold> precision.sbatch <hold-job-id> <plan.json>
#SBATCH --job-name=g4bench821-prec
#SBATCH --partition=gpu
#SBATCH --nodelist=DESKTOP-QNIHO3D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=24:00:00
#SBATCH --output=precision-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
hold=$1 plan=$2
trap 'touch "evidence/hold-release-$hold"' EXIT
source checks/geant4_performance/env.sh
uv sync --extra nvidia --locked -q
[[ "$(squeue -h -j "$hold" -o %T)" == RUNNING ]] || { echo "hold job $hold not running"; exit 1; }
out="evidence/precision/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$plan" "$out/plan.json"; cp .pyrite-sync "$out/sync.json"
lscpu > "$out/cpu.txt"; nvidia-smi > "$out/gpu.txt"
squeue -w DESKTOP-QNIHO3D > "$out/node-jobs.txt"; free -g > "$out/memory.txt"
sha256sum "$G4BENCH" checks/geant4_performance/*.py checks/geant4_performance/g4bench/* > "$out/code.txt"
uv run --no-sync python checks/equal_precision/prepare_stopping.py > "$out/tables.json"
mapfile -t seeds < <(python3 -c "import json;print(*json.load(open('checks/geant4_performance/scenes.json'))['precision']['held_out_seeds'],sep='\n')")
mapfile -t runs < <(python3 -c "import json;[print(r['scene'],r['config'],r['histories']) for r in json.load(open('$out/plan.json'))['runs']]")
for seed in "${seeds[@]}"; do
    # Rotate the run order per seed so drift cannot favour one arm.
    r=$(( seed % ${#runs[@]} ))
    for run in "${runs[@]:r}" "${runs[@]:0:r}"; do
        read -r s config n <<< "$run"
        stem="$out/$config-$s-$n-$seed"
        case $config in
            geant4-*)
                threads=${config#geant4-}; threads=${threads%t}
                uv run --no-sync python checks/geant4_performance/geant4_bench.py "$s" \
                    --histories "$n" --warmup $(( n / 100 > 100 ? n / 100 : 100 )) --seed "$seed" \
                    --threads "$threads" --executable "$G4BENCH" --output "$stem.json" \
                    > "$stem.log" 2>&1 || echo "FAIL $config $s $n $seed" ;;
            pyrite-cuda*)
                extra=()
                if [[ $config == pyrite-cuda-continuous-* ]]; then
                    mode=${config#pyrite-cuda-continuous-}; chunk=${mode##*-c}; mode=${mode%-c*}
                    extra=(--estimator continuous --cone-mode "$mode" --chunk "$chunk")
                fi
                block=$(python3 -c "import json;print(next(x['block'] for x in json.load(open('checks/geant4_performance/scenes.json'))['scenes'] if x['id']=='$s'))")
                PYRITE_MC_BACKEND=cuda uv run --no-sync python checks/geant4_performance/pyrite_bench.py \
                    "$s" --histories "$n" --warmup-histories $(( n < block ? n : block )) --seed "$seed" \
                    --rows-per-round 1024 --device-staging "${extra[@]}" --output "$stem.json" \
                    > "$stem.log" 2>&1 || echo "FAIL $config $s $n $seed" ;;
        esac
        echo "DONE $config $s $n $seed"
    done
done
uv run --no-sync python checks/geant4_performance/analyze.py "$out" --plan "$out/plan.json" \
    --output "$out/summary.json"
