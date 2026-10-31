#!/bin/bash
# #825 CXR line-reduction launch sweep: cxr_threads x cxr_energies_per_block on captured
# production flushes (checks/cxr_reduction_benchmark.py replays 128-1024 threads x 1-4
# energies/block, CUDA events, alternating order). Contained (~15 min).
# Usage: sbatch checks/cxr_launch825.sbatch
#SBATCH --job-name=cxr825
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=cxr825-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export PYRITE_HOME="$PWD" PYRITE_CATALOG="$PWD/src/pyrite/data/catalog"
export PYRITE_USER_CATALOG="$PWD/user-catalog" XDG_CONFIG_HOME="$PWD/.xdg-config"
export UV_CACHE_DIR=/tmp/pyrite-uv-cache PYRITE_MC_BACKEND=cuda
mkdir -p "$XDG_CONFIG_HOME" "$PYRITE_USER_CATALOG"
out="evidence/cxr825/$SLURM_JOB_ID"; mkdir -p "$out"; cp .pyrite-sync "$out/sync.json"
uv sync --extra nvidia --locked -q
rev=$(sed -n 's/^code_revision: //p' .pyrite-sync)
py=(uv run --no-sync python -W ignore checks/cxr_reduction_benchmark.py)
# material:profile:electrons
for a in hbn:high_energy_demo:4000 hopg:standard:4000 silicon:standard:4000; do
    IFS=: read -r m p n <<< "$a"
    fx="$out/$m-$p.npz"
    timeout 900 "${py[@]}" capture "$fx" --revision "$rev" --material "$m" --profile "$p" \
        --electrons "$n" > "$out/$m-$p.capture.log" 2>&1
    echo "capture rc=$? $a"
    [[ -f $fx ]] || continue
    timeout 600 "${py[@]}" replay "$fx" --revision "$rev" --repeats 5 \
        --output "$out/$m-$p.replay.json" > "$out/$m-$p.replay.log" 2>&1
    echo "replay rc=$? $a"
done
python3 - "$out" <<'PY'
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("*.replay.json")):
    meta = json.loads(f.with_name(f.name.replace(".replay.json", ".json")).read_text())
    rep = json.loads(f.read_text())
    for w in rep["workloads"]:
        pts = sorted(w["points"], key=lambda r: r["median_gpu_ms"])
        base = next(r for r in w["points"] if (r["threads"], r["energies"]) == (512, 3))
        best = pts[0]
        print(f"{f.stem} route={meta['route']} {w['name']} lines={w['lines']} nodes={w['nodes']}: "
              f"default 512x3 {base['median_gpu_ms']:.2f} ms; best {best['threads']}x{best['energies']} "
              f"{best['median_gpu_ms']:.2f} ms ({base['median_gpu_ms']/best['median_gpu_ms']:.2f}x); "
              f"worst {pts[-1]['threads']}x{pts[-1]['energies']} {pts[-1]['median_gpu_ms']:.2f} ms")
        for r in pts[:4]:
            print(f"   {r['threads']}x{r['energies']}: {r['median_gpu_ms']:.2f} ms, peak-scaled diff {r['peak_scaled_abs']:.1e}")
        if w.get("bin_mean"):
            print("   bin-mean:", {r["method"]: round(sorted(r["wall_ms"])[len(r["wall_ms"])//2], 2) for r in w["bin_mean"]["results"]})
PY
