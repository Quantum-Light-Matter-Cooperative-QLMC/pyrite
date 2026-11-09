#!/bin/bash
# Usage: reduce.sh CASE [ARM ...]   (no arms: merge all + fates + compile)
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export PYRITE_HOME=$HOME/pyrite-993 PYRITE_CATALOG=$HOME/pyrite-993/src/pyrite/data/catalog PYRITE_USER_CATALOG=$HOME/pyrite-993/user-catalog
cd $HOME/pyrite-993
C=$1; shift
mkdir -p merged/$C
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(control return_flux unresolved matching_surface grid_supercell source_model all_on)
for a in "${arms[@]}"; do
  n=$(ls runs/$C-$a | wc -l); [ "$n" -eq 200 ] || { echo "$C-$a has $n/200 parts"; exit 1; }
  uv run --frozen python checks/axial_fiber_source.py merge merged/$C/$a.json runs/$C-$a/part-*.json
  echo "merged $C-$a"
done
[ $# -gt 0 ] && exit 0
n=$(ls runs/$C-fates | wc -l); [ "$n" -eq 50 ] || { echo "$C-fates has $n/50 parts"; exit 1; }
uv run --frozen python checks/axial_fiber_fates.py merge merged/$C/fates-control.json runs/$C-fates/part-*.json
uv run --frozen python checks/axial_fiber_fates.py events merged/$C/all_on.json merged/$C/fates-all_on.json
uv run --frozen python - "$C" <<'PY'
import json, sys
c = sys.argv[1]
args = json.load(open(f"args/{c}-control.json"))
arms = ["control","return_flux","unresolved","matching_surface","grid_supercell","source_model","all_on"]
sha = open("SYNCED_SHA").read().strip()[:9]
spec = {"case_key": f"si-fibre/{c}", "model_id": f"si-fibre-992@{sha}", "evidence_id": f"993/{c}@{sha}",
        "checkpoints_ang": args["checkpoints_ang"], "span_ang": 15.0,
        "bands_eV": {"whole": [args["e_min_eV"], args["e_max_eV"]]},
        "arms": {a: f"{a}.json" for a in arms},
        "fates": {"control": "fates-control.json", "all_on": "fates-all_on.json"}}
json.dump(spec, open(f"merged/{c}/spec.json", "w"), indent=1)
PY
uv run --frozen python checks/axial_depth_calibration.py merged/$C/spec.json merged/$C/calibration.json | tee merged/$C/calibration.txt
