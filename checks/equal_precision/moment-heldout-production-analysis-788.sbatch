#!/bin/bash
#SBATCH --job-name=moment-heldana788
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=moment-heldana788-%j.log
# Analyze a complete moment-heldout-production-788 directory (OUT_TAG).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/tmp/pyrite-uv-cache
out="evidence/moment-heldprod788/${OUT_TAG:?}"
uv run --no-sync python checks/equal_precision/qualify_moment_fractions.py "$out" "$out/fractions.json"
# Continuous/hard-count screening on the same held-out seeds and margins.
uv run --no-sync python -c "
import json
policy = json.load(open('checks/equal_precision/moment-policy-788.json'))
held = json.load(open('checks/equal_precision/moment-heldout-788.json'))
policy['ensemble']['seeds'] = held['ensemble']['seeds']
json.dump(policy, open('$out/continuous-policy.json', 'w'), indent=2)
"
uv run --no-sync python checks/equal_precision/analyze_moment_transport.py \
    --policy "$out/continuous-policy.json" "$out" "$out/continuous.json"
echo COMPLETE
