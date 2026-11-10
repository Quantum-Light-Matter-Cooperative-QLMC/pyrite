#!/bin/bash
# Weekly lab-host entry point. Run from cron against a dedicated synced checkout.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p evidence/validation-344
# Serialize cron/manual invocations, then refuse to accumulate pending jobs.
exec 9>evidence/validation-344/submit.lock
flock -n 9 || exit 0
active=$(squeue --noheader --user "$(id -un)" --name validation-344 --format '%i')
if [[ -n "$active" ]]; then
    echo "$(date -u +%FT%TZ) skipped: validation-344 already queued/running ($active)"
    exit 0
fi
# Keep jobs mutually exclusive even against a concurrent manual submission.
job=$(sbatch --parsable --dependency=singleton --partition=gpu --gres=gpu:1 \
    checks/scheduled_cuda_parity.sbatch)
printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$job" >> evidence/validation-344/submissions.tsv
echo "Submitted validation-344 job $job"
