"""Pure shell blocks for resumable chunk execution and prelimit handling."""


def _chunked_timeout_setup() -> str:
    """Only SLURM's chunked prelimit USR1 can authorize a resumable 143."""
    return """timeout_warned=0
child_pid=
trap 'timeout_warned=1; [ -z "$child_pid" ] || kill -TERM "$child_pid" 2>/dev/null || true' USR1
progress_count() {
  python3 - "$JOBDIR/progress" <<'PY'
import glob
import json
import sys

count = 0
for path in glob.glob(sys.argv[1] + "/*.json"):
    try:
        with open(path, encoding="utf-8") as record_file:
            record = json.load(record_file)
        count += int(record.get("cached_cases", 0)) + int(record.get("completed_new_cases", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        # Legacy or interrupted records cannot prove checkpoint progress.
        continue
print(count)
PY
}
case_minutes() {
  python3 - "$JOBDIR/progress/$1.json" <<'PY_CASE'
import json
import math
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as record_file:
        record = json.load(record_file)
    seconds = float(record.get("sample_compute_seconds", 0))
    cases = int(record.get("sample_new_cases", 0))
    if cases > 0 and math.isfinite(seconds) and seconds > 0:
        print(f"{seconds / cases / 60:.2f} min")
    else:
        print("unavailable (no completed-case timing)")
except (OSError, ValueError, TypeError, AttributeError, OverflowError):
    print("unavailable (no completed-case timing)")
PY_CASE
}
slice_finished_before=$(grep -Ec "^(completed|failed): " "$JOBDIR/log" 2>/dev/null || true)
slice_finished_before=${slice_finished_before:-0}
slice_progress_before=$(progress_count) || exit 1
"""


def _chunked_run_wait() -> str:
    """Read the child result even when USR1 interrupts bash's wait."""
    return """  child_pid=$!
  wait "$child_pid" 2>/dev/null || true
  [ -f "$JOBDIR/.child_rc" ] || wait "$child_pid" 2>/dev/null || true
  rc=$(cat "$JOBDIR/.child_rc") || exit 1
  child_pid=
"""


def _chunked_timeout_result() -> str:
    return """  elif [ "$rc" -eq 143 ] && [ "$timeout_warned" -eq 1 ]; then
    progress_after=$(progress_count) || exit 1
    if [ "$progress_after" -gt "$progress_before" ]; then
      echo "resumable timeout: $m [$n/$total] $(date -Is)" | tee -a "$JOBDIR/timeouts" >> "$JOBDIR/log"
      echo "resumable timeout at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
      break
    fi
    estimate=$(case_minutes "$m") || exit 1
    reason="case exceeded slice without checkpoint progress (estimated completed-case cost $estimate; budget ${remaining_min} min of $((chunk_seconds / 60)) min slice); use --chunk-minutes 0 or a larger --chunk-minutes, or lower max_histories"
    echo "WARNING: $m: $reason; continuing remaining materials in the next slice" >> "$JOBDIR/log"
    mkdir -p "$JOBDIR/failures" || exit 1
    printf '%s\n' "$reason" > "$JOBDIR/failures/$m"
    echo "failed: $m" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] (case exceeded slice; see failures/$m) $(date -Is)" > "$JOBDIR/state"
    break
"""


def _chunked_recompute_blocks(recompute):
    """Track admission per material, including members first reached in later slices.

    Only a budget exit or a prelimit interruption permits checkpoint resume.
    The original --recompute flag remains, disabling CAS reads throughout.
    """
    if not recompute:
        return "  resume_flags=()", ""
    return (
        """  resume_flags=()
  mkdir -p "$JOBDIR/recompute" || exit 1
  [ ! -f "$JOBDIR/recompute/$m" ] || resume_flags+=(--resume-checkpoint)""",
        """  if [ "$rc" -eq 75 ] || { [ "$rc" -eq 143 ] && [ "$timeout_warned" -eq 1 ]; }; then
    touch "$JOBDIR/recompute/$m" || exit 1
  fi
""",
    )
