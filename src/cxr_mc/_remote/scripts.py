"""SLURM script and remote-command string builders (pure, no ssh)."""

import datetime
import shlex
import uuid

from . import config, transport


# ---- detached job queue -------------------------------------------------------
def _stems(materials, quick, fidelity="full", high_energy_min_kev=None, catalog_profile="standard"):
    """Checkpoint stems a queue produces (the scan runner writes
    <material>_quick.pkl for --quick runs). ``high_energy_min_kev`` predicts the
    non-canonical stem for any material in mats_to_sim.toml's
    high_energy_materials list (a no-op stem-wise for every other material)."""
    if quick:
        return [f"{m}_quick" for m in materials]
    if high_energy_min_kev is None:
        if fidelity == "full" and catalog_profile == "standard":
            return list(materials)
        from ..profiles import named_profile_stem

        return [
            named_profile_stem(material, fidelity, catalog_profile=catalog_profile)
            for material in materials
        ]
    from ..profiles import high_energy_floor_stem, named_profile_stem
    from ..scan import load_manifest_groups

    tagged = set(load_manifest_groups(config.MATS_FILE).get("high_energy_materials", []))
    stems = []
    for material in materials:
        if material in tagged:
            stems.append(
                high_energy_floor_stem(
                    material, high_energy_min_kev, fidelity, catalog_profile=catalog_profile
                )
            )
        elif fidelity == "full" and catalog_profile == "standard":
            stems.append(material)
        else:
            stems.append(named_profile_stem(material, fidelity, catalog_profile=catalog_profile))
    return stems


def _list_checkpoint_dirs_command() -> str:
    """Remote command: list immediate ``checkpoints/`` subdirectory names, one
    per line. Used to discover identity-qualified variant checkpoints (for
    example ``<material>--survey-<hash>/``) that a bare material name alone
    cannot address, since their on-disk name carries a parameter digest a
    caller cannot predict without recomputing the identity."""
    ckpt = config.shell_arg(config.remote_path("checkpoints"))
    return f"[ -d {ckpt} ] || exit 0; find {ckpt} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n'"


def _new_jobid() -> str:
    """Return a collision-resistant, shell-safe local job directory name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _validate_parallel_materials(parallel_materials):
    """Return a supported in-allocation material-process limit."""
    if (
        not isinstance(parallel_materials, int)
        or not 1 <= parallel_materials <= config.MAX_PARALLEL_MATERIALS
    ):
        raise SystemExit(
            f"parallel materials must be between 1 and {config.MAX_PARALLEL_MATERIALS}"
        )
    return parallel_materials


def _uv_sync_block() -> str:
    """Bash that records dependency-sync wall time without changing exit semantics."""
    return f"""uv_sync_start_ns=$(date +%s%N)
uv_sync_rc=0
{config.shell_remote_uv()} sync >> "$JOBDIR/log" 2>&1 || uv_sync_rc=$?
uv_sync_elapsed_ms=$((($(date +%s%N) - uv_sync_start_ns) / 1000000))
printf 'timing: uv sync %d.%03d s\\n' \
  "$((uv_sync_elapsed_ms / 1000))" "$((uv_sync_elapsed_ms % 1000))" >> "$JOBDIR/log"
if [ "$uv_sync_rc" -ne 0 ]; then
  echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"
  exit 1
fi"""


def _queue_script(
    jobid,
    materials,
    quick,
    workers,
    parallel_materials=config.DEFAULT_PARALLEL_MATERIALS,
    fidelity="full",
    high_energy_min_kev=None,
    catalog_profile="standard",
):
    """CXR payload for one bounded-concurrency queue in a SLURM allocation."""
    parallel_materials = _validate_parallel_materials(parallel_materials)
    flags = ""
    if quick:
        flags += " --quick"
    if fidelity != "full":
        flags += f" --fidelity {fidelity}"
    if workers is not None:
        flags += f" --workers {workers}"
    if high_energy_min_kev is not None:
        flags += f" --high-energy-min-kev {high_energy_min_kev}"
    if catalog_profile != "standard":
        flags += f" --profile {catalog_profile}"
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block()}
mats=({mats})
total=${{#mats[@]}}
parallel_materials={parallel_materials}
n=0
failures=0
active=0
run_material() {{
  local i="$1"
  local m="$2"
  echo "running $m [$i/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$i" "$total" "$m" "$(date -Is)" \
>> "$JOBDIR/log"
  if ! {config.shell_remote_uv()} run --no-sync python -m cxr_mc._entry.scan "$m"{flags} \
    --progress-file "$JOBDIR/progress/$m.json" --no-progress >> "$JOBDIR/log" 2>&1
  then
    echo "WARNING: scan failed for $m; continuing" >> "$JOBDIR/log"
    echo "warning at $m [$i/$total] $(date -Is)" > "$JOBDIR/state"
    return 1
  fi
  echo "completed: $m" >> "$JOBDIR/log"
}}
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  run_material "$n" "$m" &
  active=$((active + 1))
  if [ "$active" -ge "$parallel_materials" ]; then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  fi
done
while [ "$active" -gt 0 ]; do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done
if [ "$failures" -gt 0 ]; then
  echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
else
  echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
fi
"""


def _chunked_queue_script(
    jobid,
    materials,
    quick,
    workers,
    chunk_minutes,
    fidelity="full",
    high_energy_min_kev=None,
    catalog_profile="standard",
):
    """One SLURM slice of a self-resubmitting chain (spec: chunked remote jobs).

    Reused verbatim by every slice: it resumes from checkpoint, does about
    chunk_minutes of work via the scan runner's --max-minutes, and either terminates the
    chain (all materials completed:/failed:) or hands off: write the
    'queued slice' state FIRST, then sbatch fail-closed, then append the SID.
    State-first ordering keeps the EXIT trap from releasing reservations while
    the next slice is already pending (spec Component 2 step 4).
    """
    flags = ""
    if quick:
        flags += " --quick"
    if fidelity != "full":
        flags += f" --fidelity {fidelity}"
    if workers is not None:
        flags += f" --workers {workers}"
    if high_energy_min_kev is not None:
        flags += f" --high-energy-min-kev {high_energy_min_kev}"
    if catalog_profile != "standard":
        flags += f" --profile {catalog_profile}"
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    chunk_seconds = int(round(chunk_minutes * 60))
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block()}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
slice_start=$(date +%s)
n=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && continue
  now=$(date +%s)
  remaining=$((slice_start + chunk_seconds - now))
  [ "$remaining" -gt 0 ] || break
  remaining_min=$(awk "BEGIN {{ printf \\"%.2f\\", $remaining / 60 }}")
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  {config.shell_remote_uv()} run --no-sync python -m cxr_mc._entry.scan "$m"{flags} --max-minutes "$remaining_min" \
    --progress-file "$JOBDIR/progress/$m.json" --no-progress >> "$JOBDIR/log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: scan failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
unresolved=0
failures=0
for m in "${{mats[@]}}"; do
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && {{ failures=$((failures + 1)); continue; }}
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  unresolved=1
done
if [ "$unresolved" -eq 0 ]; then
  if [ "$failures" -gt 0 ]; then
    echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
  else
    echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
  fi
  exit 0
fi
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
SID=$(sbatch --parsable --nice=10000 "$JOBDIR/run.sh") || {{ echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\\n' "$SID" >> "$JOBDIR/meta"
"""


def _zhai_flags(ne, ne_brem, ne_supp, tmd_azimuth, refresh):
    flags = f" --ne {ne} --ne-brem {ne_brem} --ne-supp {ne_supp} --tmd-azimuth {tmd_azimuth}"
    if refresh:
        flags += " --refresh"
    return flags


def _zhai_queue_script(jobid, ne, ne_brem, ne_supp, tmd_azimuth, refresh):
    """CXR payload for one Zhai reproduction inside a SLURM allocation."""
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    flags = _zhai_flags(ne, ne_brem, ne_supp, tmd_azimuth, refresh)
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block()}
echo "running zhai reproduction since $(date -Is)" > "$JOBDIR/state"
if ! {config.shell_remote_uv()} run --no-sync python -m cxr_mc._entry.reproduce_zhai{flags} >> "$JOBDIR/log" 2>&1
then
  echo "FAILED $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
echo "done $(date -Is)" > "$JOBDIR/state"
"""


def _rebrem_flags(
    ne_brem,
    brem_step_eV,
    redo_all,
    fidelity="full",
    brem_start_eV=None,
    brem_stop_eV=None,
):
    flags = f" --fidelity {fidelity}"
    if ne_brem is not None:
        flags += f" --ne-brem {int(ne_brem)}"
    if brem_step_eV is not None:
        flags += f" --step {float(brem_step_eV):g}"
    if brem_start_eV is not None:
        flags += f" --start {float(brem_start_eV):g}"
    if brem_stop_eV is not None:
        flags += f" --stop {float(brem_stop_eV):g}"
    if redo_all:
        flags += " --redo-all"
    return flags


def _rebrem_queue_script(
    jobid,
    materials,
    ne_brem,
    brem_step_eV,
    redo_all,
    fidelity="full",
    brem_start_eV=None,
    brem_stop_eV=None,
):
    """CXR payload for a brem-only checkpoint recompute in a SLURM allocation.

    One sequential ``cxr rebrem`` per material (brem is cheap; no in-allocation
    parallelism needed), each writing the SAME per-material JSON progress
    record a scan does (``--progress-file``), so ``status``/``attach`` render
    the shared case-progress dashboard. The ``completed:``/``failed:`` log
    markers match the scan queue's so ``state._completed_materials`` drives the
    post-attach pull unchanged."""
    flags = _rebrem_flags(ne_brem, brem_step_eV, redo_all, fidelity, brem_start_eV, brem_stop_eV)
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block()}
mats=({mats})
total=${{#mats[@]}}
n=0
failures=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  if ! {config.shell_remote_uv()} run --no-sync cxr rebrem "$m"{flags} \
    --progress-file "$JOBDIR/progress/$m.json" >> "$JOBDIR/log" 2>&1
  then
    echo "WARNING: rebrem failed for $m; continuing" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    failures=$((failures + 1))
    echo "failed: $m" >> "$JOBDIR/log"
    continue
  fi
  echo "completed: $m" >> "$JOBDIR/log"
done
if [ "$failures" -gt 0 ]; then
  echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
else
  echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
fi
"""


def _rebrem_chunked_queue_script(
    jobid,
    materials,
    ne_brem,
    brem_step_eV,
    redo_all,
    chunk_minutes,
    fidelity="full",
    brem_start_eV=None,
    brem_stop_eV=None,
):
    """One SLURM slice of a self-resubmitting brem-only recompute chain.

    Mirror of ``_reline_chunked_queue_script`` for ``cxr rebrem``: each slice
    resumes from checkpoint, does about ``chunk_minutes`` of work via ``cxr
    rebrem --max-minutes``, and either terminates the chain (all materials
    completed:/failed:) or self-resubmits with ``--nice=10000``. Exit-code
    contract per material: ``rc==0`` -> ``completed:``, ``rc==75`` -> leave
    unresolved (a later slice finishes it), else -> ``failed:``.
    """
    flags = _rebrem_flags(ne_brem, brem_step_eV, redo_all, fidelity, brem_start_eV, brem_stop_eV)
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    chunk_seconds = int(round(chunk_minutes * 60))
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block()}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
slice_start=$(date +%s)
n=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && continue
  now=$(date +%s)
  remaining=$((slice_start + chunk_seconds - now))
  [ "$remaining" -gt 0 ] || break
  remaining_min=$(awk "BEGIN {{ printf \\"%.2f\\", $remaining / 60 }}")
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  {config.shell_remote_uv()} run --no-sync cxr rebrem "$m"{flags} --max-minutes "$remaining_min" \
    --progress-file "$JOBDIR/progress/$m.json" >> "$JOBDIR/log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: rebrem failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
unresolved=0
failures=0
for m in "${{mats[@]}}"; do
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && {{ failures=$((failures + 1)); continue; }}
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  unresolved=1
done
if [ "$unresolved" -eq 0 ]; then
  if [ "$failures" -gt 0 ]; then
    echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
  else
    echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
  fi
  exit 0
fi
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
SID=$(sbatch --parsable --nice=10000 "$JOBDIR/run.sh") || {{ echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\\n' "$SID" >> "$JOBDIR/meta"
"""


def _rebrem_queue_metadata(
    jobid,
    materials,
    ne_brem,
    brem_step_eV,
    redo_all,
    fidelity="full",
    brem_start_eV=None,
    brem_stop_eV=None,
):
    """Static metadata persisted before a rebrem queue is submitted. ``kind:
    rebrem`` keys the Mode line in status/attach; ``materials``/``quick`` keep
    the shared _live_jobs/_refuse_if_busy/jobs-listing plumbing working."""
    return "\n".join(
        [
            f"job: {jobid}",
            f"materials: {' '.join(materials)}",
            "quick: False",
            "kind: rebrem",
            f"fidelity: {fidelity}",
            f"ne_brem: {ne_brem}",
            f"brem_start_eV: {brem_start_eV}",
            f"brem_stop_eV: {brem_stop_eV}",
            f"brem_step_eV: {brem_step_eV}",
            f"redo_all: {bool(redo_all)}",
            "progress_dashboard: True",
            "",
        ]
    )


def _reline_flags(
    line_ne,
    line_step_eV,
    redo_all,
    fidelity="full",
    line_start_eV=None,
    line_stop_eV=None,
):
    flags = f" --fidelity {fidelity}"
    if line_ne is not None:
        flags += f" --line-ne {int(line_ne)}"
    if line_step_eV is not None:
        flags += f" --line-step {float(line_step_eV):g}"
    if line_start_eV is not None:
        flags += f" --start {float(line_start_eV):g}"
    if line_stop_eV is not None:
        flags += f" --stop {float(line_stop_eV):g}"
    if redo_all:
        flags += " --redo-all"
    return flags


def _reline_queue_script(
    jobid,
    materials,
    line_ne,
    line_step_eV,
    redo_all,
    fidelity="full",
    line_start_eV=None,
    line_stop_eV=None,
):
    """CXR payload for a line-only checkpoint recompute (``cxr reline``) in a
    SLURM allocation. One sequential reline per material; each writes the same
    per-material JSON progress record a scan/rebrem does, and the
    ``completed:``/``failed:`` markers match so ``state._completed_materials``
    drives the post-attach pull unchanged."""
    flags = _reline_flags(line_ne, line_step_eV, redo_all, fidelity, line_start_eV, line_stop_eV)
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block()}
mats=({mats})
total=${{#mats[@]}}
n=0
failures=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  if ! {config.shell_remote_uv()} run --no-sync cxr reline "$m"{flags} \
    --progress-file "$JOBDIR/progress/$m.json" >> "$JOBDIR/log" 2>&1
  then
    echo "WARNING: reline failed for $m; continuing" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    failures=$((failures + 1))
    echo "failed: $m" >> "$JOBDIR/log"
    continue
  fi
  echo "completed: $m" >> "$JOBDIR/log"
done
if [ "$failures" -gt 0 ]; then
  echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
else
  echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
fi
"""


def _reline_chunked_queue_script(
    jobid,
    materials,
    line_ne,
    line_step_eV,
    redo_all,
    chunk_minutes,
    fidelity="full",
    line_start_eV=None,
    line_stop_eV=None,
):
    """One SLURM slice of a self-resubmitting line-only recompute chain.

    Mirror of ``_chunked_queue_script`` for ``cxr reline``: each slice resumes
    from checkpoint, does about ``chunk_minutes`` of work via ``cxr reline
    --max-minutes``, and either terminates the chain (all materials
    completed:/failed:) or self-resubmits with ``--nice=10000``. Exit-code
    contract per material: ``rc==0`` -> ``completed:``, ``rc==75`` -> leave
    unresolved (a later slice finishes it), else -> ``failed:``.
    """
    flags = _reline_flags(line_ne, line_step_eV, redo_all, fidelity, line_start_eV, line_stop_eV)
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    chunk_seconds = int(round(chunk_minutes * 60))
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block()}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
slice_start=$(date +%s)
n=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && continue
  now=$(date +%s)
  remaining=$((slice_start + chunk_seconds - now))
  [ "$remaining" -gt 0 ] || break
  remaining_min=$(awk "BEGIN {{ printf \\"%.2f\\", $remaining / 60 }}")
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  {config.shell_remote_uv()} run --no-sync cxr reline "$m"{flags} --max-minutes "$remaining_min" \
    --progress-file "$JOBDIR/progress/$m.json" >> "$JOBDIR/log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: scan failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
unresolved=0
failures=0
for m in "${{mats[@]}}"; do
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && {{ failures=$((failures + 1)); continue; }}
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  unresolved=1
done
if [ "$unresolved" -eq 0 ]; then
  if [ "$failures" -gt 0 ]; then
    echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
  else
    echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
  fi
  exit 0
fi
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
SID=$(sbatch --parsable --nice=10000 "$JOBDIR/run.sh") || {{ echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1; }}
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\\n' "$SID" >> "$JOBDIR/meta"
"""


def _reline_queue_metadata(
    jobid,
    materials,
    line_ne,
    line_step_eV,
    redo_all,
    fidelity="full",
    line_start_eV=None,
    line_stop_eV=None,
):
    """Static metadata for a reline queue. ``kind: reline`` keys the Mode line."""
    return "\n".join(
        [
            f"job: {jobid}",
            f"materials: {' '.join(materials)}",
            "quick: False",
            "kind: reline",
            f"fidelity: {fidelity}",
            f"line_ne: {line_ne}",
            f"line_start_eV: {line_start_eV}",
            f"line_stop_eV: {line_stop_eV}",
            f"line_step_eV: {line_step_eV}",
            f"redo_all: {bool(redo_all)}",
            "progress_dashboard: True",
            "",
        ]
    )


def _reservation_root() -> str:
    return config.remote_path(config.JOBS_SUBDIR, config.RESERVATIONS_SUBDIR)


def _release_checkpoint_stems_command(jobid: str, stems: list[str]) -> str:
    """Return a remote command that releases only reservations owned by ``jobid``."""
    transport._check_shell_tokens([jobid, *stems])
    reservations = _reservation_root()
    releases = "; ".join(
        f'if [ "$(cat "$R/{stem}/jobid" 2>/dev/null)" = "$J" ]; then rm -rf "$R/{stem}"; fi'
        for stem in stems
    )
    command = f"R={config.shell_word(reservations)}; J={config.shell_word(jobid)}"
    return f"{command}; {releases}" if releases else command


def _release_job_reservations_command(jobid: str) -> str:
    """Return a remote command that releases every reservation owned by a job."""
    transport._check_shell_tokens([jobid])
    return (
        f"R={config.shell_word(_reservation_root())}; J={config.shell_word(jobid)}; "
        'for d in "$R"/*; do [ -d "$d" ] || continue; '
        '[ "$(cat "$d/jobid" 2>/dev/null)" = "$J" ] && rm -rf "$d"; done'
    )


def _reserve_checkpoint_stems_command(jobid: str, stems: list[str]) -> str:
    """Atomically reserve checkpoint stems while a job is being staged.

    Each ``mkdir`` is the cross-client compare-and-set: a second submitter
    cannot pass between the prior ``squeue`` snapshot and ``sbatch``.
    """
    transport._check_shell_tokens([jobid, *stems])
    reservations = _reservation_root()
    stem_words = " ".join(stems)
    return f"""R={config.shell_word(reservations)}; J={config.shell_word(jobid)}; mkdir -p "$R"; claimed=""; \
for stem in {stem_words}; do \
  if mkdir "$R/$stem" 2>/dev/null; then \
    printf '%s\\n' "$J" > "$R/$stem/jobid"; claimed="$claimed $stem"; \
  else \
    owner=$(cat "$R/$stem/jobid" 2>/dev/null || true); \
    for held in $claimed; do \
      [ "$(cat "$R/$held/jobid" 2>/dev/null)" = "$J" ] && rm -rf "$R/$held"; \
    done; \
    echo "refusing to stage job $J: checkpoint $stem is reserved by ${{owner:-another staging job}}" >&2; \
    exit 17; \
  fi; \
done"""


def _slurm_batch_script(
    jobid: str,
    payload: str,
    *,
    job_name: str,
    reservation_stems: list[str] | None = None,
    time_limit: str = config.SLURM_TIME,
) -> str:
    """Wrap a CXR queue payload in the lab box's one-GPU SLURM profile."""
    reservation_stems = reservation_stems or []
    transport._check_shell_tokens([jobid, *reservation_stems])
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    sbatch_jobdir = config.sbatch_remote_path(config.JOBS_SUBDIR, jobid)
    reservations = _reservation_root()
    release_lines = (
        "\n  ".join(
            f'if [ "$(cat "$RESERVATIONS/{stem}/jobid" 2>/dev/null)" = "$JOBID" ]; then rm -rf "$RESERVATIONS/{stem}"; fi;'
            for stem in reservation_stems
        )
        or ":"
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={config.SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={config.SLURM_GPUS}
#SBATCH --gres=gpu:{config.SLURM_GPUS}
#SBATCH --time={time_limit}
#SBATCH --output={sbatch_jobdir}/slurm-%j.out
#SBATCH --error={sbatch_jobdir}/slurm-%j.err

set -u
module purge 2>/dev/null || true
module load cuda openmpi hdf5 2>/dev/null || true

JOBDIR={config.shell_word(jobdir)}
JOBID={config.shell_word(jobid)}
RESERVATIONS={config.shell_word(reservations)}
release_reservations() {{
  {release_lines}
}}
termination_signal=""
termination_status=0
record_termination() {{
  termination_signal=$1
  termination_status=$2
  echo "FAILED (signal $termination_signal) $(date -Is)" > "$JOBDIR/state"
}}
finish() {{
  status=$?
  if [ -n "$termination_signal" ]; then
    echo "FAILED (signal $termination_signal) $(date -Is)" > "$JOBDIR/state"
    release_reservations
    trap - EXIT
    exit "$termination_status"
  fi
  current=$(cat "$JOBDIR/state" 2>/dev/null || true)
  case "$current" in
    "queued slice"*) ;;
    done*|FAILED*|cancelled*|cancelling*) release_reservations ;;
    *) echo "FAILED (exit $status) $(date -Is)" > "$JOBDIR/state"; release_reservations ;;
  esac
}}
trap 'record_termination TERM 143' TERM
trap 'record_termination INT 130' INT
trap 'record_termination HUP 129' HUP
trap finish EXIT
echo "running $(date -Is)" > "$JOBDIR/state"
{{
  echo "===== SLURM job ${{SLURM_JOB_ID:-unknown}} ====="
  echo "host: $(hostname)"
  echo "gpus: {config.SLURM_GPUS}"
  echo "partition: {config.SLURM_PARTITION}"
  echo "started: $(date -Is)"
}} >> "$JOBDIR/log"

{payload}"""


def _submit_slurm_command(
    jobid: str, reservation_stems: list[str] | None = None, *, nice: bool = False
) -> str:
    """Return the remote submission protocol for an already-written batch script."""
    reservation_stems = reservation_stems or []
    transport._check_shell_tokens([jobid, *reservation_stems])
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    jobdir_word = config.shell_single_word(jobdir)
    run_script_word = config.shell_single_word(f"{jobdir}/run.sh")
    release = _release_checkpoint_stems_command(jobid, reservation_stems)
    sbatch = "sbatch --parsable --nice=10000" if nice else "sbatch --parsable"
    return f"""D={jobdir_word}; \
echo "queued $(date -Is)" > "$D/state"; \
SID=$({sbatch} {run_script_word}) || {{ \
  echo "FAILED (sbatch submission) $(date -Is)" > "$D/state"; {release}; exit 1; \
}}; \
SID=${{SID%%;*}}; \
case "$SID" in ''|*[!0-9]*) \
  echo "FAILED (sbatch submission) $(date -Is)" > "$D/state"; {release}; exit 1 ;; \
esac; \
printf 'slurm_job_id: %s\\n' "$SID" >> "$D/meta"; \
printf '%s\\n' "$SID"
"""


def _queue_metadata(
    jobid,
    materials,
    quick,
    workers,
    parallel_materials: int | None = config.DEFAULT_PARALLEL_MATERIALS,
    chunk_minutes: float = 0,
    fidelity: str = "full",
    high_energy_min_kev: float | None = None,
    catalog_profile: str = "standard",
):
    """Static metadata persisted before a queue becomes visible to SLURM."""
    return "\n".join(
        [
            f"job: {jobid}",
            f"materials: {' '.join(materials)}",
            f"quick: {bool(quick)}",
            f"fidelity: {fidelity}",
            f"catalog_profile: {catalog_profile}",
            f"workers: {workers}",
            f"parallel_materials: {parallel_materials}",
            f"chunk_minutes: {chunk_minutes}",
            f"high_energy_min_kev: {high_energy_min_kev}",
            "progress_dashboard: True",
            "",
        ]
    )


def _zhai_queue_metadata(jobid, ne, ne_brem, ne_supp):
    """Static metadata persisted before the Zhai batch job is submitted."""
    return "\n".join(
        [
            f"job: {jobid}",
            f"materials: {config.ZHAI_STEM}",
            "quick: False",
            f"ne: {ne}",
            f"ne_brem: {ne_brem}",
            f"ne_supp: {ne_supp}",
            "",
        ]
    )


def _write_job_script_command(jobdir, metadata):
    """Exclusively create a job directory, then receive its script on stdin."""
    jobdir_word = config.shell_single_word(jobdir)
    run_script_word = config.shell_single_word(f"{jobdir}/run.sh")
    metadata_word = config.shell_single_word(f"{jobdir}/meta")
    return (
        f"mkdir {jobdir_word} && cat > {run_script_word} && "
        f"printf %s {shlex.quote(metadata)} > {metadata_word}"
    )


def _squeue_state_command(scheduler_id: str, *, retired: str) -> str:
    """Build fail-closed Bash that stores a live SLURM state in ``STATE``.

    SLURM reports a recently retired ID as a nonzero error on some clusters;
    callers provide the control-flow action that means "not queued" in their
    surrounding shell context. Every other query failure remains fatal.
    """
    return (
        f"STATE=$(squeue -h -j {scheduler_id} -o '%T' 2>&1); STATUS=$?; "
        'if [ "$STATUS" -ne 0 ]; then case "$STATE" in '
        f'*"Invalid job id specified"*) {retired} ;; '
        f'*) echo "could not query SLURM job {scheduler_id}" >&2; exit "$STATUS" ;; esac; fi; '
    )


def _clear_checkpoint_stems_command(jobid: str, stems: list[str]) -> str:
    """Atomically reserve and delete checkpoint stems on the remote box.

    The temporary reservation spans the delete itself, closing the interval
    between a clear's liveness check and ``rm`` where a concurrent ``start``
    could otherwise claim the same checkpoint.  The EXIT trap releases only
    reservations owned by this clear, including after a failed deletion.
    """
    transport._check_shell_tokens([jobid, *stems])
    reservations = _reservation_root()
    stem_words = " ".join(stems)
    releases = " ".join(
        f'if [ "$(cat "$R/{stem}/jobid" 2>/dev/null)" = "$J" ]; then rm -rf "$R/{stem}"; fi;'
        for stem in stems
    )
    return f"""R={config.shell_word(reservations)}; J={config.shell_word(jobid)}; C={config.shell_remote_path("checkpoints")}; \
mkdir -p "$R" || exit $?; \
release() {{ {releases} }}; trap release EXIT; \
for stem in {stem_words}; do \
  if mkdir "$R/$stem" 2>/dev/null; then printf '%s\\n' "$J" > "$R/$stem/jobid"; \
  else printf 'RESERVED\\t%s\\n' "$stem"; exit 0; fi; \
done; \
cd "$C" 2>/dev/null || exit 0; \
for stem in {stem_words}; do \
  if [ -d "$stem" ]; then rm -rf -- "$stem" || exit $?; printf 'CLEARED\\t%s/\\n' "$stem"; fi; \
  f="$stem.pkl"; [ -f "$f" ] || continue; rm -f "$f" || exit $?; \
  printf 'CLEARED\\t%s\\n' "$f"; done"""


def _reap_job_command(jobid: str) -> str:
    """Remote command: release a job's reservations, then stamp its state terminal."""
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    release = _release_job_reservations_command(jobid)
    state_path = config.shell_word(f"{jobdir}/state")
    return f'{release}; echo "reaped (orphan reservations released) $(date -Is)" > {state_path}'


def _recorded_job_dirs_command() -> str:
    """Shell fragment that emits only submitted job directories, in order.

    ``jobs/reservations`` is bookkeeping for checkpoint ownership, not a job.
    A submitted job has its metadata file written before ``sbatch`` runs, so
    that marker also excludes incomplete or unrelated directories safely.
    """
    return (
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        '[ -f "$d/meta" ] || continue; printf "%s\\n" "$(basename "$d")"; done'
    )


def _job_assign(jobid):
    """Bash that sets JOB to the given id, or the latest job dir if none given."""
    if jobid:
        transport._check_shell_tokens([jobid])
        return f'JOB="{jobid}"'
    return f"JOB=$({_recorded_job_dirs_command()} | tail -1)"
