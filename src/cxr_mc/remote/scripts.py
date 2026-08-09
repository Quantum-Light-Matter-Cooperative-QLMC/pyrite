"""SLURM script and remote-command string builders (pure, no ssh)."""

import datetime
import shlex
import uuid

from ..validation._zhai import ZHAI_CACHE_SCHEMA, ZHAI_DETECTOR
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
        from ..campaign.profiles import named_profile_stem

        return [
            named_profile_stem(material, fidelity, catalog_profile=catalog_profile)
            for material in materials
        ]
    from ..campaign.profiles import high_energy_floor_stem, named_profile_stem
    from ..runs.scan import load_manifest_groups

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


def _uv_sync_block(once: bool = False) -> str:
    """Bash that records dependency-sync wall time without changing exit semantics.

    once=True guards the sync behind a ``$JOBDIR/.synced`` sentinel so a
    self-resubmitting chunked chain syncs on its FIRST slice only -- every later
    slice runs in the same checked-out tree with the same lockfile, so re-syncing
    is pure per-slice startup overhead. The sentinel is written only after a
    successful sync, so a failed sync still exits 1 and the next slice retries.
    Delete ``$JOBDIR/.synced`` to force a re-sync (e.g. after a mid-chain
    dependency bump)."""
    sync = f"""uv_sync_start_ns=$(date +%s%N)
uv_sync_rc=0
{config.shell_remote_uv()} sync --package cxr-mc --no-dev --extra {config.remote_gpu_vendor()} >> "$JOBDIR/log" 2>&1 || uv_sync_rc=$?
uv_sync_elapsed_ms=$((($(date +%s%N) - uv_sync_start_ns) / 1000000))
printf 'timing: uv sync %d.%03d s\\n' \
  "$((uv_sync_elapsed_ms / 1000))" "$((uv_sync_elapsed_ms % 1000))" >> "$JOBDIR/log"
if [ "$uv_sync_rc" -ne 0 ]; then
  echo "FAILED (uv sync) $(date -Is)" > "$JOBDIR/state"
  exit 1
fi"""
    if not once:
        return sync
    return f"""if [ -f "$JOBDIR/.synced" ]; then
  printf 'timing: uv sync skipped (already synced this chain)\\n' >> "$JOBDIR/log"
else
{sync}
  : > "$JOBDIR/.synced"
fi"""


def _cpu_profile_block(catalog_profile, performance_profile, cpu_flags):
    """First-class bounded serial CPU cProfile phase for one material."""
    pstats_dump = (
        "import pstats,sys; p=pstats.Stats(sys.argv[1]); "
        'p.sort_stats("tottime").print_stats(60); '
        'p.sort_stats("cumulative").print_stats(60)'
    )
    return f"""  echo "profiling CPU $m [$i/$total] since $(date -Is)" > "$JOBDIR/state"
  cpu_prof_base="$JOBDIR/performance/{performance_profile}/$m.cpu"
  cpu_ckpt="$JOBDIR/cpu-profile-checkpoints/$m"
  mkdir -p "$(dirname "$cpu_prof_base")" "$cpu_ckpt"
  printf '%s\\n' "cProfile transport-only pass (serial CPU)">> "$JOBDIR/log"
  cpu_prof_rc=0
  env -u CXR_MC_NSYS CXR_MC_BACKEND=cpu {config.shell_remote_uv()} run --no-sync python \\
    -m cProfile -o "$cpu_prof_base.prof" \\
    -m cxr_mc._entry.scan {config.shell_word(catalog_profile)} -m "$m"{cpu_flags} --workers 0 --transport-only\\
    --max-minutes 10 \\
    --checkpoint-dir "$cpu_ckpt" --progress-file "$JOBDIR/progress/$m.cpu.json" \\
    --progress-phase cpu --no-progress >> "$JOBDIR/log" 2>&1 || cpu_prof_rc=$?
  if [ "$cpu_prof_rc" -eq 0 ] && [ -f "$cpu_prof_base.prof" ]; then
    {config.shell_remote_uv()} run --no-sync python -c '{pstats_dump}' \\
      "$cpu_prof_base.prof" > "$cpu_prof_base.txt" 2>&1 || cpu_prof_rc=$?
  fi
  if [ "$cpu_prof_rc" -ne 0 ] || [ ! -f "$cpu_prof_base.prof" ] || [ ! -f "$cpu_prof_base.txt" ]; then
    mkdir -p "$JOBDIR/cpu-failures"
    : > "$JOBDIR/cpu-failures/$m"
    echo "CPU profile failed for $m (exit $cpu_prof_rc) $(date -Is)" > "$JOBDIR/state"
    echo "ERROR: CPU cProfile phase failed for $m (exit $cpu_prof_rc)" >> "$JOBDIR/log"
    return 1
  fi
  echo "completed CPU profile: $m $(date -Is)" > "$JOBDIR/state"
  echo "completed CPU profile: $m" >> "$JOBDIR/log"
"""


def _queue_script(
    jobid,
    materials,
    quick,
    workers,
    parallel_materials=config.DEFAULT_PARALLEL_MATERIALS,
    fidelity="full",
    high_energy_min_kev=None,
    catalog_profile="standard",
    performance_profile=None,
    performance_repetitions=1,
    performance_interval=5.0,
    spec_chunk=None,
    brem_chunk=None,
    nsys=False,
    cpu=False,
    cpu_only=False,
):
    """CXR payload for one bounded-concurrency queue in a SLURM allocation."""
    parallel_materials = _validate_parallel_materials(parallel_materials)
    if nsys and performance_profile is None:
        raise ValueError("nsys requires a performance profile")
    if (cpu or cpu_only) and performance_profile is None:
        raise ValueError("CPU profiling requires a performance profile")
    if cpu and cpu_only:
        raise ValueError("cpu and cpu_only are mutually exclusive")
    if cpu_only and nsys:
        raise ValueError("cpu_only cannot be combined with nsys")
    flags = ""
    if quick:
        flags += " --quick"
    if fidelity != "full":
        flags += f" --fidelity {fidelity}"
    if workers is not None:
        flags += f" --workers {workers}"
    if performance_profile is not None and not cpu_only:
        flags += (
            f' --performance-profile {performance_profile} --performance-dir "$JOBDIR/performance"'
        )
        if performance_interval != 5.0:
            flags += f" --perf-interval {performance_interval:g}"
    # cProfile phase uses the same material, forced onto the serial NumPy path so it can
    # attribute the compute the GPU pipeline otherwise drives in-process. It
    # forces --workers 0 (serial, everything in THIS process) and drops
    # --performance-profile so its sampler JSON never clobbers the GPU tick file.
    #
    # It always uses the tiny --quick smoke grid regardless of the parent job's
    # fidelity: cProfile only needs a representative call distribution (which
    # functions dominate tottime), not full statistics. A full-fidelity serial
    # NumPy pass ran >1h and had to be killed; --quick finishes in minutes while
    # firing the same spectrum functions. --max-minutes is a hard backstop.
    cpu_flags = " --quick"
    cpu_profile_block = (
        _cpu_profile_block(catalog_profile, performance_profile, cpu_flags)
        if cpu or cpu_only
        else ""
    )
    runtime_exports = ""
    if spec_chunk is not None:
        runtime_exports += f"\nexport CXR_MC_SPEC_CHUNK={spec_chunk}"
    if brem_chunk is not None:
        runtime_exports += f"\nexport CXR_MC_BREM_CHUNK={brem_chunk}"
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block(once=True)}
mats=({mats})
total=${{#mats[@]}}
parallel_materials={parallel_materials}
# co-tenant scan processes share the one GPU; each divides its CuPy pool cap
# (_GPU_POOL_FRAC) by this so N processes cap at FRAC total, not N*FRAC.
export CXR_MC_GPU_SHARE={parallel_materials}
performance_repetitions={performance_repetitions}{runtime_exports}
nsys_enabled={int(bool(nsys))}
cpu_enabled={int(bool(cpu))}
cpu_only_enabled={int(bool(cpu_only))}
n=0
failures=0
active=0
run_material() {{
  local i="$1"
  local m="$2"
  local repetition
  local scan_rc
  local trace_base
  local -a checkpoint_flags=()
  local -a scan_launcher=()
  local -a scan_command=()
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$i" "$total" "$m" "$(date -Is)" \
>> "$JOBDIR/log"
  if [ "$cpu_only_enabled" -eq 0 ]; then
    echo "running primary $m [$i/$total] since $(date -Is)" > "$JOBDIR/state"
  for (( repetition=1; repetition<=performance_repetitions; repetition++ )); do
    checkpoint_flags=()
    if [ "$performance_repetitions" -gt 1 ] || [ "$nsys_enabled" -eq 1 ]; then
      checkpoint_flags=(--checkpoint-dir "$JOBDIR/performance-checkpoints/$m/$repetition")
    fi
    if [ "$performance_repetitions" -gt 1 ]; then
      printf '%s\\n' "performance repetition $repetition/$performance_repetitions" >> "$JOBDIR/log"
    elif [ "$nsys_enabled" -eq 1 ]; then
      printf '%s\\n' "Nsight Systems uncached trace" >> "$JOBDIR/log"
    fi
    if [ "$nsys_enabled" -eq 1 ]; then
      scan_launcher=({config.shell_remote_path(".venv", "bin", "python")})
    else
      scan_launcher=({config.shell_remote_uv()} run --no-sync python)
    fi
    scan_command=(
      "${{scan_launcher[@]}}" -m cxr_mc._entry.scan {config.shell_word(catalog_profile)} -m "$m"{flags}
      "${{checkpoint_flags[@]}}" --progress-file "$JOBDIR/progress/$m.json" \\
      --progress-phase primary --no-progress
    )
    scan_rc=0
    if [ "$nsys_enabled" -eq 1 ]; then
      trace_base="$JOBDIR/performance/{performance_profile}/$m"
      mkdir -p "$(dirname "$trace_base")"
      if ! command -v nsys >/dev/null 2>&1; then
        echo "ERROR: --nsys requested but nsys is unavailable on the worker" >> "$JOBDIR/log"
        scan_rc=127
      else
        export CXR_MC_NSYS=1
        # --wait=all lets nsys wait for the spawn worker tree to exit cleanly
        # (else --wait=primary SIGTERMs survivors -> leaked semaphores).
        nsys_cmd=(
          nsys profile
          --trace=cuda,nvtx,osrt
          --sample=process-tree
          --cpuctxsw=process-tree
          --wait=all
          --force-overwrite=true
          --output="$trace_base"
        )
        # nsys 2025.6.x Python stack-walkers SIGSEGV unwinding CPython 3.14's
        # frame layout, so they are off by default. Opt in with
        # CXR_MC_NSYS_PYSTACK=1 only on a supported Python/nsys pair.
        if [ -n "${{CXR_MC_NSYS_PYSTACK:-}}" ]; then
          nsys_cmd+=(
            --python-sampling=true --python-sampling-frequency=200 \\
            --python-backtrace=cuda --cudabacktrace=sync,kernel,memory
          )
        fi
        "${{nsys_cmd[@]}}" \\
          "${{scan_command[@]}}" >> "$JOBDIR/log" 2>&1 || scan_rc=$?
      fi
    else
      "${{scan_command[@]}}" >> "$JOBDIR/log" 2>&1 || scan_rc=$?
    fi
    if [ "$scan_rc" -ne 0 ]; then
      if [ "$performance_repetitions" -gt 1 ]; then
        echo "WARNING: scan failed for $m repetition $repetition; continuing" >> "$JOBDIR/log"
        echo "warning at $m [$i/$total] repetition $repetition $(date -Is)" > "$JOBDIR/state"
      else
        echo "WARNING: scan failed for $m; continuing" >> "$JOBDIR/log"
        echo "warning at $m [$i/$total] $(date -Is)" > "$JOBDIR/state"
      fi
      return 1
    fi
    if [ "$nsys_enabled" -eq 1 ] && [ -f "$trace_base.nsys-rep" ]; then
      nsys stats \\
        --report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_sum \\
        "$trace_base.nsys-rep" > "$trace_base.nsys-stats.txt" 2>&1 || \\
        echo "WARNING: nsys stats failed; .nsys-rep remains available" >> "$JOBDIR/log"
    fi
  done
    if [ "$cpu_enabled" -eq 1 ]; then
      echo "completed primary: $m" >> "$JOBDIR/log"
    fi
  fi
{cpu_profile_block}  echo "completed: $m" >> "$JOBDIR/log"
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
if compgen -G "$JOBDIR/cpu-failures/*" >/dev/null; then
  cpu_failure_count=$(find "$JOBDIR/cpu-failures" -maxdepth 1 -type f | wc -l)
  echo "FAILED CPU profile ($cpu_failure_count material(s)) $(date -Is)" > "$JOBDIR/state"
  exit 1
elif [ "$failures" -gt 0 ]; then
  echo "done with $failures warning(s) [$total/$total] $(date -Is)" > "$JOBDIR/state"
elif [ "$cpu_enabled" -eq 1 ] || [ "$cpu_only_enabled" -eq 1 ]; then
  echo "done CPU profile [$total/$total] $(date -Is)" > "$JOBDIR/state"
else
  echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
fi
"""


_SBATCH_RETRY_ATTEMPTS = 5
_SBATCH_RETRY_SECONDS = 5


def _sbatch_retry_block(sbatch_cmd: str, script_word: str) -> str:
    """Bash: submit via sbatch, retrying slurmctld's transient RPC timeouts.

    The lab box's slurmctld occasionally answers a submission with "Socket
    timed out on send/recv operation" under load; a few retries with backoff
    clear it without failing a checkpointed chain over one RPC hiccup.
    """
    return f"""SID=""
attempt=0
while [ "$attempt" -lt {_SBATCH_RETRY_ATTEMPTS} ]; do
  SID=$({sbatch_cmd} {script_word}) && break
  attempt=$((attempt + 1))
  [ "$attempt" -lt {_SBATCH_RETRY_ATTEMPTS} ] && sleep {_SBATCH_RETRY_SECONDS}
done"""


def _chunked_queue_script(
    jobid,
    materials,
    quick,
    workers,
    chunk_minutes,
    fidelity="full",
    high_energy_min_kev=None,
    catalog_profile="standard",
    performance_profile=None,
    performance_interval=5.0,
    spec_chunk=None,
    brem_chunk=None,
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
    if performance_profile is not None:
        flags += (
            f' --performance-profile {performance_profile} --performance-dir "$JOBDIR/performance"'
        )
        if performance_interval != 5.0:
            flags += f" --perf-interval {performance_interval:g}"
    runtime_exports = ""
    if spec_chunk is not None:
        runtime_exports += f"\nexport CXR_MC_SPEC_CHUNK={spec_chunk}"
    if brem_chunk is not None:
        runtime_exports += f"\nexport CXR_MC_BREM_CHUNK={brem_chunk}"
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    chunk_seconds = int(round(chunk_minutes * 60))
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block(once=True)}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
{runtime_exports.lstrip()}
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
  {config.shell_remote_uv()} run --no-sync python -m cxr_mc._entry.scan {config.shell_word(catalog_profile)} -m "$m"{flags} --max-minutes "$remaining_min" \
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
{_sbatch_retry_block("sbatch --parsable --nice=10000", '"$JOBDIR/run.sh"')}
if [ -z "$SID" ]; then
  echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
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
{_uv_sync_block(once=True)}
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
{_uv_sync_block(once=True)}
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
{_uv_sync_block(once=True)}
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
{_sbatch_retry_block("sbatch --parsable --nice=10000", '"$JOBDIR/run.sh"')}
if [ -z "$SID" ]; then
  echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
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
    rebrem`` keys the Mode line in plain/attached status; ``materials``/``quick`` keep
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
{_uv_sync_block(once=True)}
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
{_uv_sync_block(once=True)}
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
{_sbatch_retry_block("sbatch --parsable --nice=10000", '"$JOBDIR/run.sh"')}
if [ -z "$SID" ]; then
  echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
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
    cpus_per_task: int = config.SLURM_CPUS_PER_MATERIAL,
) -> str:
    """Wrap a CXR queue payload in the lab box's one-GPU SLURM profile."""
    vendor = config.remote_gpu_vendor()
    if vendor != "nvidia":
        raise ValueError(
            f"cxr remote lab-box scripts do not yet support {vendor}; "
            "use a site-specific SLURM template from docs/running-on-a-cluster.md"
        )
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
#SBATCH --cpus-per-task={cpus_per_task}
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
SID=""; attempt=0; \
while [ "$attempt" -lt {_SBATCH_RETRY_ATTEMPTS} ]; do \
  SID=$({sbatch} {run_script_word}) && break; \
  attempt=$((attempt + 1)); \
  [ "$attempt" -lt {_SBATCH_RETRY_ATTEMPTS} ] && sleep {_SBATCH_RETRY_SECONDS}; \
done; \
if [ -z "$SID" ]; then \
  echo "FAILED (sbatch submission) $(date -Is)" > "$D/state"; {release}; exit 1; \
fi; \
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
    performance_profile: str | None = None,
    performance_repetitions: int = 1,
    performance_interval: float = 5.0,
    spec_chunk: int | None = None,
    brem_chunk: int | None = None,
    nsys: bool = False,
    cpu: bool = False,
    cpu_only: bool = False,
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
            f"performance_profile: {performance_profile}",
            f"performance_repetitions: {performance_repetitions}",
            f"performance_interval: {performance_interval}",
            f"spec_chunk: {spec_chunk}",
            f"brem_chunk: {brem_chunk}",
            f"nsys: {bool(nsys)}",
            f"cpu: {bool(cpu)}",
            f"cpu_only: {bool(cpu_only)}",
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
            f"zhai_cache_schema: {ZHAI_CACHE_SCHEMA}",
            f"detector_observation_angle_deg: {ZHAI_DETECTOR.observation_angle_deg:g}",
            f"detector_polar_acceptance_deg: {ZHAI_DETECTOR.polar_acceptance_deg:g}",
            f"detector_solid_angle_sr: {ZHAI_DETECTOR.solid_angle_sr:g}",
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


def _scancel_jobs_command(jobids: list[str]) -> str:
    """Cancel many live jobs in ONE remote session (stop --all / multi-stop).

    Per-job ``_stop_jobid`` pays one ssh plus a full scancel-propagation poll
    per job, which serializes to 30+ s on a batch. Here every STOP sentinel
    lands BEFORE the single ``scancel`` (a resubmitting slice must see STOP
    even if it was about to re-queue), one poll loop waits for all scheduler
    IDs to leave squeue together, and reservation release + terminal state
    writes follow per job. Jobs without a recorded scheduler ID are skipped
    with a diagnostic rather than aborting the batch.
    """
    transport._check_shell_tokens(jobids)
    jobs = config.shell_remote_path(config.JOBS_SUBDIR)
    reservation_root = config.shell_word(_reservation_root())
    job_words = " ".join(jobids)
    return (
        f"JOBS={jobs}; SIDS=; PAIRS=; "
        f"for j in {job_words}; do "
        'D="$JOBS/$j"; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1); '
        'case "$SID" in \'\'|*[!0-9]*) echo "skipping $j: no recorded scheduler id" >&2; continue ;; esac; '
        ': > "$D/STOP"; '
        'SIDS="$SIDS $SID"; PAIRS="$PAIRS $j:$SID"; done; '
        '[ -n "$SIDS" ] || { echo "no cancellable SLURM jobs" >&2; exit 1; }; '
        "scancel $SIDS || exit $?; "
        "for pair in $PAIRS; do j=${pair%%:*}; SID=${pair##*:}; "
        'echo "cancelling [$SID] $(date -Is)" > "$JOBS/$j/state"; done; '
        "while :; do "
        'LIVE=" $(squeue -h -u "$USER" -o \'%i\' 2>/dev/null | tr "\\n" " ") "; '
        "pending=0; "
        'for SID in $SIDS; do case "$LIVE" in *" $SID "*) pending=1 ;; esac; done; '
        '[ "$pending" -eq 0 ] && break; sleep 1; done; '
        f"R={reservation_root}; "
        "for pair in $PAIRS; do j=${pair%%:*}; SID=${pair##*:}; "
        'for d in "$R"/*; do [ -d "$d" ] || continue; '
        '[ "$(cat "$d/jobid" 2>/dev/null)" = "$j" ] && rm -rf "$d"; done; '
        'echo "cancelled [$SID] $(date -Is)" > "$JOBS/$j/state"; '
        'echo "cancelled SLURM job $SID for job $j"; done'
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


def _prune_checkpoint_stems_command(
    jobid: str,
    stems: list[str],
    *,
    all_profiles: bool = False,
    catalog_profile: str | None = None,
    yes: bool = False,
) -> str:
    """Reserve exact current stems while running ``cxr prune`` remotely."""
    transport._check_shell_tokens([jobid, *stems])
    if all_profiles and catalog_profile is not None:
        raise ValueError("all_profiles and catalog_profile are mutually exclusive")
    reserve = _reserve_checkpoint_stems_command(jobid, stems)
    release = _release_checkpoint_stems_command(jobid, stems)
    args = ["prune"]
    if all_profiles:
        args.append("--all")
    elif catalog_profile is not None:
        args.extend(("--profile", catalog_profile))
    if yes:
        args.append("--yes")
    command = " ".join(config.shell_arg(arg) for arg in args)
    return (
        f"{reserve}; "
        f"release_prune() {{ {release}; }}; trap release_prune EXIT; "
        f"cd {config.shell_remote_dir()} || exit $?; "
        f"{config.shell_remote_uv()} run --no-sync cxr {command}"
    )


def _reap_job_command(jobid: str) -> str:
    """Remote command: release a job's reservations, then stamp its state terminal."""
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    release = _release_job_reservations_command(jobid)
    state_path = config.shell_word(f"{jobdir}/state")
    return f'{release}; echo "reaped (orphan reservations released) $(date -Is)" > {state_path}'


def _recorded_job_dirs_command() -> str:
    """Shell fragment that emits submitted job dirs oldest-first by meta mtime.

    ``jobs/reservations`` is bookkeeping for checkpoint ownership, not a job.
    A submitted job has its metadata file written before ``sbatch`` runs, so
    that marker also excludes incomplete or unrelated directories safely.

    Emission is ordered by the ``meta`` file's mtime so ``tail -1`` is the
    most-recently-submitted/active job. Name order is NOT chronological for
    profile-named jobs (``sub_100keV``, ``sub_100keV-2`` ...): the bare name
    sorts after every ``-N`` (the glob adds a trailing ``/`` and ``-`` < ``/``)
    and ``-10`` < ``-2``, so a plain name sort would default ``attach``/``logs``
    to the wrong job. A live chain keeps its meta fresh (it appends ``started:``
    and ``slurm_job_id:`` each slice), so it stays the newest entry.
    """
    return (
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        '[ -f "$d/meta" ] || continue; '
        'printf "%s\\t%s\\n" "$(stat -c %Y "$d/meta")" "$(basename "$d")"; done '
        "| sort -n | cut -f2"
    )


def _job_assign(jobid):
    """Bash that sets JOB to the given id, or the latest job dir if none given."""
    if jobid:
        transport._check_shell_tokens([jobid])
        return f'JOB="{jobid}"'
    return f"JOB=$({_recorded_job_dirs_command()} | tail -1)"


def _prune_job_dirs_command(profile=None, all_jobs=False, yes=False):
    """Remote command: delete terminal, non-live job directories.

    Selects the whole ``jobs/`` tree (``all_jobs``) or one profile's
    ``NAME``/``NAME-N`` family, then removes only dirs whose persisted state is
    chain-terminal (done / FAILED / cancelled, matching viewer._is_terminal_state)
    AND whose latest recorded SLURM id is not in ``squeue`` -- a live chain is
    never pruned even if an earlier slice left a transient state. squeue failure
    is fail-closed (nothing pruned). Emits one tab record per matched dir:
    ``PRUNED``/``WOULD-PRUNE`` with the state, or ``KEPT`` with ``live`` or the
    non-terminal state. ``profile`` is interpolated into the glob and must be
    token-checked by the caller.
    """
    jobs = config.shell_remote_path(config.JOBS_SUBDIR)
    glob = '"$JOBS"/*/' if all_jobs else f'"$JOBS/{profile}"/ "$JOBS/{profile}"-*/'
    action = (
        'rm -rf "$d" && printf "PRUNED\\t%s\\t%s\\n" "$jobid" "$st"'
        if yes
        else 'printf "WOULD-PRUNE\\t%s\\t%s\\n" "$jobid" "$st"'
    )
    return (
        f"JOBS={jobs}; "
        '[ -d "$JOBS" ] || exit 0; '
        "LIVE=$(squeue -h -u \"$USER\" -o '%i' 2>&1); STATUS=$?; "
        'if [ "$STATUS" -ne 0 ]; then '
        'echo "could not query SLURM jobs" >&2; printf "%s\\n" "$LIVE" >&2; '
        'exit "$STATUS"; fi; '
        'LIVE=" $(printf "%s\\n" "$LIVE" | tr "\\n" " ") "; '
        f'for d in {glob}; do [ -d "$d" ] || continue; [ -f "$d/meta" ] || continue; '
        "jobid=${d%/}; jobid=${jobid##*/}; "
        'st=$(head -n1 "$d/state" 2>/dev/null); '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$d/meta" 2>/dev/null | tail -1); '
        'case "$SID" in ""|*[!0-9]*) ;; *) case "$LIVE" in *" $SID "*) '
        'printf "KEPT\\t%s\\tlive\\n" "$jobid"; continue ;; esac ;; esac; '
        'case "$st" in done*|FAILED*|cancelled*) ;; *) '
        'printf "KEPT\\t%s\\t%s\\n" "$jobid" "${st:-<none>}"; continue ;; esac; '
        f"{action}; done"
    )
