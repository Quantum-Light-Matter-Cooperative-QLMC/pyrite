"""Queue payload and metadata string builders (pure, no ssh)."""

import datetime
import shlex
import uuid

from .._env import env_value
from ..perf.py_spy import (
    PY_SPY_REQUIREMENT,
    PY_SPY_STATUS_SUFFIX,
    PY_SPY_SUFFIX,
    py_spy_record_args,
)
from ..validation._zhai import ZHAI_CACHE_SCHEMA, ZHAI_DETECTOR
from . import config

# Private compatibility classification for legacy queue records that stored a
# high-energy floor instead of a catalog profile identity. New submissions use
# the explicit ``high_energy`` profile and never consult this mapping.
_LEGACY_HIGH_ENERGY_MATERIALS = frozenset({"tise2", "gep", "ges", "rese2"})


# Local process pins copied into generated job scripts; the remote job has no
# other route to the submitter's environment.
_FORWARDED_ENV = ("PYRITE_MC_TRANSPORT_CORE", "PYRITE_MC_RESOURCE_POLICY", "PYRITE_MC_MIN_CHUNK")


def _forwarded_env_choices(name):
    if name == "PYRITE_MC_MIN_CHUNK":
        return None
    if name == "PYRITE_MC_TRANSPORT_CORE":
        from ..montecarlo.transport.batching import TRANSPORT_CORES

        return TRANSPORT_CORES
    from ..montecarlo._resources import _VALID_POLICIES

    return _VALID_POLICIES


def _forwarded_env_exports():
    """``export`` lines for each forwarded variable set in this process."""
    exports = ""
    for name in _FORWARDED_ENV:
        value = env_value(name, "").strip().lower()
        if not value:
            continue
        choices = _forwarded_env_choices(name)
        if choices is None:
            if not value.isdigit() or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        elif value not in choices:
            raise ValueError(f"{name} must be one of {', '.join(choices)}; got {value!r}")
        exports += f"\nexport {name}={shlex.quote(value)}"
    return exports


# ---- detached job queue -------------------------------------------------------
def _stems(materials, quick, fidelity="full", high_energy_min_kev=None, catalog_profile="standard"):
    """Checkpoint stems a queue produces (the scan runner writes
    <material>_quick/ for --quick runs). ``high_energy_min_kev`` predicts the
    non-canonical stem for materials classified by legacy job records (a no-op
    stem-wise for every other material)."""
    return [
        stem
        for _, stem in _material_stems(
            materials, quick, fidelity, high_energy_min_kev, catalog_profile
        )
    ]


def _material_stems(
    materials, quick, fidelity="full", high_energy_min_kev=None, catalog_profile="standard"
):
    """``(material, stem)`` pairs behind :func:`_stems`; a multi-detector
    profile yields one pair per detector for each material."""
    from ..materials import CATALOG, load_material_catalog

    catalog = (
        CATALOG if catalog_profile == "standard" else load_material_catalog(profile=catalog_profile)
    )
    detector_ids = tuple(catalog.profile_detector_set(catalog_profile))
    multiple = len(detector_ids) > 1 or (
        catalog_profile not in catalog.profile_detectors
        and catalog_profile not in catalog.profile_physical_detectors
        and bool(catalog.profile_detector_set(catalog_profile)[detector_ids[0]])
    )
    if quick:
        return [
            (material, f"{material}_quick_{detector_id}" if multiple else f"{material}_quick")
            for material in materials
            for detector_id in detector_ids
        ]
    if high_energy_min_kev is None:
        if fidelity == "full" and catalog_profile == "standard" and not multiple:
            return [(material, material) for material in materials]
        from ..campaign.profiles import named_profile_stem

        return [
            (
                material,
                named_profile_stem(
                    material, fidelity, catalog_profile=catalog_profile, detector_id=detector_id
                ),
            )
            for material in materials
            for detector_id in detector_ids
        ]
    from ..campaign.profiles import high_energy_floor_stem, named_profile_stem

    stems = []
    for material in materials:
        for detector_id in detector_ids:
            if material in _LEGACY_HIGH_ENERGY_MATERIALS:
                stem = high_energy_floor_stem(
                    material,
                    high_energy_min_kev,
                    fidelity,
                    catalog_profile=catalog_profile,
                    detector_id=detector_id,
                )
            elif fidelity == "full" and catalog_profile == "standard" and not multiple:
                stem = material
            else:
                stem = named_profile_stem(
                    material, fidelity, catalog_profile=catalog_profile, detector_id=detector_id
                )
            stems.append((material, stem))
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


def _tables_preflight_block() -> str:
    """Bash that fails the job fast when pinned tables, datasets or sdbase are missing.

    Runs ``pyrite tables verify`` (table manifest digests; the EEDL, EADL and EPDL
    files hashed in full; SBETHE's ``sdbase/``) once per chain, guarded like
    the dependency sync by ``$JOBDIR/.tables_ok``, before any transport runs. A miss records
    ``FAILED (tables)`` and exits 1; the verify output, which names the fix,
    lands in the job log."""
    verify = f"{config.remote_runtime_env()} {config.shell_remote_uv()} run --no-sync pyrite tables verify --require bremslib,elsepa,sbethe-tables,eedl,eadl,epdl,sbethe"
    return f"""if [ ! -f "$JOBDIR/.tables_ok" ]; then
  if {verify} >> "$JOBDIR/log" 2>&1; then
    : > "$JOBDIR/.tables_ok"
  else
    echo "FAILED (tables) $(date -Is)" > "$JOBDIR/state"
    exit 1
  fi
fi"""


def _uv_sync_block(once: bool = False) -> str:
    """Bash that records dependency-sync wall time without changing exit semantics.

    once=True guards the sync behind a ``$JOBDIR/.synced`` sentinel so a
    self-resubmitting chunked chain syncs on its FIRST slice only -- every later
    slice runs in the same checked-out tree with the same lockfile, so re-syncing
    is pure per-slice startup overhead. The sentinel is written only after a
    successful sync, so a failed sync still exits 1 and the next slice retries.
    Delete ``$JOBDIR/.synced`` to force a re-sync (e.g. after a mid-chain
    dependency bump).

    AMD targets build CuPy from source against the node's ROCm toolchain, which
    ``CUPY_INSTALL_USE_HIP=1`` selects; NVIDIA installs the prebuilt wheel."""
    vendor = config.remote_gpu_vendor()
    build_env = "CUPY_INSTALL_USE_HIP=1 " if vendor == "amd" else ""
    sync = f"""uv_sync_start_ns=$(date +%s%N)
uv_sync_rc=0
{build_env}{config.shell_remote_uv()} sync --package pyrite-xray --no-default-groups --extra {vendor} >> "$JOBDIR/log" 2>&1 || uv_sync_rc=$?
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
  printf '%s\\n' "cProfile pass (serial CPU)" >> "$JOBDIR/log"
  cpu_prof_rc=0
  env -u PYRITE_MC_NSYS PYRITE_MC_BACKEND=cpu {config.shell_remote_uv()} run --no-sync python \\
    -m pyrite._entry.profile_module "$cpu_prof_base.prof" \\
    pyrite._entry.scan {config.shell_word(catalog_profile)} -m "$m"{cpu_flags} --workers 0 \\
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
    no_cache=False,
    recompute=False,
    py_spy=False,
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
    if py_spy and (performance_profile is None or nsys or cpu_only):
        raise ValueError("py_spy requires a performance profile and excludes nsys/cpu_only")
    if no_cache and recompute:
        raise ValueError("no_cache and recompute are mutually exclusive")
    flags = ""
    if quick:
        flags += " --quick"
    if fidelity != "full":
        flags += f" --fidelity {fidelity}"
    if workers is not None:
        flags += f" --workers {workers}"
    if no_cache:
        flags += " --no-cache"
    elif recompute:
        flags += " --recompute"
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
        runtime_exports += f"\nexport PYRITE_MC_SPEC_CHUNK={spec_chunk}"
    if brem_chunk is not None:
        runtime_exports += f"\nexport PYRITE_MC_BREM_CHUNK={brem_chunk}"
    runtime_exports += _forwarded_env_exports()
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    # The output path stays a double-quoted shell word so "$trace_base" expands.
    py_spy_args = " ".join(
        f'"$trace_base{PY_SPY_SUFFIX}"' if arg == "__OUTPUT__" else config.shell_word(arg)
        for arg in py_spy_record_args("__OUTPUT__")
    )
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block(once=True)}
{_tables_preflight_block()}
mats=({mats})
total=${{#mats[@]}}
parallel_materials={parallel_materials}
# co-tenant scan processes share the one GPU; each divides its CuPy pool cap
# (_GPU_POOL_FRAC) by this so N processes cap at FRAC total, not N*FRAC.
export PYRITE_MC_GPU_SHARE={parallel_materials}
performance_repetitions={performance_repetitions}{runtime_exports}
nsys_enabled={int(bool(nsys))}
py_spy_enabled={int(bool(py_spy))}
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
    if [ "$performance_repetitions" -gt 1 ] || [ "$nsys_enabled" -eq 1 ] || [ "$py_spy_enabled" -eq 1 ]; then
      checkpoint_flags=(--checkpoint-dir "$JOBDIR/performance-checkpoints/$m/$repetition")
    fi
    if [ "$performance_repetitions" -gt 1 ]; then
      printf '%s\\n' "performance repetition $repetition/$performance_repetitions" >> "$JOBDIR/log"
    elif [ "$nsys_enabled" -eq 1 ]; then
      printf '%s\\n' "Nsight Systems uncached trace" >> "$JOBDIR/log"
    elif [ "$py_spy_enabled" -eq 1 ]; then
      printf '%s\\n' "py-spy uncached sampling profile" >> "$JOBDIR/log"
    fi
    if [ "$nsys_enabled" -eq 1 ] || [ "$py_spy_enabled" -eq 1 ]; then
      scan_launcher=({config.shell_remote_path(".venv", "bin", "python")})
    else
      scan_launcher=({config.shell_remote_uv()} run --no-sync python)
    fi
    scan_command=(
      "${{scan_launcher[@]}}" -u -m pyrite._entry.scan {config.shell_word(catalog_profile)} -m "$m"{flags}
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
        export PYRITE_MC_NSYS=1
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
        # PYRITE_MC_NSYS_PYSTACK=1 only on a supported Python/nsys pair.
        if [ -n "${{PYRITE_MC_NSYS_PYSTACK:-}}" ]; then
          nsys_cmd+=(
            --python-sampling=true --python-sampling-frequency=200 \\
            --python-backtrace=cuda --cudabacktrace=sync,kernel,memory
          )
        fi
        "${{nsys_cmd[@]}}" \\
          "${{scan_command[@]}}" >> "$JOBDIR/log" 2>&1 || scan_rc=$?
      fi
    elif [ "$py_spy_enabled" -eq 1 ]; then
      trace_base="$JOBDIR/performance/{performance_profile}/$m"
      mkdir -p "$(dirname "$trace_base")"
      # py-spy launches the scan itself, so it needs no ptrace permission. With
      # --subprocesses it exits 0 whatever the scan does; the status wrapper
      # records the scan's own exit status instead.
      rm -f "$trace_base{PY_SPY_STATUS_SUFFIX}"
      {config.shell_remote_uv()} tool run --from {config.shell_word(PY_SPY_REQUIREMENT)} py-spy \\
        {py_spy_args} \\
        {config.shell_remote_path(".venv", "bin", "python")} -m pyrite.perf.py_spy \\
        "$trace_base{PY_SPY_STATUS_SUFFIX}" -- \\
        "${{scan_command[@]}}" >> "$JOBDIR/log" 2>&1
      scan_rc=$(cat "$trace_base{PY_SPY_STATUS_SUFFIX}" 2>/dev/null || echo 1)
    else
      "${{scan_command[@]}}" >> "$JOBDIR/log" 2>&1 || scan_rc=$?
    fi
    if [ "$scan_rc" -eq 143 ] && [ -f "$JOBDIR/STOP" ]; then
      echo "cancelled: scan for $m stopped by cancel request" >> "$JOBDIR/log"
      return 1
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
# `wait -n` only throttles: it returns 127 for a job bash reaped before the
# call, so its status cannot count failures. Each material marks success.
rm -rf "$JOBDIR/.material-ok"
mkdir -p "$JOBDIR/.material-ok"
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  {{ run_material "$n" "$m" && : > "$JOBDIR/.material-ok/$n"; }} &
  active=$((active + 1))
  if [ "$active" -ge "$parallel_materials" ]; then
    wait -n || true
    active=$((active - 1))
  fi
done
wait
ok_count=$(find "$JOBDIR/.material-ok" -maxdepth 1 -type f | wc -l)
failures=$((total - ok_count))
if compgen -G "$JOBDIR/cpu-failures/*" >/dev/null; then
  cpu_failure_count=$(find "$JOBDIR/cpu-failures" -maxdepth 1 -type f | wc -l)
  echo "FAILED CPU profile ($cpu_failure_count material(s)) $(date -Is)" > "$JOBDIR/state"
  exit 1
elif [ "$failures" -gt 0 ] && [ -f "$JOBDIR/STOP" ]; then
  echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"
  exit 0
elif [ "$failures" -gt 0 ]; then
  echo "FAILED ($failures of $total material(s) failed) $(date -Is)" > "$JOBDIR/state"
  exit 1
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
    echo "WARNING: slice timed out without checkpoint progress for $m; will not retry" >> "$JOBDIR/log"
    echo "failed: $m" >> "$JOBDIR/log"
    echo "FAILED (slice timeout without progress) $(date -Is)" > "$JOBDIR/state"
    exit 1
"""


def _chunked_queue_tail() -> str:
    """Common resubmit-or-finish tail shared by every self-resubmitting chunked
    queue script (scan/rebrem/reline chunked variants).

    Reports a chain-terminal ``done``/``done with N warning(s)`` state once
    every material in ``$mats`` is ``completed:``/``failed:`` in ``$JOBDIR/log``;
    otherwise honors a pending ``$JOBDIR/STOP`` sentinel, then fail-closed
    resubmits the next slice via ``_sbatch_retry_block`` and appends its SLURM
    id to ``$JOBDIR/meta``. Assumes the caller's shell scope already defines
    ``mats``, ``total``, and ``JOBDIR``.
    """
    return f"""slice_progress_after=$(progress_count) || exit 1
if [ "$slice_progress_after" -gt "$slice_progress_before" ]; then
  stalled=0
  echo 0 > "$JOBDIR/.no_progress_slices"
else
  stalled=$(cat "$JOBDIR/.no_progress_slices" 2>/dev/null || echo 0)
  stalled=$((stalled + 1))
  echo "$stalled" > "$JOBDIR/.no_progress_slices"
fi
unresolved=0
failures=0
for m in "${{mats[@]}}"; do
  grep -qx "failed: $m" "$JOBDIR/log" 2>/dev/null && {{ failures=$((failures + 1)); continue; }}
  grep -qx "completed: $m" "$JOBDIR/log" 2>/dev/null && continue
  unresolved=1
done
if [ "$unresolved" -eq 0 ]; then
  if [ "$failures" -gt 0 ]; then
    # Chain-terminal: the FAILED state is the signal; exit 0 keeps the resubmit
    # chain's own exit status out of it.
    echo "FAILED ($failures of $total material(s) failed) $(date -Is)" > "$JOBDIR/state"
    exit 0
  fi
  echo "done [$total/$total] $(date -Is)" > "$JOBDIR/state"
  exit 0
fi
[ -f "$JOBDIR/STOP" ] && {{ echo "cancelled (stop requested) $(date -Is)" > "$JOBDIR/state"; exit 0; }}
if [ "$stalled" -ge 3 ]; then
  echo "FAILED (3 slices without checkpoint progress) $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
k=$(grep -c "^slurm_job_id: " "$JOBDIR/meta" 2>/dev/null)
echo "queued slice $((k + 1)) $(date -Is)" > "$JOBDIR/state"
{_sbatch_retry_block("sbatch --parsable --nice=10000", '"$JOBDIR/run.sh"')}
if [ -z "$SID" ]; then
  echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"
  exit 1
fi
SID=${{SID%%;*}}
case "$SID" in ''|*[!0-9]*) echo "FAILED (slice resubmission) $(date -Is)" > "$JOBDIR/state"; exit 1 ;; esac
printf 'slurm_job_id: %s\\n' "$SID" >> "$JOBDIR/meta\""""


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
    no_cache=False,
    recompute=False,
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
    if no_cache and recompute:
        raise ValueError("no_cache and recompute are mutually exclusive")
    if no_cache:
        flags += " --no-cache"
    elif recompute:
        flags += " --recompute"
    if performance_profile is not None:
        flags += (
            f' --performance-profile {performance_profile} --performance-dir "$JOBDIR/performance"'
        )
        if performance_interval != 5.0:
            flags += f" --perf-interval {performance_interval:g}"
    runtime_exports = ""
    if spec_chunk is not None:
        runtime_exports += f"\nexport PYRITE_MC_SPEC_CHUNK={spec_chunk}"
    if brem_chunk is not None:
        runtime_exports += f"\nexport PYRITE_MC_BREM_CHUNK={brem_chunk}"
    runtime_exports += _forwarded_env_exports()
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    chunk_seconds = int(round(chunk_minutes * 60))
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block(once=True)}
{_tables_preflight_block()}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
{runtime_exports.lstrip()}
{_chunked_timeout_setup()}
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
  progress_before=$(progress_count) || exit 1
  timeout_warned=0
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  rm -f "$JOBDIR/.child_rc"
  (
    uv_pid=
    trap '[ -z "$uv_pid" ] || {{ kill -TERM "$uv_pid" 2>/dev/null || true; wait "$uv_pid" 2>/dev/null || true; }}; echo 143 > "$JOBDIR/.child_rc"; exit 143' TERM
    {config.shell_remote_uv()} run --no-sync python -u -m pyrite._entry.scan {config.shell_word(catalog_profile)} -m "$m"{flags} --max-minutes "$remaining_min" \
      --progress-file "$JOBDIR/progress/$m.json" --no-progress >> "$JOBDIR/log" 2>&1 &
    uv_pid=$!
    wait "$uv_pid"
    echo "$?" > "$JOBDIR/.child_rc"
  ) &
{_chunked_run_wait()}
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
{_chunked_timeout_result()}
  elif [ "$rc" -eq 143 ] && [ -f "$JOBDIR/STOP" ]; then
    echo "cancelled: scan for $m stopped by cancel request" >> "$JOBDIR/log"
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: scan failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
{_chunked_queue_tail()}
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
{_tables_preflight_block()}
echo "running zhai reproduction since $(date -Is)" > "$JOBDIR/state"
if ! {config.shell_remote_uv()} run --no-sync python -m pyrite._entry.reproduce_zhai{flags} >> "$JOBDIR/log" 2>&1
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

    One sequential ``pyrite checkpoint recompute brem`` per material (brem is
    cheap; no in-allocation parallelism needed), each writing the SAME
    per-material JSON progress record a scan does (``--progress-file``), so
    ``status``/``attach`` render the shared case-progress dashboard. The
    ``completed:``/``failed:`` log markers match the scan queue's so
    ``state._completed_materials`` drives the post-attach pull unchanged."""
    flags = _rebrem_flags(ne_brem, brem_step_eV, redo_all, fidelity, brem_start_eV, brem_stop_eV)
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block(once=True)}
{_tables_preflight_block()}
mats=({mats})
total=${{#mats[@]}}
n=0
failures=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  if ! {config.shell_remote_uv()} run --no-sync pyrite checkpoint recompute brem "$m"{flags} \
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
  echo "FAILED ($failures of $total material(s) failed) $(date -Is)" > "$JOBDIR/state"
  exit 1
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

    Mirror of ``_reline_chunked_queue_script`` for ``pyrite checkpoint recompute
    brem``: each slice resumes from checkpoint, does about ``chunk_minutes`` of
    work via that command's ``--max-minutes``, and either terminates the chain
    (all materials completed:/failed:) or self-resubmits with
    ``--nice=10000``. Exit-code
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
{_tables_preflight_block()}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
{_chunked_timeout_setup()}
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
  progress_before=$(progress_count) || exit 1
  timeout_warned=0
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  rm -f "$JOBDIR/.child_rc"
  (
    uv_pid=
    trap '[ -z "$uv_pid" ] || {{ kill -TERM "$uv_pid" 2>/dev/null || true; wait "$uv_pid" 2>/dev/null || true; }}; echo 143 > "$JOBDIR/.child_rc"; exit 143' TERM
    {config.shell_remote_uv()} run --no-sync pyrite checkpoint recompute brem "$m"{flags} --max-minutes "$remaining_min" \
      --progress-file "$JOBDIR/progress/$m.json" >> "$JOBDIR/log" 2>&1 &
    uv_pid=$!
    wait "$uv_pid"
    echo "$?" > "$JOBDIR/.child_rc"
  ) &
{_chunked_run_wait()}
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
{_chunked_timeout_result()}
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: rebrem failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
{_chunked_queue_tail()}
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
    """CXR payload for a line-only checkpoint recompute (``pyrite checkpoint
    recompute line``) in a SLURM allocation. One sequential recompute per
    material; each writes the same per-material JSON progress record a scan or
    brem recompute does, and the ``completed:``/``failed:`` markers match so
    ``state._completed_materials`` drives the post-attach pull unchanged."""
    flags = _reline_flags(line_ne, line_step_eV, redo_all, fidelity, line_start_eV, line_stop_eV)
    mats = " ".join(materials)  # safe: each token matched _SHELL_TOKEN_RE
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    return f"""JOBDIR={config.shell_word(jobdir)}
cd {config.shell_remote_dir()} || exit 1
mkdir -p "$JOBDIR/progress"
echo "started: $(date -Is)" >> "$JOBDIR/meta"
{_uv_sync_block(once=True)}
{_tables_preflight_block()}
mats=({mats})
total=${{#mats[@]}}
n=0
failures=0
for m in "${{mats[@]}}"; do
  n=$((n + 1))
  echo "running $m [$n/$total] since $(date -Is)" > "$JOBDIR/state"
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  if ! {config.shell_remote_uv()} run --no-sync pyrite checkpoint recompute line "$m"{flags} \
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
  echo "FAILED ($failures of $total material(s) failed) $(date -Is)" > "$JOBDIR/state"
  exit 1
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

    Mirror of ``_chunked_queue_script`` for ``pyrite checkpoint recompute
    line``: each slice resumes from checkpoint, does about ``chunk_minutes`` of
    work via that command's ``--max-minutes``, and either terminates the chain
    (all materials completed:/failed:) or self-resubmits with
    ``--nice=10000``. Exit-code
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
{_tables_preflight_block()}
mats=({mats})
total=${{#mats[@]}}
chunk_seconds={chunk_seconds}
{_chunked_timeout_setup()}
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
  progress_before=$(progress_count) || exit 1
  timeout_warned=0
  printf '\\n===== [%s/%s] %s  %s =====\\n' "$n" "$total" "$m" "$(date -Is)" >> "$JOBDIR/log"
  rc=0
  rm -f "$JOBDIR/.child_rc"
  (
    uv_pid=
    trap '[ -z "$uv_pid" ] || {{ kill -TERM "$uv_pid" 2>/dev/null || true; wait "$uv_pid" 2>/dev/null || true; }}; echo 143 > "$JOBDIR/.child_rc"; exit 143' TERM
    {config.shell_remote_uv()} run --no-sync pyrite checkpoint recompute line "$m"{flags} --max-minutes "$remaining_min" \
      --progress-file "$JOBDIR/progress/$m.json" >> "$JOBDIR/log" 2>&1 &
    uv_pid=$!
    wait "$uv_pid"
    echo "$?" > "$JOBDIR/.child_rc"
  ) &
{_chunked_run_wait()}
  if [ "$rc" -eq 0 ]; then
    echo "completed: $m" >> "$JOBDIR/log"
{_chunked_timeout_result()}
  elif [ "$rc" -eq 143 ] && [ -f "$JOBDIR/STOP" ]; then
    echo "cancelled: scan for $m stopped by cancel request" >> "$JOBDIR/log"
  elif [ "$rc" -ne 75 ]; then
    echo "WARNING: scan failed for $m (exit $rc); will not retry" >> "$JOBDIR/log"
    echo "warning at $m [$n/$total] $(date -Is)" > "$JOBDIR/state"
    echo "failed: $m" >> "$JOBDIR/log"
  fi
done
{_chunked_queue_tail()}
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
    no_cache: bool = False,
    recompute: bool = False,
    py_spy: bool = False,
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
            f"py_spy: {bool(py_spy)}",
            f"cpu: {bool(cpu)}",
            f"cpu_only: {bool(cpu_only)}",
            f"no_cache: {bool(no_cache)}",
            f"recompute: {bool(recompute)}",
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
