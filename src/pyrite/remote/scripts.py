"""SLURM reservation, scheduler-state, and remote-command builders (pure, no ssh)."""

import shlex
import uuid

from .._env import GENERATED_INVOCATION_ENV
from . import _queue_scripts, config, transport

datetime = _queue_scripts.datetime
ZHAI_CACHE_SCHEMA = _queue_scripts.ZHAI_CACHE_SCHEMA
ZHAI_DETECTOR = _queue_scripts.ZHAI_DETECTOR
_LEGACY_HIGH_ENERGY_MATERIALS = _queue_scripts._LEGACY_HIGH_ENERGY_MATERIALS
_SBATCH_RETRY_ATTEMPTS = _queue_scripts._SBATCH_RETRY_ATTEMPTS
_SBATCH_RETRY_SECONDS = _queue_scripts._SBATCH_RETRY_SECONDS
_stems = _queue_scripts._stems
_material_stems = _queue_scripts._material_stems
_list_checkpoint_dirs_command = _queue_scripts._list_checkpoint_dirs_command
_validate_parallel_materials = _queue_scripts._validate_parallel_materials
_uv_sync_block = _queue_scripts._uv_sync_block
_cpu_profile_block = _queue_scripts._cpu_profile_block
_queue_script = _queue_scripts._queue_script
_sbatch_retry_block = _queue_scripts._sbatch_retry_block
_chunked_queue_script = _queue_scripts._chunked_queue_script
_zhai_flags = _queue_scripts._zhai_flags
_zhai_queue_script = _queue_scripts._zhai_queue_script
_rebrem_flags = _queue_scripts._rebrem_flags
_rebrem_queue_script = _queue_scripts._rebrem_queue_script
_rebrem_chunked_queue_script = _queue_scripts._rebrem_chunked_queue_script
_rebrem_queue_metadata = _queue_scripts._rebrem_queue_metadata
_reline_flags = _queue_scripts._reline_flags
_reline_queue_script = _queue_scripts._reline_queue_script
_reline_chunked_queue_script = _queue_scripts._reline_chunked_queue_script
_reline_queue_metadata = _queue_scripts._reline_queue_metadata
_queue_metadata = _queue_scripts._queue_metadata
_zhai_queue_metadata = _queue_scripts._zhai_queue_metadata


def _new_jobid() -> str:
    """Return a collision-resistant, shell-safe local job directory name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


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


def _job_reserved_stems_command(jobid: str) -> str:
    """Return a remote command listing the checkpoint stems a job reserved.

    One ``written <stem>`` / ``unwritten <stem>`` line per reservation owned by
    ``jobid``; ``written`` means ``checkpoints/<stem>`` exists on the box. The
    reservation set is the run's own submit-time identity list, so it is
    independent of the catalog the caller has selected locally.
    """
    transport._check_shell_tokens([jobid])
    return (
        f"R={config.shell_word(_reservation_root())}; J={config.shell_word(jobid)}; "
        f"C={config.shell_remote_path('checkpoints')}; "
        'for d in "$R"/*; do [ -d "$d" ] || continue; '
        '[ "$(cat "$d/jobid" 2>/dev/null)" = "$J" ] || continue; '
        's=$(basename "$d"); '
        'if [ -e "$C/$s" ]; then echo "written $s"; else echo "unwritten $s"; fi; done'
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


def _nvidia_prelude() -> str:
    return "module purge 2>/dev/null || true\nmodule load cuda openmpi hdf5 2>/dev/null || true"


def _amd_prelude() -> str:
    """ROCm job environment: no modules, own venv, backend pinned to ``rocm``.

    AMD nodes share ``REMOTE_DIR`` with NVIDIA nodes, and the two ``uv sync``
    extras install different CuPy builds, so AMD jobs keep a separate project
    environment instead of rebuilding the shared ``.venv`` on every switch.
    Pinning the backend makes a missing HIP stack fail instead of running on CPU.
    """
    venv = config.shell_word(config.remote_path(".venv-amd"))
    return (
        f"export UV_PROJECT_ENVIRONMENT={venv}\n"
        'export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"\n'
        "export PYRITE_MC_BACKEND=rocm"
    )


_VENDOR_PRELUDES = {"nvidia": _nvidia_prelude, "amd": _amd_prelude}
# Extra job-log header lines; NVIDIA's header predates these and stays unchanged.
_VENDOR_LOG_PROBES = {
    "amd": """  echo "gpu_vendor: amd"
  echo "backend: $PYRITE_MC_BACKEND"
  if command -v rocminfo >/dev/null 2>&1; then
    echo "gfx: $(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || echo unknown)"
  else
    echo "gfx: unknown (rocminfo not on PATH)"
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram 2>&1 || true
  fi
""",
}


def _slurm_batch_script(
    jobid: str,
    payload: str,
    *,
    job_name: str,
    reservation_stems: list[str] | None = None,
    time_limit: str = config.SLURM_TIME,
    cpus_per_task: int = config.SLURM_CPUS_PER_MATERIAL,
    mem_per_cpu: str | None = None,
    chunked: bool = False,
) -> str:
    """Wrap a CXR queue payload in the configured one-GPU SLURM target profile.

    Partition, node list, gres, and vendor come from ``remote.*`` config. Every
    node is assumed to see ``REMOTE_DIR`` on a shared filesystem: the script,
    job bookkeeping, and checkpoints all live there.
    """
    vendor = config.remote_gpu_vendor()
    if vendor not in _VENDOR_PRELUDES:
        raise ValueError(
            f"pyrite remote SLURM scripts do not yet support {vendor}; "
            "use a site-specific SLURM template from docs/guides/running-on-a-cluster.md"
        )
    partition = config.slurm_partition()
    nodelist = config.slurm_nodelist()
    nodelist_line = f"#SBATCH --nodelist={nodelist}\n" if nodelist else ""
    signal_grace = max(10, min(120, int(time_limit) * 30)) if chunked else 0
    signal_line = f"#SBATCH --signal=B:USR1@{signal_grace}\n" if chunked else ""
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
#SBATCH --partition={partition}
{nodelist_line}#SBATCH --nodes=1
#SBATCH --ntasks-per-node={config.SLURM_GPUS}
#SBATCH --cpus-per-task={cpus_per_task}
{f"#SBATCH --mem-per-cpu={mem_per_cpu}" if mem_per_cpu else ""}
#SBATCH --gres={config.slurm_gres()}
#SBATCH --time={time_limit}
{signal_line}#SBATCH --output={sbatch_jobdir}/slurm-%j.out
#SBATCH --error={sbatch_jobdir}/slurm-%j.err

set -u
{_VENDOR_PRELUDES[vendor]()}

export PYRITE_HOME={config.shell_word(config.remote_dir())}
# argv below is generated; deprecated options it carries were warned locally.
export {GENERATED_INVOCATION_ENV}=1
export PYRITE_CATALOG={config.shell_word(config.remote_catalog_path())}
export PYRITE_USER_CATALOG={config.shell_word(config.remote_user_catalog_path())}
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
  echo "partition: {partition}"
{_VENDOR_LOG_PROBES.get(vendor, "")}  echo "started: $(date -Is)"
}} >> "$JOBDIR/log"

# Wait out any in-flight `pyrite remote sync` (it holds this lock exclusively), so
# the job never starts importing a half-replaced tree. Released at once: a sync
# against a live job is policed by the code-digest guard, not by this lock.
if command -v flock >/dev/null 2>&1; then
  if ! flock -s -w {config.SYNC_LOCK_WAIT_SECONDS} {config.shell_word(config.remote_sync_lock_path())} true; then
    echo "FAILED (sync lock busy) $(date -Is)" > "$JOBDIR/state"
    exit 1
  fi
fi

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


def _write_job_script_command(jobdir, metadata):
    """Exclusively create a job directory, then receive its script on stdin.

    The ``code_*`` lines of ``<REMOTE_DIR>/.pyrite-sync`` are appended to the
    job's ``meta`` here, so every queue kind records the code identity it was
    staged against, fixed at submission time. Reading the box's own stamp -- not
    the submitting client's idea of it -- keeps ``--no-sync`` submissions honest:
    they record the code that is actually there. A checkout with no stamp yet
    (predating code stamping) records no ``code_`` lines, which ``sync`` reads as
    unverifiable rather than as a conflict.
    """
    jobdir_word = config.shell_single_word(jobdir)
    run_script_word = config.shell_single_word(f"{jobdir}/run.sh")
    metadata_word = config.shell_single_word(f"{jobdir}/meta")
    stamp_word = config.shell_single_word(config.remote_sync_stamp_path())
    return (
        f"mkdir {jobdir_word} && cat > {run_script_word} && "
        f"printf %s {shlex.quote(metadata)} > {metadata_word} && "
        f"{{ sed -n '/^code_/p' {stamp_word} 2>/dev/null >> {metadata_word} || true; }}"
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
    obsolete_stems: list[str] | None = None,
    yes: bool = False,
) -> str:
    """Reserve exact stems while pruning current and obsolete profile data."""
    obsolete_stems = obsolete_stems or []
    transport._check_shell_tokens([jobid, *stems, *obsolete_stems])
    if all_profiles and catalog_profile is not None:
        raise ValueError("all_profiles and catalog_profile are mutually exclusive")
    reserve = _reserve_checkpoint_stems_command(jobid, stems)
    release = _release_checkpoint_stems_command(jobid, stems)
    args = ["checkpoint", "gc"]
    if all_profiles:
        args.append("--all")
    elif catalog_profile is not None:
        args.extend(("--profile", catalog_profile))
    if yes:
        args.append("--yes")
    command = " ".join(config.shell_arg(arg) for arg in args)
    base = (
        f"{reserve}; "
        f"release_prune() {{ {release}; }}; trap release_prune EXIT; "
        f"cd {config.shell_remote_dir()} || exit $?; "
        f"{config.remote_runtime_env()} {config.shell_remote_uv()} run --no-sync pyrite {command}"
    )
    if not obsolete_stems:
        return base
    obsolete_words = " ".join(obsolete_stems)
    checkpoint_dir = config.shell_remote_path("checkpoints")
    if yes:
        reclaim = (
            f"cd {checkpoint_dir} 2>/dev/null || exit 0; "
            f"for stem in {obsolete_words}; do "
            '[ -d "$stem" ] || continue; rm -rf -- "$stem" || exit $?; '
            "printf 'CLEARED obsolete profile checkpoint: checkpoints/%s/\\n' \"$stem\"; done"
        )
    else:
        reclaim = (
            f"cd {checkpoint_dir} 2>/dev/null || exit 0; "
            f"for stem in {obsolete_words}; do "
            '[ -d "$stem" ] || continue; '
            "printf 'would delete obsolete profile checkpoint: checkpoints/%s/\\n' \"$stem\"; done"
        )
    return f'{base}; status=$?; [ "$status" -eq 0 ] || exit "$status"; {reclaim}'


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
