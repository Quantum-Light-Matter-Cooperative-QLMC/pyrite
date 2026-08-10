"""Read-only job and reservation state queries over ssh."""

from . import config, scripts, transport


def _completed_materials(jobid, materials):
    """Return materials with a successful queue-log completion marker.

    Each concurrent child appends one marker only after ``scan.py`` exits
    successfully. Reading those markers after ``attach`` keeps a
    warning-and-continue batch from being reported as though every requested
    checkpoint were available to pull.
    """
    transport._check_shell_tokens([jobid, *materials])
    jobdir = config.shell_remote_path(config.JOBS_SUBDIR, jobid)
    completed = set(
        transport._ssh_capture(
            f'D={jobdir}; sed -n "s/^completed: //p" "$D/log" 2>/dev/null'
        ).split()
    )
    return [material for material in materials if material in completed]


def _job_state(jobid):
    """First line of a job's persisted ``state`` file, or '' when absent (one ssh)."""
    transport._check_shell_tokens([jobid])
    jobdir = config.shell_remote_path(config.JOBS_SUBDIR, jobid)
    return transport._ssh_capture(f'D={jobdir}; head -n1 "$D/state" 2>/dev/null').strip()


def _live_jobs():
    """[(jobid, quick, [materials])] for jobs still reported by SLURM.

    Legacy job directories without a recorded scheduler ID are deliberately
    non-live: PID liveness is not a safe fallback for scheduler-managed work.
    Query SLURM once, then join live scheduler IDs against recorded metadata so
    accumulated job history does not cause one ``squeue`` process per job.
    """
    remote = (
        f"JOBS={config.shell_remote_path(config.JOBS_SUBDIR)}; "
        '[ -d "$JOBS" ] || exit 0; '
        "LIVE=$(squeue -h -u \"$USER\" -o '%i' 2>&1); STATUS=$?; "
        'if [ "$STATUS" -ne 0 ]; then '
        'echo "could not query SLURM jobs" >&2; printf "%s\\n" "$LIVE" >&2; '
        'exit "$STATUS"; fi; '
        'LIVE=" $(printf "%s\\n" "$LIVE" | tr "\\n" " ") "; '
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        '[ -f "$d/meta" ] || continue; SID=; q=; m=; '
        'while IFS= read -r line; do case "$line" in '
        '"slurm_job_id: "*) SID=${line#*: } ;; '
        '"quick: "*) q=${line#*: } ;; '
        '"materials: "*) m=${line#*: } ;; esac; done < "$d/meta"; '
        "case \"$SID\" in ''|*[!0-9]*) continue ;; esac; "
        'case "$LIVE" in *" $SID "*) ;; *) continue ;; esac; '
        "jobid=${d%/}; jobid=${jobid##*/}; "
        'printf "%s\\t%s\\t%s\\n" "$jobid" "$q" "$m"; done'
    )
    jobs = []
    for line in transport._ssh_capture(remote).splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        jobid, q, mats = parts[0].strip(), parts[1].strip(), parts[2].split()
        jobs.append((jobid, q == "True", mats))
    return jobs


def _reservation_holders(stems: list[str]) -> list[tuple[str, str]]:
    """Return checkpoint reservations that must block destructive operations.

    A reservation is acquired before ``sbatch``.  It is therefore authoritative
    for avoiding checkpoint deletion even if submission was ambiguous and no
    scheduler ID was ever recorded in the job metadata.
    """
    transport._check_shell_tokens(stems)
    remote = (
        f'R={config.shell_word(scripts._reservation_root())}; [ -d "$R" ] || exit 0; '
        f"for stem in {' '.join(stems)}; do "
        '[ -d "$R/$stem" ] || continue; '
        'OWNER=$(cat "$R/$stem/jobid" 2>/dev/null) || OWNER="unknown"; '
        'printf "%s\\t%s\\n" "$stem" "$OWNER"; done'
    )
    holders = []
    for line in transport._ssh_capture(remote).splitlines():
        stem, separator, owner = line.partition("\t")
        if separator and stem in stems:
            holders.append((stem, owner or "unknown"))
    return holders


def _slurm_job_id(jobid: str) -> str | None:
    """Return a queue's recorded numeric scheduler ID, if it has one."""
    transport._check_shell_tokens([jobid])
    jobdir = config.shell_remote_path(config.JOBS_SUBDIR, jobid)
    scheduler_id = transport._ssh_capture(
        f'D={jobdir}; sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1'
    ).strip()
    return scheduler_id if scheduler_id.isdigit() else None


def _slurm_state(slurm_job_id: str) -> str | None:
    """Return the live SLURM state, or ``None`` after it leaves ``squeue``."""
    if not slurm_job_id.isdigit():
        return None
    state = transport._ssh_capture(
        scripts._squeue_state_command(slurm_job_id, retired="exit 0") + 'printf "%s\\n" "$STATE"'
    ).strip()
    return state.splitlines()[0] if state else None


def _job_state(jobid: str) -> str:
    """Return a job's persisted terminal/progress state without inferring liveness."""
    transport._check_shell_tokens([jobid])
    state_path = config.shell_remote_path(config.JOBS_SUBDIR, jobid, "state")
    return transport._ssh_capture(f"cat {state_path} 2>/dev/null").strip()


def _job_metadata(jobid: str) -> str:
    """Return the persisted queue metadata for attach-mode selection."""
    transport._check_shell_tokens([jobid])
    jobdir = config.shell_remote_path(config.JOBS_SUBDIR, jobid)
    return transport._ssh_capture(
        f'D={jobdir}; [ -d "$D" ] || {{ echo "no such job: {jobid}" >&2; exit 1; }}; '
        'cat "$D/meta" 2>/dev/null'
    )


def _job_profiles(jobids: list[str]) -> dict[str, str]:
    """Return ``{jobid: catalog_profile}`` for the given jobs in one ssh round trip.

    Jobs whose metadata predates the ``catalog_profile:`` line (or that ran the
    default) report ``standard``."""
    if not jobids:
        return {}
    transport._check_shell_tokens(jobids)
    remote = (
        f"JOBS={config.shell_remote_path(config.JOBS_SUBDIR)}; "
        f"for j in {' '.join(jobids)}; do "
        'p=$(sed -n "s/^catalog_profile: //p" "$JOBS/$j/meta" 2>/dev/null | tail -1); '
        'printf "%s\\t%s\\n" "$j" "${p:-standard}"; done'
    )
    profiles = {}
    for line in transport._ssh_capture(remote).splitlines():
        jobid, separator, profile = line.partition("\t")
        if separator:
            profiles[jobid.strip()] = profile.strip()
    return profiles


def _profile_jobdirs(profile: str) -> set[str]:
    """Return existing job dirs named ``profile`` or ``profile-N`` (one ssh).

    Profile submits name their job after the profile; a re-submission after the
    previous run finished still needs a fresh directory, so the caller suffixes
    the first free ``-N``."""
    transport._check_shell_tokens([profile])
    remote = (
        f"JOBS={config.shell_remote_path(config.JOBS_SUBDIR)}; "
        f'for d in "$JOBS/{profile}" "$JOBS/{profile}"-*/; do [ -d "$d" ] || continue; '
        'basename "$d"; done'
    )
    return set(transport._ssh_capture(remote).split())


def _job_succeeded(jobid: str) -> bool:
    """Whether the batch script recorded a successful terminal state."""
    return _job_state(jobid).startswith("done")


def _reservation_ledger() -> tuple[int, list[tuple[str, str, int]]]:
    """Return ``(now_epoch, [(stem, jobid, mtime_epoch)])`` for every held reservation.

    One ``ssh`` round trip enumerates the whole reservation root and stamps the
    box's own clock, so reservation age is measured against the box -- never the
    caller -- keeping the staging-race guard immune to client/host clock skew.
    """
    remote = (
        f'R={config.shell_word(scripts._reservation_root())}; [ -d "$R" ] || exit 0; '
        'printf "NOW\\t%s\\n" "$(date +%s)"; '
        'for d in "$R"/*/; do [ -d "$d" ] || continue; '
        's=$(basename "$d"); '
        'j=$(cat "$d/jobid" 2>/dev/null) || j="unknown"; '
        'm=$(stat -c %Y "$d" 2>/dev/null) || m=0; '
        'printf "%s\\t%s\\t%s\\n" "$s" "$j" "$m"; done'
    )
    now = 0
    rows: list[tuple[str, str, int]] = []
    for line in transport._ssh_capture(remote).splitlines():
        parts = line.split("\t")
        if parts[0] == "NOW" and len(parts) == 2 and parts[1].isdigit():
            now = int(parts[1])
        elif len(parts) == 3:
            stem, jobid, mtime = parts
            rows.append(
                (stem.strip(), jobid.strip() or "unknown", int(mtime) if mtime.isdigit() else 0)
            )
    return now, rows


def _is_reapable(live_state: str | None, age_seconds: int, min_age_seconds: float) -> bool:
    """Decide whether a reservation-holding job is a safe reap target.

    ``live_state`` is the job's current SLURM state (e.g. ``"RUNNING"``), or
    ``None`` when the scheduler has no allocation recorded for it. ``age_seconds``
    is how long the job's newest reservation has sat untouched on the box;
    ``min_age_seconds`` is the staging-race guard.
    """
    # Reap only when both hold: no live SLURM allocation (dead), and the newest
    # lock is older than the staging-race guard (not a job mid-submit).
    return live_state is None and age_seconds >= min_age_seconds


def _orphaned_reservation_jobs(
    min_age_seconds: float,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Partition reservation holders into reapable orphans and protected jobs.

    Returns ``(orphans, protected)``, each mapping ``jobid -> sorted[stem]``.
    Liveness reuses the same recorded-scheduler-ID + ``squeue`` signal as
    ``_live_jobs``; PID guessing is never a safe fallback for scheduler work.
    """
    now, rows = _reservation_ledger()
    by_job: dict[str, list[tuple[str, int]]] = {}
    for stem, jobid, mtime in rows:
        by_job.setdefault(jobid, []).append((stem, mtime))
    orphans: dict[str, list[str]] = {}
    protected: dict[str, list[str]] = {}
    for jobid, entries in by_job.items():
        stems = sorted(stem for stem, _m in entries)
        scheduler_id = _slurm_job_id(jobid) if jobid != "unknown" else None
        live_state = _slurm_state(scheduler_id) if scheduler_id else None
        newest = max(mtime for _s, mtime in entries)
        age = now - newest if now else 0
        if _is_reapable(live_state, age, min_age_seconds):
            orphans[jobid] = stems
        else:
            protected[jobid] = stems
    return orphans, protected


def _latest_jobid():
    """The most recent job id on the box (job dirs are timestamp-named), or None."""
    out = transport._ssh_capture(
        f"JOBS={config.shell_remote_path(config.JOBS_SUBDIR)}; "
        f"{scripts._recorded_job_dirs_command()} | tail -1"
    ).strip()
    return out or None
