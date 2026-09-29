"""Remote script staging, submission, stop, and reservation release."""

import subprocess

from ..console import output as _cli_core
from . import config, scripts, state, transport


def _stage_job_script(jobid: str, stems: list[str], upload: str, script: str) -> None:
    """Reserve stems and upload a batch script, releasing on upload failure."""
    transport._run(
        config.ssh_argv(
            "-n",
            config.remote_host(),
            scripts._reserve_checkpoint_stems_command(jobid, stems),
        )
    )
    try:
        subprocess.run(
            config.ssh_argv(config.remote_host(), upload),
            input=script.replace("\r\n", "\n").encode(),
            check=True,
        )
    except BaseException:
        transport._run(
            config.ssh_argv(
                "-n",
                config.remote_host(),
                scripts._release_checkpoint_stems_command(jobid, stems),
            )
        )
        raise


def _submission_outcome(jobid: str) -> str:
    """Classify a submission whose SSH response was lost, conservatively."""
    transport._check_shell_tokens([jobid])
    jobdir = config.shell_remote_path(config.JOBS_SUBDIR, jobid)
    remote = (
        f"D={jobdir}; "
        '[ -d "$D" ] || { echo missing; exit 0; }; '
        'SID=$(sed -n "s/^slurm_job_id: //p" "$D/meta" 2>/dev/null | tail -1); '
        "case \"$SID\" in *[!0-9]*|'') ;; *) echo submitted; exit 0 ;; esac; "
        'STATE=$(cat "$D/state" 2>/dev/null || true); '
        'case "$STATE" in '
        '"FAILED (sbatch submission)"*) echo failed ;; '
        "queued*|running*|cancelling*) echo pending ;; "
        "*) echo unknown ;; esac"
    )
    return transport._ssh_capture(remote).strip()


def _release_if_submission_definitely_failed(jobid: str, stems: list[str]) -> None:
    """Release staging locks only after remote state proves ``sbatch`` failed."""
    try:
        definitely_failed = _submission_outcome(jobid) == "failed"
    except BaseException:
        return
    if definitely_failed:
        transport._run(
            config.ssh_argv(
                "-n",
                config.remote_host(),
                scripts._release_checkpoint_stems_command(jobid, stems),
            )
        )


def _submit_staged_job(jobid: str, stems: list[str], *, nice: bool = False) -> str:
    """Submit an uploaded script without freeing locks after an ambiguous SSH loss."""
    try:
        scheduler_id = transport._ssh_capture(
            scripts._submit_slurm_command(jobid, stems, nice=nice)
        ).strip()
    except BaseException:
        _release_if_submission_definitely_failed(jobid, stems)
        raise
    if not scheduler_id.isdigit():
        _release_if_submission_definitely_failed(jobid, stems)
        raise SystemExit(f"SLURM submission for job {jobid} returned no scheduler ID")
    return scheduler_id


def _stop_jobid(jobid):
    """Cancel one active scheduler job and record the terminal job state."""
    transport._check_shell_tokens([jobid])
    scheduler_id = state._slurm_job_id(jobid)
    if scheduler_id is None or state._slurm_state(scheduler_id) is None:
        raise SystemExit(f"job {jobid} is not an active SLURM job")
    release = scripts._release_job_reservations_command(jobid)
    # Write the STOP sentinel BEFORE scancel (spec 3b): if the cancelled slice
    # was already past its scan loop and about to resubmit, the next slice's
    # STOP check still terminates the chain instead of re-queueing it.
    remote = (
        f"D={config.shell_remote_path(config.JOBS_SUBDIR, jobid)}; "
        ': > "$D/STOP"; '
        f"scancel {scheduler_id} || exit $?; "
        f'echo "cancelling [{scheduler_id}] $(date -Is)" > "$D/state"; '
        "while :; do "
        + scripts._squeue_state_command(str(scheduler_id), retired="break")
        + '[ -n "$STATE" ] || break; sleep 1; done; '
        f"{release}; "
        f'echo "cancelled [{scheduler_id}] $(date -Is)" > "$D/state"; '
        f'echo "cancelled SLURM job {scheduler_id} for job {jobid}"'
    )
    transport._run(
        config.ssh_argv("-n", config.remote_host(), remote), label=f"Cancelling job {jobid}..."
    )


def _stop_jobids(jobids):
    """Cancel active scheduler jobs in one batched ssh session.

    ``stop --all`` / multi-material stops pay per-job ssh + scancel polling
    through :func:`_stop_jobid`, which serializes to tens of seconds on a
    batch; the batched command writes every STOP sentinel, scancels once,
    and polls squeue once for the whole set (see
    ``scripts._scancel_jobs_command``). Single-job callers (energy_grid) keep
    the granular :func:`_stop_jobid` error semantics."""
    transport._check_shell_tokens(list(jobids))
    transport._run(
        config.ssh_argv("-n", config.remote_host(), scripts._scancel_jobs_command(list(jobids))),
        label=f"Cancelling {len(jobids)} job(s)...",
    )


def stop_jobs(materials=None, all_jobs=False, *, yes=True, profile=None):
    """Stop live queue jobs by material name, by catalog profile, or every live
    job with ``all_jobs``.

    Each material can only be owned by one live job because start/scan refuse
    checkpoint-stem collisions, so material names are the useful user-facing
    handle and job ids stay an internal implementation detail. ``profile``
    selects live jobs whose recorded ``catalog_profile`` metadata matches --
    the handle for a profile-submitted batch, where listing every member
    material would be unusable.
    """
    if profile is not None:
        if all_jobs or materials:
            raise SystemExit("stop --profile does not take material names or --all")
        transport._check_shell_tokens([profile])
        live = state._live_jobs()
        profiles = state._job_profiles([jobid for jobid, _quick, _mats in live])
        jobids = sorted(jobid for jobid, _quick, _mats in live if profiles.get(jobid) == profile)
        if not jobids:
            raise SystemExit(f"no live job found for profile: {profile}")
    elif all_jobs:
        if materials:
            raise SystemExit("stop --all does not take material names")
        live = state._live_jobs()
        jobids = [jobid for jobid, _quick, _materials in live]
        if not jobids:
            print("(no live jobs to stop)")
            return
    else:
        if not materials:
            raise SystemExit("stop needs material(s), or use --all")
        transport._check_shell_tokens(materials)
        wanted = set(materials)
        live = state._live_jobs()
        matches = [
            (jobid, wanted.intersection(jmats))
            for jobid, _quick, jmats in live
            if wanted.intersection(jmats)
        ]
        found = {material for _jobid, matched in matches for material in matched}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit("no live job found for material(s): " + ", ".join(missing))
        jobids = sorted({jobid for jobid, _matched in matches})

    print("would cancel remote job(s):")
    for jobid in jobids:
        print(f"  {jobid}")
    if not _cli_core.confirm_destructive(yes, "Cancel these remote jobs?"):
        return

    _stop_jobids(jobids)


def reap_reservations(min_age_minutes=5.0, yes=False):
    """Release checkpoint reservations orphaned by hard-killed jobs.

    A reservation is a ``mkdir`` lock released by the batch job's EXIT trap. A
    hard kill (OOM, node reboot, ``kill -9``) skips the trap and strands every
    lock, wedging ``start``/``scan`` with no recovery path -- ``stop`` refuses a
    job the scheduler no longer knows. This reaps locks whose owning job has no
    SLURM allocation and whose newest lock predates the staging-race guard.

    Without ``yes`` this is a safe dry preview. With ``yes`` it releases the
    orphans and stamps each reaped job's state terminal.
    """
    min_age_seconds = max(0.0, min_age_minutes) * 60
    orphans, protected = state._orphaned_reservation_jobs(min_age_seconds)
    if not orphans and not protected:
        print("(no active checkpoint reservations on the box)")
        return
    for jobid, stems in sorted(protected.items()):
        print(f"keeping {jobid}: {len(stems)} reservation(s) held by a live or too-recent job")
    if not orphans:
        print("(no orphaned reservations to reap)")
        return
    for jobid, stems in sorted(orphans.items()):
        print(f"would release {jobid}: {len(stems)} orphan reservation(s) -> {', '.join(stems)}")
    if not _cli_core.confirm_destructive(yes, "Release these orphaned reservations?"):
        return
    for jobid in sorted(orphans):
        transport._run(
            config.ssh_argv("-n", config.remote_host(), scripts._reap_job_command(jobid)),
            label=f"Reaping orphaned job {jobid}...",
        )
    print(f"reaped {len(orphans)} orphaned job(s)")
