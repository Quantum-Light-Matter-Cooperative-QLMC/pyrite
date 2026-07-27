"""Job lifecycle: submit, stop, clear, reap, pull."""

import math
import subprocess
import sys
import uuid
from pathlib import Path

from .. import archive
from . import config, presentation, scripts, state, transport


def _refuse_if_busy(materials, quick):
    """Abort `start` if a live job is already producing any checkpoint this run
    would write. Two runs writing the same `<stem>.pkl` share one `<stem>.pkl.tmp`
    and race on `os.replace` -- the first rename consumes the temp, the second
    dies with FileNotFoundError (see run.py:_checkpoint_save). Comparing *stems*
    (material, or material_quick) not bare materials lets a `--quick` smoke test
    run alongside a full sweep of the same material, since they write different
    files."""
    wanted = set(scripts._stems(materials, quick))
    busy = [
        (jid, sorted(clash))
        for jid, jquick, jmats in state._live_jobs()
        if (clash := wanted.intersection(scripts._stems(jmats, jquick)))
    ]
    if busy:
        detail = "\n".join(f"  job {jid} is running -> {', '.join(s)}" for jid, s in busy)
        raise SystemExit(
            "refusing to start: a live job is already producing the same "
            "checkpoint(s), and two runs writing one <stem>.pkl race on its "
            f".tmp and crash.\n{detail}\n"
            "attach to it (cxr remote attach <jobid>) or stop it "
            "(cxr remote stop <material>) first, or run different materials."
        )


def clear_remote(materials, yes=False):
    """Delete one or more materials' accumulated checkpoints on the box: both
    ``checkpoints/<material>.pkl`` and ``checkpoints/<material>_quick.pkl`` for
    each material.  Accepts a single crystal key or a list.

    Refuses (before touching anything) if a live job or a pre-submission
    reservation protects any stem.  Without ``yes`` this is a safe dry preview:
    it prints exactly which files exist and would be deleted, then stops. With
    ``yes`` it ``rm -f``s them and reports what went."""
    if isinstance(materials, str):
        materials = [materials]
    transport._check_materials(materials)  # interpolated into a remote shell command
    label = ", ".join(materials)
    wanted = {stem for m in materials for stem in (m, f"{m}_quick")}
    busy = [
        (jid, sorted(clash))
        for jid, jquick, jmats in state._live_jobs()
        if (clash := wanted.intersection(scripts._stems(jmats, jquick)))
    ]
    if busy:
        detail = "\n".join(f"  job {jid} is producing -> {', '.join(s)}" for jid, s in busy)
        raise SystemExit(
            "refusing to clear: a live job is still producing one of these "
            f"checkpoints, and clearing it would race a running sweep.\n{detail}\n"
            "stop it (cxr remote stop <material>) first, or wait for it to finish."
        )
    stems = [stem for m in materials for stem in (m, f"{m}_quick")]
    if yes:
        outcome = transport._ssh_capture(
            scripts._clear_checkpoint_stems_command(f"clear-{scripts._new_jobid()}", stems)
        ).splitlines()
        reservations = [line.split("\t", 1)[1] for line in outcome if line.startswith("RESERVED\t")]
        if reservations:
            raise SystemExit(
                "refusing to clear: a checkpoint reservation is still active for "
                f"{', '.join(reservations)}. Wait for submission to resolve, or stop "
                "the recorded job before clearing."
            )
        existing = [line.split("\t", 1)[1] for line in outcome if line.startswith("CLEARED\t")]
        if not existing:
            print(f"(nothing to clear for {label})")
            return
        print("cleared on the box:")
        for f in existing:
            print(f"  checkpoints/{f}")
        return
    reservations = state._reservation_holders(stems)
    if reservations:
        detail = "\n".join(
            f"  reservation {owner} protects -> {stem}" for stem, owner in reservations
        )
        raise SystemExit(
            "refusing to clear: a checkpoint reservation is still active, which can "
            "belong to a submission whose SLURM outcome is not yet known.\n"
            f"{detail}\n"
            "wait for submission to resolve, or stop the recorded job before clearing."
        )
    # which stems actually exist on the box; the `|| true` keeps a missing last
    # stem's failed `[ -f ]` from becoming the loop's -- and hence ssh's -- exit
    # status, which would make _ssh_capture abort the whole clear
    stem_names = " ".join(stems)
    legacy_names = " ".join(f"{stem}.pkl" for stem in stems)
    listing = (
        f"cd {config.shell_remote_path('checkpoints')} 2>/dev/null || exit 0; "
        f": legacy-candidates {legacy_names}; "
        f"for stem in {stem_names}; do "
        '[ -f "$stem/line.pkl" ] && echo "$stem/" || true; '
        '[ -f "$stem.pkl" ] && echo "$stem.pkl" || true; done'
    )
    existing = transport._ssh_capture(listing).split()
    if not existing:
        print(f"(nothing to clear for {label})")
        return
    if not yes:
        print("would delete on the box (re-run with --yes to delete):")
        for f in existing:
            print(f"  checkpoints/{f}")
        return


def clear_all_remote(yes=False):
    """Empty the box's ``checkpoints/`` directory: delete every ``*.pkl`` file
    under it (recursively, so per-reproduction subdirs are included too).

    Refuses (before touching anything) if any live job is running or any
    checkpoint reservation is held: both signal an in-flight sweep whose output
    a blanket clear would destroy or race. Without ``yes`` this is a safe dry
    preview: it lists the files that would be deleted, then stops. With ``yes``
    it deletes them and reports the count."""
    live = state._live_jobs()
    if live:
        detail = "\n".join(
            f"  job {jid} is producing -> {', '.join(sorted(scripts._stems(jmats, jquick)))}"
            for jid, jquick, jmats in live
        )
        raise SystemExit(
            "refusing to clear --all: live job(s) are still producing checkpoints, "
            f"and clearing would race running sweeps.\n{detail}\n"
            "stop them (cxr remote stop --all) first, or wait for them to finish."
        )
    _now, reservations = state._reservation_ledger()
    if reservations:
        detail = "\n".join(
            f"  reservation {jobid} protects -> {stem}" for stem, jobid, _mtime in reservations
        )
        raise SystemExit(
            "refusing to clear --all: a checkpoint reservation is still active, which can "
            "belong to a submission whose SLURM outcome is not yet known.\n"
            f"{detail}\n"
            "wait for submission to resolve, reap orphans (cxr remote reap), or stop the "
            "recorded job before clearing."
        )
    # list first (dry preview), then delete only under --yes. `|| true` keeps a
    # missing checkpoints/ dir or a find failure from becoming ssh's exit status.
    listing = (
        f"cd {config.shell_remote_path('checkpoints')} 2>/dev/null || exit 0; "
        r'find . -type f -name "*.pkl" 2>/dev/null | sed "s|^\./||" | sort || true'
    )
    existing = transport._ssh_capture(listing).split()
    if not existing:
        print("(nothing to clear: checkpoints/ holds no .pkl files)")
        return
    if not yes:
        print(f"would delete on the box (re-run with --yes to delete) -- {len(existing)} file(s):")
        for f in existing:
            print(f"  checkpoints/{f}")
        return
    transport._ssh_capture(
        f"cd {config.shell_remote_path('checkpoints')} 2>/dev/null || exit 0; "
        r'find . -type f -name "*.pkl" -delete'
    )
    print(f"cleared on the box: {len(existing)} checkpoint file(s) under checkpoints/")


def start_queue(
    materials,
    quick=False,
    workers=None,
    no_sync=False,
    dry_run=False,
    parallel_materials=None,
    chunk_minutes=10.0,
    profile="full",
    high_energy_min_kev=None,
):
    """Submit a material queue to SLURM. Returns its job id.

    By default the queue is chunked: each ~chunk_minutes slice resumes from
    checkpoint, does bounded work via ``scan.py --max-minutes``, and either
    terminates the chain or resubmits itself with ``--nice=10000`` so other
    users of the single-GPU box get priority at every slice boundary. Pass
    ``chunk_minutes=0`` for the original monolithic allocation, which is the
    only mode that accepts ``parallel_materials``.

    ``high_energy_min_kev`` forwards ``--high-energy-min-kev`` to every
    material's remote ``cxr scan`` invocation; it is a no-op there for any
    material outside mats_to_sim.toml's ``high_energy_materials``, so one
    shared value is safe across a mixed batch.
    """
    transport._check_materials(materials)
    chunked = chunk_minutes > 0
    if chunked and parallel_materials is not None:
        raise SystemExit(
            "--parallel-materials only applies to a monolithic allocation; "
            "pass --chunk-minutes 0 to use it"
        )
    if not dry_run:
        _refuse_if_busy(materials, quick)
    jobid = scripts._new_jobid()
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    stems = scripts._stems(materials, quick, profile, high_energy_min_kev)
    if chunked:
        parallel_materials = None
        payload = scripts._chunked_queue_script(
            jobid, materials, quick, workers, chunk_minutes, profile, high_energy_min_kev
        )
        time_limit = str(max(1, math.ceil(chunk_minutes * 3)))  # minutes: hard backstop
    else:
        parallel_materials = scripts._validate_parallel_materials(
            config.DEFAULT_PARALLEL_MATERIALS if parallel_materials is None else parallel_materials
        )
        payload = scripts._queue_script(
            jobid, materials, quick, workers, parallel_materials, profile, high_energy_min_kev
        )
        time_limit = config.SLURM_TIME
    script = scripts._slurm_batch_script(
        jobid, payload, job_name=f"cxr-{jobid}", reservation_stems=stems, time_limit=time_limit
    )
    upload = scripts._write_job_script_command(
        jobdir,
        scripts._queue_metadata(
            jobid,
            materials,
            quick,
            workers,
            parallel_materials,
            chunk_minutes,
            profile,
            high_energy_min_kev,
        ),
    )
    submit = scripts._submit_slurm_command(jobid, stems, nice=chunked)

    if dry_run:
        print(f"# job {jobid}: {' '.join(materials)}{' (quick)' if quick else ''}")
        print(f"# --- ssh {config.remote_host()}: {upload} <<\n")
        print(script)
        print(f"# --- ssh {config.remote_host()}: {submit}")
        return jobid

    if not no_sync:
        transport.sync_code()
    # create the job dir and write run.sh (script piped over stdin).
    # Send as LF-only bytes: text=True on Windows translates \n->\r\n,
    # which produces a CRLF run.sh that bash silently refuses to execute.
    _stage_job_script(jobid, stems, upload, script)
    scheduler_id = _submit_staged_job(jobid, stems, nice=chunked)

    print(
        f"\nJOB {jobid} · SUBMITTED\n"
        + presentation._format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", config.remote_host()),
                ("Materials", ", ".join(materials)),
                (
                    "Mode",
                    presentation._mode_summary(
                        scripts._queue_metadata(
                            jobid,
                            materials,
                            quick,
                            workers,
                            parallel_materials,
                            chunk_minutes,
                            profile,
                        )
                    ),
                ),
                ("Attach", f"cxr remote attach {jobid}"),
                ("Status", f"cxr remote status {jobid} -vv"),
                ("Logs", f"cxr remote logs {jobid} --follow"),
                ("Pull", f"cxr remote pull {' '.join(stems)}  (after completion)"),
            ]
        )
    )
    return jobid


def start_zhai_queue(
    ne=20_000,
    ne_brem=200,
    ne_supp=200,
    tmd_azimuth=0.0,
    refresh=False,
    no_sync=False,
    dry_run=False,
):
    """Submit a Zhai-reproduction batch job to SLURM. Returns the local job id."""
    if not dry_run:
        _refuse_if_busy([config.ZHAI_STEM], False)
    jobid = scripts._new_jobid()
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    stems = [config.ZHAI_STEM]
    payload = scripts._zhai_queue_script(jobid, ne, ne_brem, ne_supp, tmd_azimuth, refresh)
    script = scripts._slurm_batch_script(
        jobid, payload, job_name=f"cxr-zhai-{jobid}", reservation_stems=stems
    )
    upload = scripts._write_job_script_command(
        jobdir, scripts._zhai_queue_metadata(jobid, ne, ne_brem, ne_supp)
    )
    submit = scripts._submit_slurm_command(jobid, stems)

    if dry_run:
        print(f"# zhai job {jobid}: ne={ne} ne_brem={ne_brem} ne_supp={ne_supp}")
        print(f"# --- ssh {config.remote_host()}: {upload} <<\n")
        print(script)
        print(f"# --- ssh {config.remote_host()}: {submit}")
        return jobid

    if not no_sync:
        transport.sync_code()
    _stage_job_script(jobid, stems, upload, script)
    scheduler_id = _submit_staged_job(jobid, stems)

    print(
        f"\nJOB {jobid} · SUBMITTED\n"
        + presentation._format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", config.remote_host()),
                ("Workload", "Zhai reproduction"),
                ("Attach", f"cxr remote attach {jobid}"),
                ("Status", f"cxr remote status {jobid} -vv"),
                ("Logs", f"cxr remote logs {jobid} --follow"),
                ("Pull", "cxr remote check --pull  (after completion)"),
            ]
        )
    )
    return jobid


def start_rebrem_queue(
    materials,
    ne_brem=None,
    brem_step_eV=None,
    redo_all=False,
    no_sync=False,
    dry_run=False,
    chunk_minutes=10.0,
    profile="full",
    brem_start_eV=None,
    brem_stop_eV=None,
):
    """Submit a brem-only checkpoint recompute (``cxr rebrem``) to SLURM.

    Reserves the same ``<material>.pkl`` stems as a sweep -- rebrem rewrites
    those checkpoints in place, so it must not race a live scan of the same
    material (and vice versa). By default the recompute is chunked: each
    ~chunk_minutes slice does bounded work via ``cxr rebrem --max-minutes`` and
    self-resubmits with ``--nice=10000`` so the single-GPU box stays shareable
    at every slice boundary. Pass ``chunk_minutes=0`` for the original
    monolithic allocation. Returns the local job id."""
    transport._check_materials(materials)
    if not dry_run:
        _refuse_if_busy(materials, False)
    jobid = scripts._new_jobid()
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    stems = list(materials)
    chunked = chunk_minutes > 0
    if chunked:
        payload = scripts._rebrem_chunked_queue_script(
            jobid,
            materials,
            ne_brem,
            brem_step_eV,
            redo_all,
            chunk_minutes,
            profile,
            brem_start_eV,
            brem_stop_eV,
        )
        time_limit = str(max(1, math.ceil(chunk_minutes * 3)))  # minutes: hard backstop
    else:
        payload = scripts._rebrem_queue_script(
            jobid,
            materials,
            ne_brem,
            brem_step_eV,
            redo_all,
            profile,
            brem_start_eV,
            brem_stop_eV,
        )
        time_limit = config.SLURM_TIME
    script = scripts._slurm_batch_script(
        jobid,
        payload,
        job_name=f"cxr-rebrem-{jobid}",
        reservation_stems=stems,
        time_limit=time_limit,
    )
    upload = scripts._write_job_script_command(
        jobdir,
        scripts._rebrem_queue_metadata(
            jobid,
            materials,
            ne_brem,
            brem_step_eV,
            redo_all,
            profile,
            brem_start_eV,
            brem_stop_eV,
        ),
    )
    submit = scripts._submit_slurm_command(jobid, stems, nice=chunked)

    if dry_run:
        print(
            f"# rebrem job {jobid}: {' '.join(materials)} "
            f"ne_brem={ne_brem} step={brem_step_eV} redo_all={redo_all}"
        )
        print(f"# --- ssh {config.remote_host()}: {upload} <<\n")
        print(script)
        print(f"# --- ssh {config.remote_host()}: {submit}")
        return jobid

    if not no_sync:
        transport.sync_code()
    _stage_job_script(jobid, stems, upload, script)
    scheduler_id = _submit_staged_job(jobid, stems, nice=chunked)

    print(
        f"\nJOB {jobid} · SUBMITTED\n"
        + presentation._format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", config.remote_host()),
                ("Materials", ", ".join(materials)),
                (
                    "Mode",
                    presentation._mode_summary(
                        scripts._rebrem_queue_metadata(
                            jobid,
                            materials,
                            ne_brem,
                            brem_step_eV,
                            redo_all,
                            profile,
                            brem_start_eV,
                            brem_stop_eV,
                        )
                    ),
                ),
                ("Attach", f"cxr remote attach {jobid}"),
                ("Status", f"cxr remote status {jobid} -vv"),
                ("Logs", f"cxr remote logs {jobid} --follow"),
                ("Pull", f"cxr remote pull {' '.join(stems)}  (after completion)"),
            ]
        )
    )
    return jobid


def start_reline_queue(
    materials,
    line_ne=None,
    line_step_eV=None,
    redo_all=False,
    no_sync=False,
    dry_run=False,
    chunk_minutes=10.0,
    profile="full",
    line_start_eV=None,
    line_stop_eV=None,
):
    """Submit a line-only checkpoint recompute (``cxr reline``) to SLURM.

    Reserves the same ``<material>.pkl`` stems as a sweep/rebrem. By default the
    recompute is chunked: each ~chunk_minutes slice does bounded work via ``cxr
    reline --max-minutes`` and self-resubmits with ``--nice=10000`` so the
    single-GPU box stays shareable at every slice boundary. Pass
    ``chunk_minutes=0`` for the original monolithic allocation. Returns the
    local job id."""
    transport._check_materials(materials)
    if not dry_run:
        _refuse_if_busy(materials, False)
    jobid = scripts._new_jobid()
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    stems = list(materials)
    chunked = chunk_minutes > 0
    if chunked:
        payload = scripts._reline_chunked_queue_script(
            jobid,
            materials,
            line_ne,
            line_step_eV,
            redo_all,
            chunk_minutes,
            profile,
            line_start_eV,
            line_stop_eV,
        )
        time_limit = str(max(1, math.ceil(chunk_minutes * 3)))  # minutes: hard backstop
    else:
        payload = scripts._reline_queue_script(
            jobid,
            materials,
            line_ne,
            line_step_eV,
            redo_all,
            profile,
            line_start_eV,
            line_stop_eV,
        )
        time_limit = config.SLURM_TIME
    script = scripts._slurm_batch_script(
        jobid,
        payload,
        job_name=f"cxr-reline-{jobid}",
        reservation_stems=stems,
        time_limit=time_limit,
    )
    upload = scripts._write_job_script_command(
        jobdir,
        scripts._reline_queue_metadata(
            jobid,
            materials,
            line_ne,
            line_step_eV,
            redo_all,
            profile,
            line_start_eV,
            line_stop_eV,
        ),
    )
    submit = scripts._submit_slurm_command(jobid, stems, nice=chunked)

    if dry_run:
        print(
            f"# reline job {jobid}: {' '.join(materials)} "
            f"line_ne={line_ne} line_step={line_step_eV} redo_all={redo_all}"
        )
        print(f"# --- ssh {config.remote_host()}: {upload} <<\n")
        print(script)
        print(f"# --- ssh {config.remote_host()}: {submit}")
        return jobid

    if not no_sync:
        transport.sync_code()
    _stage_job_script(jobid, stems, upload, script)
    scheduler_id = _submit_staged_job(jobid, stems, nice=chunked)

    print(
        f"\nJOB {jobid} · SUBMITTED\n"
        + presentation._format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", config.remote_host()),
                ("Materials", ", ".join(materials)),
                (
                    "Mode",
                    presentation._mode_summary(
                        scripts._reline_queue_metadata(
                            jobid,
                            materials,
                            line_ne,
                            line_step_eV,
                            redo_all,
                            profile,
                            line_start_eV,
                            line_stop_eV,
                        )
                    ),
                ),
                ("Attach", f"cxr remote attach {jobid}"),
                ("Status", f"cxr remote status {jobid} -vv"),
                ("Logs", f"cxr remote logs {jobid} --follow"),
                ("Pull", f"cxr remote pull {' '.join(stems)} --line-only  (after completion)"),
            ]
        )
    )
    return jobid


def _stage_job_script(jobid: str, stems: list[str], upload: str, script: str) -> None:
    """Reserve stems and upload a batch script, releasing on upload failure."""
    transport._run(
        [
            "ssh",
            "-n",
            config.remote_host(),
            scripts._reserve_checkpoint_stems_command(jobid, stems),
        ]
    )
    try:
        subprocess.run(
            ["ssh", config.remote_host(), upload],
            input=script.replace("\r\n", "\n").encode(),
            check=True,
        )
    except BaseException:
        transport._run(
            [
                "ssh",
                "-n",
                config.remote_host(),
                scripts._release_checkpoint_stems_command(jobid, stems),
            ]
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
            [
                "ssh",
                "-n",
                config.remote_host(),
                scripts._release_checkpoint_stems_command(jobid, stems),
            ]
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
    transport._run(["ssh", "-n", config.remote_host(), remote])


def stop_jobs(materials=None, all_jobs=False, *, yes=True):
    """Stop live queue jobs by material name, or every live job with ``all_jobs``.

    Each material can only be owned by one live job because start/scan refuse
    checkpoint-stem collisions, so material names are the useful user-facing
    handle and job ids stay an internal implementation detail.
    """
    if all_jobs:
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

    if not yes:
        print("would cancel remote job(s):")
        for jobid in jobids:
            print(f"  {jobid}")
        print("re-run with --yes to cancel")
        return

    for jobid in jobids:
        _stop_jobid(jobid)


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
        verb = "releasing" if yes else "would release"
        print(f"{verb} {jobid}: {len(stems)} orphan reservation(s) -> {', '.join(stems)}")
    if not yes:
        print("re-run with --yes to release them")
        return
    for jobid in sorted(orphans):
        transport._run(["ssh", "-n", config.remote_host(), scripts._reap_job_command(jobid)])
    print(f"reaped {len(orphans)} orphaned job(s)")


def _resolve_survey_stems(stems):
    """Expand bare canonical-material stems to also include any matching
    identity-qualified survey checkpoint directories on the box
    (``<material>--survey-<hash>/``), so ``cxr remote pull <material>`` finds
    a survey checkpoint without the caller needing to know its hash suffix.

    On-disk names stay hash-based -- this only discovers them, via one remote
    listing of ``checkpoints/``, matched against the existing
    ``profiles._VARIANT_STEM_RE``. Already-qualified stems (a stem that
    already fullmatches that regex) and ``_quick`` stems pass through
    unexpanded: a stem that already names an exact variant, or a quick smoke
    checkpoint, never has a survey sibling worth auto-discovering."""
    from ..profiles import _VARIANT_STEM_RE

    bare = {
        stem
        for stem in stems
        if not stem.endswith("_quick") and _VARIANT_STEM_RE.fullmatch(stem) is None
    }
    if not bare:
        return list(stems)
    names = transport._ssh_capture(scripts._list_checkpoint_dirs_command()).split()
    resolved = list(stems)
    for name in sorted(names):
        match = _VARIANT_STEM_RE.fullmatch(name)
        if match is None or match["profile"] != "survey" or match["material"] not in bare:
            continue
        if name in resolved:
            continue
        resolved.append(name)
        print(f"also pulling identity-qualified survey checkpoint -> checkpoints/{name}/")
    return resolved


def pull(
    stems,
    grid=False,
    drop_wide_brem=False,
    downcast=False,
    level9=False,
    no_sync=False,
    dataset=None,
    force=False,
    summary=None,
):
    """Fetch checkpoints/<stem>.pkl back from the box for each stem (stem =
    material, or material_quick for a --quick run).

    A bare material stem also picks up any identity-qualified survey
    checkpoint the box holds for it (``<material>--survey-<hash>/``, see
    :func:`_resolve_survey_stems`) -- skipped for a ``dataset`` merge pull
    (``--brem-only``/``--line-only``), since those require an existing local
    checkpoint to merge into and a freshly discovered stem would not have one.

    With ``grid`` and/or ``level9``, prep on the box BEFORE the transfer via
    ``cxr slim``: ``grid`` filters to just the material's current grid (plus the
    optional byte trimmers), ``level9`` recompresses at gzip level 9 instead of
    the level-6 default a live sweep writes at -- lossless, just smaller for the
    wire. One SSH session creates a box temp, streams it into the local active
    slot, and removes the temp with a remote EXIT trap. ``sync_code()`` runs
    first (unless ``no_sync``) so the box rebuilds the grid from the same
    ``config.py`` the laptop has, and has the ``--compresslevel`` flag at all
    -- closing sync drift."""
    if dataset not in (None, "brem", "line"):
        raise ValueError("dataset must be None, 'brem', or 'line'")
    stems = list(stems)
    if not stems:
        raise ValueError("stems must contain at least one checkpoint stem")
    transport._check_shell_tokens(stems)
    if dataset is None:
        stems = _resolve_survey_stems(stems)
        transport._check_shell_tokens(stems)
    dest = config.LOCAL_ROOT / "checkpoints"
    dest.mkdir(exist_ok=True)
    use_slim = True  # component directories are projected to one transfer pickle
    if use_slim and not no_sync:
        transport.sync_code()  # box must rebuild the grid from the same config.py
    failed = []
    completed = []
    failure_errors = {}
    for stem in stems:
        local = dest / stem
        try:
            from .. import _checkpoint_io, _checkpoint_store
            from ..profiles import identity_from_stem
            from ..results import merge_dataset
            from ..run import _manifest_save

            flags = ""
            if grid:
                flags += " --grid"
            if drop_wide_brem:
                flags += " --drop-wide-brem"
            if downcast:
                flags += " --downcast"
            if level9:
                flags += " --compresslevel 9"
            if dataset is not None:
                flags += f" --{dataset}-only"
            remote_tmp = f"/tmp/{stem}.{dataset or 'full'}.{uuid.uuid4().hex}.pkl"
            ckpt = config.remote_path("checkpoints", stem)
            incoming_local = dest / f".{stem}.incoming.pkl"
            remote_transfer = (
                f"T={config.shell_word(remote_tmp)}; "
                'cleanup() { rm -f -- "$T"; }; trap cleanup EXIT; '
                "trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM; "
                f"cd {config.shell_remote_dir()} && "
                f"{config.shell_remote_uv()} run --no-sync cxr slim "
                f'{config.shell_arg(ckpt)}{flags} -o "$T" 1>&2 && cat "$T"'
            )
            transport._ssh_download(remote_transfer, incoming_local)

            incoming = _checkpoint_io.load(str(incoming_local))
            incoming_local.unlink(missing_ok=True)
            if dataset is not None:
                if not _checkpoint_store.checkpoint_exists(stem, dest):
                    raise FileNotFoundError(f"no local checkpoints/{stem}/ to merge into")
                archive.archive_checkpoint(stem, force=True)
                base = _checkpoint_store.load(stem, dest)
                n_merged, n_skipped = merge_dataset(base, incoming, dataset, force=force)
                _checkpoint_store.save(stem, dest, base, components=(dataset,))
                _manifest_save(str(local), base, identity_from_stem(stem))
                print(
                    f"merged {dataset} ({n_merged} rec, skipped {n_skipped}) -> checkpoints/{stem}/"
                )
            else:
                _checkpoint_store.save(stem, dest, incoming)
                _manifest_save(str(local), incoming, identity_from_stem(stem))
                label = "+".join(filter(None, ["grid" if grid else "", "level9" if level9 else ""]))
                detail = f" ({label})" if label else ""
                print(f"pulled{detail} -> checkpoints/{stem}/")
            completed.append(stem)
        except (Exception, SystemExit) as exc:
            failed.append(stem)
            detail = str(exc) or type(exc).__name__
            failure_errors[stem] = detail
            print(
                f"warning: could not pull checkpoint {stem!r}: {detail}; continuing",
                file=sys.stderr,
            )
    if summary is not None:
        summary.update(completed=completed, failed=failed, errors=failure_errors)
    if failed:
        raise SystemExit(
            f"remote pull failed for {len(failed)} of {len(stems)} requested "
            f"checkpoint(s): {', '.join(failed)}"
        )


def pull_zhai_cache():
    """Fetch every cache file under checkpoints/zhai_reproduction/ from the
    box.

    Lists remote filenames first (like clear_remote's listing step) rather
    than `scp -r`, which double-nests the directory when the local destination
    already exists -- listing + per-file scp is unambiguous either way."""
    remote_dir = config.remote_path("checkpoints", "zhai_reproduction")
    remote_dir_word = config.shell_arg(remote_dir)
    listing = (
        f"[ -d {remote_dir_word} ] || exit 0; "
        f"find {remote_dir_word} -maxdepth 1 -type f -name '*.pkl' -printf '%f\\n'"
    )
    names = [Path(p).name for p in transport._ssh_capture(listing).split()]
    if not names:
        print("(no zhai cache files on the box -- run `cxr remote check` first)")
        return
    dest = config.LOCAL_ROOT / "checkpoints" / "zhai_reproduction"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        transport._run(["scp", config.scp_remote_path(f"{remote_dir}/{name}"), str(dest / name)])
    print(f"pulled -> checkpoints/zhai_reproduction/ ({len(names)} cache files)")
