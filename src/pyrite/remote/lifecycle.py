"""Job lifecycle: submit, stop, clear, reap, pull."""

import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ..checkpoints import archive
from ..cli import _core as _cli_core
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
            "monitor it (pyrite remote status <jobid> --attach) or stop it "
            "(pyrite remote stop <material>) first, or run different materials."
        )


def clear_remote(materials, yes=False, catalog_profile="standard"):
    """Delete one or more materials' accumulated checkpoints on the box: both
    ``checkpoints/<material>/`` and ``checkpoints/<material>_quick/`` for
    each standard-profile material, or current full/survey identity stems for
    ``catalog_profile``. Accepts a single crystal key or a list.

    Refuses (before touching anything) if a live job or a pre-submission
    reservation protects any stem.  Without ``yes`` this is a safe dry preview:
    it prints exactly which files exist and would be deleted, then stops. With
    ``yes`` it ``rm -f``s them and reports what went."""
    if isinstance(materials, str):
        materials = [materials]
    transport._check_materials(materials)  # interpolated into a remote shell command
    label = f"profile={catalog_profile}" if catalog_profile != "standard" else ", ".join(materials)
    if catalog_profile != "standard":
        stems = [
            stem
            for fidelity in ("full", "survey")
            for stem in scripts._stems(
                materials,
                False,
                fidelity,
                catalog_profile=catalog_profile,
            )
        ]
    else:
        stems = [stem for m in materials for stem in (m, f"{m}_quick")]
    wanted = set(stems)
    live_jobs = state._live_jobs()
    live_profiles = (
        state._job_profiles([jobid for jobid, _quick, _materials in live_jobs])
        if catalog_profile != "standard" and live_jobs
        else {}
    )
    if catalog_profile != "standard":
        busy = [
            (jid, sorted(set(materials).intersection(jmats)))
            for jid, _jquick, jmats in live_jobs
            if live_profiles.get(jid) == catalog_profile and set(materials).intersection(jmats)
        ]
    else:
        busy = [
            (jid, sorted(clash))
            for jid, jquick, jmats in live_jobs
            if (clash := wanted.intersection(scripts._stems(jmats, jquick)))
        ]
    if busy:
        detail = "\n".join(f"  job {jid} is producing -> {', '.join(s)}" for jid, s in busy)
        raise SystemExit(
            "refusing to clear: a live job is still producing one of these "
            f"checkpoints, and clearing it would race a running sweep.\n{detail}\n"
            "stop it (pyrite remote stop <material>) first, or wait for it to finish."
        )
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
    print("would delete on the box:")
    for f in existing:
        print(f"  checkpoints/{f}")
    if not _cli_core.confirm_destructive(yes, "Delete these remote checkpoints?"):
        return
    return clear_remote(materials, yes=True, catalog_profile=catalog_profile)


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
            "stop them (pyrite remote stop --all) first, or wait for them to finish."
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
            "wait for submission to resolve, reap orphans (pyrite remote reap), or stop the "
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
    print(f"would delete on the box -- {len(existing)} file(s):")
    for f in existing:
        print(f"  checkpoints/{f}")
    if not _cli_core.confirm_destructive(yes, "Delete all remote checkpoint files?"):
        return
    transport._ssh_capture(
        f"cd {config.shell_remote_path('checkpoints')} 2>/dev/null || exit 0; "
        r'find . -type f -name "*.pkl" -delete'
    )
    print(f"cleared on the box: {len(existing)} checkpoint file(s) under checkpoints/")


def prune_remote(
    *,
    all_profiles: bool = False,
    catalog_profile: str | None = None,
    yes: bool = False,
):
    """Preview or prune stale records on box under exact stem reservations."""
    from ..checkpoints.checkpoint_cleanup import _targets

    targets = _targets(all_profiles, catalog_profile)
    stems = [target.stem for target in targets]
    live = state._live_jobs()
    if live:
        detail = "\n".join(
            f"  job {jobid} is producing -> {', '.join(sorted(scripts._stems(materials, quick)))}"
            for jobid, quick, materials in live
        )
        raise SystemExit(
            "refusing remote prune: live job(s) are producing checkpoints.\n"
            f"{detail}\n"
            "wait for completion, or stop them with pyrite remote stop --all."
        )
    reservations = state._reservation_holders(stems)
    if reservations:
        detail = "\n".join(
            f"  reservation {owner} protects -> {stem}" for stem, owner in reservations
        )
        raise SystemExit(
            "refusing remote prune: checkpoint reservation still active.\n"
            f"{detail}\n"
            "wait for submission to resolve, reap orphans, or stop its recorded job."
        )
    output = transport._ssh_capture(
        scripts._prune_checkpoint_stems_command(
            f"prune-{scripts._new_jobid()}",
            stems,
            all_profiles=all_profiles,
            catalog_profile=catalog_profile,
            yes=yes,
        )
    )
    if output:
        print(output)
    if (
        not yes
        and "would prune" in output.lower()
        and _cli_core.confirm_destructive(False, "Delete these stale remote checkpoint records?")
    ):
        return prune_remote(
            all_profiles=all_profiles,
            catalog_profile=catalog_profile,
            yes=True,
        )


def prune_job_dirs(*, profile=None, all_jobs=False, yes=False):
    """Delete terminal (done/FAILED/cancelled) job directories on the box.

    Preview unless ``yes``. Selection is exactly one of ``profile`` (its
    ``NAME``/``NAME-N`` family) or ``all_jobs`` (every job directory). A live
    chain is never removed -- the remote command skips any directory whose
    latest recorded SLURM id is still in squeue -- so this is safe to run while
    other jobs (including a fresh submission of the same profile) are live.
    """
    if all_jobs == (profile is not None):
        raise SystemExit("prune-jobs needs exactly one of --profile NAME or --all")
    if profile is not None:
        transport._check_shell_tokens([profile])
    records = [
        line.split("\t")
        for line in transport._ssh_capture(
            scripts._prune_job_dirs_command(profile=profile, all_jobs=all_jobs, yes=yes)
        ).splitlines()
        if "\t" in line
    ]
    scope = f"profile {profile!r}" if profile is not None else "all profiles"
    live_kept = sorted(r[1] for r in records if r[0] == "KEPT" and r[2] == "live")
    if yes:
        pruned = [r[1] for r in records if r[0] == "PRUNED"]
        if pruned:
            print("pruned job directories on the box:")
            for jid in pruned:
                print(f"  jobs/{jid}")
        else:
            print(f"(no terminal job directories to prune for {scope})")
    else:
        candidates = [(r[1], r[2]) for r in records if r[0] == "WOULD-PRUNE"]
        if candidates:
            print("would delete on the box:")
            for jid, st in candidates:
                print(f"  jobs/{jid}  [{st}]")
            if _cli_core.confirm_destructive(False, "Delete these terminal job directories?"):
                return prune_job_dirs(profile=profile, all_jobs=all_jobs, yes=True)
        else:
            print(f"(no terminal job directories to prune for {scope})")
    if live_kept:
        print(f"kept {len(live_kept)} live job(s): {', '.join(live_kept)}")


def _refuse_if_profile_live(catalog_profile):
    """Refuse a second live job under one catalog profile.

    Profile-named jobs make the profile the user-facing handle (``attach
    NAME``, ``stop --profile NAME``, ``pull --profile NAME``); two live jobs
    sharing it would make every one of those handles ambiguous."""
    live = state._live_jobs()
    if not live:
        return
    profiles = state._job_profiles([jobid for jobid, _quick, _mats in live])
    clash = sorted(jobid for jobid, _quick, _mats in live if profiles.get(jobid) == catalog_profile)
    if clash:
        raise SystemExit(
            f"refusing to submit: profile {catalog_profile!r} already has a live job "
            f"({', '.join(clash)}); monitor it "
            f"(pyrite remote status {clash[0]} --attach) or stop it "
            f"(pyrite remote stop --profile {catalog_profile}) first."
        )


def _profile_jobid(catalog_profile):
    """Job id for a profile submit: the profile name itself, or the first free
    ``NAME-N`` when a finished run already holds the bare name (a live clash is
    refused earlier by :func:`_refuse_if_profile_live`)."""
    existing = state._profile_jobdirs(catalog_profile)
    if catalog_profile not in existing:
        return catalog_profile
    suffix = 2
    while f"{catalog_profile}-{suffix}" in existing:
        suffix += 1
    return f"{catalog_profile}-{suffix}"


def start_queue(
    materials,
    quick=False,
    workers=None,
    no_sync=False,
    dry_run=False,
    parallel_materials=None,
    chunk_minutes=10.0,
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
    """Submit a material queue to SLURM. Returns its job id.

    By default the queue is chunked: each ~chunk_minutes slice resumes from
    checkpoint, does bounded work via ``scan.py --max-minutes``, and either
    terminates the chain or resubmits itself with ``--nice=10000`` so other
    users of the single-GPU box get priority at every slice boundary. Pass
    ``chunk_minutes=0`` for the original monolithic allocation, which is the
    only mode that accepts ``parallel_materials``.

    ``catalog_profile`` is forwarded positionally to the synchronized run
    entry point, which re-validates it against the remote catalog. A
    non-standard profile also names the job (``sub_100keV``, then
    ``sub_100keV-2`` once a finished run holds the bare name) and refuses to
    submit while another job under the same profile is live.

    ``high_energy_min_kev`` remains an internal metadata field for older job
    records; the profile-first CLI always leaves it unset.

    ``performance_repetitions > 1`` runs uncached sessions against isolated
    job-local checkpoint roots. It is intentionally monolithic: repetitions
    are experiment samples, not resumable production checkpoints.

    ``nsys`` wraps one uncached single-material session with Nsight Systems and
    stores its CUDA/NVTX trace beside the performance NDJSON. ``cpu`` appends a
    bounded serial cProfile phase; ``cpu_only`` runs that phase alone.
    """
    transport._check_materials(materials)
    transport._check_shell_tokens(
        [catalog_profile, *([performance_profile] if performance_profile is not None else [])]
    )
    chunked = chunk_minutes > 0
    if (
        not isinstance(performance_repetitions, int)
        or isinstance(performance_repetitions, bool)
        or not 1 <= performance_repetitions <= 20
    ):
        raise SystemExit("performance repetitions must be an integer from 1 to 20")
    if performance_interval <= 0:
        raise SystemExit("performance interval must be greater than zero")
    if performance_profile is None and (
        performance_repetitions != 1
        or performance_interval != 5.0
        or spec_chunk is not None
        or brem_chunk is not None
        or nsys
        or cpu
        or cpu_only
    ):
        raise SystemExit(
            "performance repetitions, interval, chunk pins, nsys, and CPU profiling "
            "require a performance profile"
        )
    if cpu and cpu_only:
        raise SystemExit("cpu and cpu-only modes are mutually exclusive")
    if cpu_only and nsys:
        raise SystemExit("cpu-only mode cannot be combined with nsys")
    if cpu_only and performance_repetitions != 1:
        raise SystemExit("cpu-only mode cannot use performance repetitions")
    if cpu_only and performance_interval != 5.0:
        raise SystemExit("cpu-only mode cannot set the performance sampling interval")
    if cpu_only and (spec_chunk is not None or brem_chunk is not None):
        raise SystemExit("cpu-only mode cannot set GPU chunk pins")
    if performance_repetitions > 1 and chunked:
        raise SystemExit("performance repetitions require a monolithic allocation")
    if performance_repetitions > 1 and parallel_materials not in (None, 1):
        raise SystemExit("performance repetitions require one material process per GPU")
    if nsys and chunked:
        raise SystemExit("nsys requires a monolithic allocation")
    if nsys and performance_repetitions != 1:
        raise SystemExit("nsys requires exactly one performance repetition")
    if nsys and parallel_materials not in (None, 1):
        raise SystemExit("nsys requires one material process per GPU")
    if nsys and len(materials) != 1:
        raise SystemExit("nsys requires exactly one material")
    if (cpu or cpu_only) and chunked:
        raise SystemExit("CPU profiling requires a monolithic allocation")
    if chunked and parallel_materials is not None:
        raise SystemExit(
            "--parallel-materials only applies to a monolithic allocation; "
            "pass --chunk-minutes 0 to use it"
        )
    if not dry_run and not cpu_only:
        _refuse_if_busy(materials, quick)
    if catalog_profile != "standard":
        # a profile submit is named after the profile: readable, and the same
        # handle stop/pull --profile already take. One live job per profile.
        if not dry_run:
            _refuse_if_profile_live(catalog_profile)
            jobid = _profile_jobid(catalog_profile)
        else:
            jobid = catalog_profile  # preview only; no remote existence probe
    else:
        jobid = scripts._new_jobid()
    jobdir = config.remote_path(config.JOBS_SUBDIR, jobid)
    stems = (
        []
        if cpu_only
        else scripts._stems(materials, quick, fidelity, high_energy_min_kev, catalog_profile)
    )
    if chunked:
        parallel_materials = None
        payload = scripts._chunked_queue_script(
            jobid,
            materials,
            quick,
            workers,
            chunk_minutes,
            fidelity,
            high_energy_min_kev,
            catalog_profile,
            performance_profile,
            performance_interval,
            spec_chunk,
            brem_chunk,
        )
        time_limit = str(max(1, math.ceil(chunk_minutes * 3)))  # minutes: hard backstop
    else:
        parallel_materials = scripts._validate_parallel_materials(
            config.DEFAULT_PARALLEL_MATERIALS if parallel_materials is None else parallel_materials
        )
        payload = scripts._queue_script(
            jobid,
            materials,
            quick,
            workers,
            parallel_materials,
            fidelity,
            high_energy_min_kev,
            catalog_profile,
            performance_profile,
            performance_repetitions,
            performance_interval,
            spec_chunk,
            brem_chunk,
            nsys,
            cpu,
            cpu_only,
        )
        time_limit = config.SLURM_TIME
    workers_per_material = config.SLURM_CPUS_PER_MATERIAL if workers is None else max(1, workers)
    cpus_per_task = workers_per_material * (parallel_materials or 1)
    script = scripts._slurm_batch_script(
        jobid,
        payload,
        job_name=f"cxr-{jobid}",
        reservation_stems=stems,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
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
            fidelity,
            high_energy_min_kev,
            catalog_profile,
            performance_profile,
            performance_repetitions,
            performance_interval,
            spec_chunk,
            brem_chunk,
            nsys,
            cpu,
            cpu_only,
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

    # A profile submit pulls by profile, not by a wall of hash-qualified stems
    # nobody can type; pull --profile resolves each member's MATERIAL@PROFILE
    # checkpoint remotely.
    pull_hint = (
        f"pyrite remote pull --profile {catalog_profile}"
        if catalog_profile != "standard"
        else f"pyrite remote pull {' '.join(stems)}"
    )
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
                            fidelity,
                            high_energy_min_kev,
                            catalog_profile,
                            performance_profile,
                            performance_repetitions,
                            performance_interval,
                            spec_chunk,
                            brem_chunk,
                            nsys,
                            cpu,
                            cpu_only,
                        )
                    ),
                ),
                ("Monitor", f"pyrite remote status {jobid} --attach"),
                ("Status", f"pyrite remote status {jobid} -vv"),
                ("Logs", f"pyrite remote logs {jobid} --follow"),
                *(
                    [
                        (
                            "Performance",
                            f"pyrite remote performance pull {performance_profile}",
                        )
                    ]
                    if performance_profile is not None
                    else []
                ),
                *(
                    [
                        (
                            "Checkpoints",
                            "job-local profiling artifacts; no automatic checkpoint pull",
                        )
                    ]
                    if performance_repetitions > 1 or nsys or cpu or cpu_only
                    else [("Pull", f"{pull_hint}  (after completion)")]
                ),
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
                ("Monitor", f"pyrite remote status {jobid} --attach"),
                ("Status", f"pyrite remote status {jobid} -vv"),
                ("Logs", f"pyrite remote logs {jobid} --follow"),
                ("Pull", "pyrite remote pull --preset zhai  (after completion)"),
            ]
        )
    )
    return jobid


def pull_performance_profile(profile: str) -> list[Path]:
    """Pull performance NDJSON plus any Nsight report artifacts."""
    transport._check_shell_tokens([profile])
    jobs = config.remote_path(config.JOBS_SUBDIR)
    listing = transport._ssh_capture(
        f"JOBS={config.shell_word(jobs)}; "
        '[ -d "$JOBS" ] || exit 0; '
        'for d in "$JOBS"/*/; do [ -d "$d" ] || continue; '
        f'p="$d/performance/{profile}"; [ -d "$p" ] || continue; '
        'find "$p" -maxdepth 1 -type f '
        "\\( -name '*.ndjson' -o -name '*.nsys-rep' -o -name '*.sqlite' "
        "-o -name '*.nsys-stats.txt' -o -name '*.cpu.prof' -o -name '*.cpu.txt' \\) "
        "-printf '%f\\n' | while IFS= read -r f; do "
        'printf "%s\\t%s\\n" "$(basename "$d")" "$f"; done; done'
    )
    artifacts = []
    suffixes = (".nsys-stats.txt", ".cpu.prof", ".cpu.txt", ".nsys-rep", ".ndjson", ".sqlite")
    for line in listing.splitlines():
        jobid, separator, filename = line.partition("\t")
        suffix = next((item for item in suffixes if filename.endswith(item)), "")
        material = filename[: -len(suffix)] if suffix else ""
        if (
            not separator
            or not suffix
            or presentation._SHELL_TOKEN_RE.fullmatch(jobid) is None
            or presentation._SHELL_TOKEN_RE.fullmatch(material) is None
        ):
            continue
        artifacts.append((jobid, filename))
    if not artifacts:
        raise SystemExit(f"no remote performance artifacts found for profile {profile!r}")

    local_root = config.LOCAL_ROOT / "performance-profiles" / profile
    pulled = []
    for jobid, filename in artifacts:
        destination = local_root / jobid / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        remote = config.remote_path(
            config.JOBS_SUBDIR,
            jobid,
            "performance",
            profile,
            filename,
        )
        try:
            transport._ssh_download(f"cat {config.shell_arg(remote)}", temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        pulled.append(destination)
        print(f"pulled {jobid}/{filename} -> {destination}")
    return pulled


def remote_performance_inventory() -> list[tuple[str, str, int, int]]:
    """Return ``(job, profile, files, bytes)`` for remote performance directories."""
    jobs = config.remote_path(config.JOBS_SUBDIR)
    output = transport._ssh_capture(
        f"JOBS={config.shell_word(jobs)}; "
        '[ -d "$JOBS" ] || exit 0; '
        'for p in "$JOBS"/*/performance/*/; do [ -d "$p" ] || continue; '
        'job=$(basename "$(dirname "$(dirname "$p")")"); profile=$(basename "$p"); '
        'set -- $(find "$p" -maxdepth 1 -type f -printf "%s\\n" | '
        "awk '{n += 1; b += $1} END {print n+0, b+0}'); "
        'printf "%s\\t%s\\t%s\\t%s\\n" "$job" "$profile" "$1" "$2"; done'
    )
    inventory = []
    for line in output.splitlines():
        fields = line.split("\t")
        if (
            len(fields) != 4
            or presentation._SHELL_TOKEN_RE.fullmatch(fields[0]) is None
            or presentation._SHELL_TOKEN_RE.fullmatch(fields[1]) is None
        ):
            raise SystemExit("refusing performance operation: remote inventory was malformed")
        try:
            files, size = int(fields[2]), int(fields[3])
        except ValueError:
            raise SystemExit(
                "refusing performance operation: remote inventory was malformed"
            ) from None
        if files < 0 or size < 0:
            raise SystemExit("refusing performance operation: remote inventory was malformed")
        inventory.append((fields[0], fields[1], files, size))
    return sorted(inventory)


def list_remote_performance() -> list[tuple[str, str, int, int]]:
    """Print remote performance artifact inventory."""
    inventory = remote_performance_inventory()
    if not inventory:
        print("(no remote performance profiles)")
        return []
    for jobid, profile, files, size in inventory:
        print(f"{profile}/{jobid}: {files} artifact(s), {size} bytes")
    return inventory


def _performance_job_states(jobids: set[str]) -> dict[str, str]:
    """Read selected job states; only explicit terminal records are prune-safe."""
    return {jobid: state._job_state(jobid) for jobid in sorted(jobids)}


def prune_remote_performance(profiles=None, *, all_profiles=False, yes=False):
    """Preview/delete selected terminal-job performance directories."""
    profiles = list(profiles or [])
    transport._check_shell_tokens(profiles)
    before_live = {jobid for jobid, _quick, _materials in state._live_jobs()}
    inventory = remote_performance_inventory()
    selected_profiles = {profile for _job, profile, _files, _size in inventory}
    if not all_profiles:
        selected_profiles.intersection_update(profiles)
        missing = sorted(set(profiles) - selected_profiles)
        if missing:
            raise SystemExit(f"no remote performance profile: {', '.join(missing)}")
    selected = [row for row in inventory if row[1] in selected_profiles]
    selected_jobs = {jobid for jobid, _profile, _files, _size in selected}
    busy = sorted(selected_jobs & before_live)
    if busy:
        raise SystemExit(
            "refusing performance prune: selected artifacts belong to live job(s): "
            + ", ".join(busy)
        )
    before_states = _performance_job_states(selected_jobs)
    incomplete = sorted(
        jobid
        for jobid, job_state in before_states.items()
        if not job_state.startswith(("done", "FAILED", "cancelled"))
    )
    if incomplete:
        raise SystemExit(
            "refusing performance prune: selected artifacts belong to "
            "non-terminal job(s): " + ", ".join(incomplete)
        )
    if not selected:
        print("(nothing to prune)")
        return
    print("would delete remote performance profiles:")
    for jobid, profile, files, size in selected:
        print(f"  {profile}/{jobid} ({files} artifact(s), {size} bytes)")
    if not _cli_core.confirm_destructive(yes, "Delete these remote performance profiles?"):
        return

    after_live = {jobid for jobid, _quick, _materials in state._live_jobs()}
    after_states = _performance_job_states(selected_jobs)
    current = remote_performance_inventory()
    if after_live != before_live or after_states != before_states or current != inventory:
        raise SystemExit(
            "refusing performance prune: remote job state or artifact inventory changed"
        )
    targets = [
        config.remote_path(config.JOBS_SUBDIR, jobid, "performance", profile)
        for jobid, profile, _files, _size in selected
    ]
    command = " ".join(config.shell_arg(target) for target in targets)
    transport._ssh_capture(f'for target in {command}; do rm -rf -- "$target"; done')
    print(f"deleted {len(selected)} remote performance profile path(s)")


def start_rebrem_queue(
    materials,
    ne_brem=None,
    brem_step_eV=None,
    redo_all=False,
    no_sync=False,
    dry_run=False,
    chunk_minutes=10.0,
    fidelity="full",
    brem_start_eV=None,
    brem_stop_eV=None,
):
    """Submit a brem-only checkpoint recompute (``pyrite rebrem``) to SLURM.

    Reserves the same ``<material>.pkl`` stems as a sweep -- rebrem rewrites
    those checkpoints in place, so it must not race a live scan of the same
    material (and vice versa). By default the recompute is chunked: each
    ~chunk_minutes slice does bounded work via ``pyrite rebrem --max-minutes`` and
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
            fidelity,
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
            fidelity,
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
            fidelity,
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
                            fidelity,
                            brem_start_eV,
                            brem_stop_eV,
                        )
                    ),
                ),
                ("Monitor", f"pyrite remote status {jobid} --attach"),
                ("Status", f"pyrite remote status {jobid} -vv"),
                ("Logs", f"pyrite remote logs {jobid} --follow"),
                ("Pull", f"pyrite remote pull {' '.join(stems)}  (after completion)"),
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
    fidelity="full",
    line_start_eV=None,
    line_stop_eV=None,
):
    """Submit a line-only checkpoint recompute (``pyrite reline``) to SLURM.

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
            fidelity,
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
            fidelity,
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
            fidelity,
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
                            fidelity,
                            line_start_eV,
                            line_stop_eV,
                        )
                    ),
                ),
                ("Monitor", f"pyrite remote status {jobid} --attach"),
                ("Status", f"pyrite remote status {jobid} -vv"),
                ("Logs", f"pyrite remote logs {jobid} --follow"),
                ("Pull", f"pyrite remote pull {' '.join(stems)} --line-only  (after completion)"),
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


def _stop_jobids(jobids):
    """Cancel active scheduler jobs in one batched ssh session.

    ``stop --all`` / multi-material stops pay per-job ssh + scancel polling
    through :func:`_stop_jobid`, which serializes to tens of seconds on a
    batch; the batched command writes every STOP sentinel, scancels once,
    and polls squeue once for the whole set (see
    ``scripts._scancel_jobs_command``). Single-job callers (energy_grid) keep
    the granular :func:`_stop_jobid` error semantics."""
    transport._check_shell_tokens(list(jobids))
    transport._run(["ssh", "-n", config.remote_host(), scripts._scancel_jobs_command(list(jobids))])


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
        transport._run(["ssh", "-n", config.remote_host(), scripts._reap_job_command(jobid)])
    print(f"reaped {len(orphans)} orphaned job(s)")


def _resolve_survey_stems(stems):
    """Expand bare canonical-material stems to also include any matching
    identity-qualified survey checkpoint directories on the box
    (``<material>--survey-<hash>/``), so ``pyrite remote pull <material>`` finds
    a survey checkpoint without the caller needing to know its hash suffix.

    On-disk names stay hash-based -- this only discovers them, via one remote
    listing of ``checkpoints/``, matched against the existing
    ``profiles._VARIANT_STEM_RE``. Already-qualified stems (a stem that
    already fullmatches that regex) and ``_quick`` stems pass through
    unexpanded: a stem that already names an exact variant, or a quick smoke
    checkpoint, never has a survey sibling worth auto-discovering."""
    from ..campaign.profiles import _VARIANT_STEM_RE

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
        # A survey variant reads as ``--survey-`` (legacy ``fidelity`` group) or
        # ``@survey-`` (@-stem ``label`` group for a standard-profile survey run).
        survey_token = None if match is None else (match["fidelity"] or match["label"])
        if match is None or survey_token != "survey" or match["material"] not in bare:
            continue
        if name in resolved:
            continue
        resolved.append(name)
        print(f"also pulling identity-qualified survey checkpoint -> checkpoints/{name}/")
    return resolved


def _split_profile_selector(stem):
    """Split a ``MATERIAL@PROFILE`` pull selector into ``(material, profile)``,
    or ``None`` for a plain stem (no ``@``, or a full ``@``-stem).

    A resolved on-disk @-stem (``<material>@<label>-<digest>``, matching
    :data:`~pyrite.campaign.profiles._VARIANT_STEM_RE`) is already an exact checkpoint,
    not a profile query, so it passes through literally -- only a bare
    ``MATERIAL@PROFILE`` with no digest tail is treated as a selector to
    resolve via meta.json."""
    from ..campaign.profiles import _VARIANT_STEM_RE

    if "@" not in stem or _VARIANT_STEM_RE.fullmatch(stem) is not None:
        return None
    material, _, profile = stem.partition("@")
    if not material or not profile:
        raise SystemExit(
            f"invalid MATERIAL@PROFILE selector {stem!r}: expected both a material "
            "and a profile name either side of '@'"
        )
    return material, profile


def _remote_meta_json(stem):
    """Fetch and parse one remote checkpoint's ``meta.json`` plus its mtime, or
    ``None`` when the file is absent or unreadable. One ssh round trip: a
    leading ``stat`` line disambiguates "missing" from "empty" without a
    second connection."""
    path = config.remote_path("checkpoints", stem, "meta.json")
    path_q = config.shell_arg(path)
    raw = transport._ssh_capture(f"[ -f {path_q} ] || exit 0; stat -c %Y {path_q}; cat {path_q}")
    mtime_line, _, body = raw.partition("\n")
    if not mtime_line.strip():
        return None
    try:
        mtime = float(mtime_line)
        meta = json.loads(body)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return mtime, meta


def _profile_pull_candidates(material):
    """Every on-disk stem for MATERIAL: the bare canonical directory (if
    present) plus every identity-qualified ``<material>--<fidelity>-<hash>``
    variant -- the stem name alone never carries the catalog profile, so this
    only narrows by material; :func:`resolve_profile_stem` reads each
    candidate's meta.json to filter by profile."""
    from ..campaign.profiles import _VARIANT_STEM_RE

    names = set(transport._ssh_capture(scripts._list_checkpoint_dirs_command()).split())
    candidates = [material] if material in names else []
    for name in sorted(names):
        match = _VARIANT_STEM_RE.fullmatch(name)
        if match is not None and match["material"] == material and name not in candidates:
            candidates.append(name)
    return candidates


def resolve_profile_stem(material, catalog_profile, *, hash_prefix=None):
    """Resolve a ``MATERIAL@PROFILE`` pull selector to one exact on-disk
    checkpoint stem.

    Decision 3/Phase 3 design: on-disk stems stay hash-based and never encode
    the catalog profile, so this reads each candidate directory's
    ``meta.json`` -> ``dataset_identity.catalog_profile`` remotely (one ssh
    round trip per candidate; materials rarely carry more than a handful of
    variants). The newest match wins unless ``hash_prefix`` (``--hash``) pins
    one; ties print every alternative hash so a caller can pin explicitly.
    """
    candidates = _profile_pull_candidates(material)
    found = []
    for stem in candidates:
        fetched = _remote_meta_json(stem)
        if fetched is None:
            continue
        mtime, meta = fetched
        identity = meta.get("dataset_identity") or {}
        profile = identity.get("catalog_profile") or "standard"
        digest = str(identity.get("parameter_sha256", ""))
        found.append((stem, profile, digest, mtime))
    matches = [item for item in found if item[1] == catalog_profile]
    if hash_prefix is not None:
        matches = [item for item in matches if item[2].startswith(hash_prefix)]
    if not matches:
        available = sorted({f"{profile}:{digest[:12]}" for _, profile, digest, _ in found})
        detail = f"; found on the box: {', '.join(available)}" if available else "; none found"
        selector = f"{material}@{catalog_profile}"
        if hash_prefix is not None:
            selector += f" --hash {hash_prefix}"
        raise SystemExit(f"no remote checkpoint matches {selector}{detail}")
    matches.sort(key=lambda item: item[3], reverse=True)
    if len(matches) > 1 and hash_prefix is None:
        alternates = ", ".join(item[2][:12] for item in matches[1:])
        print(
            f"{material}@{catalog_profile}: {len(matches)} hashes found on the box; "
            f"pulling newest ({matches[0][2][:12]}); pin another with "
            f"--hash (alternates: {alternates})"
        )
    return matches[0][0]


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
    hash_prefix=None,
):
    """Fetch checkpoints/<stem>.pkl back from the box for each stem (stem =
    material, or material_quick for a --quick run).

    A bare material stem also picks up any identity-qualified survey
    checkpoint the box holds for it (``<material>--survey-<hash>/``, see
    :func:`_resolve_survey_stems`) -- skipped for a ``dataset`` merge pull
    (``--brem-only``/``--line-only``), since those require an existing local
    checkpoint to merge into and a freshly discovered stem would not have one.

    Every pull runs the transfer through ``pyrite slim`` on the box, which encodes
    straight to that ssh session's stdout (``-o -``) -- the box's compress pass
    overlaps the wire instead of staging a whole temp artifact on box disk
    first. ``grid`` filters to just the material's current grid (plus the
    optional byte trimmers); ``level9`` recompresses at the codec's maximum
    level (:data:`_checkpoint_io.MAX_LEVEL`) instead of the default a live sweep
    writes at -- lossless, just smaller for the wire, and only worth it on a
    slow link, since the level costs far more box CPU than it saves bytes.
    ``sync_code()`` runs first (unless ``no_sync``) so the box rebuilds the grid
    from the same ``config.py`` the laptop has, and has the ``-o -`` and
    ``--compresslevel`` flags at all -- closing sync drift."""
    if dataset not in (None, "brem", "line"):
        raise ValueError("dataset must be None, 'brem', or 'line'")
    stems = list(stems)
    if not stems:
        raise ValueError("stems must contain at least one checkpoint stem")
    # MATERIAL@PROFILE selectors split and resolve to an exact stem here,
    # before the shell-token check below -- '@' is not a safe interpolation
    # token, and only the resolved stem ever reaches a remote command.
    selectors = [_split_profile_selector(stem) for stem in stems]
    qualified = [selector for selector in selectors if selector is not None]
    if hash_prefix is not None and len(qualified) != 1:
        raise SystemExit("--hash requires exactly one MATERIAL@PROFILE selector to pull")
    if qualified:
        resolved_stems = []
        for stem, selector in zip(stems, selectors, strict=True):
            if selector is None:
                resolved_stems.append(stem)
                continue
            material, profile = selector
            transport._check_shell_tokens([material])
            resolved_stems.append(resolve_profile_stem(material, profile, hash_prefix=hash_prefix))
        stems = resolved_stems
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
            from ..campaign import profiles
            from ..campaign.profiles import identity_from_stem
            from ..checkpoints import _checkpoint_io, _checkpoint_store
            from ..results import merge_dataset
            from ..runs.run import _manifest_save

            # identity_from_stem(stem, dest) covers the bare-canonical/_quick
            # stems and any re-pull with an already-registered local sidecar
            # with no extra ssh call. It cannot, on a stem's FIRST pull,
            # reconstruct a variant whose digest depends on state the catalog
            # alone doesn't carry (a coherent/both emission run, a
            # since-edited named profile -- see identity_from_stem's
            # docstring) -- so for that one case only, fall back to the box's
            # authoritative meta.json (one extra ssh round trip, gated on the
            # variant-stem pattern so canonical/_quick/merge pulls never pay
            # it -- see test_pull_quick_stem_does_not_resolve_survey_siblings
            # and test_pull_dataset_merge_skips_survey_discovery).
            resolved_identity = identity_from_stem(stem, dest)
            if resolved_identity is None and profiles._VARIANT_STEM_RE.fullmatch(stem):
                remote_meta = _remote_meta_json(stem)
                if remote_meta is not None:
                    resolved_identity = remote_meta[1].get("dataset_identity")

            flags = ""
            if grid:
                flags += " --grid"
            if drop_wide_brem:
                flags += " --drop-wide-brem"
            if downcast:
                flags += " --downcast"
            if level9:
                flags += f" --compresslevel {_checkpoint_io.MAX_LEVEL}"
            if dataset is not None:
                flags += f" --{dataset}-only"
            ckpt = config.remote_path("checkpoints", stem)
            incoming_local = dest / f".{stem}.incoming.pkl"
            # `-o -` streams the encoded artifact straight down this ssh
            # session's stdout (slim's own report goes to stderr), so the box's
            # compress pass overlaps the transfer. The older write-temp-then-cat
            # form serialized the two and staged a gigabyte-scale temp on box
            # disk; a nonzero slim exit still fails the pull, because ssh
            # propagates the remote command's status.
            remote_transfer = (
                f"cd {config.shell_remote_dir()} && "
                f"{config.shell_remote_uv()} run --no-sync pyrite slim "
                f"{config.shell_arg(ckpt)}{flags} -o -"
            )
            # Timed so a slow pull is attributable: this covers box slim CPU +
            # wire, the local decode+split below is what remains. If the rate
            # here sits near the link speed the wire is the wall; if it sits far
            # below it, the box's compress pass is.
            started = time.monotonic()
            transport._ssh_download(remote_transfer, incoming_local)
            elapsed = max(time.monotonic() - started, 1e-9)
            transferred = incoming_local.stat().st_size / 1e6
            print(
                f"transferred {transferred:.1f} MB in {elapsed:.1f}s ({transferred / elapsed:.1f} MB/s)"
            )

            incoming = _checkpoint_io.load(str(incoming_local))
            incoming_local.unlink(missing_ok=True)
            if dataset is not None:
                if not _checkpoint_store.checkpoint_exists(stem, dest):
                    raise FileNotFoundError(f"no local checkpoints/{stem}/ to merge into")
                archive.archive_checkpoint(stem, force=True)
                base = _checkpoint_store.load(stem, dest)
                n_merged, n_skipped = merge_dataset(base, incoming, dataset, force=force)
                _checkpoint_store.save(stem, dest, base, components=(dataset,))
                _manifest_save(str(local), base, resolved_identity)
                print(
                    f"merged {dataset} ({n_merged} rec, skipped {n_skipped}) -> checkpoints/{stem}/"
                )
            else:
                _checkpoint_store.save(stem, dest, incoming)
                _manifest_save(str(local), incoming, resolved_identity)
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
        print("(no zhai cache files on the box -- run `pyrite run --preset zhai --remote` first)")
        return
    dest = config.LOCAL_ROOT / "checkpoints" / "zhai_reproduction"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        transport._run(["scp", config.scp_remote_path(f"{remote_dir}/{name}"), str(dest / name)])
    print(f"pulled -> checkpoints/zhai_reproduction/ ({len(names)} cache files)")
