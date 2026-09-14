"""Remote profile and validation queue submission."""

import math

from ..console import dashboard as presentation
from . import config, scripts, state, transport
from .cleanup import _refuse_if_busy
from .jobs import _stage_job_script, _submit_staged_job


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
            f"(pyrite job attach {clash[0]}) or stop it "
            f"(pyrite job stop --profile {catalog_profile}) first."
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
    no_cache=False,
    recompute=False,
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
    if no_cache and recompute:
        raise SystemExit("no-cache and recompute modes are mutually exclusive")
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
            no_cache,
            recompute,
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
            no_cache,
            recompute,
        )
        time_limit = config.SLURM_TIME
    workers_per_material = config.SLURM_CPUS_PER_MATERIAL if workers is None else max(1, workers)
    cpus_per_task = workers_per_material * (parallel_materials or 1)
    script = scripts._slurm_batch_script(
        jobid,
        payload,
        job_name=f"pyrite-{jobid}",
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
            no_cache,
            recompute,
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
        + presentation.format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", config.remote_host()),
                ("Materials", ", ".join(materials)),
                (
                    "Mode",
                    presentation.mode_summary(
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
                            no_cache,
                            recompute,
                        )
                    ),
                ),
                ("Monitor", f"pyrite job attach {jobid}"),
                ("Status", f"pyrite job status {jobid} -vv"),
                ("Logs", f"pyrite job logs {jobid}"),
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
        jobid, payload, job_name=f"pyrite-zhai-{jobid}", reservation_stems=stems
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
        + presentation.format_fields(
            [
                ("SLURM", scheduler_id),
                ("Host", config.remote_host()),
                ("Workload", "Zhai reproduction"),
                ("Monitor", f"pyrite job attach {jobid}"),
                ("Status", f"pyrite job status {jobid} -vv"),
                ("Logs", f"pyrite job logs {jobid} --follow"),
                ("Pull", "pyrite remote pull --preset zhai  (after completion)"),
            ]
        )
    )
    return jobid
