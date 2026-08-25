"""Remote component-recompute queue submission."""

import math

from ..cli import dashboard as presentation
from . import config, scripts, transport
from .cleanup import _refuse_if_busy
from .jobs import _stage_job_script, _submit_staged_job


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
        job_name=f"pyrite-rebrem-{jobid}",
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
                ("Monitor", f"pyrite job attach {jobid}"),
                ("Status", f"pyrite job status {jobid} -vv"),
                ("Logs", f"pyrite job logs {jobid} --follow"),
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
    recompute is chunked: each ~chunk_minutes slice does bounded work via ``pyrite
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
        job_name=f"pyrite-reline-{jobid}",
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
                ("Monitor", f"pyrite job attach {jobid}"),
                ("Status", f"pyrite job status {jobid} -vv"),
                ("Logs", f"pyrite job logs {jobid} --follow"),
                ("Pull", f"pyrite remote pull {' '.join(stems)} --line-only  (after completion)"),
            ]
        )
    )
    return jobid
