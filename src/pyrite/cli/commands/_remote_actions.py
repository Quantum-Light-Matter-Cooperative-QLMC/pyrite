"""Click command group, compatibility wiring, and remote orchestrators."""

import io
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

import click

from ...remote import config, lifecycle, scripts, state, transport, viewer
from .. import dashboard as presentation
from .. import json as cli_json
from .._core import (
    emit_diagnostic,
    emit_json_result,
    emit_result,
)


def remote_scan(material, quick=False, workers=None, fidelity="full"):
    """Submit one material through SLURM and follow it to completion.

    This compatibility helper deliberately does not pull: callers that need a
    checkpoint can apply their own grid/trim policy after it returns.
    """
    transport._check_materials([material])
    jobid = lifecycle.start_queue(
        [material],
        quick=quick,
        workers=workers,
        chunk_minutes=0,
        fidelity=fidelity,
    )
    viewer.attach(jobid)
    return jobid


def remote_check(
    ne=20_000,
    ne_brem=200,
    ne_supp=200,
    tmd_azimuth=0.0,
    refresh=False,
    no_sync=False,
    detach=False,
    dry_run=False,
):
    """Submit the Zhai reproduction through SLURM, follow it, then pull cache."""
    jobid = lifecycle.start_zhai_queue(
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
        tmd_azimuth=tmd_azimuth,
        refresh=refresh,
        no_sync=no_sync,
        dry_run=dry_run,
    )
    if detach or dry_run:
        return jobid
    if not viewer.attach(jobid):
        emit_diagnostic(
            "Zhai job is still running or its viewer disconnected; skipping automatic cache pull"
        )
        return
    if not state._job_succeeded(jobid):
        job_state = state._job_state(jobid) or "no terminal state recorded"
        raise SystemExit(
            f"Zhai SLURM job {jobid} did not complete successfully ({job_state}); cache not pulled"
        )
    lifecycle.pull_zhai_cache()


def _ensure_utf8_stdio():
    """The box's output is UTF-8 (job logs embed tqdm block-glyph progress bars
    like `████▌`). On Windows stdout defaults to cp1252 -- and when it is,
    printing that text raises UnicodeEncodeError, so `status`/`jobs`/`logs`/
    `attach` would crash on the glyphs. Force UTF-8 with replacement so they
    never do. Scoped to remote subcommands (called from their CLI wrappers, not
    at import time) so it doesn't change stdio encoding for unrelated ``pyrite``
    commands."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass


# ---- CLI wiring ----------------------------------------------------------------
def _dispatch(handler):
    def _run_cli(args):
        _ensure_utf8_stdio()
        return handler(args)

    return _run_cli


def _performance_profile_name(ctx, param, value):
    if value is None:
        return None
    if presentation._SHELL_TOKEN_RE.fullmatch(value) is None:
        raise click.BadParameter(
            "expected letters, digits, underscores, or hyphens",
            ctx=ctx,
            param=param,
        )
    return value


def _selected_materials(remote_command, all_, explicit):
    """Resolve explicit material arguments or standard-profile membership."""
    if all_:
        if explicit:
            raise click.UsageError(f"{remote_command} --all does not take material names")
        from ...runs.scan import resolve_profile_materials

        return resolve_profile_materials("standard")
    if explicit:
        return explicit if isinstance(explicit, list) else [explicit]
    raise click.UsageError(f"{remote_command} needs material name(s), or use --all")


def _profile_default_materials(catalog_profile):
    """Resolve a profile's explicit membership or implicit full catalog."""
    from ...runs.scan import resolve_profile_materials

    return resolve_profile_materials(catalog_profile)


def _profile_selected_materials(catalog_profile, materials):
    """Resolve profile defaults or validate an explicit narrowing selection."""
    if not materials:
        return _profile_default_materials(catalog_profile)

    from ...runs.scan import validate_catalog_profile, validate_materials

    validate_materials(materials)
    return validate_catalog_profile(catalog_profile, materials, intersect=False)


def _start_selected(materials, catalog_profile):
    """Validate the profile-owned material selection prepared by ``remote run``."""
    from ...runs.scan import validate_catalog_profile

    return validate_catalog_profile(catalog_profile, list(materials or []), intersect=False), None


def _cli_rebrem(
    *,
    material,
    all_,
    fidelity,
    ne_brem,
    start,
    stop,
    step,
    redo_all,
    no_sync,
    dry_run,
    chunk_minutes,
    detach,
):
    """Submit a brem-only checkpoint recompute, follow it, pull what completed."""
    materials = _selected_materials("rebrem", all_, material)
    jobid = lifecycle.start_rebrem_queue(
        materials,
        fidelity=fidelity,
        ne_brem=ne_brem,
        brem_start_eV=start,
        brem_stop_eV=stop,
        brem_step_eV=step,
        redo_all=redo_all,
        no_sync=no_sync,
        dry_run=dry_run,
        chunk_minutes=chunk_minutes,
    )
    if dry_run:
        return
    if detach:
        return
    if not viewer.attach(jobid):
        emit_diagnostic(
            "rebrem is still running or its viewer disconnected; skipping automatic pull"
        )
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        emit_diagnostic(
            "warning: the SLURM rebrem updated no checkpoints successfully; nothing to pull"
        )
        return
    # merge ONLY the fresh brem into the local pickle, preserving any local line spectra
    lifecycle.pull(completed, dataset="brem")


def _cli_reline(
    *,
    material,
    all_,
    fidelity,
    line_ne,
    start,
    stop,
    line_step,
    redo_all,
    no_sync,
    dry_run,
    chunk_minutes,
    detach,
):
    """Submit a line-only checkpoint recompute, follow it, pull what completed."""
    materials = _selected_materials("reline", all_, material)
    jobid = lifecycle.start_reline_queue(
        materials,
        fidelity=fidelity,
        line_ne=line_ne,
        line_start_eV=start,
        line_stop_eV=stop,
        line_step_eV=line_step,
        redo_all=redo_all,
        no_sync=no_sync,
        dry_run=dry_run,
        chunk_minutes=chunk_minutes,
    )
    if dry_run:
        return
    if detach:
        return
    if not viewer.attach(jobid):
        emit_diagnostic(
            "reline is still running or its viewer disconnected; skipping automatic pull"
        )
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        emit_diagnostic(
            "warning: the SLURM reline updated no checkpoints successfully; nothing to pull"
        )
        return
    lifecycle.pull(completed, dataset="line")


def _cli_pull(
    *,
    remote_command,
    material,
    all_,
    full,
    drop_wide_brem,
    downcast,
    level9,
    no_sync,
    brem_only,
    line_only,
    force,
    hash_prefix,
):
    dataset = "brem" if brem_only else ("line" if line_only else None)
    lifecycle.pull(
        _selected_materials(remote_command, all_, material),
        grid=(not full) and dataset is None,
        drop_wide_brem=drop_wide_brem,
        downcast=downcast,
        level9=level9,
        no_sync=no_sync,
        dataset=dataset,
        force=force,
        hash_prefix=hash_prefix,
    )


def _cli_pull_json(
    *,
    remote_command,
    material,
    all_,
    full,
    drop_wide_brem,
    downcast,
    level9,
    no_sync,
    brem_only,
    line_only,
    force,
    hash_prefix,
):
    materials = _selected_materials(remote_command, all_, material)
    dataset = "brem" if brem_only else ("line" if line_only else None)
    started = time.monotonic()
    summary = {}
    caught = None
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            lifecycle.pull(
                materials,
                grid=(not full) and dataset is None,
                drop_wide_brem=drop_wide_brem,
                downcast=downcast,
                level9=level9,
                no_sync=no_sync,
                dataset=dataset,
                force=force,
                summary=summary,
                hash_prefix=hash_prefix,
            )
    except (Exception, SystemExit) as exc:
        caught = exc
    completed = summary.get("completed", [])
    failed = summary.get("failed", [item for item in materials if item not in completed])
    errors = summary.get("errors", {})
    if caught is not None:
        detail = str(caught) or type(caught).__name__
        for material in failed:
            errors.setdefault(material, detail)
    result = cli_json.operation_summary(
        "remote-pull",
        materials,
        completed,
        failed_materials=failed,
        checkpoints=[config.LOCAL_ROOT / "checkpoints" / item for item in materials],
        elapsed_seconds=time.monotonic() - started,
        material_errors=errors,
    )
    emit_json_result(result)


def _cli_start(
    *,
    materials,
    catalog_profile,
    quick,
    fidelity,
    workers,
    parallel_materials,
    chunk_minutes,
    no_sync,
    dry_run,
    headless,
    no_pull,
    grid,
    drop_wide_brem,
    downcast,
    level9,
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
):
    materials, high_energy_min_kev = _start_selected(materials, catalog_profile)
    if nsys and len(materials) != 1:
        raise click.UsageError("--nsys requires exactly one material")
    jobid = lifecycle.start_queue(
        materials,
        quick=quick,
        fidelity=fidelity,
        workers=workers,
        parallel_materials=parallel_materials,
        chunk_minutes=chunk_minutes,
        no_sync=no_sync,
        dry_run=dry_run,
        high_energy_min_kev=high_energy_min_kev,
        catalog_profile=catalog_profile,
        performance_profile=performance_profile,
        performance_repetitions=performance_repetitions,
        performance_interval=performance_interval,
        spec_chunk=spec_chunk,
        brem_chunk=brem_chunk,
        nsys=nsys,
        cpu=cpu,
        cpu_only=cpu_only,
        no_cache=no_cache,
        recompute=recompute,
    )
    if dry_run:
        return
    if headless:
        if performance_profile is not None:
            emit_result(
                "performance artifacts remain remote; pull after completion with: "
                f"pyrite remote performance pull {performance_profile}"
            )
        return
    if not viewer.attach(jobid):
        emit_diagnostic("run is still active or its viewer disconnected; skipping automatic pull")
        return
    if performance_profile is not None and not no_pull:
        succeeded = state._job_succeeded(jobid)
        if succeeded or cpu:
            if not succeeded:
                emit_diagnostic("CPU phase failed; pulling retained primary performance artifacts")
            lifecycle.pull_performance_profile(performance_profile)
        else:
            emit_diagnostic(
                "performance run did not complete successfully; "
                "skipping automatic performance-artifact pull"
            )
    profiling_only = performance_repetitions > 1 or nsys or cpu or cpu_only
    if no_pull or profiling_only:
        if profiling_only:
            emit_diagnostic(
                "performance profiling used isolated job-local checkpoints; "
                "skipping automatic checkpoint pull"
            )
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        emit_diagnostic(
            "warning: the SLURM scan produced no successful checkpoints; nothing to pull"
        )
        return
    if quick or (fidelity == "full" and catalog_profile == "standard"):
        stems = scripts._stems(
            completed,
            quick,
            fidelity,
            high_energy_min_kev=high_energy_min_kev,
            catalog_profile=catalog_profile,
        )
    else:
        # The synchronized box is authoritative after a potentially long job.
        # Recomputing identity hashes locally after attachment can target a
        # nonexistent directory when profile/catalog inputs changed meanwhile.
        stems = [
            lifecycle.resolve_profile_stem(material, catalog_profile, fidelity=fidelity)
            for material in completed
        ]
    lifecycle.pull(
        stems,
        grid=grid,
        drop_wide_brem=drop_wide_brem,
        downcast=downcast,
        level9=level9,
        no_sync=True,
    )
    for stem in stems:
        print(
            f"\ndone. checkpoints/{stem}/ is local; run `pyrite app analysis {stem}` "
            f"(or run `pyrite app analysis export`) -- visualization and static-HTML export "
            "stay local."
        )


def _cli_jobs(*, json_output, kind=None):
    if not json_output:
        if kind is None:
            viewer.list_jobs()
        else:
            viewer.list_jobs(kind)
        return
    try:
        result = cli_json.remote_jobs(viewer.jobs_raw(), kind=kind)
    except (Exception, SystemExit) as exc:
        result = cli_json.failure("cxr.remote.jobs", {"jobs": []}, str(exc))
    emit_json_result(result)


def _cli_status(*, jobid, verbose, attach, json_output):
    if attach:
        # Continuous, reconnecting monitor: the same acquisition/render path as
        # the one-shot snapshot, repainted in place until the job is terminal or
        # the viewer is interrupted (viewer-only disconnect; job keeps running).
        viewer.attach(jobid, verbose)
        return
    if not json_output:
        viewer.job_status(jobid, verbose)
        return
    try:
        sections, output = viewer.status_sections(jobid, max(verbose, 2))
        if not sections:
            raise RuntimeError(output.strip() or "remote status response was malformed")
        result = cli_json.remote_status(sections)
    except (Exception, SystemExit) as exc:
        result = cli_json.failure("cxr.remote.status", {}, str(exc))
    emit_json_result(result)


def _cli_logs(*, jobid, follow):
    return viewer.tail_logs(jobid, follow)


def _cli_profile_pull(*, profile):
    return lifecycle.pull_performance_profile(profile)


def _cli_performance_list():
    return lifecycle.list_remote_performance()


def _cli_performance_prune(*, profiles, all_profiles, yes):
    return lifecycle.prune_remote_performance(
        profiles,
        all_profiles=all_profiles,
        yes=yes,
    )


def _cli_stop(*, materials, all_, yes, catalog_profile):
    lifecycle.stop_jobs(materials, all_, yes=yes, profile=catalog_profile)


def _cli_reap(*, min_age_minutes, yes):
    lifecycle.reap_reservations(min_age_minutes=min_age_minutes, yes=yes)


def _cli_clear(*, materials, all_checkpoints, catalog_profile, yes):
    if catalog_profile is not None:
        if all_checkpoints or materials:
            raise click.UsageError("rm --profile takes no material arguments or --all")
        membership = _profile_default_materials(catalog_profile)
        if membership is None:
            from ...materials import CATALOG

            membership = CATALOG.material_keys
        lifecycle.clear_remote(
            list(membership),
            yes,
            catalog_profile=catalog_profile,
        )
        return
    if all_checkpoints:
        if materials:
            raise click.UsageError("rm --all takes no material argument")
        lifecycle.clear_all_remote(yes)
        return
    if not materials:
        raise click.UsageError("rm needs material(s), --profile, or --all")
    lifecycle.clear_remote(materials, yes)


def _cli_prune(*, all_profiles, catalog_profile, yes):
    lifecycle.prune_remote(
        all_profiles=all_profiles,
        catalog_profile=catalog_profile,
        yes=yes,
    )


def _cli_prune_jobs(*, catalog_profile, all_jobs, yes):
    lifecycle.prune_job_dirs(
        profile=catalog_profile,
        all_jobs=all_jobs,
        yes=yes,
    )


def _cli_sync():
    transport.sync_code()


def _cli_check(
    *,
    follow,
    detached,
    pull,
    ne,
    ne_brem,
    ne_supp,
    tmd_azimuth,
    refresh,
    no_sync,
):
    if follow and not detached:
        raise click.UsageError("--follow requires --detached")
    if pull:
        lifecycle.pull_zhai_cache()
        return
    if detached:
        jobid = lifecycle.start_zhai_queue(
            ne=ne,
            ne_brem=ne_brem,
            ne_supp=ne_supp,
            tmd_azimuth=tmd_azimuth,
            refresh=refresh,
            no_sync=no_sync,
        )
        if follow:
            viewer.attach(jobid)
        return
    remote_check(
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
        tmd_azimuth=tmd_azimuth,
        refresh=refresh,
        no_sync=no_sync,
    )
