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


def _selected_materials(args, attribute):
    """Resolve explicit material arguments or standard-profile membership."""
    explicit = getattr(args, attribute)
    if args.all:
        if explicit:
            raise SystemExit(f"{args.remote_command} --all does not take material names")
        from ...runs.scan import resolve_profile_materials

        return resolve_profile_materials("standard")
    if explicit:
        return explicit if isinstance(explicit, list) else [explicit]
    raise SystemExit(f"{args.remote_command} needs material name(s), or use --all")


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


def _start_selected(args):
    """Validate the profile-owned material selection prepared by ``remote run``."""
    from ...runs.scan import validate_catalog_profile

    materials = list(getattr(args, "materials", None) or [])
    catalog_profile = getattr(args, "catalog_profile", "standard")
    return validate_catalog_profile(catalog_profile, materials, intersect=False), None


def _cli_rebrem(args):
    """Submit a brem-only checkpoint recompute, follow it, pull what completed."""
    materials = _selected_materials(args, "material")
    jobid = lifecycle.start_rebrem_queue(
        materials,
        fidelity=getattr(args, "fidelity", "full"),
        ne_brem=args.ne_brem,
        brem_start_eV=getattr(args, "start", None),
        brem_stop_eV=getattr(args, "stop", None),
        brem_step_eV=args.step,
        redo_all=args.redo_all,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
        chunk_minutes=args.chunk_minutes,
    )
    if args.dry_run:
        return
    if getattr(args, "detach", False):
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


def _cli_reline(args):
    """Submit a line-only checkpoint recompute, follow it, pull what completed."""
    materials = _selected_materials(args, "material")
    jobid = lifecycle.start_reline_queue(
        materials,
        fidelity=getattr(args, "fidelity", "full"),
        line_ne=args.line_ne,
        line_start_eV=getattr(args, "start", None),
        line_stop_eV=getattr(args, "stop", None),
        line_step_eV=args.line_step,
        redo_all=args.redo_all,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
        chunk_minutes=args.chunk_minutes,
    )
    if args.dry_run:
        return
    if getattr(args, "detach", False):
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


def _cli_pull(args):
    dataset = "brem" if args.brem_only else ("line" if args.line_only else None)
    lifecycle.pull(
        _selected_materials(args, "material"),
        grid=(not args.full) and dataset is None,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        level9=args.level9,
        no_sync=args.no_sync,
        dataset=dataset,
        force=args.force,
        hash_prefix=getattr(args, "hash_prefix", None),
    )


def _cli_pull_json(args):
    materials = _selected_materials(args, "material")
    dataset = "brem" if args.brem_only else ("line" if args.line_only else None)
    started = time.monotonic()
    summary = {}
    caught = None
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            lifecycle.pull(
                materials,
                grid=(not args.full) and dataset is None,
                drop_wide_brem=args.drop_wide_brem,
                downcast=args.downcast,
                level9=args.level9,
                no_sync=args.no_sync,
                dataset=dataset,
                force=args.force,
                summary=summary,
                hash_prefix=getattr(args, "hash_prefix", None),
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


def _cli_start(args):
    materials, high_energy_min_kev = _start_selected(args)
    if getattr(args, "nsys", False) and len(materials) != 1:
        raise click.UsageError("--nsys requires exactly one material")
    jobid = lifecycle.start_queue(
        materials,
        quick=args.quick,
        fidelity=getattr(args, "fidelity", "full"),
        workers=args.workers,
        parallel_materials=args.parallel_materials,
        chunk_minutes=args.chunk_minutes,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
        high_energy_min_kev=high_energy_min_kev,
        catalog_profile=getattr(args, "catalog_profile", "standard"),
        performance_profile=getattr(args, "performance_profile", None),
        performance_repetitions=getattr(args, "performance_repetitions", 1),
        performance_interval=getattr(args, "performance_interval", 5.0),
        spec_chunk=getattr(args, "spec_chunk", None),
        brem_chunk=getattr(args, "brem_chunk", None),
        nsys=getattr(args, "nsys", False),
        cpu=getattr(args, "cpu", False),
        cpu_only=getattr(args, "cpu_only", False),
        no_cache=getattr(args, "no_cache", False),
        recompute=getattr(args, "recompute", False),
    )
    if args.dry_run:
        return
    if args.headless:
        if getattr(args, "performance_profile", None) is not None:
            emit_result(
                "performance artifacts remain remote; pull after completion with: "
                f"pyrite remote performance pull {args.performance_profile}"
            )
        return
    if not viewer.attach(jobid):
        emit_diagnostic("run is still active or its viewer disconnected; skipping automatic pull")
        return
    if getattr(args, "performance_profile", None) is not None and not args.no_pull:
        succeeded = state._job_succeeded(jobid)
        if succeeded or getattr(args, "cpu", False):
            if not succeeded:
                emit_diagnostic("CPU phase failed; pulling retained primary performance artifacts")
            lifecycle.pull_performance_profile(args.performance_profile)
        else:
            emit_diagnostic(
                "performance run did not complete successfully; "
                "skipping automatic performance-artifact pull"
            )
    profiling_only = (
        getattr(args, "performance_repetitions", 1) > 1
        or getattr(args, "nsys", False)
        or getattr(args, "cpu", False)
        or getattr(args, "cpu_only", False)
    )
    if args.no_pull or profiling_only:
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
    fidelity = getattr(args, "fidelity", "full")
    catalog_profile = getattr(args, "catalog_profile", "standard")
    if args.quick or (fidelity == "full" and catalog_profile == "standard"):
        stems = scripts._stems(
            completed,
            args.quick,
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
        grid=args.grid,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        level9=getattr(args, "level9", False),
        no_sync=True,
    )
    for stem in stems:
        print(
            f"\ndone. checkpoints/{stem}/ is local; run `pyrite app analysis {stem}` "
            f"(or run `pyrite app analysis export`) -- visualization and static-HTML export "
            "stay local."
        )


def _cli_jobs(args):
    if not args.json_output:
        kind = getattr(args, "kind", None)
        if kind is None:
            viewer.list_jobs()
        else:
            viewer.list_jobs(kind)
        return
    try:
        result = cli_json.remote_jobs(viewer.jobs_raw(), kind=getattr(args, "kind", None))
    except (Exception, SystemExit) as exc:
        result = cli_json.failure("cxr.remote.jobs", {"jobs": []}, str(exc))
    emit_json_result(result)


def _cli_status(args):
    if getattr(args, "attach", False):
        # Continuous, reconnecting monitor: the same acquisition/render path as
        # the one-shot snapshot, repainted in place until the job is terminal or
        # the viewer is interrupted (viewer-only disconnect; job keeps running).
        viewer.attach(args.jobid, args.verbose)
        return
    if not args.json_output:
        viewer.job_status(args.jobid, args.verbose)
        return
    try:
        sections, output = viewer.status_sections(args.jobid, max(args.verbose, 2))
        if not sections:
            raise RuntimeError(output.strip() or "remote status response was malformed")
        result = cli_json.remote_status(sections)
    except (Exception, SystemExit) as exc:
        result = cli_json.failure("cxr.remote.status", {}, str(exc))
    emit_json_result(result)


def _cli_logs(args):
    return viewer.tail_logs(args.jobid, args.follow)


def _cli_profile_pull(args):
    return lifecycle.pull_performance_profile(args.profile)


def _cli_performance_list(args):
    del args
    return lifecycle.list_remote_performance()


def _cli_performance_prune(args):
    return lifecycle.prune_remote_performance(
        args.profiles,
        all_profiles=args.all_profiles,
        yes=args.yes,
    )


def _cli_stop(args):
    lifecycle.stop_jobs(args.materials, args.all, yes=args.yes, profile=args.catalog_profile)


def _cli_reap(args):
    lifecycle.reap_reservations(min_age_minutes=args.min_age_minutes, yes=args.yes)


def _cli_clear(args):
    if args.catalog_profile is not None:
        if args.all_checkpoints or args.materials:
            raise click.UsageError("rm --profile takes no material arguments or --all")
        membership = _profile_default_materials(args.catalog_profile)
        if membership is None:
            from ...materials import CATALOG

            membership = CATALOG.material_keys
        lifecycle.clear_remote(
            list(membership),
            args.yes,
            catalog_profile=args.catalog_profile,
        )
        return
    if args.all_checkpoints:
        if args.materials:
            raise click.UsageError("rm --all takes no material argument")
        lifecycle.clear_all_remote(args.yes)
        return
    if not args.materials:
        raise click.UsageError("rm needs material(s), --profile, or --all")
    lifecycle.clear_remote(args.materials, args.yes)


def _cli_prune(args):
    lifecycle.prune_remote(
        all_profiles=args.all_profiles,
        catalog_profile=args.catalog_profile,
        yes=args.yes,
    )


def _cli_prune_jobs(args):
    lifecycle.prune_job_dirs(
        profile=args.catalog_profile,
        all_jobs=args.all_jobs,
        yes=args.yes,
    )


def _cli_sync(args):
    transport.sync_code()


def _cli_check(args):
    if args.follow and not args.detached:
        raise click.UsageError("--follow requires --detached")
    if args.pull:
        lifecycle.pull_zhai_cache()
        return
    if args.detached:
        jobid = lifecycle.start_zhai_queue(
            ne=args.ne,
            ne_brem=args.ne_brem,
            ne_supp=args.ne_supp,
            tmd_azimuth=args.tmd_azimuth,
            refresh=args.refresh,
            no_sync=args.no_sync,
        )
        if args.follow:
            viewer.attach(jobid)
        return
    remote_check(
        ne=args.ne,
        ne_brem=args.ne_brem,
        ne_supp=args.ne_supp,
        tmd_azimuth=args.tmd_azimuth,
        refresh=args.refresh,
        no_sync=args.no_sync,
    )
