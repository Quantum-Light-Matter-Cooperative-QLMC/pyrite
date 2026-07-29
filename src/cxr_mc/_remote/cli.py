"""Click command group, compatibility wiring, and remote orchestrators."""

import io
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from copy import copy

import click

from ..cli import _completion as _cli_completion
from ..cli import json as cli_json
from ..cli._core import (
    FINITE_FLOAT,
    NONNEGATIVE_FLOAT,
    NONNEGATIVE_INT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
    emit_diagnostic,
    emit_json_result,
    fidelity_option,
    invoke_legacy,
    run,
)
from ..scan import DEFAULT_HIGH_ENERGY_MIN_KEV, load_all_materials, load_manifest_groups
from . import config, lifecycle, presentation, scripts, state, transport, viewer


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
):
    """Submit the Zhai reproduction through SLURM, follow it, then pull cache."""
    jobid = lifecycle.start_zhai_queue(
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
        tmd_azimuth=tmd_azimuth,
        refresh=refresh,
        no_sync=no_sync,
    )
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
    at import time) so it doesn't change stdio encoding for unrelated ``cxr``
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
    """Resolve explicit material arguments or the shared ``--all`` manifest."""
    explicit = getattr(args, attribute)
    if args.all:
        if explicit:
            raise SystemExit(f"{args.remote_command} --all does not take material names")
        return load_all_materials(config.MATS_FILE)
    if explicit:
        return explicit if isinstance(explicit, list) else [explicit]
    raise SystemExit(f"{args.remote_command} needs material name(s), or use --all")


def _profile_default_materials(catalog_profile):
    """A non-standard profile's explicit ``materials`` membership, or ``None``
    when the profile has no membership row (implicit all-in-use -- the caller
    falls through to requiring ``--all``/explicit materials, since there is no
    narrower campaign list to default to). Also raises the usual
    unknown-profile usage error via ``validate_catalog_profile`` -- an empty
    material list never triggers its membership check, only its name check."""
    from ..materials import CATALOG
    from ..scan import validate_catalog_profile

    validate_catalog_profile(catalog_profile, [], intersect=False)
    return CATALOG.profile_materials(catalog_profile)


def _start_selected(args):
    """Resolve ``start``/``submit``'s materials plus a resolved high-energy
    floor, mirroring ``scan._selected``. Unlike ``scan.py``'s per-material
    floor map, a queue shares one flags string across every material in the
    batch, so this returns a single floor (or ``None``); ``--high-energy-min-kev``
    is a no-op for any queued material outside ``high_energy_materials``, so
    it is safe to forward blindly.

    With no ``--material``/``--all``/``-A``, the profile's explicit membership
    is the selection. A profile with no membership row uses the verified
    ``materials`` manifest group."""
    from ..scan import validate_catalog_profile

    explicit = list(getattr(args, "materials", None) or [])
    all_ = getattr(args, "all", False)
    actually_all = getattr(args, "actually_all", False)
    catalog_profile = getattr(args, "catalog_profile", "standard")
    high_energy_min_kev = None
    if all_ or actually_all:
        if explicit:
            raise SystemExit(f"{args.remote_command} --all/-A does not take material names")
        groups = load_manifest_groups(config.MATS_FILE)
        if actually_all:
            materials = list(
                dict.fromkeys(
                    [
                        *groups["materials"],
                        *groups["no_verified_dw"],
                        *groups["high_energy_materials"],
                        *groups["materials_to_leave_out"],
                    ]
                )
            )
            high_energy_selected = set(groups["high_energy_materials"])
        else:
            materials = list(groups["materials"])
            if getattr(args, "include_unverified_dw", False):
                materials += groups["no_verified_dw"]
            high_energy_selected = set()
            if getattr(args, "include_high_energy", False):
                materials += groups["high_energy_materials"]
                high_energy_selected = set(groups["high_energy_materials"])
            materials = list(dict.fromkeys(materials))
        materials = validate_catalog_profile(catalog_profile, materials, intersect=True)
        high_energy_selected &= set(materials)
        if high_energy_selected:
            floor = getattr(args, "high_energy_min_kev", None)
            high_energy_min_kev = DEFAULT_HIGH_ENERGY_MIN_KEV if floor is None else floor
    elif explicit:
        materials = validate_catalog_profile(catalog_profile, explicit, intersect=False)
        floor = getattr(args, "high_energy_min_kev", None)
        if floor is not None:
            tagged = set(load_manifest_groups(config.MATS_FILE)["high_energy_materials"])
            if tagged.intersection(materials):
                high_energy_min_kev = floor
    else:
        membership = _profile_default_materials(catalog_profile)
        materials = (
            list(membership)
            if membership is not None
            else list(load_manifest_groups(config.MATS_FILE)["materials"])
        )
        materials = validate_catalog_profile(catalog_profile, materials, intersect=True)
        floor = getattr(args, "high_energy_min_kev", None)
        if floor is not None:
            tagged = set(load_manifest_groups(config.MATS_FILE)["high_energy_materials"])
            if tagged.intersection(materials):
                high_energy_min_kev = floor
    return materials, high_energy_min_kev


def _cli_scan(args):
    if args.quick and args.grid:
        raise SystemExit(
            "scan --quick --grid: quick checkpoints aren't grid-filterable "
            "(their grid isn't reproducible from material_sweep), so the "
            "trailing pull would fail after the whole sweep ran. Drop --grid."
        )
    materials = _selected_materials(args, "material")
    # `start_queue` validates material tokens, refuses checkpoint collisions,
    # syncs once, and submits one bounded-concurrency scheduler job for the materials.
    jobid = lifecycle.start_queue(
        materials,
        quick=args.quick,
        fidelity=getattr(args, "fidelity", "full"),
        workers=args.workers,
        parallel_materials=getattr(args, "parallel_materials", None),
        chunk_minutes=getattr(args, "chunk_minutes", 10.0),
        no_sync=args.no_sync,
        performance_profile=getattr(args, "performance_profile", None),
    )
    if not viewer.attach(jobid):
        emit_diagnostic("scan is still running or its viewer disconnected; skipping automatic pull")
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        emit_diagnostic(
            "warning: the SLURM scan produced no successful checkpoints; nothing to pull"
        )
        return
    stems = scripts._stems(completed, args.quick, getattr(args, "fidelity", "full"))
    # Code is already synced by the queue, so a grid pull skips its own sync;
    # preserve the requested grid/trim policy.
    lifecycle.pull(
        stems,
        grid=args.grid,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        no_sync=True,
    )
    for stem in stems:
        print(
            f"\ndone. checkpoints/{stem}/ is local; run `cxr analyze {stem}` "
            f"(or run `cxr export`) -- visualization and static-HTML export "
            "stay local."
        )


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
    )
    if args.dry_run or args.headless:
        return
    if not viewer.attach(jobid):
        emit_diagnostic(
            "submit is still running or its viewer disconnected; skipping automatic pull"
        )
        return
    profiling_only = getattr(args, "performance_repetitions", 1) > 1 or getattr(args, "nsys", False)
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
    stems = scripts._stems(
        completed,
        args.quick,
        getattr(args, "fidelity", "full"),
        high_energy_min_kev=high_energy_min_kev,
        catalog_profile=getattr(args, "catalog_profile", "standard"),
    )
    lifecycle.pull(
        stems,
        grid=args.grid,
        drop_wide_brem=args.drop_wide_brem,
        downcast=args.downcast,
        no_sync=True,
    )
    for stem in stems:
        print(
            f"\ndone. checkpoints/{stem}/ is local; run `cxr analyze {stem}` "
            f"(or run `cxr export`) -- visualization and static-HTML export "
            "stay local."
        )


def _cli_attach(args):
    viewer.attach(args.jobid, args.verbose)


def _cli_jobs(args):
    if not args.json_output:
        viewer.list_jobs()
        return
    try:
        result = cli_json.remote_jobs(viewer.jobs_raw())
    except (Exception, SystemExit) as exc:
        result = cli_json.failure("cxr.remote.jobs", {"jobs": []}, str(exc))
    emit_json_result(result)


def _cli_status(args):
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


def _cli_stop(args):
    lifecycle.stop_jobs(args.materials, args.all, yes=args.yes, profile=args.catalog_profile)


def _cli_reap(args):
    lifecycle.reap_reservations(min_age_minutes=args.min_age_minutes, yes=args.yes)


def _cli_clear(args):
    if args.catalog_profile is not None:
        if args.all_checkpoints or args.materials:
            args._clear_parser.error("clear --profile takes no material arguments or --all")
        membership = _profile_default_materials(args.catalog_profile)
        if membership is None:
            from ..materials import CATALOG

            membership = CATALOG.material_keys
        lifecycle.clear_remote(
            list(membership),
            args.yes,
            catalog_profile=args.catalog_profile,
        )
        return
    if args.all_checkpoints:
        if args.materials:
            args._clear_parser.error("clear --all takes no material argument")
        lifecycle.clear_all_remote(args.yes)
        return
    if not args.materials:
        args._clear_parser.error("clear needs material(s), --profile, or --all")
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
        args._check_parser.error("--follow requires --detached")
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


class _ClickParser:
    """Parser-error compatibility for reused orchestration handlers."""

    @staticmethod
    def error(message):
        raise click.UsageError(message)


def _click_args(command_name, **values):
    """Build handler namespace for Click orchestration callbacks."""
    values["remote_command"] = command_name
    values["_clear_parser"] = _ClickParser
    values["_check_parser"] = _ClickParser
    return values


def _invoke_click(handler, args):
    """Translate legacy runtime exits and returned statuses to Click contract."""
    _ensure_utf8_stdio()
    status = invoke_legacy(handler, **args)
    if isinstance(status, int) and not isinstance(status, bool) and status:
        raise click.exceptions.Exit(status)
    return status


def _reject_all_with_values(command_name, all_, values):
    if all_ and values:
        raise click.UsageError(f"{command_name} --all does not take material names")
    if not all_ and not values:
        raise click.UsageError(f"{command_name} needs material name(s), or use --all")


@click.group(
    "remote",
    help=(
        "[dev] Push code and run or manage MC sweeps on a remote GPU box over SSH.\n\n"
        "Host, remote directory, and executable come from CXR_REMOTE_HOST, "
        "CXR_REMOTE_DIR, and CXR_REMOTE_UV. Command-line options take precedence "
        "over workflow defaults where offered.\n\n"
        "\b\n"
        "Examples:\n"
        "  cxr remote submit sub_100keV --dry-run\n"
        "  cxr remote submit compute_test_300keV -p\n"
        "  cxr remote submit standard -m hopg\n"
        "  cxr remote status -vv"
    ),
    no_args_is_help=False,
)
def command():
    """Push code and run or manage MC sweeps on a remote GPU box."""
    _ensure_utf8_stdio()


@command.command(
    "scan",
    help="Deprecated alias for `cxr remote submit`.",
    deprecated="Use 'cxr remote submit'.",
)
@click.argument(
    "material",
    required=False,
    metavar="[MATERIAL]",
    shell_complete=_cli_completion.complete_material,
)
@click.option("-a", "--all", "all_", is_flag=True, help="Run every material in mats_to_sim.toml.")
@fidelity_option(help="Named settings/grid policy. survey is provisional and reduced.")
@click.option(
    "--quick",
    is_flag=True,
    help="Smoke-test grid; incompatible with --grid.",
)
@click.option(
    "--workers",
    type=NONNEGATIVE_INT,
    default=None,
    help="Workers; 0 runs serially (default auto).",
)
@click.option(
    "--parallel-materials",
    type=click.IntRange(1, config.MAX_PARALLEL_MATERIALS),
    default=None,
    metavar="N",
    help="Simultaneous scans in one allocation; requires --chunk-minutes 0.",
    shell_complete=_cli_completion.choice_completer(range(1, config.MAX_PARALLEL_MATERIALS + 1)),
)
@click.option(
    "--chunk-minutes",
    type=NONNEGATIVE_FLOAT,
    default=10.0,
    show_default=True,
    help="Self-resubmitting SLURM slice length; 0 runs one monolithic job.",
)
@click.option(
    "--performance-profile",
    callback=_performance_profile_name,
    default=None,
    metavar="NAME",
    help=(
        "Log CPU pressure, RAM/swap, GPU clocks/VRAM, process, phase timing, "
        "queue, worker, chunk, and case metrics every 5 s."
    ),
)
@click.option("--no-sync", is_flag=True, help="Skip code upload.")
@click.option(
    "--grid",
    is_flag=True,
    help="Grid-filter checkpoint before pulling; incompatible with --quick.",
)
@click.option("--drop-wide-brem", is_flag=True, help="With --grid, drop wide-brem.")
@click.option("--downcast", is_flag=True, help="With --grid, downcast to float32.")
def scan_command(
    material,
    all_,
    fidelity,
    quick,
    workers,
    parallel_materials,
    chunk_minutes,
    performance_profile,
    no_sync,
    grid,
    drop_wide_brem,
    downcast,
):
    values = [material] if material else []
    _reject_all_with_values("scan", all_, values)
    if quick and grid:
        raise click.UsageError(
            "scan --quick --grid: quick checkpoints aren't grid-filterable; drop --grid"
        )
    if quick and fidelity != "full":
        raise click.UsageError("--quick cannot be combined with --fidelity survey")
    if parallel_materials is not None and chunk_minutes != 0:
        raise click.UsageError("--parallel-materials requires --chunk-minutes 0")
    return _invoke_click(
        _cli_scan,
        _click_args(
            "scan",
            material=material,
            all=all_,
            fidelity=fidelity,
            quick=quick,
            workers=workers,
            parallel_materials=parallel_materials,
            chunk_minutes=chunk_minutes,
            performance_profile=performance_profile,
            no_sync=no_sync,
            grid=grid,
            drop_wide_brem=drop_wide_brem,
            downcast=downcast,
        ),
    )


def _recompute_options(function):
    function = click.option(
        "--chunk-minutes",
        type=NONNEGATIVE_FLOAT,
        default=10.0,
        show_default=True,
        help="Self-resubmitting SLURM slice length; 0 runs one monolithic job.",
    )(function)
    function = click.option("--no-sync", is_flag=True, help="Skip code upload.")(function)
    function = click.option(
        "--dry-run",
        is_flag=True,
        help="Print batch script and submission command; do not connect.",
    )(function)
    function = click.option(
        "--redo-all",
        is_flag=True,
        help="Recompute every record even when already at target.",
    )(function)
    function = click.option(
        "-a", "--all", "all_", is_flag=True, help="Use every material in mats_to_sim.toml."
    )(function)
    return click.argument(
        "material",
        nargs=-1,
        metavar="[MATERIAL]...",
        shell_complete=_cli_completion.complete_material,
    )(function)


@command.command(
    "rebrem",
    help="Recompute brem-only remotely, follow, and pull completed checkpoints.",
)
@_recompute_options
@fidelity_option(help="Fidelity preset supplying omitted grid and electron defaults.")
@click.option("--ne-brem", type=POSITIVE_INT, default=None, help="New brem electron count.")
@click.option("--start", type=NONNEGATIVE_FLOAT, default=None, help="Brem lower bound in eV.")
@click.option("--stop", type=POSITIVE_FLOAT, default=None, help="Brem exclusive upper bound in eV.")
@click.option("--step", type=POSITIVE_FLOAT, default=None, help="Wide-brem grid spacing in eV.")
def rebrem_command(
    material,
    all_,
    fidelity,
    redo_all,
    dry_run,
    no_sync,
    chunk_minutes,
    ne_brem,
    start,
    stop,
    step,
):
    materials = list(material)
    _reject_all_with_values("rebrem", all_, materials)
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("rebrem --stop must be greater than --start")
    return _invoke_click(
        _cli_rebrem,
        _click_args(
            "rebrem",
            material=materials,
            all=all_,
            fidelity=fidelity,
            ne_brem=ne_brem,
            start=start,
            stop=stop,
            step=step,
            redo_all=redo_all,
            chunk_minutes=chunk_minutes,
            no_sync=no_sync,
            dry_run=dry_run,
        ),
    )


@command.command(
    "reline",
    help="Recompute line-only remotely, follow, and pull completed checkpoints.",
)
@_recompute_options
@fidelity_option(help="Fidelity preset supplying omitted grid and electron defaults.")
@click.option("--line-ne", type=POSITIVE_INT, default=None, help="New line electron count.")
@click.option("--start", type=NONNEGATIVE_FLOAT, default=None, help="Line lower bound in eV.")
@click.option("--stop", type=POSITIVE_FLOAT, default=None, help="Line exclusive upper bound in eV.")
@click.option(
    "--line-step",
    type=POSITIVE_FLOAT,
    default=None,
    help="Explicit uniform line-grid spacing in eV.",
)
def reline_command(
    material,
    all_,
    fidelity,
    redo_all,
    dry_run,
    no_sync,
    chunk_minutes,
    line_ne,
    start,
    stop,
    line_step,
):
    materials = list(material)
    _reject_all_with_values("reline", all_, materials)
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("reline --stop must be greater than --start")
    return _invoke_click(
        _cli_reline,
        _click_args(
            "reline",
            material=materials,
            all=all_,
            fidelity=fidelity,
            line_ne=line_ne,
            start=start,
            stop=stop,
            line_step=line_step,
            redo_all=redo_all,
            chunk_minutes=chunk_minutes,
            no_sync=no_sync,
            dry_run=dry_run,
        ),
    )


@click.command(
    "submit",
    help=(
        "Sync code, submit sweep(s), track progress, and pull checkpoints.\n\n"
        "Use --headless to return after submission. Use --no-pull to track "
        "through completion without automatically pulling checkpoints.\n\n"
        "PROFILE selects the catalog campaign and its material membership. "
        "Use -m/--material to run one member only. Profiles without an explicit "
        "membership use mats_to_sim.toml's verified materials list.\n\n"
        "A profile submit names the job after PROFILE (NAME, then "
        "NAME-2 once a finished run holds the bare name) and refuses while "
        "another job under the same profile is live."
    ),
)
@click.argument(
    "catalog_profile",
    required=False,
    metavar="[PROFILE]",
    shell_complete=_cli_completion.complete_profile,
)
@click.option(
    "-m",
    "--material",
    shell_complete=_cli_completion.complete_material,
    help="Run one material from PROFILE instead of its full membership.",
)
@click.option(
    "-a", "--all", "all_", is_flag=True, help="Queue mats_to_sim.toml's verified `materials` list."
)
@click.option(
    "-A",
    "--actually-all",
    "actually_all",
    is_flag=True,
    help=(
        "Queue every material in mats_to_sim.toml -- materials, no_verified_dw, "
        "high_energy_materials, and materials_to_leave_out combined. Not "
        "combined with --all/--include-unverified-dw/--include-high-energy or "
        "explicit materials."
    ),
)
@click.option(
    "--include-unverified-dw",
    is_flag=True,
    help="With --all, also queue mats_to_sim.toml's no_verified_dw materials.",
)
@click.option(
    "--include-high-energy",
    is_flag=True,
    help=(
        "With --all, also queue mats_to_sim.toml's high_energy_materials, "
        "filtered to --high-energy-min-kev and above."
    ),
)
@click.option(
    "--high-energy-min-kev",
    type=POSITIVE_FLOAT,
    default=None,
    metavar="KEV",
    help=(
        "Energy floor applied to any queued high_energy_materials member "
        "[default: 150.0 when selected via --include-high-energy/-A]. With "
        "explicit MATERIAL(s), applies only to those that are themselves "
        "high_energy_materials entries; a no-op on every other material."
    ),
)
@fidelity_option(help="Named settings/grid policy. survey is provisional and reduced.")
@click.option(
    "--profile",
    "legacy_catalog_profile",
    default=None,
    hidden=True,
    help="Deprecated compatibility form for positional PROFILE.",
)
@click.option("--quick", is_flag=True, help="Use tiny smoke-test grid.")
@click.option(
    "--workers",
    type=NONNEGATIVE_INT,
    default=None,
    help="Transport workers (default: auto; 0 runs serially).",
)
@click.option(
    "--parallel-materials",
    type=click.IntRange(1, config.MAX_PARALLEL_MATERIALS),
    default=None,
    metavar="N",
    help="Simultaneous scans in one allocation; requires --chunk-minutes 0.",
    shell_complete=_cli_completion.choice_completer(range(1, config.MAX_PARALLEL_MATERIALS + 1)),
)
@click.option(
    "--chunk-minutes",
    type=NONNEGATIVE_FLOAT,
    default=10.0,
    show_default=True,
    help="Self-resubmitting SLURM slice length; 0 runs one monolithic job.",
)
@click.option(
    "-p",
    "--perf",
    is_flag=True,
    help=(
        "Log CPU pressure, RAM/swap, GPU clocks/VRAM, process, phase timing, "
        "queue, worker, chunk, and case metrics for PROFILE."
    ),
)
@click.option(
    "--performance-repetitions",
    type=click.IntRange(1, 20),
    default=1,
    show_default=True,
    metavar="N",
    help=(
        "Run N uncached sessions per material with isolated job-local checkpoints; "
        "requires --perf and --chunk-minutes 0. Profiling checkpoints "
        "are not pulled."
    ),
)
@click.option(
    "--performance-interval",
    type=POSITIVE_FLOAT,
    default=5.0,
    show_default=True,
    metavar="SECONDS",
    help="Performance telemetry sampling interval; requires --perf.",
)
@click.option(
    "--spec-chunk",
    type=POSITIVE_INT,
    default=None,
    metavar="N",
    help="Pin line-spectrum segments per GPU chunk; requires --perf.",
)
@click.option(
    "--brem-chunk",
    type=POSITIVE_INT,
    default=None,
    metavar="N",
    help="Pin bremsstrahlung segments per GPU chunk; requires --perf.",
)
@click.option(
    "--nsys",
    is_flag=True,
    help=(
        "Capture one uncached full-profile session with Nsight Systems CUDA/NVTX "
        "and Python-stack tracing; requires exactly one material, --perf, "
        "--performance-repetitions 1, and --chunk-minutes 0."
    ),
)
@click.option("--no-sync", is_flag=True, help="Skip code upload.")
@click.option("--dry-run", is_flag=True, help="Print submission preview; do not connect.")
@click.option(
    "--headless",
    is_flag=True,
    help="Return after submission without attaching or pulling.",
)
@click.option(
    "--no-pull",
    is_flag=True,
    help="Attach and track, but do not pull completed checkpoints.",
)
@click.option(
    "--grid",
    is_flag=True,
    help="Grid-filter checkpoint before pulling; incompatible with --quick.",
)
@click.option("--drop-wide-brem", is_flag=True, help="With --grid, drop wide-brem.")
@click.option("--downcast", is_flag=True, help="With --grid, downcast to float32.")
@click.option("-f", "--follow", is_flag=True, hidden=True)
def start_command(
    catalog_profile,
    material,
    all_,
    actually_all,
    include_unverified_dw,
    include_high_energy,
    high_energy_min_kev,
    fidelity,
    legacy_catalog_profile,
    quick,
    workers,
    parallel_materials,
    chunk_minutes,
    perf,
    performance_repetitions,
    performance_interval,
    spec_chunk,
    brem_chunk,
    nsys,
    no_sync,
    dry_run,
    headless,
    no_pull,
    grid,
    drop_wide_brem,
    downcast,
    follow,
):
    if catalog_profile is not None and legacy_catalog_profile is not None:
        raise click.UsageError("PROFILE and --profile cannot be combined")
    catalog_profile = catalog_profile or legacy_catalog_profile
    missing_selection = (
        catalog_profile is None and material is None and not all_ and not actually_all
    )
    catalog_profile = catalog_profile or "standard"
    materials = [material] if material is not None else []
    performance_profile = catalog_profile if perf else None
    if performance_profile is None:
        if performance_repetitions != 1:
            raise click.UsageError("--performance-repetitions requires --perf")
        if performance_interval != 5.0:
            raise click.UsageError("--performance-interval requires --perf")
        if spec_chunk is not None:
            raise click.UsageError("--spec-chunk requires --perf")
        if brem_chunk is not None:
            raise click.UsageError("--brem-chunk requires --perf")
        if nsys:
            raise click.UsageError("--nsys requires --perf")
    if performance_repetitions > 1 and chunk_minutes != 0:
        raise click.UsageError("--performance-repetitions requires --chunk-minutes 0")
    if performance_repetitions > 1 and parallel_materials not in (None, 1):
        raise click.UsageError("--performance-repetitions requires one material process per GPU")
    if nsys and chunk_minutes != 0:
        raise click.UsageError("--nsys requires --chunk-minutes 0")
    if nsys and performance_repetitions != 1:
        raise click.UsageError("--nsys requires --performance-repetitions 1")
    if nsys and parallel_materials not in (None, 1):
        raise click.UsageError("--nsys requires one material process per GPU")
    if actually_all and materials:
        raise click.UsageError("start -A/--actually-all does not take material names")
    if actually_all and all_:
        raise click.UsageError("start -A/--actually-all already includes --all; drop --all")
    if actually_all and include_unverified_dw:
        raise click.UsageError("start -A/--actually-all already includes --include-unverified-dw")
    if actually_all and include_high_energy:
        raise click.UsageError("start -A/--actually-all already includes --include-high-energy")
    if include_unverified_dw and not all_:
        raise click.UsageError("--include-unverified-dw requires --all")
    if include_high_energy and not all_:
        raise click.UsageError("--include-high-energy requires --all")
    if missing_selection:
        raise click.UsageError("submit needs PROFILE, -m/--material, --all, or -A")
    if not actually_all:
        if all_ and materials:
            raise click.UsageError("start --all does not take material names")
    if parallel_materials is not None and chunk_minutes != 0:
        raise click.UsageError("--parallel-materials requires --chunk-minutes 0")
    if quick and fidelity != "full":
        raise click.UsageError("--quick cannot be combined with --fidelity survey")
    if quick and grid:
        raise click.UsageError(
            "submit --quick --grid: quick checkpoints aren't grid-filterable; drop --grid"
        )
    if headless and no_pull:
        raise click.UsageError("--headless cannot be combined with --no-pull")
    if headless and follow:
        raise click.UsageError("--headless cannot be combined with --follow")
    return _invoke_click(
        _cli_start,
        _click_args(
            "start",
            materials=materials,
            all=all_,
            actually_all=actually_all,
            include_unverified_dw=include_unverified_dw,
            include_high_energy=include_high_energy,
            high_energy_min_kev=high_energy_min_kev,
            fidelity=fidelity,
            catalog_profile=catalog_profile,
            quick=quick,
            workers=workers,
            parallel_materials=parallel_materials,
            chunk_minutes=chunk_minutes,
            performance_profile=performance_profile,
            performance_repetitions=performance_repetitions,
            performance_interval=performance_interval,
            spec_chunk=spec_chunk,
            brem_chunk=brem_chunk,
            nsys=nsys,
            no_sync=no_sync,
            dry_run=dry_run,
            headless=headless,
            no_pull=no_pull,
            grid=grid,
            drop_wide_brem=drop_wide_brem,
            downcast=downcast,
            follow=follow,
        ),
    )


command.add_command(start_command)
_scan_alias = copy(start_command)
_scan_alias.name = "scan"
_scan_alias.deprecated = "Use 'cxr remote submit'."
_scan_alias.help = "Deprecated alias for `cxr remote submit`.\n\n" + (start_command.help or "")
_scan_alias.params = [copy(parameter) for parameter in start_command.params]
command.add_command(_scan_alias)

_start_alias = copy(start_command)
_start_alias.name = "start"
_start_alias.hidden = True
_start_alias.params = [copy(parameter) for parameter in start_command.params]
for _parameter in _start_alias.params:
    if isinstance(_parameter, click.Option) and "--follow" in _parameter.opts:
        _parameter.hidden = False
        _parameter.help = (
            "Compatibility flag: track after launching. `submit` now tracks by default; "
            "use --headless to detach."
        )
command.add_command(_start_alias)


@command.command("attach", help="Live-track a remote job; defaults to latest.")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Add allocation detail; repeat for recent logs.",
)
def attach_command(jobid, verbose):
    return _invoke_click(_cli_attach, _click_args("attach", jobid=jobid, verbose=verbose))


@command.command("jobs", help="List jobs with SLURM IDs, materials, and last events.")
@click.option("--json", "json_output", is_flag=True, help="Emit one versioned JSON object.")
def jobs_command(json_output):
    return _invoke_click(_cli_jobs, _click_args("jobs", json_output=json_output))


@command.command("status", help="Show one job; use -v for allocation and -vv for logs.")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Add allocation detail; repeat for case progress and recent logs.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit one versioned JSON object.")
def status_command(jobid, verbose, json_output):
    return _invoke_click(
        _cli_status,
        _click_args("status", jobid=jobid, verbose=verbose, json_output=json_output),
    )


@command.command("logs", help="Show a job diagnostic log; defaults to latest.")
@click.argument(
    "jobid",
    required=False,
    metavar="[JOBID]",
    shell_complete=_cli_completion.complete_job_id,
)
@click.option("-f", "--follow", is_flag=True, help="Stream live; Ctrl-C disconnects viewer.")
def logs_command(jobid, follow):
    return _invoke_click(_cli_logs, _click_args("logs", jobid=jobid, follow=follow))


@command.group("profile", help="Manage named compute-performance logs.")
def profile_command():
    pass


@profile_command.command(
    "pull",
    help=(
        "Fetch NDJSON logs and Nsight artifacts for PERFORMANCE_PROFILE from every "
        "matching remote job into performance-profiles/PROFILE/<job>/."
    ),
)
@click.argument(
    "profile",
    callback=_performance_profile_name,
    metavar="PERFORMANCE_PROFILE",
)
def profile_pull_command(profile):
    return _invoke_click(
        _cli_profile_pull,
        _click_args("profile pull", profile=profile),
    )


@command.command("stop", help="cancel active SLURM job(s) by material, profile, or every live job.")
@click.argument("materials", nargs=-1, metavar="[MATERIAL]...")
@click.option("-a", "--all", "all_", is_flag=True, help="Stop every live job.")
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    help="Stop live job(s) submitted with this catalog profile.",
)
@click.option("--yes", is_flag=True, help="Cancel exact previewed jobs; otherwise preview.")
def stop_command(materials, all_, catalog_profile, yes):
    if all_ and materials:
        raise click.UsageError("stop --all does not take material names")
    if catalog_profile is not None and (materials or all_):
        raise click.UsageError("stop --profile does not take material names or --all")
    if not all_ and not materials and catalog_profile is None:
        raise click.UsageError("stop needs material name(s), --profile, or --all")
    return _invoke_click(
        _cli_stop,
        _click_args(
            "stop", materials=list(materials), all=all_, catalog_profile=catalog_profile, yes=yes
        ),
    )


@command.command(
    "reap",
    help="Release orphaned checkpoint reservations; preview unless --yes.",
)
@click.option(
    "--min-age-minutes",
    type=NONNEGATIVE_FLOAT,
    default=5.0,
    show_default=True,
    help="Only reap locks at least this old.",
)
@click.option("--yes", is_flag=True, help="Release reservations; otherwise preview.")
def reap_command(min_age_minutes, yes):
    return _invoke_click(
        _cli_reap,
        _click_args("reap", min_age_minutes=min_age_minutes, yes=yes),
    )


@command.command(
    "pull",
    help=(
        "Fetch existing checkpoints from remote box.\n\n"
        "STEM is usually a bare material name, but MATERIAL@PROFILE selects the "
        "checkpoint the box produced for that catalog profile (--profile on "
        "`cxr remote submit`) -- on-disk names never carry the profile, so this "
        "reads each candidate's meta.json remotely and pulls the newest match; "
        "--hash pins a specific parameter-hash prefix when more than one exists."
    ),
)
@click.argument(
    "material",
    nargs=-1,
    metavar="[STEM|MATERIAL@PROFILE]...",
    shell_complete=_cli_completion.complete_remote_checkpoint_stem,
)
@click.option("-a", "--all", "all_", is_flag=True, help="Pull every configured material.")
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    help=(
        "Pull each member material's checkpoint for this catalog profile "
        "(MATERIAL@PROFILE for every member; with explicit MATERIALs, "
        "qualifies just those). Mutually exclusive with --all."
    ),
)
@click.option(
    "--hash",
    "hash_prefix",
    default=None,
    metavar="HEXPREFIX",
    help="Pin one MATERIAL@PROFILE selector to a parameter-hash prefix.",
)
@click.option(
    "-f",
    "--full",
    "full_",
    is_flag=True,
    help="Pull full unfiltered checkpoint (default: grid-filtered).",
)
@click.option("--drop-wide-brem", is_flag=True, help="With grid pull, drop wide-brem.")
@click.option("--downcast", is_flag=True, help="With grid pull, downcast to float32.")
@click.option("--level9", is_flag=True, help="Recompress remotely at gzip level 9.")
@click.option("--no-sync", is_flag=True, help="With grid pull, skip code sync.")
@click.option(
    "--brem-only",
    is_flag=True,
    help="Merge only brem arrays locally; mutually exclusive with --line-only.",
)
@click.option(
    "--line-only",
    is_flag=True,
    help="Merge only line spectra locally; mutually exclusive with --brem-only.",
)
@click.option(
    "--force",
    is_flag=True,
    help="With partial merge, insert records absent locally.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit one versioned JSON object.")
def pull_command(
    material,
    all_,
    catalog_profile,
    hash_prefix,
    full_,
    drop_wide_brem,
    downcast,
    level9,
    no_sync,
    brem_only,
    line_only,
    force,
    json_output,
):
    materials = list(material)
    if catalog_profile is not None:
        if all_:
            raise click.UsageError(
                "pull --profile already selects the profile's materials; drop --all"
            )
        if any("@" in m for m in materials):
            raise click.UsageError(
                "--profile qualifies bare material names; drop the @PROFILE selector"
            )
        selected = materials or _profile_default_materials(catalog_profile)
        if not selected:
            raise click.UsageError(
                f"profile {catalog_profile!r} has no explicit material membership; "
                "name materials alongside --profile, or use --all"
            )
        materials = [f"{m}@{catalog_profile}" for m in selected]
    _reject_all_with_values("pull", all_, materials)
    if brem_only and line_only:
        raise click.UsageError("--brem-only and --line-only are mutually exclusive")
    if hash_prefix is not None and sum("@" in m for m in materials) != 1:
        raise click.UsageError("--hash requires exactly one MATERIAL@PROFILE selector to pull")
    return _invoke_click(
        _cli_pull_json if json_output else _cli_pull,
        _click_args(
            "pull",
            material=materials,
            all=all_,
            hash_prefix=hash_prefix,
            full=full_,
            drop_wide_brem=drop_wide_brem,
            downcast=downcast,
            level9=level9,
            no_sync=no_sync,
            brem_only=brem_only,
            line_only=line_only,
            force=force,
            json_output=json_output,
        ),
    )


@command.command("clear", help="Delete remote checkpoints; preview unless --yes.")
@click.argument("materials", nargs=-1, metavar="[MATERIAL]...")
@click.option(
    "--all",
    "all_checkpoints",
    is_flag=True,
    help="Empty remote checkpoints directory; takes no material arguments.",
)
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    help="Clear checkpoints belonging to catalog profile NAME.",
)
@click.option("--yes", is_flag=True, help="Delete exact previewed targets; otherwise preview.")
def clear_command(materials, all_checkpoints, catalog_profile, yes):
    if all_checkpoints and materials:
        raise click.UsageError("clear --all takes no material argument")
    if catalog_profile is not None and (all_checkpoints or materials):
        raise click.UsageError("clear --profile takes no material arguments or --all")
    if not all_checkpoints and not materials and catalog_profile is None:
        raise click.UsageError("clear needs material(s), --profile, or --all")
    return _invoke_click(
        _cli_clear,
        _click_args(
            "clear",
            materials=list(materials),
            all_checkpoints=all_checkpoints,
            catalog_profile=catalog_profile,
            yes=yes,
        ),
    )


@command.command(
    "prune",
    help=(
        "Drop remote records obsolete under current scan profiles; preview "
        "unless --yes. Defaults to profile=standard."
    ),
)
@click.option(
    "--all",
    "all_profiles",
    is_flag=True,
    help="Prune current checkpoints for standard and every named catalog profile.",
)
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Prune current full and survey checkpoints for catalog profile NAME.",
)
@click.option("--yes", is_flag=True, help="Delete exact previewed stale records.")
def prune_command(all_profiles, catalog_profile, yes):
    if all_profiles and catalog_profile is not None:
        raise click.UsageError("prune --all cannot be combined with --profile")
    return _invoke_click(
        _cli_prune,
        _click_args(
            "prune",
            all_profiles=all_profiles,
            catalog_profile=catalog_profile,
            yes=yes,
        ),
    )


@command.command(
    "prune-jobs",
    help=(
        "Delete terminal (done/failed/cancelled) job directories; preview "
        "unless --yes. Live jobs are always kept."
    ),
)
@click.option("--all", "all_jobs", is_flag=True, help="Prune every terminal job directory.")
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Prune the NAME / NAME-N job-directory family only.",
)
@click.option("--yes", is_flag=True, help="Delete exact previewed directories.")
def prune_jobs_command(all_jobs, catalog_profile, yes):
    if all_jobs == (catalog_profile is not None):
        raise click.UsageError("prune-jobs needs exactly one of --profile NAME or --all")
    return _invoke_click(
        _cli_prune_jobs,
        _click_args(
            "prune-jobs",
            all_jobs=all_jobs,
            catalog_profile=catalog_profile,
            yes=yes,
        ),
    )


@command.command("sync", help="Push current code to remote box.")
def sync_command():
    return _invoke_click(_cli_sync, _click_args("sync"))


@click.command("validate", help="Run Zhai reproduction remotely or pull existing caches.")
@click.option(
    "--ne",
    type=POSITIVE_INT,
    default=20_000,
    show_default=True,
    help="Fig. 1c line electrons per energy.",
)
@click.option(
    "--ne-brem",
    type=POSITIVE_INT,
    default=200,
    show_default=True,
    help="Fig. 1c bremsstrahlung electrons per energy.",
)
@click.option(
    "--ne-supp",
    type=POSITIVE_INT,
    default=200,
    show_default=True,
    help="Supplementary electrons per polar-tilt spectrum.",
)
@click.option(
    "--tmd-azimuth",
    type=FINITE_FLOAT,
    default=0.0,
    show_default=True,
    help="Exploratory TMD azimuth in degrees.",
)
@click.option("--refresh", is_flag=True, help="Recompute matching cache.")
@click.option("--no-sync", is_flag=True, help="Skip code upload.")
@click.option(
    "-d",
    "--detached",
    is_flag=True,
    help="Launch detached; mutually exclusive with --pull.",
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Track detached job; requires --detached.",
)
@click.option(
    "--pull",
    is_flag=True,
    help="Only fetch existing Zhai caches; mutually exclusive with --detached.",
)
def check_command(ne, ne_brem, ne_supp, tmd_azimuth, refresh, no_sync, detached, follow, pull):
    if follow and not detached:
        raise click.UsageError("--follow requires --detached")
    if pull and detached:
        raise click.UsageError("--pull and --detached are mutually exclusive")
    return _invoke_click(
        _cli_check,
        _click_args(
            "check",
            ne=ne,
            ne_brem=ne_brem,
            ne_supp=ne_supp,
            tmd_azimuth=tmd_azimuth,
            refresh=refresh,
            no_sync=no_sync,
            detached=detached,
            follow=follow,
            pull=pull,
        ),
    )


command.add_command(check_command)
_check_alias = copy(check_command)
_check_alias.name = "check"
_check_alias.hidden = True
command.add_command(_check_alias)


def main(argv=None):
    return run(command, argv, prog_name="cxr-remote")


if __name__ == "__main__":
    raise SystemExit(main())
