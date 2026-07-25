"""Click command group, compatibility wiring, and remote orchestrators."""

import sys

import click

from .._cli_core import (
    FINITE_FLOAT,
    NONNEGATIVE_FLOAT,
    NONNEGATIVE_INT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
    invoke_legacy,
    run,
)
from ..scan import load_all_materials
from . import config, lifecycle, scripts, state, transport, viewer


def remote_scan(material, quick=False, workers=None):
    """Submit one material through SLURM and follow it to completion.

    This compatibility helper deliberately does not pull: callers that need a
    checkpoint can apply their own grid/trim policy after it returns.
    """
    transport._check_materials([material])
    jobid = lifecycle.start_queue([material], quick=quick, workers=workers, chunk_minutes=0)
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
        print("Zhai job is still running or its viewer disconnected; skipping automatic cache pull")
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
        workers=args.workers,
        parallel_materials=getattr(args, "parallel_materials", None),
        chunk_minutes=getattr(args, "chunk_minutes", 10.0),
        no_sync=args.no_sync,
    )
    if not viewer.attach(jobid):
        print("scan is still running or its viewer disconnected; skipping automatic pull")
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        print("warning: the SLURM scan produced no successful checkpoints; nothing to pull")
        return
    stems = scripts._stems(completed, args.quick)
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
            f"\ndone. checkpoints/{stem}.pkl is local; run `cxr analyze {stem}` "
            f"(or run `cxr export`) -- visualization and static-HTML export "
            "stay local."
        )


def _cli_rebrem(args):
    """Submit a brem-only checkpoint recompute, follow it, pull what completed."""
    materials = _selected_materials(args, "material")
    jobid = lifecycle.start_rebrem_queue(
        materials,
        ne_brem=args.ne_brem,
        brem_step_eV=args.step,
        redo_all=args.redo_all,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
        chunk_minutes=args.chunk_minutes,
    )
    if args.dry_run:
        return
    if not viewer.attach(jobid):
        print("rebrem is still running or its viewer disconnected; skipping automatic pull")
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        print("warning: the SLURM rebrem updated no checkpoints successfully; nothing to pull")
        return
    # merge ONLY the fresh brem into the local pickle, preserving any local line spectra
    lifecycle.pull(completed, dataset="brem")


def _cli_reline(args):
    """Submit a line-only checkpoint recompute, follow it, pull what completed."""
    materials = _selected_materials(args, "material")
    jobid = lifecycle.start_reline_queue(
        materials,
        line_ne=args.line_ne,
        line_step_eV=args.line_step,
        redo_all=args.redo_all,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
        chunk_minutes=args.chunk_minutes,
    )
    if args.dry_run:
        return
    if not viewer.attach(jobid):
        print("reline is still running or its viewer disconnected; skipping automatic pull")
        return
    completed = state._completed_materials(jobid, materials)
    if not completed:
        print("warning: the SLURM reline updated no checkpoints successfully; nothing to pull")
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
    )


def _cli_start(args):
    materials = _selected_materials(args, "materials")
    jobid = lifecycle.start_queue(
        materials,
        quick=args.quick,
        workers=args.workers,
        parallel_materials=args.parallel_materials,
        chunk_minutes=args.chunk_minutes,
        no_sync=args.no_sync,
        dry_run=args.dry_run,
    )
    if args.follow and not args.dry_run:
        viewer.attach(jobid)


def _cli_attach(args):
    viewer.attach(args.jobid, args.verbose)


def _cli_jobs(args):
    viewer.list_jobs()


def _cli_status(args):
    viewer.job_status(args.jobid, args.verbose)


def _cli_logs(args):
    return viewer.tail_logs(args.jobid, args.follow)


def _cli_stop(args):
    lifecycle.stop_jobs(args.materials, args.all)


def _cli_reap(args):
    lifecycle.reap_reservations(min_age_minutes=args.min_age_minutes, yes=args.yes)


def _cli_clear(args):
    if args.all_checkpoints:
        if args.materials:
            args._clear_parser.error("clear --all takes no material argument")
        lifecycle.clear_all_remote(args.yes)
        return
    if not args.materials:
        args._clear_parser.error("clear needs material(s), or --all")
    lifecycle.clear_remote(args.materials, args.yes)


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
        "  cxr remote start hopg --dry-run\n"
        "  cxr remote scan hopg\n"
        "  cxr remote status -vv"
    ),
    no_args_is_help=False,
)
def command():
    """Push code and run or manage MC sweeps on a remote GPU box."""
    _ensure_utf8_stdio()


@command.command("scan", help="Sync code, submit sweep(s), wait, and pull checkpoints.")
@click.argument("material", required=False, metavar="[MATERIAL]")
@click.option("-a", "--all", "all_", is_flag=True, help="Run every material in mats_to_sim.toml.")
@click.option(
    "--quick",
    is_flag=True,
    help="Use tiny smoke-test grid; incompatible with --grid.",
)
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
)
@click.option(
    "--chunk-minutes",
    type=NONNEGATIVE_FLOAT,
    default=10.0,
    show_default=True,
    help="Self-resubmitting SLURM slice length; 0 runs one monolithic job.",
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
    quick,
    workers,
    parallel_materials,
    chunk_minutes,
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
    if parallel_materials is not None and chunk_minutes != 0:
        raise click.UsageError("--parallel-materials requires --chunk-minutes 0")
    return _invoke_click(
        _cli_scan,
        _click_args(
            "scan",
            material=material,
            all=all_,
            quick=quick,
            workers=workers,
            parallel_materials=parallel_materials,
            chunk_minutes=chunk_minutes,
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
    return click.argument("material", nargs=-1, metavar="[MATERIAL]...")(function)


@command.command(
    "rebrem",
    help="Recompute brem-only remotely, follow, and pull completed checkpoints.",
)
@_recompute_options
@click.option("--ne-brem", type=POSITIVE_INT, default=None, help="New brem electron count.")
@click.option("--step", type=POSITIVE_FLOAT, default=None, help="Wide-brem grid spacing in eV.")
def rebrem_command(
    material,
    all_,
    redo_all,
    dry_run,
    no_sync,
    chunk_minutes,
    ne_brem,
    step,
):
    materials = list(material)
    _reject_all_with_values("rebrem", all_, materials)
    return _invoke_click(
        _cli_rebrem,
        _click_args(
            "rebrem",
            material=materials,
            all=all_,
            ne_brem=ne_brem,
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
@click.option("--line-ne", type=POSITIVE_INT, default=None, help="New line electron count.")
@click.option(
    "--line-step",
    type=POSITIVE_FLOAT,
    default=None,
    help="Explicit uniform line-grid spacing in eV.",
)
def reline_command(
    material,
    all_,
    redo_all,
    dry_run,
    no_sync,
    chunk_minutes,
    line_ne,
    line_step,
):
    materials = list(material)
    _reject_all_with_values("reline", all_, materials)
    return _invoke_click(
        _cli_reline,
        _click_args(
            "reline",
            material=materials,
            all=all_,
            line_ne=line_ne,
            line_step=line_step,
            redo_all=redo_all,
            chunk_minutes=chunk_minutes,
            no_sync=no_sync,
            dry_run=dry_run,
        ),
    )


@command.command("start", help="Sync code and submit a detached SLURM material queue.")
@click.argument("materials", nargs=-1, metavar="[MATERIAL]...")
@click.option("-a", "--all", "all_", is_flag=True, help="Queue every configured material.")
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
)
@click.option(
    "--chunk-minutes",
    type=NONNEGATIVE_FLOAT,
    default=10.0,
    show_default=True,
    help="Self-resubmitting SLURM slice length; 0 runs one monolithic job.",
)
@click.option("--no-sync", is_flag=True, help="Skip code upload.")
@click.option("--dry-run", is_flag=True, help="Print submission preview; do not connect.")
@click.option("-f", "--follow", is_flag=True, help="Track job after launch.")
def start_command(
    materials,
    all_,
    quick,
    workers,
    parallel_materials,
    chunk_minutes,
    no_sync,
    dry_run,
    follow,
):
    materials = list(materials)
    _reject_all_with_values("start", all_, materials)
    if parallel_materials is not None and chunk_minutes != 0:
        raise click.UsageError("--parallel-materials requires --chunk-minutes 0")
    return _invoke_click(
        _cli_start,
        _click_args(
            "start",
            materials=materials,
            all=all_,
            quick=quick,
            workers=workers,
            parallel_materials=parallel_materials,
            chunk_minutes=chunk_minutes,
            no_sync=no_sync,
            dry_run=dry_run,
            follow=follow,
        ),
    )


@command.command("attach", help="Live-track a remote job; defaults to latest.")
@click.argument("jobid", required=False, metavar="[JOBID]")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Add allocation detail; repeat for recent logs.",
)
def attach_command(jobid, verbose):
    return _invoke_click(_cli_attach, _click_args("attach", jobid=jobid, verbose=verbose))


@command.command("jobs", help="List jobs with SLURM IDs, materials, and last events.")
def jobs_command():
    return _invoke_click(_cli_jobs, _click_args("jobs"))


@command.command("status", help="Show one job; use -v for allocation and -vv for logs.")
@click.argument("jobid", required=False, metavar="[JOBID]")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Add allocation detail; repeat for case progress and recent logs.",
)
def status_command(jobid, verbose):
    return _invoke_click(_cli_status, _click_args("status", jobid=jobid, verbose=verbose))


@command.command("logs", help="Show a job diagnostic log; defaults to latest.")
@click.argument("jobid", required=False, metavar="[JOBID]")
@click.option("-f", "--follow", is_flag=True, help="Stream live; Ctrl-C disconnects viewer.")
def logs_command(jobid, follow):
    return _invoke_click(_cli_logs, _click_args("logs", jobid=jobid, follow=follow))


@command.command("stop", help="Cancel active SLURM job(s) by material, or every live job.")
@click.argument("materials", nargs=-1, metavar="[MATERIAL]...")
@click.option("-a", "--all", "all_", is_flag=True, help="Stop every live job.")
def stop_command(materials, all_):
    if all_ and materials:
        raise click.UsageError("stop --all does not take material names")
    if not all_ and not materials:
        raise click.UsageError("stop needs material name(s), or use --all")
    return _invoke_click(
        _cli_stop,
        _click_args("stop", materials=list(materials), all=all_),
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


@command.command("pull", help="Fetch existing checkpoints from remote box.")
@click.argument("material", nargs=-1, metavar="[STEM]...")
@click.option("-a", "--all", "all_", is_flag=True, help="Pull every configured material.")
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
def pull_command(
    material,
    all_,
    full_,
    drop_wide_brem,
    downcast,
    level9,
    no_sync,
    brem_only,
    line_only,
    force,
):
    materials = list(material)
    _reject_all_with_values("pull", all_, materials)
    if brem_only and line_only:
        raise click.UsageError("--brem-only and --line-only are mutually exclusive")
    return _invoke_click(
        _cli_pull,
        _click_args(
            "pull",
            material=materials,
            all=all_,
            full=full_,
            drop_wide_brem=drop_wide_brem,
            downcast=downcast,
            level9=level9,
            no_sync=no_sync,
            brem_only=brem_only,
            line_only=line_only,
            force=force,
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
@click.option("--yes", is_flag=True, help="Delete exact previewed targets; otherwise preview.")
def clear_command(materials, all_checkpoints, yes):
    if all_checkpoints and materials:
        raise click.UsageError("clear --all takes no material argument")
    if not all_checkpoints and not materials:
        raise click.UsageError("clear needs material(s), or --all")
    return _invoke_click(
        _cli_clear,
        _click_args(
            "clear",
            materials=list(materials),
            all_checkpoints=all_checkpoints,
            yes=yes,
        ),
    )


@command.command("sync", help="Push current code to remote box.")
def sync_command():
    return _invoke_click(_cli_sync, _click_args("sync"))


@command.command("check", help="Run Zhai reproduction remotely or pull existing caches.")
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


def main(argv=None):
    return run(command, argv, prog_name="cxr-remote")


if __name__ == "__main__":
    raise SystemExit(main())
