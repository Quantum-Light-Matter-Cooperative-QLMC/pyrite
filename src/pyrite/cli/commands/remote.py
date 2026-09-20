"""Click command group for remote orchestration."""

from dataclasses import dataclass

import click

from ...console.output import (
    FINITE_FLOAT,
    NONNEGATIVE_FLOAT,
    NONNEGATIVE_INT,
    POSITIVE_FLOAT,
    POSITIVE_INT,
    CLIError,
    emit_diagnostic,
    fidelity_option,
    output_option,
    run,
)
from ...remote import config, lifecycle, transport
from .. import _completion as _cli_completion
from .._deprecations import DeprecatingGroup
from ._remote_actions import (
    _cli_check,
    _cli_clear,
    _cli_performance_list,
    _cli_performance_prune,
    _cli_profile_pull,
    _cli_prune,
    _cli_prune_jobs,
    _cli_pull,
    _cli_pull_json,
    _cli_reap,
    _cli_rebrem,
    _cli_reline,
    _cli_start,
    _cli_sync,
    _ensure_utf8_stdio,
    _performance_profile_name,
    _profile_selected_materials,
    _selected_materials,  # noqa: F401 - compatibility export
    remote_check,  # noqa: F401 - compatibility export
    remote_scan,  # noqa: F401 - compatibility export
)


def _invoke_action(handler, **values):
    """Invoke one explicitly-typed remote action under the Click exit contract."""
    _ensure_utf8_stdio()
    ssh_verbose = values.pop("ssh_verbose", False)
    with transport.verbose_ssh(ssh_verbose):
        status = handler(**values)
    if isinstance(status, int) and not isinstance(status, bool) and status:
        raise click.exceptions.Exit(status)
    return status


def _verbose_option(function):
    """Shared ``-v/--verbose`` flag for commands that ship work over ssh/scp."""
    return click.option(
        "-v",
        "--verbose",
        "ssh_verbose",
        is_flag=True,
        help="Print raw ssh/scp commands instead of a status line.",
    )(function)


def _option_group(*options):
    """Bundle Click decorators into one, preserving their declared help order."""

    def decorate(function):
        for option in reversed(options):
            function = option(function)
        return function

    return decorate


def _reject_all_with_values(command_name, all_, values):
    if all_ and values:
        raise click.UsageError(f"{command_name} --all does not take material names")
    if not all_ and not values:
        raise click.UsageError(f"{command_name} needs material name(s), or use --all")


@click.group(
    "remote",
    cls=DeprecatingGroup,
    help=(
        "[dev] Push code and run or manage MC sweeps on a remote GPU box over SSH.\n\n"
        "Host, remote directory, and executable come from PYRITE_REMOTE_HOST, "
        "PYRITE_REMOTE_DIR, and PYRITE_REMOTE_UV. Command-line options take precedence over workflow defaults "
        "where offered.\n\n"
        "\b\n"
        "Examples:\n"
        "  pyrite run sub_100keV --remote --dry-run\n"
        "  pyrite run compute_test_300keV --remote -p\n"
        "  pyrite run standard -m hopg --remote\n"
        "  pyrite job status -vv"
    ),
    no_args_is_help=False,
)
def command():
    """Push code and run or manage MC sweeps on a remote GPU box."""
    _ensure_utf8_stdio()


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
        "-a", "--all", "all_", is_flag=True, help="Use every standard-profile material."
    )(function)
    return click.argument(
        "material",
        nargs=-1,
        metavar="[MATERIAL]...",
        shell_complete=_cli_completion.complete_material,
    )(function)


@click.command(
    "rebrem",
    help="Recompute brem-only remotely, follow, and pull completed checkpoints.",
)
@_recompute_options
@fidelity_option(help="Fidelity preset supplying omitted grid and electron defaults.")
@click.option("--ne-brem", type=POSITIVE_INT, default=None, help="New brem electron count.")
@click.option("--start", type=NONNEGATIVE_FLOAT, default=None, help="Brem lower bound in eV.")
@click.option("--stop", type=POSITIVE_FLOAT, default=None, help="Brem exclusive upper bound in eV.")
@click.option("--step", type=POSITIVE_FLOAT, default=None, help="Wide-brem grid spacing in eV.")
@click.option("--detach", is_flag=True, help="Return after remote submission.")
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
    detach,
):
    materials = list(material)
    _reject_all_with_values("rebrem", all_, materials)
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("rebrem --stop must be greater than --start")
    return _invoke_action(
        _cli_rebrem,
        material=materials,
        all_=all_,
        fidelity=fidelity,
        ne_brem=ne_brem,
        start=start,
        stop=stop,
        step=step,
        redo_all=redo_all,
        chunk_minutes=chunk_minutes,
        no_sync=no_sync,
        dry_run=dry_run,
        detach=detach,
    )


@click.command(
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
@click.option("--detach", is_flag=True, help="Return after remote submission.")
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
    detach,
):
    materials = list(material)
    _reject_all_with_values("reline", all_, materials)
    if start is not None and stop is not None and stop <= start:
        raise click.UsageError("reline --stop must be greater than --start")
    return _invoke_action(
        _cli_reline,
        material=materials,
        all_=all_,
        fidelity=fidelity,
        line_ne=line_ne,
        start=start,
        stop=stop,
        line_step=line_step,
        redo_all=redo_all,
        chunk_minutes=chunk_minutes,
        no_sync=no_sync,
        dry_run=dry_run,
        detach=detach,
    )


@dataclass(frozen=True)
class _StartFlags:
    """Raw ``remote run`` option values, before compatibility resolution."""

    catalog_profile: str | None
    material: str | None
    fidelity: str
    quick: bool
    workers: int | None
    parallel_materials: int | None
    chunk_minutes: float | None
    perf: bool
    performance_repetitions: int
    performance_interval: float
    spec_chunk: int | None
    brem_chunk: int | None
    nsys: bool
    cpu: bool
    cpu_only: bool
    no_cache: bool
    recompute: bool
    no_sync: bool
    dry_run: bool
    headless: bool
    no_pull: bool
    grid: bool
    drop_wide_brem: bool
    downcast: bool
    level9: bool


@dataclass(frozen=True)
class _StartPlan:
    """``remote run`` settings derived from one validated flag combination."""

    performance_profile: str | None
    chunk_minutes: float


def _reject_exclusive_start_flags(flags):
    """Reject ``remote run`` flag pairs that cannot describe one session."""
    if flags.cpu and flags.cpu_only:
        raise click.UsageError("--cpu and --cpu-only are mutually exclusive")
    if flags.no_cache and flags.recompute:
        raise click.UsageError("--no-cache and --recompute are mutually exclusive")
    if flags.cpu_only and flags.nsys:
        raise click.UsageError("--cpu-only cannot be combined with --nsys")
    if flags.cpu_only and flags.performance_repetitions != 1:
        raise click.UsageError("--cpu-only cannot be combined with --perf-reps")
    if flags.cpu_only and flags.performance_interval != 5.0:
        raise click.UsageError("--cpu-only cannot be combined with --perf-interval")
    if flags.cpu_only and (flags.spec_chunk is not None or flags.brem_chunk is not None):
        raise click.UsageError("--cpu-only cannot be combined with GPU chunk pins")


def _resolve_performance_profile(flags, catalog_profile):
    """Name the profile to instrument, or reject telemetry knobs given without --perf."""
    if flags.perf or flags.nsys or flags.cpu or flags.cpu_only:
        return catalog_profile
    if flags.performance_repetitions != 1:
        raise click.UsageError("--perf-reps requires --perf")
    if flags.performance_interval != 5.0:
        raise click.UsageError("--perf-interval requires --perf")
    if flags.spec_chunk is not None:
        raise click.UsageError("--spec-chunk requires --perf")
    if flags.brem_chunk is not None:
        raise click.UsageError("--brem-chunk requires --perf")
    return None


def _resolve_start_chunk_minutes(flags):
    """Default the SLURM slice length; profiling modes need one monolithic job."""
    if flags.chunk_minutes is not None:
        return flags.chunk_minutes
    monolithic = flags.performance_repetitions > 1 or flags.nsys or flags.cpu or flags.cpu_only
    return 0.0 if monolithic else 10.0


def _reject_start_allocation_conflicts(flags, chunk_minutes):
    """Reject modes that need one unchunked material process per allocation."""
    if flags.performance_repetitions > 1 and chunk_minutes != 0:
        raise click.UsageError("--perf-reps requires --chunk-minutes 0")
    if flags.performance_repetitions > 1 and flags.parallel_materials not in (None, 1):
        raise click.UsageError("--perf-reps requires one material process per GPU")
    if flags.nsys and chunk_minutes != 0:
        raise click.UsageError("--nsys requires --chunk-minutes 0")
    if flags.nsys and flags.performance_repetitions != 1:
        raise click.UsageError("--nsys requires --perf-reps 1")
    if flags.nsys and flags.parallel_materials not in (None, 1):
        raise click.UsageError("--nsys requires one material process per GPU")
    if flags.cpu and chunk_minutes != 0:
        raise click.UsageError("--cpu requires --chunk-minutes 0")
    if flags.cpu_only and chunk_minutes != 0:
        raise click.UsageError("--cpu-only requires --chunk-minutes 0")
    if flags.parallel_materials is not None and chunk_minutes != 0:
        raise click.UsageError("--parallel-materials requires --chunk-minutes 0")


def _reject_start_mode_conflicts(flags):
    """Reject fidelity, grid, and attachment combinations ``remote run`` cannot honour."""
    if flags.quick and flags.fidelity != "full":
        raise click.UsageError("--quick cannot be combined with --fidelity survey")
    if flags.quick and flags.grid:
        raise click.UsageError(
            "run --quick --grid: quick checkpoints aren't grid-filterable; drop --grid"
        )
    if flags.headless and flags.no_pull:
        raise click.UsageError("--headless cannot be combined with --no-pull")


def _plan_start(flags, catalog_profile):
    """Validate one ``remote run`` flag combination and resolve its derived settings.

    Checks run in the order the command has always applied them, so the first
    rule a user violates still decides the ``UsageError`` they see.
    """
    _reject_exclusive_start_flags(flags)
    performance_profile = _resolve_performance_profile(flags, catalog_profile)
    if flags.nsys and flags.material is None:
        raise click.UsageError("--nsys requires one explicit -m/--material")
    chunk_minutes = _resolve_start_chunk_minutes(flags)
    _reject_start_allocation_conflicts(flags, chunk_minutes)
    _reject_start_mode_conflicts(flags)
    return _StartPlan(performance_profile=performance_profile, chunk_minutes=chunk_minutes)


_start_selection_options = _option_group(
    click.argument(
        "catalog_profile",
        required=False,
        default=None,
        metavar="[PROFILE]",
        shell_complete=_cli_completion.complete_profile,
    ),
    click.option(
        "-m",
        "--material",
        shell_complete=_cli_completion.complete_material,
        help="Run one material from PROFILE instead of its full membership.",
    ),
    fidelity_option(help="Named settings/grid policy. survey is provisional and reduced."),
    click.option("--quick", is_flag=True, help="Use tiny smoke-test grid."),
)


_start_allocation_options = _option_group(
    click.option(
        "--workers",
        type=NONNEGATIVE_INT,
        default=None,
        help="Transport workers (default: auto; 0 runs serially).",
    ),
    click.option(
        "--parallel-materials",
        type=click.IntRange(1, config.MAX_PARALLEL_MATERIALS),
        default=None,
        metavar="N",
        help="Simultaneous scans in one allocation; requires --chunk-minutes 0.",
        shell_complete=_cli_completion.choice_completer(
            range(1, config.MAX_PARALLEL_MATERIALS + 1)
        ),
    ),
    click.option(
        "--chunk-minutes",
        type=NONNEGATIVE_FLOAT,
        default=None,
        help=(
            "Self-resubmitting SLURM slice length; defaults to 10, or 0 for "
            "--perf-reps >1, --nsys, and CPU profiling."
        ),
    ),
)


_start_performance_options = _option_group(
    click.option(
        "-p",
        "--perf",
        is_flag=True,
        help=(
            "Log CPU pressure, RAM/swap, GPU clocks/VRAM, process, phase timing, "
            "queue, worker, chunk, and case metrics for PROFILE."
        ),
    ),
    click.option(
        "-r",
        "--perf-reps",
        "performance_repetitions",
        type=click.IntRange(1, 20),
        default=1,
        show_default=True,
        metavar="N",
        help=(
            "Run N uncached sessions per material with isolated job-local checkpoints; "
            "requires --perf and --chunk-minutes 0. Profiling checkpoints "
            "are not pulled."
        ),
    ),
    click.option(
        "-i",
        "--perf-interval",
        "performance_interval",
        type=POSITIVE_FLOAT,
        default=5.0,
        show_default=True,
        metavar="SECONDS",
        help="Performance telemetry sampling interval; requires --perf.",
    ),
    click.option(
        "--spec-chunk",
        type=POSITIVE_INT,
        default=None,
        metavar="N",
        help="Pin line-spectrum segments per GPU chunk; requires --perf.",
    ),
    click.option(
        "--brem-chunk",
        type=POSITIVE_INT,
        default=None,
        metavar="N",
        help="Pin bremsstrahlung segments per GPU chunk; requires --perf.",
    ),
    click.option(
        "--nsys",
        is_flag=True,
        help=(
            "Capture one uncached full-profile session with Nsight Systems CUDA/NVTX "
            "and Python-stack tracing; implies --perf, --perf-reps 1, and "
            "--chunk-minutes 0; requires one explicit -m/--material."
        ),
    ),
    click.option(
        "-c",
        "--cpu",
        is_flag=True,
        help=(
            "After the primary run, capture one bounded serial CPU cProfile pass; "
            "implies --perf and --chunk-minutes 0."
        ),
    ),
    click.option(
        "--cpu-only",
        is_flag=True,
        help=(
            "Capture only the bounded serial CPU cProfile pass; starts no primary "
            "GPU/Nsight scan and implies --perf and --chunk-minutes 0."
        ),
    ),
)


_start_cache_options = _option_group(
    click.option(
        "--no-cache",
        is_flag=True,
        help=(
            "Neither read nor write the shared per-case checkpoint cache: an "
            "ephemeral run that recomputes every case and stores nothing shared."
        ),
    ),
    click.option(
        "--recompute",
        is_flag=True,
        help=(
            "Ignore cached cases and recompute fresh, but repopulate the shared "
            "per-case cache with the results."
        ),
    ),
)


_start_transfer_options = _option_group(
    click.option("--no-sync", is_flag=True, help="Skip code upload."),
    click.option("--dry-run", is_flag=True, help="Print submission preview; do not connect."),
    click.option(
        "--headless",
        is_flag=True,
        help="Return after submission without attaching or pulling.",
    ),
    click.option(
        "--no-pull",
        is_flag=True,
        help="Attach and track, but do not pull completed checkpoints or performance artifacts.",
    ),
    click.option(
        "--grid",
        is_flag=True,
        help="Grid-filter checkpoint before pulling; incompatible with --quick.",
    ),
    click.option("--drop-wide-brem", is_flag=True, help="With --grid, drop wide-brem."),
    click.option("--downcast", is_flag=True, help="With --grid, downcast to float32."),
    click.option(
        "--level9",
        is_flag=True,
        help="Recompress completed checkpoints at max compression before automatic pull.",
    ),
)


@click.command(
    "run",
    help=(
        "Sync code, submit sweep(s), track progress, and pull checkpoints.\n\n"
        "Use --headless to return after submission. Use --no-pull to track "
        "through completion without automatically pulling checkpoints.\n\n"
        "PROFILE selects the catalog campaign and its material membership; when "
        "omitted it uses the current configured profile (standard built-in). Use "
        "-m/--material to run one member only. Profiles without an explicit "
        "membership run every catalog material.\n\n"
        "A profile run names the job after PROFILE (NAME, then "
        "NAME-2 once a finished run holds the bare name) and refuses while "
        "another job under the same profile is live."
    ),
)
@_start_selection_options
@_start_allocation_options
@_start_performance_options
@_start_cache_options
@_start_transfer_options
def start_command(**params):
    from ...console import config as cli_config
    from ...runs.scan import resolve_profile_materials

    flags = _StartFlags(**params)
    try:
        catalog_profile = cli_config.resolve("profile.current", flags.catalog_profile).value
    except cli_config.ConfigError as exc:
        raise CLIError(str(exc)) from exc

    materials = resolve_profile_materials(catalog_profile, flags.material)
    plan = _plan_start(flags, catalog_profile)
    return _invoke_action(
        _cli_start,
        materials=materials,
        fidelity=flags.fidelity,
        catalog_profile=catalog_profile,
        quick=flags.quick,
        workers=flags.workers,
        parallel_materials=flags.parallel_materials,
        chunk_minutes=plan.chunk_minutes,
        performance_profile=plan.performance_profile,
        performance_repetitions=flags.performance_repetitions,
        performance_interval=flags.performance_interval,
        spec_chunk=flags.spec_chunk,
        brem_chunk=flags.brem_chunk,
        nsys=flags.nsys,
        cpu=flags.cpu,
        cpu_only=flags.cpu_only,
        no_cache=flags.no_cache,
        recompute=flags.recompute,
        no_sync=flags.no_sync,
        dry_run=flags.dry_run,
        headless=flags.headless,
        no_pull=flags.no_pull,
        grid=flags.grid,
        drop_wide_brem=flags.drop_wide_brem,
        downcast=flags.downcast,
        level9=flags.level9,
    )


@command.group(
    "performance",
    cls=DeprecatingGroup,
    help="List, pull, or delete remote performance artifacts.",
)
def performance_command():
    pass


@performance_command.command("list", help="List remote performance artifact directories.")
def performance_list_command():
    return _invoke_action(_cli_performance_list)


@performance_command.command(
    "pull",
    help="Fetch one profile's NDJSON, Nsight, and CPU-profile artifacts.",
)
@click.argument("profile", callback=_performance_profile_name, metavar="PERFORMANCE_PROFILE")
@_verbose_option
def performance_pull_command(profile, ssh_verbose):
    return _invoke_action(_cli_profile_pull, profile=profile, ssh_verbose=ssh_verbose)


@performance_command.command(
    "rm",
    help="Delete selected terminal-job performance artifacts; preview by default.",
)
@click.argument(
    "profiles",
    nargs=-1,
    callback=lambda ctx, param, values: tuple(
        _performance_profile_name(ctx, param, value) for value in values
    ),
    metavar="[PROFILE]...",
)
@click.option("--all", "all_profiles", is_flag=True, help="Select every remote profile.")
@click.option("-y", "--yes", is_flag=True, help="Delete exact previewed directories.")
@_verbose_option
def performance_rm_command(profiles, all_profiles, yes, ssh_verbose):
    if all_profiles and profiles:
        raise click.UsageError("remote performance rm --all does not take PROFILE names")
    if not all_profiles and not profiles:
        raise click.UsageError("remote performance rm needs PROFILE name(s), or use --all")
    return _invoke_action(
        _cli_performance_prune,
        profiles=list(profiles),
        all_profiles=all_profiles,
        yes=yes,
        ssh_verbose=ssh_verbose,
    )


@command.command(
    "pull",
    help=(
        "Fetch existing checkpoints from remote box.\n\n"
        "STEM is usually a bare material name, but MATERIAL@PROFILE selects the "
        "checkpoint the box produced for that catalog profile (PROFILE on "
        "`pyrite run --remote`) -- on-disk names never carry the profile, so this "
        "reads each candidate's meta.json remotely and pulls the newest match; "
        "--hash pins a specific parameter-hash prefix when more than one exists."
    ),
)
@click.option(
    "--preset",
    type=click.Choice(("zhai",), case_sensitive=True),
    default=None,
    help="Fetch an existing reproduction cache instead of checkpoints.",
)
@click.argument(
    "material",
    nargs=-1,
    metavar="[PROFILE|STEM|MATERIAL@PROFILE]...",
    shell_complete=_cli_completion.complete_remote_checkpoint_stem,
)
@click.option("-a", "--all", "all_", is_flag=True, help="Pull every configured material.")
@click.option(
    "-m",
    "--material",
    "narrow_materials",
    multiple=True,
    metavar="MATERIAL",
    shell_complete=_cli_completion.complete_material,
    help="Narrow positional PROFILE or --profile to MATERIAL; repeatable.",
)
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    help=(
        "Alias for positional PROFILE. Pull its explicit members, or the full "
        "catalog when membership is implicit; -m/--material narrows it."
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
    help="Default is grid-filtered; pull full unfiltered checkpoint.",
)
@click.option("--drop-wide-brem", is_flag=True, help="With grid pull, drop wide-brem.")
@click.option("--downcast", is_flag=True, help="With grid pull, downcast to float32.")
@click.option("--level9", is_flag=True, help="Recompress remotely at max compression.")
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
@output_option
@_verbose_option
@click.pass_context
def pull_command(
    ctx,
    preset,
    material,
    all_,
    narrow_materials,
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
    ssh_verbose,
):
    materials = list(material)
    narrowed = list(narrow_materials)
    if preset == "zhai":
        incompatible = {
            "material": materials,
            "all": all_,
            "narrow_materials": narrowed,
            "catalog_profile": catalog_profile,
            "hash": hash_prefix,
            "full": full_,
            "drop_wide_brem": drop_wide_brem,
            "downcast": downcast,
            "level9": level9,
            "no_sync": no_sync,
            "brem_only": brem_only,
            "line_only": line_only,
            "force": force,
            "output": (
                ctx.get_parameter_source("json_output") is click.core.ParameterSource.COMMANDLINE
            ),
        }
        selected = [name for name, value in incompatible.items() if value]
        if selected:
            raise click.UsageError(
                "remote pull --preset zhai does not take checkpoint option(s): "
                + ", ".join(selected)
            )
        with transport.verbose_ssh(ssh_verbose):
            return lifecycle.pull_zhai_cache()
    if catalog_profile is None:
        from ...materials import CATALOG

        if materials and materials[0] in CATALOG.profile_names:
            catalog_profile = materials.pop(0)
        elif narrowed and materials:
            if len(materials) != 1:
                raise click.UsageError(
                    "-m/--material takes one positional PROFILE or --profile NAME"
                )
            catalog_profile = materials.pop(0)
        elif narrowed:
            catalog_profile = "standard"
    if catalog_profile is not None:
        if all_:
            emit_diagnostic("warning: pull profile already selects its materials; ignoring --all")
            all_ = False
        if materials and narrowed:
            raise click.UsageError(
                "use positional material names or -m/--material to narrow, not both"
            )
        materials = narrowed or materials
        if any("@" in m for m in materials):
            raise click.UsageError(
                "PROFILE qualifies bare material names; drop the @PROFILE selector"
            )
        selected = _profile_selected_materials(catalog_profile, materials)
        materials = [f"{m}@{catalog_profile}" for m in selected]
    elif narrowed:
        raise click.UsageError("-m/--material requires a PROFILE")
    _reject_all_with_values("pull", all_, materials)
    if brem_only and line_only:
        raise click.UsageError("--brem-only and --line-only are mutually exclusive")
    if hash_prefix is not None and sum("@" in m for m in materials) != 1:
        raise click.UsageError("--hash requires exactly one MATERIAL@PROFILE selector to pull")
    return _invoke_action(
        _cli_pull_json if json_output else _cli_pull,
        remote_command="pull",
        material=materials,
        all_=all_,
        hash_prefix=hash_prefix,
        full=full_,
        drop_wide_brem=drop_wide_brem,
        downcast=downcast,
        level9=level9,
        no_sync=no_sync,
        brem_only=brem_only,
        line_only=line_only,
        force=force,
        ssh_verbose=ssh_verbose,
    )


@command.command("rm", help="Delete remote checkpoints; preview unless --yes.")
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
    help="Delete checkpoints belonging to catalog profile NAME.",
)
@click.option(
    "-y", "--yes", is_flag=True, help="Delete exact previewed targets; otherwise preview."
)
@_verbose_option
def rm_command(materials, all_checkpoints, catalog_profile, yes, ssh_verbose):
    if all_checkpoints and materials:
        raise click.UsageError("rm --all takes no material argument")
    if catalog_profile is not None and (all_checkpoints or materials):
        raise click.UsageError("rm --profile takes no material arguments or --all")
    if not all_checkpoints and not materials and catalog_profile is None:
        raise click.UsageError("rm needs material(s), --profile, or --all")
    return _invoke_action(
        _cli_clear,
        materials=list(materials),
        all_checkpoints=all_checkpoints,
        catalog_profile=catalog_profile,
        yes=yes,
        ssh_verbose=ssh_verbose,
    )


@command.command(
    "gc",
    help=(
        "Reclaim remote records obsolete under current scan profiles and release "
        "orphaned checkpoint reservations; preview unless --yes. Record selection "
        "defaults to profile=standard."
    ),
)
@click.option(
    "--all",
    "all_profiles",
    is_flag=True,
    help="Reclaim records for standard and every named catalog profile.",
)
@click.option(
    "--profile",
    "catalog_profile",
    default=None,
    metavar="NAME",
    shell_complete=_cli_completion.complete_profile,
    help="Reclaim current full and survey records for catalog profile NAME.",
)
@click.option(
    "--min-age-minutes",
    type=NONNEGATIVE_FLOAT,
    default=5.0,
    show_default=True,
    help="Only release reservations at least this old.",
)
@click.option("-y", "--yes", is_flag=True, help="Reclaim exactly what was previewed.")
@_verbose_option
def gc_command(all_profiles, catalog_profile, min_age_minutes, yes, ssh_verbose):
    """Run both reclamations the retired `prune` and `reap` spellings ran separately.

    Each half previews unless ``--yes``, so the combined command keeps the
    preview-then-confirm contract both retired spellings had.
    """
    if all_profiles and catalog_profile is not None:
        raise click.UsageError("gc --all cannot be combined with --profile")
    _invoke_action(
        _cli_prune,
        all_profiles=all_profiles,
        catalog_profile=catalog_profile,
        yes=yes,
        ssh_verbose=ssh_verbose,
    )
    return _invoke_action(
        _cli_reap,
        min_age_minutes=min_age_minutes,
        yes=yes,
        ssh_verbose=ssh_verbose,
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
@click.option("-y", "--yes", is_flag=True, help="Delete exact previewed directories.")
@_verbose_option
def prune_jobs_command(all_jobs, catalog_profile, yes, ssh_verbose):
    if all_jobs == (catalog_profile is not None):
        raise click.UsageError("prune-jobs needs exactly one of --profile NAME or --all")
    return _invoke_action(
        _cli_prune_jobs,
        all_jobs=all_jobs,
        catalog_profile=catalog_profile,
        yes=yes,
        ssh_verbose=ssh_verbose,
    )


@command.command("sync", help="Push current code to remote box.")
@click.option(
    "--force",
    is_flag=True,
    help="Sync even while a live job is running different code.",
)
@_verbose_option
def sync_command(force, ssh_verbose):
    return _invoke_action(_cli_sync, force=force, ssh_verbose=ssh_verbose)


@click.command(
    "validate",
    help="Run Zhai reproduction remotely or pull existing caches.",
)
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
@click.pass_context
def check_command(ctx, ne, ne_brem, ne_supp, tmd_azimuth, refresh, no_sync, detached, follow, pull):
    if follow and not detached:
        raise click.UsageError("--follow requires --detached")
    if pull and detached:
        raise click.UsageError("--pull and --detached are mutually exclusive")
    return _invoke_action(
        _cli_check,
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
        tmd_azimuth=tmd_azimuth,
        refresh=refresh,
        no_sync=no_sync,
        detached=detached,
        follow=follow,
        pull=pull,
    )


def main(argv=None):
    return run(command, argv, prog_name="pyrite-remote")


if __name__ == "__main__":
    raise SystemExit(main())
