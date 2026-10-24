"""Click wiring shared by ``pyrite run`` and ``pyrite-dev perf``."""

import copy
import re
from pathlib import Path

import click

from ..._env import set_canonical_env
from ...console import config as _cli_config
from ...console import output as _cli_core
from ...console.outputs import output_default, output_dir, output_label
from ...runs import scan as _scan
from .. import _completion as _cli_completion
from .. import _implicit_defaults
from .._deprecations import RetiredOption, canonical_option
from .._options import remote_option

_PERFORMANCE_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _performance_profile(ctx, param, value):
    if value is None:
        return None
    if _PERFORMANCE_PROFILE_RE.fullmatch(value) is None:
        raise click.BadParameter(
            "expected letters, digits, underscores, or hyphens; must start with letter or digit",
            ctx=ctx,
            param=param,
        )
    return value


# ``--trajectories`` with no value; NUL can never be a real path.
_BARE_TRAJECTORIES = "\0"


class _TrajectoryPath(click.Path):
    def convert(self, value, param, ctx):
        if value == _BARE_TRAJECTORIES:
            return value
        return super().convert(value, param, ctx)


def _trajectories_dir(ctx, param, value):
    del param
    if value is None:
        return None
    if str(value) == _BARE_TRAJECTORIES:
        return _BARE_TRAJECTORIES
    return value


def _reproduce_zhai(ne, ne_brem, ne_supp, tmd_azimuth, refresh):
    """Populate the Zhai reproduction cache in this process.

    Same workload the box runs as ``python -m pyrite._entry.reproduce_zhai``;
    ``--remote`` queues it there instead. Heavy at the default sample counts.
    """
    import importlib

    anchor_figures = importlib.import_module("pyrite.validation.anchor_figures")
    results = anchor_figures.reproduce_all(
        ne=ne,
        ne_brem=ne_brem,
        ne_supp=ne_supp,
        tmd_exploratory_azimuth_deg=tmd_azimuth,
        refresh=refresh,
    )
    for label, path, cache_hit in results:
        click.echo(anchor_figures.format_reproduction_row(label, path, cache_hit))


@click.command(
    "run",
    help=(
        "Run a catalog profile's MC sweeps or one ephemeral pixel scene.\n\n"
        "PROFILE defaults to PYRITE_PROFILE or the configured profile.current; with "
        "neither set, naming no profile is a usage error. PROFILE "
        "owns material membership, campaign ranges, and workload settings. "
        "Use -m/--material to run one profile member instead of the full resolved "
        "membership.\n\n"
        "Resumes compatible checkpoints in CHECKPOINTS. Full writes component "
        "data beneath <material>/; variants use "
        "identity-qualified stems. With --ephemeral, require -m and singleton "
        "scene/workload grids, run one pixel detector without checkpoint, cache, "
        "or observation-store I/O, and optionally export factorized arrays."
    ),
)
@click.argument(
    "catalog_profile",
    required=False,
    default=None,
    metavar="[PROFILE]",
    shell_complete=_cli_completion.complete_profile,
)
@click.option(
    "-m",
    "--material",
    type=_cli_completion.MATERIAL,
    default=None,
    shell_complete=_cli_completion.complete_material,
    help="Run one member of PROFILE instead of its full membership.",
)
@click.option(
    "--ephemeral",
    is_flag=True,
    help="Run one local pixel scene without checkpoint/cache/observation I/O; requires -m and singleton grids.",
)
@click.option(
    "--detector",
    "detector_id",
    metavar="ID",
    help="Pixel detector ID for --ephemeral; required when several pixel detectors exist.",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    help="With --ephemeral, write full factorized arrays as a new compressed .npz file; never overwrite.",
)
@click.option(
    "--workers",
    type=_cli_core.NONNEGATIVE_INT,
    default=None,
    help="run_cases max_workers (default auto; 0 = serial, no transport pool).",
)
@click.option(
    "--execution",
    default=None,
    metavar="NAME",
    help="Select a named execution profile (built-in: local, slurm-cpu).",
)
@click.option(
    "--execution-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    metavar="TOML",
    help="TOML file containing [execution.NAME] scheduler settings.",
)
@click.option(
    "--execution-plan",
    is_flag=True,
    help="Inspect requested execution resources and task counts without running cases.",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Use a tiny smoke-test grid and write the <material>_quick/ component directory.",
)
@click.option(
    "--n-families",
    type=_cli_core.POSITIVE_INT,
    default=None,
    help="Override positive dominant reflection-family count.",
)
@click.option(
    "--checkpoint-dir",
    default=output_default("checkpoints"),
    show_default=output_label("checkpoints"),
    metavar="DIR",
    help="Read and write component checkpoints in DIR.",
)
@click.option(
    "--max-minutes",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="MINUTES",
    help="Soft wall-clock budget in minutes; exit 75 if resumable work remains.",
)
@click.option(
    "-p",
    "--perf",
    is_flag=True,
    help=(
        "Sample CPU pressure, RAM/swap, GPU clocks/VRAM, process-tree, phase "
        "timing, queue, case, worker, and chunk metrics for PROFILE's resolved "
        "membership into pyrite-output/performance/PROFILE/<material>.ndjson "
        "(cxr.performance.v1). Combine with -m to profile a single member. Runs "
        "without shared-cache reads or writes unless --recompute is explicit."
    ),
)
@click.option(
    "--performance-profile",
    callback=_performance_profile,
    default=None,
    metavar="NAME",
    hidden=True,
    help=(
        "Run catalog profile NAME while sampling CPU pressure, RAM/swap, GPU clocks/"
        "VRAM, process-tree, phase timing, queue, case, worker, and chunk metrics "
        "every 5 s into "
        "pyrite-output/performance/NAME/<material>.ndjson."
    ),
)
@click.option(
    "--performance-dir",
    type=click.Path(path_type=Path),
    default=None,
    hidden=True,
)
@click.option(
    "-i",
    "--perf-interval",
    "performance_interval",
    type=_cli_core.POSITIVE_FLOAT,
    default=5.0,
    show_default=True,
    metavar="SECONDS",
    help="Performance-telemetry sampling interval; requires -p/--perf.",
)
@canonical_option(
    "--cxr-chunk",
    "spec_chunk",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help="Pin line-spectrum segments per GPU chunk; requires -p/--perf.",
)
@click.option(
    "--brem-chunk",
    type=_cli_core.POSITIVE_INT,
    default=None,
    metavar="N",
    help="Pin bremsstrahlung segments per GPU chunk; requires -p/--perf.",
)
@click.option(
    "--nsys",
    is_flag=True,
    help=(
        "Capture one uncached Nsight Systems CUDA/NVTX trace of the run (writes a "
        ".nsys-rep next to the perf log); defaults to the profile's full "
        "membership (-m narrows to one member). Requires -p/--perf."
    ),
)
@click.option(
    "--py-spy",
    "py_spy",
    is_flag=True,
    help=(
        "Sample the whole uncached run with py-spy (every thread and worker "
        "process, including idle waits) and write a .py-spy.json speedscope "
        "profile next to the perf log; exclusive with --nsys. Requires -p/--perf."
    ),
)
@click.option(
    "-c",
    "--cpu",
    is_flag=True,
    help=(
        "After the primary run, capture one bounded serial CPU cProfile pass; "
        "implies --perf. Requires -R/--remote."
    ),
)
@click.option(
    "--cpu-only",
    is_flag=True,
    help=(
        "Capture only the bounded serial CPU cProfile pass; starts no primary "
        "GPU/Nsight scan and implies --perf. Requires -R/--remote."
    ),
)
@click.option(
    "--no-cache",
    is_flag=True,
    help=(
        "Neither read nor write the shared per-case checkpoint cache: a "
        "run that recomputes every case but still writes campaign checkpoints. "
        "Use --ephemeral for no checkpoint or observation-store I/O."
    ),
)
@click.option(
    "--recompute",
    is_flag=True,
    help=(
        "Ignore cached cases and recompute fresh, but repopulate the shared "
        "per-case cache with the results."
    ),
)
@click.option(
    "--trajectories",
    type=_TrajectoryPath(file_okay=False, path_type=Path),
    default=None,
    is_flag=False,
    flag_value=_BARE_TRAJECTORIES,
    callback=_trajectories_dir,
    metavar="[DIR]",
    help=(
        "Opt in to saving each transported case's full electron-transport result "
        "as HDF5 under DIR/<stem>/ (bare flag: pyrite-output/trajectories). "
        "With --remote, DIR is on the host, relative to its checkout; captures stay "
        "remote until `pyrite remote trajectories pull`. Put PROFILE before a bare "
        "--trajectories, or use --trajectories=DIR. Files can "
        "be much larger than checkpoints; cached cases are not re-transported (use "
        "--recompute to capture them)."
    ),
)
@click.option(
    "--overwrite-trajectories",
    is_flag=True,
    help="Replace existing trajectory files for cases this run transports; requires --trajectories.",
)
@click.option("--progress-file", type=click.Path(path_type=Path), default=None, hidden=True)
@click.option(
    "--progress-phase",
    type=click.Choice(("primary", "cpu"), case_sensitive=True),
    default=None,
    hidden=True,
)
@click.option("--no-progress", is_flag=True, help="Disable progress bars/dashboard.")
@click.option("-v", "--verbose", count=True, help="Increase dashboard detail.")
@click.option(
    "--preset",
    type=click.Choice(("zhai",), case_sensitive=True),
    default=None,
    help="Run a named reproduction workflow; zhai runs here unless -R/--remote.",
)
@click.option(
    "--ne",
    type=_cli_core.POSITIVE_INT,
    default=20_000,
    show_default=True,
    help="With --preset zhai, Fig. 1c line electrons per energy.",
)
@canonical_option(
    "--brem-trials",
    "ne_brem",
    type=_cli_core.POSITIVE_INT,
    default=200,
    show_default=True,
    help="With --preset zhai, Monte Carlo electron histories for Fig. 1c bremsstrahlung per energy.",
)
@click.option(
    "--ne-supp",
    type=_cli_core.POSITIVE_INT,
    default=200,
    show_default=True,
    help="With --preset zhai, supplementary electrons per polar-tilt spectrum.",
)
@click.option(
    "--tmd-azimuth",
    type=_cli_core.FINITE_FLOAT,
    default=0.0,
    show_default=True,
    metavar="DEGREES",
    help="With --preset zhai, exploratory TMD azimuth.",
)
@click.option("--refresh", is_flag=True, help="With --preset zhai, recompute matching cache.")
@click.option("--no-sync", is_flag=True, help="With -R/--remote, skip code upload.")
@click.option("--dry-run", is_flag=True, help="With -R/--remote, print submission preview only.")
@remote_option
@click.option("--wait", is_flag=True, help="Wait for remote completion and pull results.")
@click.option("--detach", is_flag=True, help="Return after remote submission.")
@click.option(
    "--chunk-minutes",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="MINUTES",
    help=(
        "With -R/--remote, self-resubmitting SLURM slice length; defaults to 10, "
        "or 0 (one unchunked allocation) for profiling captures."
    ),
)
@click.pass_context
@_cli_core.output_option(wide=True)
def _command(ctx, **options):
    """Click entry point for the staged root migration."""
    if not options["ephemeral"] and (
        options["detector_id"] is not None or options["output_file"] is not None
    ):
        raise click.UsageError("--detector/--output-file require --ephemeral")
    if options["execution_config"] is not None and options["execution"] is None:
        raise click.UsageError("--execution-config requires --execution NAME")
    if options["execution_plan"] and options["execution"] is None:
        raise click.UsageError("--execution-plan requires --execution NAME")
    if options["execution_plan"] and options["json_output"]:
        raise click.UsageError("--execution-plan has its own JSON output; omit --json")
    if options["execution"] is not None and (
        options["ephemeral"]
        or options["remote_target"] is not None
        or options["preset"] is not None
    ):
        raise click.UsageError(
            "--execution is for ordinary sweeps; it cannot be combined with "
            "--ephemeral, -R/--remote, or --preset"
        )
    if options["ephemeral"]:
        return _run_ephemeral(ctx, options)
    raw_catalog_profile = options["catalog_profile"]
    resolved_profile = None
    if options["preset"] is None:
        resolved_profile = _resolve_current_profile(options["catalog_profile"])
        options["catalog_profile"] = resolved_profile.value
    _check_mode_combinations(ctx, options)
    if options["preset"] == "zhai":
        return _run_zhai_preset(ctx, options, raw_catalog_profile)
    if resolved_profile is not None:
        _implicit_defaults.require_explicit_profile(resolved_profile)
    if options["remote_target"] is not None or options["nsys"]:
        _refuse_unstaged_gdf_beam(options["catalog_profile"])
    if options["remote_target"] is not None:
        return _run_remote(ctx, options)
    if resolved_profile is not None:
        _implicit_defaults.require_explicit_instrument(options["catalog_profile"])
    return _run_local(options)


# Parameter sources that count as a user (not a built-in default) choice for
# --ephemeral's allow-list. Other combination checks read COMMANDLINE only.
_USER_PARAMETER_SOURCES = frozenset(
    {
        click.core.ParameterSource.COMMANDLINE,
        click.core.ParameterSource.ENVIRONMENT,
        click.core.ParameterSource.DEFAULT_MAP,
    }
)
_EPHEMERAL_PARAMETER_NAMES = frozenset(
    {
        "catalog_profile",
        "material",
        "ephemeral",
        "detector_id",
        "output_file",
        "json_output",
        "no_progress",
        "verbose",
    }
)
_ZHAI_PARAMETER_FLAGS = {
    "ne": "--ne",
    "ne_brem": "--brem-trials",
    "ne_supp": "--ne-supp",
    "tmd_azimuth": "--tmd-azimuth",
    "refresh": "--refresh",
}
_NORMAL_RUN_PARAMETER_FLAGS = {
    "catalog_profile": "PROFILE",
    "material": "--material",
    "workers": "--workers",
    "quick": "--quick",
    "n_families": "--n-families",
    "checkpoint_dir": "--checkpoint-dir",
    "max_minutes": "--max-minutes",
    "perf": "--perf",
    "performance_profile": "--performance-profile",
    "performance_dir": "--performance-dir",
    "performance_interval": "--perf-interval",
    "spec_chunk": "--cxr-chunk",
    "brem_chunk": "--brem-chunk",
    "nsys": "--nsys",
    "py_spy": "--py-spy",
    "cpu": "--cpu",
    "cpu_only": "--cpu-only",
    "chunk_minutes": "--chunk-minutes",
    "no_cache": "--no-cache",
    "recompute": "--recompute",
    "trajectories": "--trajectories",
    "overwrite_trajectories": "--overwrite-trajectories",
    "progress_file": "--progress-file",
    "progress_phase": "--progress-phase",
    "no_progress": "--no-progress",
    "json_output": "--output",
    "verbose": "--verbose",
}
_LOCAL_ONLY_PARAMETER_FLAGS = {
    "n_families": "--n-families",
    "checkpoint_dir": "--checkpoint-dir",
    "max_minutes": "--max-minutes",
    "performance_profile": "--performance-profile",
    "performance_dir": "--performance-dir",
    "progress_file": "--progress-file",
    "progress_phase": "--progress-phase",
    "no_progress": "--no-progress",
    "verbose": "--verbose",
}


def _explicit_flags(ctx, parameter_flags):
    """Flags in ``parameter_flags`` given on the command line, in table order."""
    return [
        flag
        for parameter, flag in parameter_flags.items()
        if ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
    ]


def _resolve_current_profile(catalog_profile):
    try:
        return _cli_config.resolve("profile.current", catalog_profile)
    except _cli_config.ConfigError as exc:
        raise _cli_core.CLIError(str(exc)) from exc


def _run_ephemeral(ctx, options):
    """``--ephemeral``: one local pixel scene through the simulate pipeline."""
    from ._simulation import execute_simulation, resolve_scene, simulation_usage_error

    # output_option returns a bool; Click retains the original human format
    # in its metadata so wide output can share the simulate presentation.
    output_format = (
        "json" if options["json_output"] else ctx.meta.get("pyrite.output_format", "table")
    )
    incompatible = [
        parameter.opts[0]
        for parameter in ctx.command.params
        if parameter.name not in _EPHEMERAL_PARAMETER_NAMES
        and ctx.get_parameter_source(parameter.name) in _USER_PARAMETER_SOURCES
    ]
    if incompatible:
        simulation_usage_error(
            "--ephemeral does not support option(s): " + ", ".join(incompatible), output_format
        )
    if options["material"] is None:
        simulation_usage_error("--ephemeral requires -m/--material MATERIAL", output_format)
    resolved = _resolve_current_profile(options["catalog_profile"])
    if (error := _implicit_defaults.implicit_profile_error(resolved)) is not None:
        simulation_usage_error(error, output_format)

    def resolver(document, material, profile_name, detector_id):
        return resolve_scene(
            document, material, profile_name, detector_id, command_name="run --ephemeral"
        )

    return execute_simulation(
        options["material"],
        resolved.value,
        options["detector_id"],
        output_format,
        options["output_file"],
        resolver=resolver,
        usage_errors=True,
    )


def _check_remote_only_flags(options):
    remote = options["remote_target"] is not None
    if options["wait"] and options["detach"]:
        raise click.UsageError("--wait and --detach are mutually exclusive")
    if not remote and (options["wait"] or options["detach"]):
        raise click.UsageError("--wait/--detach require -R/--remote")
    if not remote and (options["no_sync"] or options["dry_run"]):
        raise click.UsageError("--no-sync/--dry-run require -R/--remote")
    if not remote and (options["cpu"] or options["cpu_only"]):
        raise click.UsageError("--cpu/--cpu-only require -R/--remote")
    if not remote and options["chunk_minutes"] is not None:
        raise click.UsageError("--chunk-minutes requires -R/--remote")


def _check_mode_combinations(ctx, options):
    """Refuse option pairs invalid in every non-ephemeral mode, before any I/O."""
    _check_remote_only_flags(options)
    if options["no_cache"] and options["recompute"]:
        raise click.UsageError("--no-cache and --recompute are mutually exclusive")
    if options["py_spy"] and options["nsys"]:
        raise click.UsageError("--py-spy and --nsys are mutually exclusive")
    if options["overwrite_trajectories"] and options["trajectories"] is None:
        raise click.UsageError("--overwrite-trajectories requires --trajectories")
    explicit_zhai = _explicit_flags(ctx, _ZHAI_PARAMETER_FLAGS)
    if options["preset"] is None and explicit_zhai:
        raise click.UsageError(f"Zhai option(s) require --preset zhai: {', '.join(explicit_zhai)}")


def _run_zhai_preset(ctx, options, raw_catalog_profile):
    """``--preset zhai``: reproduce locally, or queue it with -R/--remote."""
    explicit_normal = _explicit_flags(ctx, _NORMAL_RUN_PARAMETER_FLAGS)
    if raw_catalog_profile is not None and "PROFILE" not in explicit_normal:
        explicit_normal.insert(0, "PROFILE")
    if explicit_normal:
        raise click.UsageError(
            "--preset zhai does not support normal-run option(s): " + ", ".join(explicit_normal)
        )
    zhai = {name: options[name] for name in _ZHAI_PARAMETER_FLAGS}
    if options["remote_target"] is None:
        return _reproduce_zhai(**zhai)
    from ...remote import config as remote_config
    from . import remote as remote_cli

    with remote_config.override_remote_host(_remote_host(options["remote_target"])):
        return remote_cli.remote_check(
            **zhai,
            no_sync=options["no_sync"],
            detach=options["detach"],
            dry_run=options["dry_run"],
        )


def _remote_host(remote_target):
    return None if remote_target == "__configured__" else remote_target


def _refuse_unstaged_gdf_beam(catalog_profile):
    from ...campaign.config import _catalog

    beam_fields = _catalog(catalog_profile).profile_beam(catalog_profile)
    if beam_fields is not None and beam_fields.get("source") == "gpt_gdf":
        raise click.UsageError(
            "gpt_gdf beams require a local run without --nsys; external GDF file staging is unsupported"
        )


_REMOTE_START_PARAMETER_NAMES = (
    "catalog_profile",
    "material",
    "quick",
    "workers",
    "perf",
    "performance_interval",
    "spec_chunk",
    "brem_chunk",
    "nsys",
    "py_spy",
    "cpu",
    "cpu_only",
    "no_cache",
    "recompute",
    "no_sync",
    "dry_run",
)


def _run_remote(ctx, options):
    """``-R/--remote``: hand the run to ``pyrite remote start`` on the target."""
    if options["json_output"]:
        raise click.UsageError("remote run does not yet support --output json")
    explicit_local = _explicit_flags(ctx, _LOCAL_ONLY_PARAMETER_FLAGS)
    if explicit_local:
        raise click.UsageError(
            f"remote run does not support local-only option(s): {', '.join(explicit_local)}"
        )
    from ...remote import config as remote_config
    from . import remote as remote_cli

    trajectories = options["trajectories"]
    if str(trajectories) == _BARE_TRAJECTORIES:
        trajectories = "pyrite-output/trajectories"
    with remote_config.override_remote_host(_remote_host(options["remote_target"])):
        return ctx.invoke(
            remote_cli.start_command,
            **{name: options[name] for name in _REMOTE_START_PARAMETER_NAMES},
            headless=options["detach"],
            no_pull=False,
            chunk_minutes=options["chunk_minutes"],
            trajectories=trajectories,
            overwrite_trajectories=options["overwrite_trajectories"],
        )


def _local_performance_profile(options):
    """Resolve -p/--perf sugar and refuse perf-only options without a profile."""
    performance_profile = options["performance_profile"]
    if options["perf"] and performance_profile is None:
        # -p is sugar for --performance-profile PROFILE: profile the positional
        # profile's full membership (or one member when -m is given).
        performance_profile = options["catalog_profile"]
    if performance_profile is not None:
        return performance_profile
    if options["performance_interval"] != 5.0:
        raise click.UsageError("--perf-interval requires -p/--perf")
    if options["spec_chunk"] is not None or options["brem_chunk"] is not None:
        raise click.UsageError("--cxr-chunk/--brem-chunk require -p/--perf")
    if options["nsys"]:
        raise click.UsageError("--nsys requires -p/--perf")
    if options["py_spy"]:
        raise click.UsageError("--py-spy requires -p/--perf")
    return None


def _cache_gates(no_cache, recompute, performance_profile):
    """Shared per-case cache gates (see run_sweep's content-addressable store).

    default -> read + write; --recompute -> skip read, still repopulate
    (write); --no-cache -> neither read nor write (ephemeral). A -p/--perf run
    defaults to --no-cache semantics so it measures real compute and never
    pollutes the shared store with a measurement run; an explicit
    --no-cache/--recompute on the same line wins (explicit beats the perf
    default).
    """
    perf_no_cache = performance_profile is not None and not (no_cache or recompute)
    cache_read = not (no_cache or recompute or perf_no_cache)
    cache_write = not (no_cache or perf_no_cache)
    return cache_read, cache_write


_PROFILER_REEXEC_PARAMETER_NAMES = (
    "catalog_profile",
    "material",
    "performance_dir",
    "performance_interval",
    "workers",
    "quick",
    "n_families",
    "max_minutes",
    "no_cache",
    "recompute",
)
_LOCAL_RUN_PARAMETER_NAMES = (
    "material",
    "workers",
    "execution",
    "execution_config",
    "execution_plan",
    "catalog_profile",
    "quick",
    "n_families",
    "checkpoint_dir",
    "max_minutes",
    "performance_dir",
    "performance_interval",
    "progress_file",
    "no_progress",
)


def _run_local(options):
    """Local sweep, optionally re-executed under Nsight Systems or py-spy."""
    performance_profile = _local_performance_profile(options)
    cache_read, cache_write = _cache_gates(
        options["no_cache"], options["recompute"], performance_profile
    )
    # Pin the GPU spectrum/brem chunk before the runtime import so the main
    # process and every spawned transport worker (env inherited on spawn/
    # forkserver) read it. Mirrors the remote job script's `export
    # PYRITE_MC_CXR_CHUNK` / `PYRITE_MC_BREM_CHUNK`.
    if options["spec_chunk"] is not None:
        set_canonical_env("PYRITE_MC_CXR_CHUNK", str(options["spec_chunk"]))
    if options["brem_chunk"] is not None:
        set_canonical_env("PYRITE_MC_BREM_CHUNK", str(options["brem_chunk"]))
    reexec = (
        _scan._reexec_under_nsys
        if options["nsys"]
        else _scan._reexec_under_py_spy
        if options["py_spy"]
        else None
    )
    if reexec is not None and options["execution"] is not None:
        raise click.UsageError("--execution cannot be combined with profiling re-execution")
    if reexec is not None:
        reexec(
            performance_profile=performance_profile,
            **{name: options[name] for name in _PROFILER_REEXEC_PARAMETER_NAMES},
        )
        return None  # os.execvp already replaced the process; defensive.
    trajectories = options["trajectories"]
    if str(trajectories) == _BARE_TRAJECTORIES:
        trajectories = output_dir("trajectories")
    # Trajectory capture is opt-in; ordinary runs keep their exact arguments.
    capture = (
        {}
        if trajectories is None
        else {
            "trajectories": trajectories,
            "overwrite_trajectories": options["overwrite_trajectories"],
        }
    )
    resolved_profile = _scan._resolve_catalog_profile(
        options["catalog_profile"], performance_profile
    )
    _scan.resolve_profile_materials(resolved_profile, options["material"])
    progress_phase = options["progress_phase"]
    run_kwargs = {
        **{name: options[name] for name in _LOCAL_RUN_PARAMETER_NAMES},
        "all": False,
        "actually_all": False,
        "beam_uvw": None,
        "performance_profile": performance_profile,
        "cache_read": cache_read,
        "cache_write": cache_write,
        **({"progress_phase": progress_phase} if progress_phase is not None else {}),
        **capture,
    }
    if options["json_output"]:
        return _cli_core.invoke_legacy(_scan._run_json, **run_kwargs)
    return _cli_core.invoke_legacy(_scan.run, verbose=options["verbose"], **run_kwargs)


_PERF_PARAMETER_NAMES = frozenset(
    {
        "perf",
        "performance_profile",
        "performance_dir",
        "performance_interval",
        "spec_chunk",
        "brem_chunk",
        "nsys",
        "py_spy",
        "cpu",
        "cpu_only",
    }
)
_PRESET_PARAMETER_NAMES = frozenset(
    {"preset", "ne", "ne_brem", "ne_supp", "tmd_azimuth", "refresh"}
)


def _derived_command(name, *, excluded, implied, help):
    """Build a public command view over the shared run implementation."""
    params = []
    for parameter in _command.params:
        if parameter.name in excluded or (
            isinstance(parameter, RetiredOption) and parameter.canonical_dest in excluded
        ):
            continue
        cloned = copy.copy(parameter)
        if name == "perf" and isinstance(cloned, click.Option):
            if cloned.name == "performance_dir":
                cloned.hidden = False
                cloned.help = "Write local telemetry below DIR."
            elif cloned.name == "performance_interval":
                cloned.help = "Performance-telemetry sampling interval."
            elif cloned.name == "spec_chunk":
                cloned.help = "Pin line-spectrum segments per GPU chunk."
            elif cloned.name == "brem_chunk":
                cloned.help = "Pin bremsstrahlung segments per GPU chunk."
            elif cloned.name == "nsys":
                cloned.help = (
                    "Capture one uncached Nsight Systems CUDA/NVTX trace of the run "
                    "(writes a .nsys-rep next to the perf log)."
                )
            elif cloned.name == "py_spy":
                cloned.help = (
                    "Sample the whole uncached run with py-spy (every thread and "
                    "worker process, including idle waits) and write a .py-spy.json "
                    "speedscope profile next to the perf log; exclusive with --nsys."
                )
            elif cloned.name == "cpu":
                cloned.help = (
                    "After the primary run, capture one bounded serial CPU cProfile "
                    "pass; requires -R/--remote."
                )
            elif cloned.name == "cpu_only":
                cloned.help = (
                    "Capture only the bounded serial CPU cProfile pass; starts no "
                    "primary GPU/Nsight scan and requires -R/--remote."
                )
        params.append(cloned)

    command_callback = _command.callback
    assert command_callback is not None

    def callback(**kwargs):
        return command_callback(**implied, **kwargs)

    return click.Command(
        name,
        params=params,
        callback=callback,
        help=help,
        context_settings={"help_option_names": ["-h", "--help"]},
    )


command = _derived_command(
    "run",
    excluded=_PERF_PARAMETER_NAMES,
    implied={
        "perf": False,
        "performance_profile": None,
        "performance_dir": None,
        "performance_interval": 5.0,
        "spec_chunk": None,
        "brem_chunk": None,
        "nsys": False,
        "py_spy": False,
        "cpu": False,
        "cpu_only": False,
    },
    help=_command.help,
)

performance_command = _derived_command(
    "perf",
    excluded={
        "perf",
        "performance_profile",
        "trajectories",
        "overwrite_trajectories",
        "ephemeral",
        "detector_id",
        "output_file",
        *_PRESET_PARAMETER_NAMES,
    },
    implied={
        "perf": True,
        "ephemeral": False,
        "detector_id": None,
        "output_file": None,
        "trajectories": None,
        "overwrite_trajectories": False,
        "performance_profile": None,
        "preset": None,
        "ne": 20_000,
        "ne_brem": 200,
        "ne_supp": 200,
        "tmd_azimuth": 0.0,
        "refresh": False,
    },
    help=(
        "Measure a catalog profile's MC sweeps and write performance telemetry.\n\n"
        "PROFILE defaults to the current configured profile. Use -m/--material "
        "to measure one member; otherwise the full resolved membership is measured. "
        "Runs omit shared-cache reads and writes unless --recompute is explicit."
    ),
)
# `pyrite-dev perf` runs this command as its own program root, so deprecation
# lookups need the prefix the relocated root cannot infer.
performance_command.deprecation_prefix = "pyrite-dev perf"  # type: ignore[attr-defined]
