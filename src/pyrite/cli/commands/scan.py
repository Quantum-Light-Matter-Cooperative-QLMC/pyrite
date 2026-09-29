"""Click wiring shared by ``pyrite run`` and ``pyrite-dev perf``."""

import copy
import re
from pathlib import Path

import click

from ..._env import set_canonical_env
from ...console import config as _cli_config
from ...console import output as _cli_core
from ...runs import scan as _scan
from .. import _completion as _cli_completion
from .. import _implicit_defaults
from .._deprecations import DeprecatedOption
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
        "Run a catalog profile's MC sweeps and write checkpoints.\n\n"
        "PROFILE defaults to the current configured profile (standard built-in; "
        "the built-in fallback warns and is deprecated) "
        "and owns material membership, campaign ranges, and workload settings. "
        "Use -m/--material to run one profile member instead of the full resolved "
        "membership.\n\n"
        "Resumes compatible checkpoints in CHECKPOINTS. Full writes component "
        "data beneath <material>/; variants use "
        "identity-qualified stems."
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
    "--workers",
    type=_cli_core.NONNEGATIVE_INT,
    default=None,
    help="run_cases max_workers (default auto; 0 = serial, no transport pool).",
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
    default="checkpoints",
    show_default=True,
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
        "membership into performance-profiles/PROFILE/<material>.ndjson "
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
        "performance-profiles/NAME/<material>.ndjson."
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
@click.option(
    "--spec-chunk",
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
        "Neither read nor write the shared per-case checkpoint cache: an "
        "ephemeral run that recomputes every case and stores nothing shared."
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
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="DIR",
    help=(
        "Opt in to saving each transported case's full electron-transport result "
        "as HDF5 under DIR/<stem>/. Files can be much larger than checkpoints; "
        "cached cases are not re-transported (use --recompute to capture them)."
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
@click.option(
    "--ne-brem",
    type=_cli_core.POSITIVE_INT,
    default=200,
    show_default=True,
    help="With --preset zhai, Fig. 1c bremsstrahlung electrons per energy.",
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
    "--source",
    type=click.Choice(("analytic", "gpt_gdf")),
    default=None,
    help="Override beam source; gpt_gdf replaces analytic phase space (local runs).",
)
@click.option(
    "--gdf-shape-only/--no-gdf-shape-only",
    default=None,
    help="Use profile sweep energies with GDF positions/directions/weights; discard crossing times. Default: import energies.",
)
@click.option(
    "--gdf-path",
    type=click.Path(path_type=Path),
    default=None,
    help="GPT time-output file; relative to current directory.",
)
@click.option(
    "--gdf-time-s",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    help="Select time in seconds; required for multiple time blocks.",
)
@click.option(
    "--gdf-time-tolerance-s",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    help="Absolute time tolerance in seconds [default: 1e-15].",
)
@click.option(
    "--gdf-normalization",
    type=click.Choice(("pyrite_current", "gdf_charge")),
    default=None,
    help="Configured current or GDF bunch charge times repetition rate.",
)
@click.option(
    "--gdf-repetition-rate-hz",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    help="Required positive repetition rate for gdf_charge normalization.",
)
@click.option(
    "--gdf-z-origin-m",
    type=_cli_core.FINITE_FLOAT,
    default=None,
    help="Explicit target origin along GPT lab z in meters; required for GDF.",
)
@click.option(
    "--gdf-screen-position-m",
    type=_cli_core.FINITE_FLOAT,
    default=None,
    help="Select GPT screen coordinate in meters; excludes --gdf-time-s.",
)
@click.option(
    "--gdf-screen-tolerance-m",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    help="Absolute screen-coordinate tolerance in meters [default: 1e-9].",
)
@click.pass_context
@_cli_core.fidelity_option(cls=DeprecatedOption)
@_cli_core.output_option
def _command(
    ctx,
    catalog_profile,
    material,
    workers,
    fidelity,
    quick,
    n_families,
    checkpoint_dir,
    max_minutes,
    perf,
    performance_profile,
    performance_dir,
    performance_interval,
    spec_chunk,
    brem_chunk,
    nsys,
    cpu,
    cpu_only,
    no_cache,
    recompute,
    trajectories,
    overwrite_trajectories,
    progress_file,
    progress_phase,
    no_progress,
    json_output,
    verbose,
    preset,
    ne,
    ne_brem,
    ne_supp,
    tmd_azimuth,
    refresh,
    no_sync,
    dry_run,
    remote_target,
    wait,
    detach,
    source,
    gdf_shape_only,
    gdf_path,
    gdf_time_s,
    gdf_time_tolerance_s,
    gdf_normalization,
    gdf_repetition_rate_hz,
    gdf_z_origin_m,
    gdf_screen_position_m,
    gdf_screen_tolerance_m,
):
    """Click entry point for the staged root migration."""
    gdf_overrides = {
        key: value
        for key, value in {
            "source": source,
            "gdf_path": str(gdf_path.resolve()) if gdf_path is not None else None,
            "gdf_shape_only": gdf_shape_only,
            "gdf_time_s": gdf_time_s,
            "gdf_time_tolerance_s": gdf_time_tolerance_s,
            "gdf_normalization": gdf_normalization,
            "gdf_repetition_rate_hz": gdf_repetition_rate_hz,
            "gdf_z_origin_m": gdf_z_origin_m,
            "gdf_screen_position_m": gdf_screen_position_m,
            "gdf_screen_tolerance_m": gdf_screen_tolerance_m,
        }.items()
        if value is not None
    }
    if gdf_overrides and (remote_target is not None or preset is not None or nsys):
        raise click.UsageError("GDF overrides require a local normal run without --nsys")
    if gdf_time_s is not None and gdf_screen_position_m is not None:
        raise click.UsageError("--gdf-time-s and --gdf-screen-position-m are mutually exclusive")
    raw_catalog_profile = catalog_profile
    resolved_profile = None
    if preset is None:
        try:
            resolved_profile = _cli_config.resolve("profile.current", catalog_profile)
            catalog_profile = resolved_profile.value
        except _cli_config.ConfigError as exc:
            raise _cli_core.CLIError(str(exc)) from exc
    if wait and detach:
        raise click.UsageError("--wait and --detach are mutually exclusive")
    if remote_target is None and (wait or detach):
        raise click.UsageError("--wait/--detach require -R/--remote")
    if remote_target is None and (no_sync or dry_run):
        raise click.UsageError("--no-sync/--dry-run require -R/--remote")
    if remote_target is None and (cpu or cpu_only):
        raise click.UsageError("--cpu/--cpu-only require -R/--remote")
    if no_cache and recompute:
        raise click.UsageError("--no-cache and --recompute are mutually exclusive")
    if overwrite_trajectories and trajectories is None:
        raise click.UsageError("--overwrite-trajectories requires --trajectories")
    zhai_parameters = {
        "ne": "--ne",
        "ne_brem": "--ne-brem",
        "ne_supp": "--ne-supp",
        "tmd_azimuth": "--tmd-azimuth",
        "refresh": "--refresh",
    }
    explicit_zhai = [
        flag
        for parameter, flag in zhai_parameters.items()
        if ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
    ]
    if preset is None and explicit_zhai:
        raise click.UsageError(f"Zhai option(s) require --preset zhai: {', '.join(explicit_zhai)}")
    if preset == "zhai":
        normal_run_parameters = {
            "catalog_profile": "PROFILE",
            "material": "--material",
            "workers": "--workers",
            "fidelity": "--fidelity",
            "quick": "--quick",
            "n_families": "--n-families",
            "checkpoint_dir": "--checkpoint-dir",
            "max_minutes": "--max-minutes",
            "perf": "--perf",
            "performance_profile": "--performance-profile",
            "performance_dir": "--performance-dir",
            "performance_interval": "--perf-interval",
            "spec_chunk": "--spec-chunk",
            "brem_chunk": "--brem-chunk",
            "nsys": "--nsys",
            "cpu": "--cpu",
            "cpu_only": "--cpu-only",
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
        explicit_normal = [
            flag
            for parameter, flag in normal_run_parameters.items()
            if ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
        ]
        if raw_catalog_profile is not None and "PROFILE" not in explicit_normal:
            explicit_normal.insert(0, "PROFILE")
        if explicit_normal:
            raise click.UsageError(
                "--preset zhai does not support normal-run option(s): " + ", ".join(explicit_normal)
            )
        if remote_target is None:
            return _reproduce_zhai(ne, ne_brem, ne_supp, tmd_azimuth, refresh)
        from ...remote import config as remote_config
        from . import remote as remote_cli

        target = None if remote_target == "__configured__" else remote_target
        with remote_config.override_remote_host(target):
            return remote_cli.remote_check(
                ne=ne,
                ne_brem=ne_brem,
                ne_supp=ne_supp,
                tmd_azimuth=tmd_azimuth,
                refresh=refresh,
                no_sync=no_sync,
                detach=detach,
                dry_run=dry_run,
            )
    if resolved_profile is not None:
        _implicit_defaults.warn_implicit_profile(resolved_profile)
    if remote_target is not None:
        if json_output:
            raise click.UsageError("remote run does not yet support --output json")
        local_only = {
            "n_families": "--n-families",
            "checkpoint_dir": "--checkpoint-dir",
            "max_minutes": "--max-minutes",
            "trajectories": "--trajectories",
            "overwrite_trajectories": "--overwrite-trajectories",
            "performance_profile": "--performance-profile",
            "performance_dir": "--performance-dir",
            "progress_file": "--progress-file",
            "progress_phase": "--progress-phase",
            "no_progress": "--no-progress",
            "verbose": "--verbose",
        }
        explicit_local = [
            flag
            for parameter, flag in local_only.items()
            if ctx.get_parameter_source(parameter) is click.core.ParameterSource.COMMANDLINE
        ]
        if explicit_local:
            raise click.UsageError(
                f"remote run does not support local-only option(s): {', '.join(explicit_local)}"
            )
        from ...remote import config as remote_config
        from . import remote as remote_cli

        target = None if remote_target == "__configured__" else remote_target
        with remote_config.override_remote_host(target):
            return ctx.invoke(
                remote_cli.start_command,
                catalog_profile=catalog_profile,
                material=material,
                fidelity=fidelity,
                quick=quick,
                workers=workers,
                perf=perf,
                performance_interval=performance_interval,
                spec_chunk=spec_chunk,
                brem_chunk=brem_chunk,
                nsys=nsys,
                cpu=cpu,
                cpu_only=cpu_only,
                no_cache=no_cache,
                recompute=recompute,
                no_sync=no_sync,
                dry_run=dry_run,
                headless=detach,
                no_pull=False,
            )
    if resolved_profile is not None:
        _implicit_defaults.warn_implicit_instrument(catalog_profile)
    if quick and fidelity != "full":
        raise click.UsageError("--quick cannot be combined with --fidelity survey")
    if perf and performance_profile is None:
        # -p is sugar for --performance-profile PROFILE: profile the positional
        # profile's full membership (or one member when -m is given).
        performance_profile = catalog_profile
    if performance_interval != 5.0 and performance_profile is None:
        raise click.UsageError("--perf-interval requires -p/--perf")
    if (spec_chunk is not None or brem_chunk is not None) and performance_profile is None:
        raise click.UsageError("--spec-chunk/--brem-chunk require -p/--perf")
    if nsys and performance_profile is None:
        raise click.UsageError("--nsys requires -p/--perf")
    # Shared per-case cache gates (see run_sweep's content-addressable store).
    #   default    -> read + write
    #   --recompute-> skip read, still repopulate (write)
    #   --no-cache -> neither read nor write (ephemeral)
    # A -p/--perf run defaults to --no-cache semantics so it measures real
    # compute and never pollutes the shared store with a measurement run; an
    # explicit --no-cache/--recompute on the same line wins (explicit beats the
    # perf default).
    perf_no_cache = performance_profile is not None and not (no_cache or recompute)
    cache_read = not (no_cache or recompute or perf_no_cache)
    cache_write = not (no_cache or perf_no_cache)
    # Pin the GPU spectrum/brem chunk before the runtime import so the main
    # process and every spawned transport worker (env inherited on spawn/
    # forkserver) read it. Mirrors the remote job script's `export
    # PYRITE_MC_SPEC_CHUNK` / `PYRITE_MC_BREM_CHUNK`.
    if spec_chunk is not None:
        set_canonical_env("PYRITE_MC_SPEC_CHUNK", str(spec_chunk))
    if brem_chunk is not None:
        set_canonical_env("PYRITE_MC_BREM_CHUNK", str(brem_chunk))
    if nsys:
        _scan._reexec_under_nsys(
            catalog_profile=catalog_profile,
            material=material,
            performance_profile=performance_profile,
            performance_dir=performance_dir,
            performance_interval=performance_interval,
            workers=workers,
            fidelity=fidelity,
            quick=quick,
            n_families=n_families,
            max_minutes=max_minutes,
            no_cache=no_cache,
            recompute=recompute,
        )
        return None  # os.execvp already replaced the process; defensive.
    # Trajectory capture is opt-in; ordinary runs keep their exact arguments.
    capture = (
        {}
        if trajectories is None
        else {"trajectories": trajectories, "overwrite_trajectories": overwrite_trajectories}
    )
    resolved_profile = _scan._resolve_catalog_profile(catalog_profile, performance_profile)
    _scan.resolve_profile_materials(resolved_profile, material)
    if not json_output:
        return _cli_core.invoke_legacy(
            _scan.run,
            material=material,
            all=False,
            actually_all=False,
            workers=workers,
            fidelity=fidelity,
            catalog_profile=catalog_profile,
            quick=quick,
            n_families=n_families,
            beam_uvw=None,
            **({"gdf_overrides": gdf_overrides} if gdf_overrides else {}),
            checkpoint_dir=checkpoint_dir,
            max_minutes=max_minutes,
            performance_profile=performance_profile,
            performance_dir=performance_dir,
            performance_interval=performance_interval,
            cache_read=cache_read,
            cache_write=cache_write,
            progress_file=progress_file,
            no_progress=no_progress,
            verbose=verbose,
            **({"progress_phase": progress_phase} if progress_phase is not None else {}),
            **capture,
        )
    return _cli_core.invoke_legacy(
        _scan._run_json,
        material=material,
        all=False,
        actually_all=False,
        workers=workers,
        fidelity=fidelity,
        catalog_profile=catalog_profile,
        quick=quick,
        n_families=n_families,
        beam_uvw=None,
        **({"gdf_overrides": gdf_overrides} if gdf_overrides else {}),
        checkpoint_dir=checkpoint_dir,
        max_minutes=max_minutes,
        performance_profile=performance_profile,
        performance_dir=performance_dir,
        performance_interval=performance_interval,
        cache_read=cache_read,
        cache_write=cache_write,
        progress_file=progress_file,
        no_progress=no_progress,
        **({"progress_phase": progress_phase} if progress_phase is not None else {}),
        **capture,
    )


_PERF_PARAMETER_NAMES = frozenset(
    {
        "perf",
        "performance_profile",
        "performance_dir",
        "performance_interval",
        "spec_chunk",
        "brem_chunk",
        "nsys",
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
        if parameter.name in excluded:
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
        *_PRESET_PARAMETER_NAMES,
    },
    implied={
        "perf": True,
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
