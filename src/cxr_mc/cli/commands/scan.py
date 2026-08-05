"""Click wiring for ``cxr run``."""

import os
import re
from pathlib import Path

import click

from ... import scan as _scan
from .. import _completion as _cli_completion
from .. import _config as _cli_config
from .. import _core as _cli_core

_PERFORMANCE_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _beam_uvw(ctx, param, value):
    if value is None:
        return None
    return _cli_core.BEAM_UVW.convert(value, param, ctx)


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


def _remote_target(ctx, param, value):
    if value in (None, "__configured__"):
        return value
    from ..._remote.config import validate_remote_target

    try:
        return validate_remote_target(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc


@click.command(
    "run",
    help=(
        "Run a catalog profile's MC sweeps and write checkpoints.\n\n"
        "PROFILE defaults to the current configured profile (standard built-in) "
        "and owns material membership, campaign ranges, and workload settings. "
        "Use -m/--material to run one profile member instead of the full resolved "
        "membership.\n\n"
        "Resumes compatible checkpoints in CHECKPOINTS. Full writes "
        "<material>.pkl-compatible data in <material>/; variants use "
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
    help="Use tiny smoke-test grid and write <material>_quick.pkl.",
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
    help="Read and write checkpoint pickles in DIR.",
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
    "-R",
    "--remote",
    "remote_target",
    is_flag=False,
    flag_value="__configured__",
    default=None,
    callback=_remote_target,
    metavar="[TARGET]",
    help="Run through SLURM; bare uses the configured target, =TARGET overrides it.",
)
@click.option("--wait", is_flag=True, help="Wait for remote completion and pull results.")
@click.option("--detach", is_flag=True, help="Return after remote submission.")
@click.pass_context
@_cli_core.fidelity_option()
@_cli_core.output_option
def command(
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
    no_cache,
    recompute,
    progress_file,
    progress_phase,
    no_progress,
    json_output,
    verbose,
    remote_target,
    wait,
    detach,
):
    """Click entry point for the staged root migration."""
    try:
        catalog_profile = _cli_config.resolve("profile.current", catalog_profile).value
    except _cli_config.ConfigError as exc:
        raise _cli_core.CLIError(str(exc)) from exc
    if wait and detach:
        raise click.UsageError("--wait and --detach are mutually exclusive")
    if remote_target is None and (wait or detach):
        raise click.UsageError("--wait/--detach require -R/--remote")
    if remote_target is not None:
        if json_output:
            raise click.UsageError("remote run does not yet support --output json")
        local_only = {
            "n_families": "--n-families",
            "checkpoint_dir": "--checkpoint-dir",
            "max_minutes": "--max-minutes",
            "performance_profile": "--performance-profile",
            "performance_dir": "--performance-dir",
            "no_cache": "--no-cache",
            "recompute": "--recompute",
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
        from ..._remote import cli as remote_cli
        from ..._remote import config as remote_config

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
                headless=detach,
                no_pull=False,
            )
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
    if no_cache and recompute:
        raise click.UsageError("--no-cache and --recompute are mutually exclusive")
    perf_no_cache = performance_profile is not None and not (no_cache or recompute)
    cache_read = not (no_cache or recompute or perf_no_cache)
    cache_write = not (no_cache or perf_no_cache)
    # Pin the GPU spectrum/brem chunk before the runtime import so the main
    # process and every spawned transport worker (env inherited on spawn/
    # forkserver) read it. Mirrors the remote job script's `export
    # CXR_MC_SPEC_CHUNK` / `CXR_MC_BREM_CHUNK`.
    if spec_chunk is not None:
        os.environ["CXR_MC_SPEC_CHUNK"] = str(spec_chunk)
    if brem_chunk is not None:
        os.environ["CXR_MC_BREM_CHUNK"] = str(brem_chunk)
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
            no_cache=no_cache,
            recompute=recompute,
        )
        return None  # os.execvp already replaced the process; defensive.
    resolved_profile = _scan._resolve_catalog_profile(catalog_profile, performance_profile)
    _scan.resolve_profile_materials(resolved_profile, material)
    if not json_output:
        return _cli_core.invoke_legacy(
            _scan.run,
            material=material,
            all=False,
            actually_all=False,
            include_unverified_dw=False,
            include_high_energy=False,
            high_energy_min_kev=None,
            workers=workers,
            fidelity=fidelity,
            catalog_profile=catalog_profile,
            quick=quick,
            n_families=n_families,
            beam_uvw=None,
            beam_transverse_fwhm_x_mm=None,
            beam_transverse_fwhm_y_mm=None,
            beam_bunch_length_fs=None,
            beam_long_shape=None,
            beam_rep_rate_hz=None,
            beam_bunch_charge_pc=None,
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
        )
    return _cli_core.invoke_legacy(
        _scan._run_json,
        material=material,
        all=False,
        actually_all=False,
        include_unverified_dw=False,
        include_high_energy=False,
        high_energy_min_kev=None,
        workers=workers,
        fidelity=fidelity,
        catalog_profile=catalog_profile,
        quick=quick,
        n_families=n_families,
        beam_uvw=None,
        beam_transverse_fwhm_x_mm=None,
        beam_transverse_fwhm_y_mm=None,
        beam_bunch_length_fs=None,
        beam_long_shape=None,
        beam_rep_rate_hz=None,
        beam_bunch_charge_pc=None,
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
    )
