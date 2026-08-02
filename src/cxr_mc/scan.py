"""Headless CXR scan runner (the library twin of ``notebooks/scan_app.py``).

Runs the Monte-Carlo CXR parameter sweep for one material and writes the
per-material checkpoint (checkpoints/<material>.pkl). Use this to run sweeps
non-interactively -- in particular over SSH on the GPU box; see cxr_mc.remote,
which drives this and pulls the checkpoint back so interactive analysis and
static-HTML export can stay on the laptop.

    cxr run                        # standard profile membership
    cxr run sub_100keV             # named profile membership
    cxr run standard -m mose2     # one standard-profile member
    cxr run standard -m mose2 --fidelity survey
    cxr run standard -m mose2 --quick
    cxr run standard -m mose2 --workers 0

(equivalently ``python -m cxr_mc._entry.scan standard -m mose2`` via the
module shim).

The ``if __name__ == "__main__"`` guard on the entry point is REQUIRED, not
stylistic: run_cases farms the electron transport out to a process pool, and the
default start method is 'spawn' on Windows and -- as of Python 3.14 --
'forkserver' on Linux. BOTH re-import the entry module in every worker, so without
the guard the sweep relaunches itself recursively (a process-spawn cascade that
surfaces as a forkserver ConnectionResetError). Inside the guard the re-import is
a harmless no-op.
"""

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from .cli import _completion as _cli_completion
from .cli import _core as _cli_core
from .cli import _dashboard
from .cli import json as cli_json

MATS_FILE = Path("mats_to_sim.toml")
_PERFORMANCE_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# Lazy runtime bindings keep help fast while preserving monkeypatchable module
# seams used by focused driver tests.
default_settings = None
format_penetration_watchdog_summary = None
gate_cases_by_penetration = None
material_sweep = None
run_sweep = None
build_cases = None
case_cost = None


def _load_runtime() -> None:
    global default_settings
    global format_penetration_watchdog_summary
    global gate_cases_by_penetration
    global material_sweep
    global run_sweep
    global build_cases
    global case_cost

    from . import config as config_module
    from . import run as run_module
    from . import sweep as sweep_module

    default_settings = default_settings or config_module.default_settings
    format_penetration_watchdog_summary = (
        format_penetration_watchdog_summary or config_module.format_penetration_watchdog_summary
    )
    gate_cases_by_penetration = gate_cases_by_penetration or config_module.gate_cases_by_penetration
    material_sweep = material_sweep or config_module.material_sweep
    run_sweep = run_sweep or run_module.run_sweep
    build_cases = build_cases or sweep_module.build_cases
    case_cost = case_cost or sweep_module.case_cost


def _local_probe():
    fields = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                parts = out.split("\n")[0].split(",")
                if len(parts) == 3:
                    gpu = float(parts[0])
                    vram_used = float(parts[1])
                    vram_total = float(parts[2])
                    vram_pct = 100.0 * vram_used / vram_total if vram_total > 0 else 0.0
                    fields.extend(
                        [
                            f"gpu_percent={gpu:.1f}",
                            f"vram_used_mib={vram_used:.0f}",
                            f"vram_total_mib={vram_total:.0f}",
                            f"vram_percent={vram_pct:.1f}",
                        ]
                    )
        except Exception:
            pass
    return "|".join(fields)


def _build_sections(args, materials, job_records, detail):
    meta = [
        f"kind: {'quick' if getattr(args, 'quick', False) else 'material-sweep'}",
        f"materials: {' '.join(materials)}",
        "job: local",
    ]
    if getattr(args, "workers", None) is not None:
        meta.append(f"workers: {args.workers}")
    if getattr(args, "max_minutes", None) is not None:
        meta.append(f"slice_minutes: {args.max_minutes}")
    if getattr(args, "catalog_profile", None) is not None:
        meta.append(f"profile: {args.catalog_profile}")

    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        meta.append(f"slurm_job_id: {slurm_job}")

    sections = {
        "META": "\n".join(meta),
        "STATE": "running",
        "RESOURCES": _local_probe(),
    }

    if slurm_job and shutil.which("squeue"):
        try:
            sq = subprocess.check_output(
                ["squeue", "-j", slurm_job, "-o", "%i|%T|%P|%M|%L|%D|%r", "--noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if sq:
                parts = sq.split("|")
                if len(parts) >= 7:
                    sections["SQUEUE"] = (
                        f"job_id={parts[0]}|state={parts[1]}|partition={parts[2]}|elapsed={parts[3]}|left={parts[4]}|nodes={parts[5]}|reason={parts[6]}"
                    )
        except Exception:
            pass

    progress_lines = []
    for _mat, rec in job_records.items():
        progress_lines.append(json.dumps(rec))

    if progress_lines:
        sections["PROGRESS"] = "\n".join(progress_lines)

    return sections


_dashboard_stop = threading.Event()


def _dashboard_loop(args, materials, job_records, detail):
    keys = _dashboard._KeyListener()
    try:
        while not _dashboard_stop.is_set():
            for key in keys.poll():
                if key.lower() == "v":
                    detail = (detail + 1) % 3
            sections = _build_sections(args, materials, job_records, detail)
            frame = _dashboard._style_states(_dashboard._format_job_status(sections, detail))
            _dashboard._render_frame(frame, tty=True)
            _dashboard_stop.wait(1.0)
    finally:
        keys.stop()


def _read_manifest_toml(path: Path):
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise SystemExit(f"material manifest not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"invalid material manifest {path}: {exc}") from None


def _manifest_list(raw, path: Path, key: str, *, required: bool) -> list[str]:
    materials = raw.get(key, [] if not required else None)
    if (
        not isinstance(materials, list)
        or (required and not materials)
        or not all(isinstance(material, str) and material.strip() for material in materials)
    ):
        raise SystemExit(
            f"invalid material manifest {path}: expected a list of non-empty strings {key!r}"
        )
    duplicates = list(
        dict.fromkeys(material for material in materials if materials.count(material) > 1)
    )
    if duplicates:
        raise SystemExit(f"duplicate material(s) in {path} {key!r}: {', '.join(duplicates)}")
    valid_materials = set(_cli_completion._material_keys())
    unknown = [material for material in materials if material not in valid_materials]
    if unknown:
        raise SystemExit(f"unknown material(s) in {path} {key!r}: {', '.join(unknown)}")
    return materials


def load_all_materials(path: Path | None = None) -> list[str]:
    """Read the ordered verified-material manifest group."""
    path = MATS_FILE if path is None else path
    raw = _read_manifest_toml(path)
    return _manifest_list(raw, path, "materials", required=True)


def load_manifest_groups(path: Path | None = None) -> dict[str, list[str]]:
    """Read every material group from ``mats_to_sim.toml``.

    ``materials`` is the verified/production subset (required, non-empty; goes
    through ``load_all_materials`` so callers that monkeypatch it still take
    effect here). ``no_verified_dw``, ``high_energy_materials``, and
    ``materials_to_leave_out`` are optional and default to empty.
    """
    path = MATS_FILE if path is None else path
    raw = _read_manifest_toml(path)
    return {
        "materials": load_all_materials(path),
        "no_verified_dw": _manifest_list(raw, path, "no_verified_dw", required=False),
        "high_energy_materials": _manifest_list(raw, path, "high_energy_materials", required=False),
        "materials_to_leave_out": _manifest_list(
            raw, path, "materials_to_leave_out", required=False
        ),
    }


def validate_materials(materials: list[str]) -> None:
    """Reject runnable selections that are absent from the material catalog."""
    valid_materials = _cli_completion._material_keys()
    unknown = [material for material in materials if material not in valid_materials]
    if unknown:
        raise SystemExit(f"unknown material(s): {', '.join(unknown)}")


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


@click.command(
    "run",
    help=(
        "Run a catalog profile's MC sweeps and write checkpoints.\n\n"
        "PROFILE defaults to standard and owns material membership, campaign "
        "ranges, and workload settings. Use -m/--material to run one profile "
        "member instead of the full resolved membership.\n\n"
        "Resumes compatible checkpoints in CHECKPOINTS. Full writes "
        "<material>.pkl-compatible data in <material>/; variants use "
        "identity-qualified stems."
    ),
)
@click.argument(
    "catalog_profile",
    required=False,
    default="standard",
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
@_cli_core.fidelity_option()
@_cli_core.json_option
def command(
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
):
    """Click entry point for the staged root migration."""
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
        _reexec_under_nsys(
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
    resolved_profile = _resolve_catalog_profile(catalog_profile, performance_profile)
    resolve_profile_materials(resolved_profile, material)
    if not json_output:
        return _cli_core.invoke_legacy(
            run,
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
        _run_json,
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


def _nsys_reexec_command(
    *,
    catalog_profile,
    material,
    performance_profile,
    performance_dir,
    performance_interval,
    workers,
    fidelity,
    quick,
    n_families,
    no_cache=False,
    recompute=False,
):
    """Build the ``nsys profile ... python -m cxr_mc._entry.scan`` argv and the
    trace-output stem for a local ``--nsys`` capture.

    Pure (no side effects) so the argv/trace contract is unit-testable. The
    child re-runs this same command WITHOUT ``--nsys`` (so it cannot recurse)
    under CXR_MC_NSYS=1, mirroring the remote job script's nsys launcher
    (:mod:`cxr_mc._remote.scripts`). It points the child at an isolated,
    always-uncached checkpoint dir so the trace covers real GPU work rather
    than a fast checkpoint resume. ``material`` is optional: when omitted the
    capture covers the profile's full membership and the trace/checkpoint stem
    falls back to the profile name."""
    perf_root = (
        Path(performance_dir) if performance_dir is not None else Path("performance-profiles")
    )
    stem = material if material is not None else performance_profile
    trace_base = perf_root / performance_profile / stem
    checkpoint_dir = perf_root / performance_profile / "nsys-checkpoints" / stem
    child = [
        sys.executable,
        "-m",
        "cxr_mc._entry.scan",
        catalog_profile,
    ]
    if material is not None:
        child += ["-m", material]
    child += ["--performance-profile", performance_profile]
    if performance_dir is not None:
        child += ["--performance-dir", str(performance_dir)]
    if performance_interval != 5.0:
        child += ["--perf-interval", f"{performance_interval:g}"]
    if workers is not None:
        child += ["--workers", str(workers)]
    if fidelity != "full":
        child += ["--fidelity", fidelity]
    if quick:
        child += ["--quick"]
    if n_families is not None:
        child += ["--n-families", str(n_families)]
    if no_cache:
        child += ["--no-cache"]
    elif recompute:
        child += ["--recompute"]
    child += ["--checkpoint-dir", str(checkpoint_dir)]
    nsys_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--wait=all",
        "--force-overwrite=true",
        f"--output={trace_base}",
    ]
    return [*nsys_cmd, *child], trace_base


def _reexec_under_nsys(**kwargs):
    """Replace this process with the :func:`_nsys_reexec_command` launcher so
    Nsight Systems captures one uncached CUDA/NVTX trace of the run. Sets
    CXR_MC_NSYS=1 so the runner emits NVTX ranges in the child. Does not
    return on success (``os.execvp``)."""
    if shutil.which("nsys") is None:
        raise click.UsageError("--nsys requested but the nsys executable is not on PATH")
    argv, trace_base = _nsys_reexec_command(**kwargs)
    trace_base.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CXR_MC_NSYS"] = "1"
    os.execvp(argv[0], argv)


def _resolve_catalog_profile(catalog_profile: str, performance_profile: str | None) -> str:
    """Reconcile ``--profile``/``--performance-profile`` into one catalog
    profile name -- the same rule the command's early validation, material
    selection (``_selected``), and per-material settings resolution
    (``_resolved_run``) all need to agree on."""
    if performance_profile is not None:
        if catalog_profile not in ("standard", performance_profile):
            raise click.UsageError(
                "--performance-profile and --profile must name the same catalog profile"
            )
        catalog_profile = performance_profile
    return catalog_profile


def _effective_catalog_profile(args) -> str:
    """``args``-based wrapper around :func:`_resolve_catalog_profile`."""
    return _resolve_catalog_profile(
        getattr(args, "catalog_profile", "standard"),
        getattr(args, "performance_profile", None),
    )


def validate_catalog_profile(
    catalog_profile: str, materials: list[str], *, intersect: bool
) -> list[str]:
    """Validate ``--profile`` (catalog campaign name) against the material
    catalog and reconcile it with an already-selected material list.

    Unknown profile names are always a usage error listing what's available.
    With ``intersect=True`` (``--all``/``-A``), a profile's explicit
    ``materials`` membership list silently narrows the selection -- absent
    membership means every candidate stays. With ``intersect=False`` (an
    explicit MATERIAL/material list), any member outside the profile is a
    hard usage error instead: the user named it, so silently dropping it
    would be a surprise, not a convenience.
    """
    from .materials import CATALOG

    if catalog_profile not in CATALOG.profile_names:
        available = ", ".join(sorted(CATALOG.profile_names)) or "(none defined)"
        raise click.UsageError(f"unknown profile {catalog_profile!r}; available: {available}")
    membership = CATALOG.profile_materials(catalog_profile)
    if membership is None:
        return materials
    allowed = set(membership)
    if intersect:
        narrowed = [m for m in materials if m in allowed]
        if not narrowed:
            raise click.UsageError(
                f"profile {catalog_profile!r} shares no materials with this selection "
                f"(members: {', '.join(membership)})"
            )
        return narrowed
    non_members = [m for m in materials if m not in allowed]
    if non_members:
        raise click.UsageError(
            f"profile {catalog_profile!r} does not include {', '.join(non_members)} "
            f"(members: {', '.join(membership)})"
        )
    return materials


def resolve_profile_materials(catalog_profile: str, material: str | None = None) -> list[str]:
    """Resolve one run selection from a profile and optional material override.

    Shared boundary contract for ``cxr run`` and ``cxr remote run``: ``-m``
    selects one profile member; omitting it selects the profile's explicit
    ``materials`` membership, or the in-use manifest set (``mats_to_sim.toml``
    ``materials`` -- what ``--all`` loads) when membership is implicit, e.g.
    ``standard``. An implicit profile defaults to the campaign's in-use
    subset, not the full catalog.
    """
    from .materials import CATALOG

    validate_catalog_profile(catalog_profile, [], intersect=False)
    if material is not None:
        materials = validate_catalog_profile(catalog_profile, [material], intersect=False)
    else:
        membership = CATALOG.profile_materials(catalog_profile)
        materials = list(membership) if membership is not None else load_all_materials()
    validate_materials(materials)
    return materials


def _selected(args):
    """Resolve profile-owned membership plus an optional one-material override."""
    catalog_profile = _effective_catalog_profile(args)
    args.high_energy_floor_map = {}
    return resolve_profile_materials(catalog_profile, getattr(args, "material", None))


def run(args):
    materials = _selected(args)
    deadline = None
    if getattr(args, "max_minutes", None) is not None:
        deadline = time.monotonic() + args.max_minutes * 60.0

    job_records = {}
    use_dashboard = (
        not getattr(args, "no_progress", False)
        and sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
    )

    if use_dashboard:
        args._job_records = job_records
        os.environ["CXR_LOCAL_DASHBOARD"] = "1"
        detail = getattr(args, "verbose", 0)
        t = threading.Thread(
            target=_dashboard_loop, args=(args, materials, job_records, detail), daemon=True
        )
        t.start()

    try:
        incomplete = False
        for material in materials:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not _run_material(args, material, max_seconds=remaining):
                incomplete = True
        if incomplete:
            raise SystemExit(75)  # EX_TEMPFAIL: budget hit, work remains
    finally:
        if use_dashboard:
            _dashboard_stop.set()
            t.join(timeout=2.0)
            sections = _build_sections(args, materials, job_records, getattr(args, "verbose", 0))
            sections["STATE"] = "done" if not incomplete else "paused"
            frame = _dashboard._style_states(
                _dashboard._format_job_status(sections, getattr(args, "verbose", 0))
            )
            _dashboard._render_frame(frame, tty=True)
            if "CXR_LOCAL_DASHBOARD" in os.environ:
                del os.environ["CXR_LOCAL_DASHBOARD"]


def _run_json(args):
    """Run batch with machine-only output and retain partial material results."""
    materials = _selected(args)
    started = time.monotonic()
    deadline = (
        None if getattr(args, "max_minutes", None) is None else started + args.max_minutes * 60.0
    )
    completed = []
    failed = []
    errors = {}
    resumable = False
    args.no_progress = True
    for material in materials:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                complete = _run_material(args, material, max_seconds=remaining)
        except (Exception, SystemExit) as exc:
            failed.append(material)
            errors[material] = str(exc) or type(exc).__name__
            continue
        if complete:
            completed.append(material)
        else:
            failed.append(material)
            errors[material] = "resumable work remains"
            resumable = True
    checkpoints = [
        os.path.join(args.checkpoint_dir, _checkpoint_stem(args, material))
        for material in [*completed, *failed]
    ]
    result = cli_json.operation_summary(
        "run",
        materials,
        completed,
        failed_materials=failed,
        checkpoints=checkpoints,
        elapsed_seconds=time.monotonic() - started,
        resumable=resumable,
        material_errors=errors,
    )
    _cli_core.emit_json_result(
        result,
        failure_exit=75
        if resumable and all(message == "resumable work remains" for message in errors.values())
        else 1,
    )


def _select_quick_energies(by_energy, e_grid_line, *, nominal=(30, 50), count=2):
    """Bounded, deterministic quick beam energies with valid line grids.

    ``--quick`` needs a small representative beam-energy subset whose line grids
    the effective profile/material actually configures, so ``build_cases`` never
    requests an absent per-energy grid. Policy:

    * A fixed ``E_grid_line`` or an absent ``E_grid_line_by_energy`` mapping means
      every beam energy resolves the same grid; keep the historical ``[30, 50]``
      (int literals -- preserves the standard profile's checkpoint identity
      bit-for-bit).
    * Otherwise the valid energies are the mapping's keys. Preserve the nominal
      ``30``/``50`` literals where they are valid, then fill any remaining slots
      with the valid energies nearest the unmet nominal targets (ties break low)
      to keep energy diversity. A profile exposing fewer than ``count`` valid
      energies runs exactly those; an empty mapping falls through to the nominal
      pair so the strict missing-grid error still fires downstream.
    """
    if by_energy is None or e_grid_line is not None:
        return list(nominal)
    valid = sorted(float(key) for key in by_energy)
    if not valid:
        return list(nominal)
    valid_set = set(valid)
    chosen = [energy for energy in nominal if float(energy) in valid_set]
    chosen_values = {float(energy) for energy in chosen}
    remaining = [energy for energy in valid if energy not in chosen_values]
    for target in (energy for energy in nominal if float(energy) not in valid_set):
        if len(chosen) >= count or not remaining:
            break
        nearest = min(remaining, key=lambda energy: (abs(energy - target), energy))
        chosen.append(nearest)
        remaining.remove(nearest)
    for energy in remaining:
        if len(chosen) >= count:
            break
        chosen.append(energy)
    return chosen[:count]


def _resolved_run(args, material):
    """Resolve settings, sweep, identity, and collision-free checkpoint stem."""
    import numpy as np

    _load_runtime()
    assert default_settings is not None
    assert material_sweep is not None

    fidelity = getattr(args, "fidelity", "full")
    catalog_profile = _effective_catalog_profile(args)
    settings = default_settings() if fidelity == "full" else default_settings(fidelity)
    # Emission (incoherent/coherent/both) is PROFILE-owned -- there is no CLI
    # override flag. The resolved settings.emission (via the profile) drives the
    # dataset_identity divergence key (profiles.dataset_identity) and the
    # canonical_full collision guard below, so a coherent/both run never shares
    # the plain incoherent <material> stem.
    overrides = {}
    if getattr(args, "quick", False):
        # Resolve quick beam energies from the effective profile/material line
        # grids instead of forcing a fixed [30, 50]: a named profile may own an
        # E_grid_line_by_energy mapping without 50 keV, and build_cases rejects a
        # beam energy with no configured line grid before compute. The probe
        # sweep (same catalog_profile/fidelity, no beam/tilt overrides) exposes
        # the valid line-grid energies; _select_quick_energies keeps the standard
        # profile's [30, 50] bit-for-bit where both are valid.
        probe = (
            material_sweep(material, catalog_profile=catalog_profile)
            if fidelity == "full"
            else material_sweep(material, fidelity=fidelity, catalog_profile=catalog_profile)
        )
        overrides.update(
            # Start at 5 deg, not 0: tilt=0 is a banned emission geometry
            # (issue_notes.md #1), and build_cases rejects it.
            tilt_deg=np.linspace(5.0, 85.0, 5),
            tilt_azim_deg=np.array([10.0, 30.0]),
            energy_keV=_select_quick_energies(probe.E_grid_line_by_energy, probe.E_grid_line),
        )
    if getattr(args, "n_families", None) is not None:
        overrides["n_families"] = args.n_families
    if getattr(args, "beam_uvw", None) is not None:
        overrides["beam_uvw"] = tuple(args.beam_uvw)
    beam_overrides = {
        "transverse_fwhm_x_mm": getattr(args, "beam_transverse_fwhm_x_mm", None),
        "transverse_fwhm_y_mm": getattr(args, "beam_transverse_fwhm_y_mm", None),
        "bunch_length_fs": getattr(args, "beam_bunch_length_fs", None),
        "long_shape": getattr(args, "beam_long_shape", None),
        "rep_rate_hz": getattr(args, "beam_rep_rate_hz", None),
        "bunch_charge_pc": getattr(args, "beam_bunch_charge_pc", None),
    }
    overrides.update({key: value for key, value in beam_overrides.items() if value is not None})
    sweep = (
        material_sweep(material, catalog_profile=catalog_profile, **overrides)
        if fidelity == "full"
        else material_sweep(
            material, fidelity=fidelity, catalog_profile=catalog_profile, **overrides
        )
    )
    if getattr(args, "beam_long_shape", None) is not None and sweep.beam.bunch_length_fs is None:
        raise click.UsageError(
            "--beam-long-shape requires --beam-bunch-length-fs or a profile bunch"
        )

    # High-energy-only materials (mats_to_sim.toml's high_energy_materials list,
    # pulled in via --include-high-energy or -A) are "only worthwhile to sim for
    # higher beam energies" -- filter their grid to the floor regardless of which
    # flag selected them. --quick already substitutes its own fixed low-energy
    # smoke-test grid, so the floor doesn't apply there.
    floor = getattr(args, "high_energy_floor_map", None) or {}
    floor = floor.get(material)
    if floor is not None and not getattr(args, "quick", False):
        from .sweep import beam_replace

        energies = np.asarray(sweep.beam.energy_keV, dtype=float)
        kept = energies[energies >= floor]
        if kept.size == 0:
            raise SystemExit(
                f"{material}: no energies >= {floor} keV in its grid (high-energy "
                "floor); lower --high-energy-min-kev or drop this material"
            )
        # Also record in `overrides` (not just apply via `replace` below) so the
        # canonical_full check further down treats a floored grid as a variant,
        # not the plain canonical stem -- otherwise a floored tise2 checkpoint
        # would collide with an unfiltered tise2 full-grid checkpoint.
        overrides["energy_keV"] = kept
        sweep = replace(sweep, beam=beam_replace(sweep.beam, energy_keV=kept))

    from .profiles import dataset_identity, variant_stem

    identity = dataset_identity(
        material,
        fidelity,
        settings,
        sweep,
        variant="quick" if getattr(args, "quick", False) else None,
        catalog_profile=catalog_profile,
    )
    canonical_full = (
        fidelity == "full"
        and not overrides
        and not getattr(args, "quick", False)
        and catalog_profile == "standard"
        and settings.emission == "incoherent"
    )
    stem = variant_stem(identity, canonical_full=canonical_full)
    return settings, sweep, identity, stem


def _checkpoint_stem(args, material):
    return _resolved_run(args, material)[3]


def _run_material(args, material, max_seconds=None):
    _load_runtime()
    assert format_penetration_watchdog_summary is not None
    assert gate_cases_by_penetration is not None
    assert run_sweep is not None
    assert build_cases is not None

    settings, sweep, identity, stem = _resolved_run(args, material)
    fidelity = identity["fidelity"]

    cases = build_cases(
        sweep,
        settings.n_electrons,
        settings.n_electrons_brem,
        coherent_emission=settings.coherent_emission,
    )
    cases, dropped = gate_cases_by_penetration(cases)
    summary = format_penetration_watchdog_summary(dropped, material=material)
    if summary is not None:
        print(summary)
    print(
        f"{material}: {len(cases)} cases across "
        f"{len({c['name'] for c in cases})} configs "
        f"[profile={fidelity}, parameters={identity['parameter_sha256'][:12]}]"
        + (" (quick grid)" if args.quick else "")
        + ("" if settings.emission == "incoherent" else f" ({settings.emission})")
    )
    # read the RESOLVED orientation off the first case, not the Sweep request:
    # HOPG/h-BN hand-pin hkl_list and bypass dominant_reflections entirely, so
    # echoing sweep.n_families would claim a knob that's actually inert for them.
    print(
        f"crystal orientation: beam_uvw={cases[0]['beam_uvw']} "
        f"hkl_list ({len(cases[0]['hkl_list'])} reflections)={cases[0]['hkl_list']}"
    )

    # Always pass an explicit path named for the catalog material key: run_sweep's
    # default derives the name from the film crystal, which would make a named
    # stack (e.g. mos2-on-sio2-si) clobber/resume the plain film's checkpoint.
    # A --quick smoke test writes to its OWN checkpoint (<material>_quick.pkl), so
    # its coarse off-grid points never contaminate the real per-material sweep.
    ckpt = os.path.join(args.checkpoint_dir, stem)
    results = {}
    progress_file = getattr(args, "progress_file", None)
    progress_phase = getattr(args, "progress_phase", None)
    has_dashboard = hasattr(args, "_job_records")
    latest_progress = {
        "total_cases": len(cases),
        "cached_cases": 0,
        "completed_new_cases": 0,
    }
    latest_case = {}
    # Compute-weighted progress (item 6): cost is a pure function of a case dict
    # (sweep.case_cost), so the exact cached vs. done split run_sweep already
    # tracks (identity, not just a count) lets `attach` render a percent that
    # tracks relative matmul work instead of a flat case count -- see
    # notebooks/scan_app.py for the same cost-weighted meter run locally.
    latest_cost = {}
    initial_done_cost = None
    progress_timer = _ProgressTimer(progress_file) if progress_file is not None else None
    runtime_info = {}
    timing_info: dict[str, Any] = {
        "checkpoint_count": 0,
        "checkpoint_seconds_total": 0.0,
        "gpu_oom_retry_count_total": 0,
        "gpu_feed_wait_fraction": None,
    }
    activity_info = {
        "phase": "setup",
        "active_case": None,
        "in_flight_case_count": 0,
    }
    performance_state = {"state": "running"}
    performance_lock = threading.Lock()

    def _case_summary(case):
        from .montecarlo import runner as montecarlo_runner

        return {
            "configuration": str(case["name"]),
            "energy_keV": round(float(case["E0_keV"]), 3),
            "tilt_deg": round(float(case["tilt_deg"]), 2),
            "azimuth_deg": round(float(case["tilt_azim_deg"]), 2),
            "thickness_um": round(float(case["thickness_ang"]) / 1e4, 4),
            **montecarlo_runner.case_runtime_plan(case),
        }

    def _note_case(case):
        # Frontier crystal case just finished -- surface its parameters so a live
        # viewer can show what's under test (energy, both tilts, thickness).
        with performance_lock:
            latest_case.clear()
            latest_case.update(_case_summary(case))

    def _record_cost(done_cost, total_cost):
        nonlocal initial_done_cost
        with performance_lock:
            if initial_done_cost is None:
                initial_done_cost = done_cost
            latest_cost.update(done_cost=done_cost, total_cost=total_cost)

    def _record_progress(completed_new_cases, total_cases, cached_cases):
        with performance_lock:
            latest_progress.update(
                total_cases=total_cases,
                cached_cases=cached_cases,
                completed_new_cases=completed_new_cases,
            )
            progress_snapshot = dict(latest_progress)
            cost_snapshot = dict(latest_cost)
            case_snapshot = dict(latest_case) or None

        computed_cost = (
            None
            if initial_done_cost is None or "done_cost" not in cost_snapshot
            else max(0.0, cost_snapshot["done_cost"] - initial_done_cost)
        )
        if hasattr(args, "_job_records"):
            rec = {
                "material": material,
                "state": "running",
                "total_cases": progress_snapshot["total_cases"],
                "cached_cases": progress_snapshot["cached_cases"],
                "completed_new_cases": progress_snapshot["completed_new_cases"],
            }
            if case_snapshot:
                rec["current"] = case_snapshot
            if "done_cost" in cost_snapshot and "total_cost" in cost_snapshot:
                rec["done_cost"] = cost_snapshot["done_cost"]
                rec["total_cost"] = cost_snapshot["total_cost"]
            if progress_timer:
                rec.update(
                    progress_timer.snapshot(
                        completed_new_cases=progress_snapshot["completed_new_cases"],
                        computed_cost=computed_cost,
                    )
                )
            args._job_records[material] = rec

        if progress_file is not None:
            assert progress_timer is not None
            _write_progress_record(
                progress_file,
                material=material,
                phase=progress_phase,
                state="running",
                current=case_snapshot,
                **progress_snapshot,
                **cost_snapshot,
                **progress_timer.snapshot(
                    completed_new_cases=progress_snapshot["completed_new_cases"],
                    computed_cost=computed_cost,
                ),
            )

    def _record_runtime(info):
        with performance_lock:
            runtime_info.clear()
            runtime_info.update(info)

    def _record_timing(info):
        with performance_lock:
            checkpoint_seconds = info.get("checkpoint_seconds")
            if checkpoint_seconds is not None:
                timing_info["checkpoint_count"] += 1
                timing_info["checkpoint_seconds"] = checkpoint_seconds
                timing_info["checkpoint_seconds_total"] += checkpoint_seconds
                return
            retries = info.get("gpu_oom_retry_count", 0)
            timing_info["gpu_oom_retry_count_total"] += retries
            for key, value in info.items():
                if key == "gpu_oom_retry_count":
                    continue
                if key in {
                    "transport_seconds",
                    "spectrum_seconds",
                    "driver_wait_seconds",
                }:
                    timing_info[f"last_{key}"] = value
                else:
                    timing_info[key] = value
            wait = timing_info.get("driver_wait_seconds_total", 0.0)
            spectrum = timing_info.get("spectrum_seconds_total", 0.0)
            denominator = wait + spectrum
            timing_info["gpu_feed_wait_fraction"] = wait / denominator if denominator > 0 else None

    def _record_activity(info):
        info = dict(info)
        active = info.pop("case", None)
        active_case = _case_summary(active) if active is not None else None
        with performance_lock:
            activity_info.clear()
            activity_info.update(info, active_case=active_case)

    def _performance_context():
        with performance_lock:
            return {
                **runtime_info,
                **latest_progress,
                **latest_cost,
                **timing_info,
                **activity_info,
                "current": dict(latest_case) or None,
                **performance_state,
            }

    if progress_file is not None:
        assert progress_timer is not None
        _write_progress_record(
            progress_file,
            material=material,
            phase=progress_phase,
            state="running",
            **latest_progress,
            **progress_timer.snapshot(),
        )
    performance_logger = None
    performance_profile = getattr(args, "performance_profile", None)
    if performance_profile is not None:
        from .performance_profile import PerformanceLogger
        from .profiles import _jsonable

        performance_root = Path(getattr(args, "performance_dir", None) or "performance-profiles")
        profile_dir = performance_root / performance_profile
        performance_logger = PerformanceLogger(
            profile_dir / f"{material}.ndjson",
            profile=performance_profile,
            material=material,
            static={
                "fidelity": fidelity,
                "catalog_profile": identity.get("catalog_profile", "standard"),
                "parameter_sha256": identity["parameter_sha256"],
                "checkpoint_stem": stem,
                "total_case_count": len(cases),
                # Beam parameters now resolve through Sweep.beam, including
                # profile TOML values plus CLI overrides. Log that resolved
                # dataclass instead of reconstructing legacy top-level fields.
                "beam_parameters": _jsonable(sweep.beam),
            },
            context=_performance_context,
            latest_path=profile_dir / f"{material}.latest.json",
            interval_seconds=getattr(args, "performance_interval", 5.0),
        )
        performance_logger.start()
    # Shared per-case cache gates. Default reads + writes; --recompute / --no-cache
    # / a -p perf run narrow this (see the `run` command). cache_read also drives
    # the per-stem resume, so a --no-cache/--recompute/perf run recomputes rather
    # than resume-skipping (the perf-run "measures near-nothing" bug).
    from .profiles import case_content_key

    cache_read = getattr(args, "cache_read", True)
    cache_write = getattr(args, "cache_write", True)
    try:
        if progress_timer is not None:
            progress_timer.start()
        result = run_sweep(
            cases,
            results,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_path=ckpt,
            max_workers=args.workers,
            resume=cache_read,
            content_key_fn=case_content_key,
            cache_read=cache_read,
            cache_write=cache_write,
            progress=not getattr(args, "no_progress", False),
            on_progress=(
                _record_progress
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            on_case=(
                _note_case
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            on_runtime=_record_runtime if performance_logger is not None else None,
            on_timing=_record_timing if performance_logger is not None else None,
            on_activity=_record_activity if performance_logger is not None else None,
            max_seconds=max_seconds,
            dataset_identity=identity,
            case_cost_fn=(
                case_cost
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            on_cost=(
                _record_cost
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
        )
        # run_sweep returns a bool (complete?). Only a bare None -- test doubles
        # that predate the budget feature and don't bother returning anything --
        # is read as complete; every other falsy return fails loud as incomplete.
        complete = True if result is None else bool(result)
    except BaseException:
        with performance_lock:
            performance_state["state"] = "failed"
        if progress_file is not None:
            assert progress_timer is not None
            _write_progress_record(
                progress_file,
                material=material,
                phase=progress_phase,
                state="failed",
                **latest_progress,
                **latest_cost,
                **progress_timer.snapshot(
                    completed_new_cases=latest_progress["completed_new_cases"],
                    computed_cost=(
                        None
                        if initial_done_cost is None or "done_cost" not in latest_cost
                        else max(0.0, latest_cost["done_cost"] - initial_done_cost)
                    ),
                ),
            )
        if performance_logger is not None:
            performance_logger.close("failed")
        raise
    with performance_lock:
        performance_state["state"] = "done" if complete else "paused"
    if performance_logger is not None:
        performance_logger.close("done" if complete else "paused")

    if hasattr(args, "_job_records") and material in args._job_records:
        args._job_records[material]["state"] = "done" if complete else "paused"

    if progress_file is not None:
        assert progress_timer is not None
        _write_progress_record(
            progress_file,
            material=material,
            phase=progress_phase,
            state="done" if complete else "paused",
            **latest_progress,
            **latest_cost,
            **progress_timer.snapshot(
                completed_new_cases=latest_progress["completed_new_cases"],
                computed_cost=(
                    None
                    if initial_done_cost is None or "done_cost" not in latest_cost
                    else max(0.0, latest_cost["done_cost"] - initial_done_cost)
                ),
            ),
        )
    n = sum(len(v) for v in results.values())
    if complete:
        print(f"{_cli_core.paint('done', 'done')} -> {args.checkpoint_dir}/{stem}/ ({n} records)")
    else:
        print(
            f"{_cli_core.paint('paused', 'warning')} (budget) -> "
            f"{args.checkpoint_dir}/{stem}/ ({n} records)"
        )
    return complete


class _ProgressTimer:
    """Persist additive active-process time and measured work across resumes."""

    def __init__(self, path, *, time_fn=None):
        self._time_fn = time.monotonic if time_fn is None else time_fn
        self._started = None
        self._base_seconds = 0.0
        self._base_cases = 0
        self._base_cost = 0.0
        if path is None:
            return
        try:
            previous = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(previous, dict):
            return
        seconds = previous.get("active_compute_seconds")
        cases = previous.get("measured_new_cases")
        cost = previous.get("measured_new_cost")
        if (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and 0 <= seconds < math.inf
        ):
            self._base_seconds = float(seconds)
        if isinstance(cases, int) and not isinstance(cases, bool) and cases >= 0:
            self._base_cases = cases
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and 0 <= cost < math.inf:
            self._base_cost = float(cost)

    def start(self):
        if self._started is None:
            self._started = self._time_fn()

    def snapshot(self, *, completed_new_cases=0, computed_cost=None):
        elapsed = 0.0 if self._started is None else max(0.0, self._time_fn() - self._started)
        fields = {
            "active_compute_seconds": self._base_seconds + elapsed,
            "measured_new_cases": self._base_cases + max(0, int(completed_new_cases)),
        }
        if computed_cost is not None or self._base_cost > 0:
            fields["measured_new_cost"] = self._base_cost + max(0.0, float(computed_cost or 0.0))
        return fields


def _write_progress_record(
    path,
    *,
    material,
    total_cases,
    cached_cases,
    completed_new_cases,
    state,
    phase=None,
    current=None,
    done_cost=None,
    total_cost=None,
    active_compute_seconds=None,
    measured_new_cases=None,
    measured_new_cost=None,
):
    """Atomically replace one scan's compact JSON progress record.

    ``current`` (optional) is the frontier crystal case's parameters (energy,
    both tilts, thickness) so a live viewer can show what's under test; it is
    omitted from the record when None (start/done/failed/paused snapshots).
    ``phase`` identifies parallel/serial orchestration phases sharing one
    material name; legacy callers omit it.
    ``done_cost``/``total_cost`` (optional) are relative compute-weight sums
    (``sweep.case_cost``) enabling a compute-aware progress bar; omitted when
    either is None -- callers that don't cost-weight (rebrem, reline, blaze)
    keep writing the same record shape as before.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "material": material,
        "total_cases": total_cases,
        "cached_cases": cached_cases,
        "completed_new_cases": completed_new_cases,
        "state": state,
    }
    if phase is not None:
        record["phase"] = phase
    if current:
        record["current"] = current
    if done_cost is not None and total_cost is not None:
        record["done_cost"] = done_cost
        record["total_cost"] = total_cost
    if active_compute_seconds is not None:
        record["active_compute_seconds"] = active_compute_seconds
    if measured_new_cases is not None:
        record["measured_new_cases"] = measured_new_cases
    if measured_new_cost is not None:
        record["measured_new_cost"] = measured_new_cost
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="cxr run")


if __name__ == "__main__":
    raise SystemExit(main())
