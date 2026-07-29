"""Headless CXR scan runner (the library twin of ``notebooks/scan_app.py``).

Runs the Monte-Carlo CXR parameter sweep for one material and writes the
per-material checkpoint (checkpoints/<material>.pkl). Use this to run sweeps
non-interactively -- in particular over SSH on the GPU box; see cxr_mc.remote,
which drives this and pulls the checkpoint back so interactive analysis and
static-HTML export can stay on the laptop.

    cxr scan mose2                # the full per-material grid (config)
    cxr scan mose2 --fidelity survey  # provisional reduced survey
    cxr scan mose2 --quick           # tiny grid: smoke test / pipeline check
    cxr scan mose2 --workers 0    # serial (no transport worker pool)
    cxr scan --all                # verified `materials` list only
    cxr scan --all --include-unverified-dw --include-high-energy  # + no_verified_dw + high_energy_materials (>=150 keV)
    cxr scan --all --include-high-energy --high-energy-min-kev 200  # override the 150 keV floor
    cxr scan -A                    # every material in mats_to_sim.toml, no exceptions

(equivalently ``python -m cxr_mc._entry.scan mose2`` via the module shim).

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
import os
import re
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
    """Read the ordered ``materials`` list used by ``cxr scan --all``."""
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
    "scan",
    help=(
        "Run one material's MC sweep and write its checkpoint.\n\n"
        "Pass MATERIAL, --all, or -A/--actually-all, never more than one. --all "
        "runs the verified `materials` list, optionally widened with "
        "--include-unverified-dw and/or --include-high-energy (floored at "
        "--high-energy-min-kev, default 150). -A runs every material in "
        "mats_to_sim.toml, no exceptions.\n\n"
        "Resumes compatible checkpoints in CHECKPOINTS. Full writes "
        "<material>.pkl-compatible data in <material>/; variants use "
        "identity-qualified stems.\n\n"
        "Beam overrides: --beam-transverse-fwhm-{x,y}-mm, "
        "--beam-bunch-length-fs/--beam-long-shape, "
        "--beam-rep-rate-hz, and --beam-bunch-charge-pc replace resolved "
        "catalog-profile distribution values. Beam energy always remains the "
        "material scan grid; this command deliberately has no energy override."
    ),
)
@click.argument(
    "material",
    required=False,
    type=_cli_completion.MATERIAL,
    shell_complete=_cli_completion.complete_material,
)
@click.option(
    "-a",
    "--all",
    "all_",
    is_flag=True,
    help="Run mats_to_sim.toml's verified `materials` list; takes no MATERIAL.",
)
@click.option(
    "-A",
    "--actually-all",
    "actually_all",
    is_flag=True,
    help=(
        "Run every material in mats_to_sim.toml -- materials, no_verified_dw, "
        "high_energy_materials, and materials_to_leave_out combined; takes no "
        "MATERIAL. Not combined with --all/--include-unverified-dw/--include-high-energy."
    ),
)
@click.option(
    "--include-unverified-dw",
    is_flag=True,
    help="With --all, also run mats_to_sim.toml's no_verified_dw materials.",
)
@click.option(
    "--include-high-energy",
    is_flag=True,
    help=(
        "With --all, also run mats_to_sim.toml's high_energy_materials, "
        "filtered to --high-energy-min-kev and above."
    ),
)
@click.option(
    "--high-energy-min-kev",
    type=_cli_core.POSITIVE_FLOAT,
    default=None,
    metavar="KEV",
    help=(
        "Energy floor applied to any selected high_energy_materials member "
        "[default: 150.0 when selected via --include-high-energy/-A]. With an "
        "explicit MATERIAL, applies only if that material is itself a "
        "high_energy_materials entry; a no-op on every other material."
    ),
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
    "--beam-uvw",
    type=int,
    nargs=3,
    callback=_beam_uvw,
    default=None,
    metavar="H K L",
    help="Override nonzero integer beam zone axis [uvw].",
)
@click.option(
    "--beam-transverse-fwhm-x-mm",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="MM",
    help="Beam horizontal Gaussian spot FWHM [mm]; overrides the resolved profile.",
)
@click.option(
    "--beam-transverse-fwhm-y-mm",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="MM",
    help="Beam vertical Gaussian spot FWHM [mm]; overrides the resolved profile.",
)
@click.option(
    "--beam-bunch-length-fs",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="FS",
    help="Longitudinal bunch RMS length [fs]; 0 is a point bunch.",
)
@click.option(
    "--beam-long-shape",
    type=click.Choice(("gaussian", "uniform"), case_sensitive=False),
    default=None,
    metavar="SHAPE",
    help="Longitudinal bunch shape: gaussian or uniform; needs --beam-bunch-length-fs.",
)
@click.option(
    "--beam-rep-rate-hz",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="HZ",
    help="Beam repetition rate [Hz]; overrides the resolved profile.",
)
@click.option(
    "--beam-bunch-charge-pc",
    type=_cli_core.NONNEGATIVE_FLOAT,
    default=None,
    metavar="PC",
    help="Single-bunch charge [pC]; overrides the resolved profile.",
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
    "--performance-profile",
    callback=_performance_profile,
    default=None,
    metavar="NAME",
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
    "--performance-interval",
    type=_cli_core.POSITIVE_FLOAT,
    default=5.0,
    hidden=True,
)
@click.option("--progress-file", type=click.Path(path_type=Path), default=None, hidden=True)
@click.option("--no-progress", is_flag=True, hidden=True)
@_cli_core.fidelity_option()
@click.option(
    "--profile",
    "catalog_profile",
    default="standard",
    show_default=True,
    help="Catalog profile to run (e.g. standard, sub_100keV).",
)
@_cli_core.json_option
def command(
    material,
    all_,
    actually_all,
    include_unverified_dw,
    include_high_energy,
    high_energy_min_kev,
    workers,
    fidelity,
    catalog_profile,
    quick,
    n_families,
    beam_uvw,
    beam_transverse_fwhm_x_mm,
    beam_transverse_fwhm_y_mm,
    beam_bunch_length_fs,
    beam_long_shape,
    beam_rep_rate_hz,
    beam_bunch_charge_pc,
    checkpoint_dir,
    max_minutes,
    performance_profile,
    performance_dir,
    performance_interval,
    progress_file,
    no_progress,
    json_output,
):
    """Click entry point for the staged root migration."""
    if quick and fidelity != "full":
        raise click.UsageError("--quick cannot be combined with --fidelity survey")
    if actually_all and material is not None:
        raise click.UsageError("scan -A/--actually-all does not take a material name")
    if actually_all and all_:
        raise click.UsageError("scan -A/--actually-all already includes --all; drop --all")
    if actually_all and include_unverified_dw:
        raise click.UsageError("scan -A/--actually-all already includes --include-unverified-dw")
    if actually_all and include_high_energy:
        raise click.UsageError("scan -A/--actually-all already includes --include-high-energy")
    if all_ and material is not None:
        raise click.UsageError("scan --all does not take a material name")
    if include_unverified_dw and not all_:
        raise click.UsageError("--include-unverified-dw requires --all")
    if include_high_energy and not all_:
        raise click.UsageError("--include-high-energy requires --all")
    if not all_ and not actually_all and material is None:
        raise click.UsageError("scan needs a material name, or use --all/-A")
    if not json_output:
        return _cli_core.invoke_legacy(
            run,
            material=material,
            all=all_,
            actually_all=actually_all,
            include_unverified_dw=include_unverified_dw,
            include_high_energy=include_high_energy,
            high_energy_min_kev=high_energy_min_kev,
            workers=workers,
            fidelity=fidelity,
            catalog_profile=catalog_profile,
            quick=quick,
            n_families=n_families,
            beam_uvw=beam_uvw,
            beam_transverse_fwhm_x_mm=beam_transverse_fwhm_x_mm,
            beam_transverse_fwhm_y_mm=beam_transverse_fwhm_y_mm,
            beam_bunch_length_fs=beam_bunch_length_fs,
            beam_long_shape=beam_long_shape,
            beam_rep_rate_hz=beam_rep_rate_hz,
            beam_bunch_charge_pc=beam_bunch_charge_pc,
            checkpoint_dir=checkpoint_dir,
            max_minutes=max_minutes,
            performance_profile=performance_profile,
            performance_dir=performance_dir,
            performance_interval=performance_interval,
            progress_file=progress_file,
            no_progress=no_progress,
        )
    return _cli_core.invoke_legacy(
        _run_json,
        material=material,
        all=all_,
        actually_all=actually_all,
        include_unverified_dw=include_unverified_dw,
        include_high_energy=include_high_energy,
        high_energy_min_kev=high_energy_min_kev,
        workers=workers,
        fidelity=fidelity,
        catalog_profile=catalog_profile,
        quick=quick,
        n_families=n_families,
        beam_uvw=beam_uvw,
        beam_transverse_fwhm_x_mm=beam_transverse_fwhm_x_mm,
        beam_transverse_fwhm_y_mm=beam_transverse_fwhm_y_mm,
        beam_bunch_length_fs=beam_bunch_length_fs,
        beam_long_shape=beam_long_shape,
        beam_rep_rate_hz=beam_rep_rate_hz,
        beam_bunch_charge_pc=beam_bunch_charge_pc,
        checkpoint_dir=checkpoint_dir,
        max_minutes=max_minutes,
        performance_profile=performance_profile,
        performance_dir=performance_dir,
        performance_interval=performance_interval,
        progress_file=progress_file,
        no_progress=no_progress,
    )


DEFAULT_HIGH_ENERGY_MIN_KEV = 150.0


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


def _selected(args):
    """Resolve MATERIAL/--all/-A plus the include-* flags into an ordered,
    deduplicated material list; also stashes ``args.high_energy_floor_map``
    (material -> keV floor) so ``_resolved_run`` can filter high-energy-only
    materials' energy grids regardless of which flag pulled them in."""
    all_ = getattr(args, "all", False)
    actually_all = getattr(args, "actually_all", False)
    catalog_profile = getattr(args, "catalog_profile", "standard")
    args.high_energy_floor_map = {}
    if all_ or actually_all:
        if args.material is not None:
            raise SystemExit("scan --all/-A does not take a material name")
        groups = load_manifest_groups()
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
        allowed = set(materials)
        high_energy_selected &= allowed
        if high_energy_selected:
            floor = getattr(args, "high_energy_min_kev", None)
            floor = DEFAULT_HIGH_ENERGY_MIN_KEV if floor is None else floor
            args.high_energy_floor_map = {m: floor for m in high_energy_selected}
    elif args.material is not None:
        materials = [args.material]
        materials = validate_catalog_profile(catalog_profile, materials, intersect=False)
        # A direct MATERIAL invocation only applies the floor when the flag is
        # explicitly given (no surprise default) AND the material is itself a
        # high_energy_materials entry (a no-op otherwise) -- this is what lets
        # `cxr remote submit` forward one shared --high-energy-min-kev across a
        # mixed batch without it misfiring on non-high-energy materials.
        floor = getattr(args, "high_energy_min_kev", None)
        if floor is not None:
            high_energy_materials = load_manifest_groups().get("high_energy_materials", [])
            if args.material in high_energy_materials:
                args.high_energy_floor_map = {args.material: floor}
    else:
        raise SystemExit("scan needs a material name, or use --all/-A")
    validate_materials(materials)
    return materials


def run(args):
    materials = _selected(args)
    # One deadline spans the whole invocation (including --all): each material
    # below gets whatever's left of it, not a fresh --max-minutes apiece.
    deadline = None
    if getattr(args, "max_minutes", None) is not None:
        deadline = time.monotonic() + args.max_minutes * 60.0
    incomplete = False
    for material in materials:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not _run_material(args, material, max_seconds=remaining):
            incomplete = True
    if incomplete:
        raise SystemExit(75)  # EX_TEMPFAIL: budget hit, work remains


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
        "scan",
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


def _resolved_run(args, material):
    """Resolve settings, sweep, identity, and collision-free checkpoint stem."""
    import numpy as np

    _load_runtime()
    assert default_settings is not None
    assert material_sweep is not None

    fidelity = getattr(args, "fidelity", "full")
    catalog_profile = getattr(args, "catalog_profile", "standard")
    performance_profile = getattr(args, "performance_profile", None)
    if performance_profile is not None:
        if catalog_profile not in ("standard", performance_profile):
            raise click.UsageError(
                "--performance-profile and --profile must name the same catalog profile"
            )
        catalog_profile = performance_profile
    settings = default_settings() if fidelity == "full" else default_settings(fidelity)
    overrides = {}
    if getattr(args, "quick", False):
        overrides.update(
            # Start at 5 deg, not 0: tilt=0 is a banned emission geometry
            # (issue_notes.md #1), and build_cases rejects it.
            tilt_deg=np.linspace(5.0, 85.0, 5),
            tilt_azim_deg=np.array([10.0, 30.0]),
            energy_keV=[30, 50],
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
    profile = identity["fidelity"]

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    cases, dropped = gate_cases_by_penetration(cases)
    summary = format_penetration_watchdog_summary(dropped, material=material)
    if summary is not None:
        print(summary)
    print(
        f"{material}: {len(cases)} cases across "
        f"{len({c['name'] for c in cases})} configs "
        f"[profile={profile}, parameters={identity['parameter_sha256'][:12]}]"
        + (" (quick grid)" if args.quick else "")
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
        with performance_lock:
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
        if progress_file is not None:
            _write_progress_record(
                progress_file,
                material=material,
                state="running",
                current=case_snapshot,
                **progress_snapshot,
                **cost_snapshot,
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
        _write_progress_record(
            progress_file,
            material=material,
            state="running",
            **latest_progress,
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
                "fidelity": profile,
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
    try:
        result = run_sweep(
            cases,
            results,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_path=ckpt,
            max_workers=args.workers,
            progress=not getattr(args, "no_progress", False),
            on_progress=(
                _record_progress
                if progress_file is not None or performance_logger is not None
                else None
            ),
            on_case=(
                _note_case if progress_file is not None or performance_logger is not None else None
            ),
            on_runtime=_record_runtime if performance_logger is not None else None,
            on_timing=_record_timing if performance_logger is not None else None,
            on_activity=_record_activity if performance_logger is not None else None,
            max_seconds=max_seconds,
            dataset_identity=identity,
            case_cost_fn=(
                case_cost if progress_file is not None or performance_logger is not None else None
            ),
            on_cost=(
                _record_cost
                if progress_file is not None or performance_logger is not None
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
            _write_progress_record(
                progress_file,
                material=material,
                state="failed",
                **latest_progress,
                **latest_cost,
            )
        if performance_logger is not None:
            performance_logger.close("failed")
        raise
    with performance_lock:
        performance_state["state"] = "done" if complete else "paused"
    if performance_logger is not None:
        performance_logger.close("done" if complete else "paused")
    if progress_file is not None:
        _write_progress_record(
            progress_file,
            material=material,
            state="done" if complete else "paused",
            **latest_progress,
            **latest_cost,
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


def _write_progress_record(
    path,
    *,
    material,
    total_cases,
    cached_cases,
    completed_new_cases,
    state,
    current=None,
    done_cost=None,
    total_cost=None,
):
    """Atomically replace one scan's compact JSON progress record.

    ``current`` (optional) is the frontier crystal case's parameters (energy,
    both tilts, thickness) so a live viewer can show what's under test; it is
    omitted from the record when None (start/done/failed/paused snapshots).
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
    if current:
        record["current"] = current
    if done_cost is not None and total_cost is not None:
        record["done_cost"] = done_cost
        record["total_cost"] = total_cost
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None):
    return _cli_core.run(command, argv, prog_name="cxr scan")


if __name__ == "__main__":
    raise SystemExit(main())
