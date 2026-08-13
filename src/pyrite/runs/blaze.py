"""Headless blazed-crystal (sawtooth entrance-face) CXR sweep driver.

Runs the Monte-Carlo CXR parameter sweep for one material with a blazed
groove entrance face, writing to a checkpoint kept separate from the
crystal's flat-face (``cxr run``) checkpoint:
``checkpoints/<material>_blazed.pkl``. Groove face angles are set
automatically by the existing geometry code
(``montecarlo.groove.blazed_groove_spec``); this driver only supplies groove
spacing and the scan grid.

    cxr material blaze hopg --spacing 2e-6 --energy 30 --polar 25 45

adds a HOPG crystal with 2 micron-spaced grooves, scanned over polar angles
25 deg and 45 deg.

Grooves are v1 single-slab: no substrate/stack, ``theta_obs = 90 deg``,
``tilt_azim = 180 deg``, ``0 < tilt < 90 deg``. These are enforced by
``sweep._reject_invalid_groove_geometry``; this driver constructs every Sweep
to satisfy them and does not relax them. The crystal keeps the default finite
5x5 mm footprint (same as flat sweeps), so blazed records carry a real electron
hit/miss fraction (``hit_frac``, shown in the analysis_app heatmap) rather than
the trivial 1.0 of a laterally infinite slab. See
``docs/validation/geometry/blazed-groove-geometry.md``.

The ``if __name__ == "__main__"`` guard on the entry point is REQUIRED, not
stylistic -- same process-pool re-import reason as ``scan.py``; see its
module docstring.
"""

import io
import os
import time
from contextlib import redirect_stderr, redirect_stdout

from ..cli import _core as _cli_core
from ..cli import json as cli_json

# Lazy runtime bindings keep command help light and focused tests patchable.
default_settings = None
format_penetration_watchdog_summary = None
gate_cases_by_penetration = None
material_sweep = None
run_sweep = None
build_cases = None
_write_progress_record = None
_ProgressTimer = None
validate_materials = None


def _load_runtime() -> None:
    global default_settings
    global format_penetration_watchdog_summary
    global gate_cases_by_penetration
    global material_sweep
    global run_sweep
    global build_cases
    global _write_progress_record
    global _ProgressTimer
    global validate_materials

    from ..campaign import config as config_module
    from ..campaign import sweep as sweep_module
    from . import run as run_module
    from . import scan as scan_module

    default_settings = default_settings or config_module.default_settings
    format_penetration_watchdog_summary = (
        format_penetration_watchdog_summary or config_module.format_penetration_watchdog_summary
    )
    gate_cases_by_penetration = gate_cases_by_penetration or config_module.gate_cases_by_penetration
    material_sweep = material_sweep or config_module.material_sweep
    run_sweep = run_sweep or run_module.run_sweep
    build_cases = build_cases or sweep_module.build_cases
    _write_progress_record = _write_progress_record or scan_module._write_progress_record
    _ProgressTimer = _ProgressTimer or scan_module._ProgressTimer
    validate_materials = validate_materials or scan_module.validate_materials


def _pair_energies_and_spacings(energies, spacings):
    """Zip beam energies (keV) with groove spacings (already in angstroms).

    Equal-length lists zip element-wise; a single spacing broadcasts to every
    energy; any other length mismatch is a user error.
    """
    if len(spacings) == len(energies):
        return list(zip(energies, spacings, strict=True))
    if len(spacings) == 1:
        return [(energy, spacings[0]) for energy in energies]
    raise SystemExit(
        f"--energy takes {len(energies)} value(s) but --spacing takes "
        f"{len(spacings)}; supply one spacing, or one spacing per energy"
    )


def _run_json(args):
    started = time.monotonic()
    material = args.material
    checkpoint = os.path.join(args.checkpoint_dir, f"{material}_blazed")
    completed = []
    failed = []
    errors = {}
    resumable = False
    args.no_progress = True
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            run(args)
        completed.append(material)
    except SystemExit as exc:
        failed.append(material)
        resumable = exc.code == 75
        errors[material] = "resumable work remains" if resumable else str(exc.code)
    except Exception as exc:
        failed.append(material)
        errors[material] = str(exc) or type(exc).__name__
    result = cli_json.operation_summary(
        "blaze",
        [material],
        completed,
        failed_materials=failed,
        checkpoints=[checkpoint],
        elapsed_seconds=time.monotonic() - started,
        resumable=resumable,
        material_errors=errors,
    )
    _cli_core.emit_json_result(result, failure_exit=75 if resumable else 1)


def run(args):
    _load_runtime()
    assert default_settings is not None
    assert format_penetration_watchdog_summary is not None
    assert gate_cases_by_penetration is not None
    assert material_sweep is not None
    assert run_sweep is not None
    assert build_cases is not None
    assert _write_progress_record is not None
    assert _ProgressTimer is not None
    assert validate_materials is not None
    write_progress_record = _write_progress_record

    validate_materials([args.material])
    spacings_ang = [spacing_m * 1e10 for spacing_m in args.spacing]
    pairs = _pair_energies_and_spacings(args.energy, spacings_ang)

    settings = default_settings()
    cases = []
    for energy_keV, spacing_ang in pairs:
        # crystal_width_mm/crystal_height_mm are left at the material_sweep
        # default (finite 5x5 mm footprint, same as flat sweeps) so the blazed
        # run records a real electron hit/miss fraction (hit_frac) instead of the
        # trivial 1.0 of a laterally infinite slab -- surfaced in the analysis_app
        # heatmap. substrate/stack stay None (grooves are v1 single-slab).
        overrides = dict(
            theta_obs_deg=90.0,
            energy_keV=[energy_keV],
            groove_spacing_ang=spacing_ang,
            tilt_azim_deg=180.0,
            substrate=None,
            stack=None,
        )
        if args.angles is not None:
            overrides["tilt_deg"] = args.angles
        sweep = material_sweep(args.material, **overrides)
        cases.extend(
            build_cases(
                sweep,
                settings.n_electrons,
                settings.n_electrons_brem,
                coherent_emission=settings.coherent_emission,
                xray_dispersion=settings.xray_dispersion,
            )
        )

    cases, dropped = gate_cases_by_penetration(cases)
    summary = format_penetration_watchdog_summary(dropped, material=args.material)
    if summary is not None:
        print(summary)
    print(
        f"{args.material}: {len(cases)} cases across "
        f"{len({c['name'] for c in cases})} configs (blazed)"
    )

    # Explicit path so the blazed sweep NEVER shares (or clobbers/resumes) the
    # flat-face checkpoint -- see the "Checkpoint" section of the design doc.
    stem = f"{args.material}_blazed"
    ckpt = os.path.join(args.checkpoint_dir, stem)
    results = {}
    max_seconds = None if args.max_minutes is None else args.max_minutes * 60.0

    progress_file = getattr(args, "progress_file", None)
    progress_timer = _ProgressTimer(progress_file) if progress_file is not None else None
    latest_progress = {
        "total_cases": len(cases),
        "cached_cases": 0,
        "completed_new_cases": 0,
    }

    def _record_progress(completed_new_cases, total_cases, cached_cases):
        latest_progress.update(
            total_cases=total_cases,
            cached_cases=cached_cases,
            completed_new_cases=completed_new_cases,
        )
        if progress_file is not None:
            assert progress_timer is not None
            write_progress_record(
                progress_file,
                material=args.material,
                state="running",
                **latest_progress,
                **progress_timer.snapshot(completed_new_cases=completed_new_cases),
            )

    if progress_file is not None:
        assert progress_timer is not None
        write_progress_record(
            progress_file,
            material=args.material,
            state="running",
            **latest_progress,
            **progress_timer.snapshot(),
        )
    try:
        if progress_timer is not None:
            progress_timer.start()
        result = run_sweep(
            cases,
            results,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_path=ckpt,
            max_workers=args.workers,
            progress=not getattr(args, "no_progress", False),
            on_progress=_record_progress if progress_file is not None else None,
            max_seconds=max_seconds,
        )
        # run_sweep returns a bool (complete?); a bare None is only ever a test
        # double predating the budget feature -- treat as complete.
        complete = True if result is None else bool(result)
    except BaseException:
        if progress_file is not None:
            assert progress_timer is not None
            write_progress_record(
                progress_file,
                material=args.material,
                state="failed",
                **latest_progress,
                **progress_timer.snapshot(
                    completed_new_cases=latest_progress["completed_new_cases"]
                ),
            )
        raise
    if progress_file is not None:
        assert progress_timer is not None
        write_progress_record(
            progress_file,
            material=args.material,
            state="done" if complete else "paused",
            **latest_progress,
            **progress_timer.snapshot(completed_new_cases=latest_progress["completed_new_cases"]),
        )

    n = sum(len(v) for v in results.values())
    if complete:
        print(f"{_cli_core.paint('done', 'done')} -> {ckpt} ({n} records)")
    else:
        print(f"{_cli_core.paint('paused', 'warning')} (budget) -> {ckpt} ({n} records)")
    if not complete:
        raise SystemExit(75)  # EX_TEMPFAIL: budget hit, work remains


def __getattr__(name):
    # ``command`` moved to pyrite.cli.commands.blaze; keep the module-level seam
    # so ``blaze.command`` and dispatch keep resolving without an import cycle.
    if name == "command":
        from ..cli.commands.blaze import command

        return command
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv=None):
    from ..cli.commands.blaze import command

    return _cli_core.run(command, argv, prog_name="blaze.py")


if __name__ == "__main__":
    raise SystemExit(main())
