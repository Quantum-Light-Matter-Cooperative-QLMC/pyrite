"""Headless blazed-crystal (sawtooth entrance-face) CXR sweep driver.

Runs the Monte-Carlo CXR parameter sweep for one material with a blazed
groove entrance face, writing to a checkpoint kept separate from the
crystal's flat-face (``cxr scan``) checkpoint:
``checkpoints/<material>_blazed.pkl``. Groove face angles are set
automatically by the existing geometry code
(``montecarlo.groove.blazed_groove_spec``); this driver only supplies groove
spacing and the scan grid.

    cxr blaze hopg --spacing 2e-6 --energy 30 --angles 25 45

adds a HOPG crystal with 2 micron-spaced grooves, scanned over polar angles
25 deg and 45 deg.

Grooves are v1 single-slab: laterally infinite, no substrate/stack, no finite
footprint, ``theta_obs = 90 deg``, ``tilt_azim = 180 deg``, ``0 < tilt < 90
deg``. These are enforced by ``sweep._reject_invalid_groove_geometry``; this
driver constructs every Sweep to satisfy them and does not relax them. See
``docs/superpowers/specs/2026-07-23-cxr-blaze-grooved-sweep-design.md``.

The ``if __name__ == "__main__"`` guard on the entry point is REQUIRED, not
stylistic -- same process-pool re-import reason as ``scan.py``; see its
module docstring.
"""

import argparse
import os
from pathlib import Path

from .config import (
    default_settings,
    format_penetration_watchdog_summary,
    gate_cases_by_penetration,
    material_sweep,
)
from .run import run_sweep
from .scan import _write_progress_record, validate_materials
from .sweep import build_cases


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


def _build_parser(ap):
    ap.add_argument("material", help="catalog crystal key, e.g. hopg")
    ap.add_argument(
        "--energy",
        type=float,
        nargs="+",
        required=True,
        metavar="E",
        help="beam energies in keV (one or more)",
    )
    ap.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        required=True,
        metavar="S",
        help="groove spacing(s) in meters (one, or one per --energy)",
    )
    ap.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=None,
        metavar="A",
        help="polar tilt_deg values (default: the material's std catalog grid)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="run_cases max_workers (default auto; 0 = serial, no transport pool)",
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="soft wall-clock budget; exit 75 if work remains (chained remote slices)",
    )
    ap.add_argument("--progress-file", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(func=run)
    return ap


def add_subparser(sub):
    """Register the ``blaze`` subcommand on an argparse subparsers object."""
    return _build_parser(
        sub.add_parser(
            "blaze", help="run one material's blazed-crystal MC sweep -> _blazed checkpoint"
        )
    )


def run(args):
    validate_materials([args.material])
    spacings_ang = [spacing_m * 1e10 for spacing_m in args.spacing]
    pairs = _pair_energies_and_spacings(args.energy, spacings_ang)

    settings = default_settings()
    cases = []
    for energy_keV, spacing_ang in pairs:
        overrides = dict(
            theta_obs_deg=90.0,
            energy_keV=[energy_keV],
            groove_spacing_ang=spacing_ang,
            tilt_azim_deg=180.0,
            crystal_width_mm=None,
            crystal_height_mm=None,
            substrate=None,
            stack=None,
        )
        if args.angles is not None:
            overrides["tilt_deg"] = args.angles
        sweep = material_sweep(args.material, **overrides)
        cases.extend(build_cases(sweep, settings.n_electrons, settings.n_electrons_brem))

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
    ckpt = os.path.join(args.checkpoint_dir, f"{stem}.pkl")
    results = {}
    max_seconds = None if args.max_minutes is None else args.max_minutes * 60.0

    progress_file = getattr(args, "progress_file", None)
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
            _write_progress_record(
                progress_file,
                material=args.material,
                state="running",
                **latest_progress,
            )

    if progress_file is not None:
        _write_progress_record(
            progress_file,
            material=args.material,
            state="running",
            **latest_progress,
        )
    try:
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
            _write_progress_record(
                progress_file,
                material=args.material,
                state="failed",
                **latest_progress,
            )
        raise
    if progress_file is not None:
        _write_progress_record(
            progress_file,
            material=args.material,
            state="done" if complete else "paused",
            **latest_progress,
        )

    n = sum(len(v) for v in results.values())
    if complete:
        print(f"done -> {ckpt} ({n} records)")
    else:
        print(f"paused (budget) -> {ckpt} ({n} records)")
    if not complete:
        raise SystemExit(75)  # EX_TEMPFAIL: budget hit, work remains


def main(argv=None):
    ap = _build_parser(
        argparse.ArgumentParser(
            prog="blaze.py", description="headless blazed-crystal CXR scan runner"
        )
    )
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
