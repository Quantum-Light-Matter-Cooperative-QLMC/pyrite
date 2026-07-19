"""Headless CXR scan runner (the library twin of ``notebooks/scan_app.py``).

Runs the Monte-Carlo CXR parameter sweep for one material and writes the
per-material checkpoint (checkpoints/<material>.pkl). Use this to run sweeps
non-interactively -- in particular over SSH on the GPU box; see cxr_mc.remote,
which drives this and pulls the checkpoint back so interactive analysis and
static-HTML export can stay on the laptop.

    cxr scan mose2                # the full per-material grid (config)
    cxr scan mose2 --quick        # tiny grid: smoke test / pipeline check
    cxr scan mose2 --workers 0    # serial (no transport worker pool)

(equivalently ``python scan.py mose2`` via the root shim).

The ``if __name__ == "__main__"`` guard on the entry point is REQUIRED, not
stylistic: run_cases farms the electron transport out to a process pool, and the
default start method is 'spawn' on Windows and -- as of Python 3.14 --
'forkserver' on Linux. BOTH re-import the entry module in every worker, so without
the guard the sweep relaunches itself recursively (a process-spawn cascade that
surfaces as a forkserver ConnectionResetError). Inside the guard the re-import is
a harmless no-op.
"""

import argparse
import json
import os
import time
import tomllib
from pathlib import Path

import numpy as np

from .config import (
    default_settings,
    format_penetration_watchdog_summary,
    gate_cases_by_penetration,
    material_sweep,
)
from .materials import CATALOG
from .run import run_sweep
from .sweep import build_cases

MATS_FILE = Path("mats_to_sim.toml")


def load_all_materials(path: Path | None = None) -> list[str]:
    """Read the ordered ``materials`` list used by ``cxr scan --all``."""
    path = MATS_FILE if path is None else path
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise SystemExit(f"material manifest not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"invalid material manifest {path}: {exc}") from None
    materials = raw.get("materials")
    if (
        not isinstance(materials, list)
        or not materials
        or not all(isinstance(material, str) and material.strip() for material in materials)
    ):
        raise SystemExit(
            f"invalid material manifest {path}: expected a list of non-empty strings 'materials'"
        )
    duplicates = list(
        dict.fromkeys(material for material in materials if materials.count(material) > 1)
    )
    if duplicates:
        raise SystemExit(f"duplicate material(s) in {path}: {', '.join(duplicates)}")
    unknown = [material for material in materials if material not in CATALOG.materials]
    if unknown:
        raise SystemExit(f"unknown material(s) in {path}: {', '.join(unknown)}")
    return materials


def validate_materials(materials: list[str]) -> None:
    """Reject runnable selections that are absent from the material catalog."""
    unknown = [material for material in materials if material not in CATALOG.materials]
    if unknown:
        valid = ", ".join(CATALOG.material_keys)
        raise SystemExit(f"unknown material(s): {', '.join(unknown)}; valid: {valid}")


def _build_parser(ap):
    ap.add_argument("material", nargs="?", help="crystal key, e.g. mose2 / hopg / silicon / ptse2")
    ap.add_argument(
        "-a", "--all", action="store_true", help="run every material in mats_to_sim.toml"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="run_cases max_workers (default auto; 0 = serial, no transport pool)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="tiny grid (5 polar tilts x 2 azimuths x 2 energies) for a smoke test",
    )
    ap.add_argument(
        "--n-families",
        type=int,
        default=None,
        help="override the number of dominant reflection families "
        "(crystallography.dominant_reflections; default is the material's Sweep default)",
    )
    ap.add_argument(
        "--beam-uvw",
        type=int,
        nargs=3,
        metavar=("H", "K", "L"),
        default=None,
        help="override the beam zone axis [uvw] (default: the catalog crystal's beam_uvw)",
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
    """Register the ``scan`` subcommand on an argparse subparsers object."""
    return _build_parser(sub.add_parser("scan", help="run one material's MC sweep -> checkpoint"))


def run(args):
    if getattr(args, "all", False):
        if args.material is not None:
            raise SystemExit("scan --all does not take a material name")
        materials = load_all_materials()
    elif args.material is not None:
        materials = [args.material]
    else:
        raise SystemExit("scan needs a material name, or use --all")
    validate_materials(materials)
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


def _run_material(args, material, max_seconds=None):
    settings = default_settings()
    overrides = {}
    if args.quick:
        overrides.update(
            tilt_deg=np.linspace(0.0, 85.0, 5),
            tilt_azim_deg=np.array([10.0, 30.0]),
            energy_keV=[30, 50],
        )
    if args.n_families is not None:
        overrides["n_families"] = args.n_families
    if args.beam_uvw is not None:
        overrides["beam_uvw"] = tuple(args.beam_uvw)
    sweep = material_sweep(material, **overrides)

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    cases, dropped = gate_cases_by_penetration(cases)
    summary = format_penetration_watchdog_summary(dropped, material=material)
    if summary is not None:
        print(summary)
    print(
        f"{material}: {len(cases)} cases across "
        f"{len({c['name'] for c in cases})} configs" + (" (quick grid)" if args.quick else "")
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
    stem = f"{material}_quick" if args.quick else material
    ckpt = os.path.join(args.checkpoint_dir, f"{stem}.pkl")
    results = {}
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
                material=material,
                state="running",
                **latest_progress,
            )

    if progress_file is not None:
        _write_progress_record(
            progress_file,
            material=material,
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
        # run_sweep returns a bool (complete?). Only a bare None -- test doubles
        # that predate the budget feature and don't bother returning anything --
        # is read as complete; every other falsy return fails loud as incomplete.
        complete = True if result is None else bool(result)
    except BaseException:
        if progress_file is not None:
            _write_progress_record(
                progress_file,
                material=material,
                state="failed",
                **latest_progress,
            )
        raise
    if progress_file is not None:
        _write_progress_record(
            progress_file,
            material=material,
            state="done" if complete else "paused",
            **latest_progress,
        )
    n = sum(len(v) for v in results.values())
    if complete:
        print(f"done -> {args.checkpoint_dir}/{stem}.pkl ({n} records)")
    else:
        print(f"paused (budget) -> {args.checkpoint_dir}/{stem}.pkl ({n} records)")
    return complete


def _write_progress_record(
    path,
    *,
    material,
    total_cases,
    cached_cases,
    completed_new_cases,
    state,
):
    """Atomically replace one scan's compact JSON progress record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "material": material,
        "total_cases": total_cases,
        "cached_cases": cached_cases,
        "completed_new_cases": completed_new_cases,
        "state": state,
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None):
    ap = _build_parser(
        argparse.ArgumentParser(prog="scan.py", description="headless CXR scan runner")
    )
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
