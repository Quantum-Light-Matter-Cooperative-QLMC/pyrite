"""Headless CXR scan runner (the library twin of scan.ipynb).

Runs the Monte-Carlo CXR parameter sweep for one material and writes the
per-material checkpoint (checkpoints/<material>.pkl). Use this to run sweeps
non-interactively -- in particular over SSH on the GPU box; see cxr_mc.remote,
which drives this and pulls the checkpoint back so the (matplotlib / PDF)
data-vis can stay on the laptop.

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
import os
import tomllib
from pathlib import Path

import numpy as np

from .config import MATERIALS, default_settings, material_sweep
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
        or not all(isinstance(material, str) for material in materials)
    ):
        raise SystemExit(
            f"invalid material manifest {path}: expected a non-empty string list 'materials'"
        )
    unknown = [material for material in materials if material not in MATERIALS]
    if unknown:
        raise SystemExit(f"unknown material(s) in {path}: {', '.join(unknown)}")
    return materials


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
        help="override the beam zone axis [uvw] (default: the material's crystal_params default)",
    )
    ap.add_argument("--checkpoint-dir", default="checkpoints")
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
    for material in materials:
        _run_material(args, material)


def _run_material(args, material):
    settings = default_settings()
    overrides = {}
    if args.quick:
        overrides.update(
            tilt_deg=np.linspace(0.0, 85.0, 5),
            tilt_azim_deg=np.array([10.0, 30.0]),
            energy_keV=[30, 60],
        )
    if args.n_families is not None:
        overrides["n_families"] = args.n_families
    if args.beam_uvw is not None:
        overrides["beam_uvw"] = tuple(args.beam_uvw)
    sweep = material_sweep(material, **overrides)

    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
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

    # Always pass an explicit path named for the REGISTRY key: run_sweep's
    # default derives the name from the film crystal, which would make a named
    # stack (e.g. mos2-on-sio2-si) clobber/resume the plain film's checkpoint.
    # A --quick smoke test writes to its OWN checkpoint (<material>_quick.pkl), so
    # its coarse off-grid points never contaminate the real per-material sweep.
    stem = f"{material}_quick" if args.quick else material
    ckpt = os.path.join(args.checkpoint_dir, f"{stem}.pkl")
    results = {}
    run_sweep(
        cases,
        results,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=ckpt,
        max_workers=args.workers,
    )
    n = sum(len(v) for v in results.values())
    print(f"done -> {args.checkpoint_dir}/{stem}.pkl ({n} records)")


def main(argv=None):
    ap = _build_parser(
        argparse.ArgumentParser(prog="scan.py", description="headless CXR scan runner")
    )
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
