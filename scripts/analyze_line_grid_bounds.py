#!/usr/bin/env python3
# scripts/analyze_line_grid_bounds.py
"""Empirically derive per-beam-energy line-grid upper bounds (``stop``) for
the standard profile's ``E_grid_line_by_energy`` in
``src/cxr_mc/data/materials.toml``, from simulated coherent-line intensity
coverage rather than an undocumented cap.

See docs/superpowers/specs/2026-07-16-line-grid-max-energy-design.md for the
full method. Summary: for every requested material and standard beam energy,
run a small-Ne diagnostic spectrum on a wide ``E_grid_line`` at the two
smallest standard polar tilts (near tilt=0, where E_res is maximized) across
every standard azimuth, plus a couple of larger-tilt spot checks. Rank by the
energy at which 99% of coherent-line intensity is captured, refine the top
candidates at higher Ne, then report a +15%-margined ``stop`` and the ``num``
that preserves ~3 eV endpoint-inclusive spacing. Each coarse/refine batch of
geometries is run through cxr_mc.montecarlo.runner.run_cases, which pipelines
the independent per-geometry transports across a CPU worker pool instead of
running them one at a time.

    uv run python scripts/analyze_line_grid_bounds.py
    uv run python scripts/analyze_line_grid_bounds.py --materials hopg,diamond --energies 30,50
    uv run python scripts/analyze_line_grid_bounds.py --json-out /tmp/line_grid_bounds.json
    uv run python scripts/analyze_line_grid_bounds.py --max-workers 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from cxr_mc.config import material_sweep
from cxr_mc.line_grid_bounds import coverage_energy, margined_stop, spacing_num
from cxr_mc.materials import CATALOG
from cxr_mc.montecarlo.runner import run_cases
from cxr_mc.sweep import build_cases

COVERAGE = 0.99
MARGIN = 0.15
ROUND_TO_EV = 100.0
TARGET_SPACING_EV = 3.0
WIDE_GRID_EV = np.arange(10.0, 10000.0, 5.0)
COARSE_NE = 500
REFINE_NE = 5000
TOP_K = 5
SPOT_CHECK_TILT_INDICES = (4, 9)  # the standard grid's ~40deg and 89deg points


@dataclass(frozen=True)
class Candidate:
    material: str
    tilt_deg: float
    tilt_azim_deg: float
    coverage_energy_eV: float
    total_intensity: float


def _build_case(material, energy_keV, tilt_deg, tilt_azim_deg, n_electrons):
    """One run_case dict for a fixed material/energy/geometry, on the wide
    diagnostic grid. Pins the material's first configured thickness so only
    the tilt/azimuth axis varies across the scan. Building is cheap and pure;
    running happens in a batch via _run_specs so the whole scan shares one
    worker pool instead of paying pool-startup and per-call overhead once per
    geometry."""
    thickness_ang = float(np.atleast_1d(CATALOG.material(material).scan.thickness_ang)[0])
    sweep = material_sweep(
        material,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        E_grid_line=WIDE_GRID_EV,
        E_grid_line_by_energy=None,
    )
    return build_cases(sweep, n_electrons=n_electrons)[0]


def _candidate_from_result(material, tilt_deg, tilt_azim_deg, result):
    E_grid, spec = result["E_grid"], result["spec"]
    return Candidate(
        material=material,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        coverage_energy_eV=coverage_energy(E_grid, spec, COVERAGE),
        total_intensity=float(np.trapezoid(spec, E_grid)),
    )


def _run_specs(specs, energy_keV, n_electrons, max_workers):
    """Build and run every (material, tilt_deg, tilt_azim_deg) geometry in
    ``specs`` as one run_cases batch, so the CPU worker pool stays saturated
    across the whole batch. run_cases returns results in the same order as
    ``specs`` (index-aligned, per its docstring), so zipping is safe."""
    cases = [
        _build_case(material, energy_keV, tilt_deg, tilt_azim_deg, n_electrons)
        for material, tilt_deg, tilt_azim_deg in specs
    ]
    results = run_cases(cases, max_workers=max_workers)
    return [
        _candidate_from_result(material, tilt_deg, tilt_azim_deg, result)
        for (material, tilt_deg, tilt_azim_deg), result in zip(specs, results, strict=True)
    ]


def _scan(materials, energy_keV, tilts, azimuths, n_electrons, max_workers):
    specs = [
        (material, tilt, azim) for material in materials for tilt in tilts for azim in azimuths
    ]
    return _run_specs(specs, energy_keV, n_electrons, max_workers)


def _warn_degenerate_zero_tilt(materials, near_zero_tilts, near_zero_candidates):
    """Flag any material whose exact tilt=0 geometry radiates zero coherent-
    line intensity while its next-smallest tilt does not -- a possible
    geometric degeneracy at tilt=0, distinct from ordinary MC noise. Does not
    change ranking: coverage_energy already returns the grid floor for
    zero-intensity geometries, so they can't win the top-k selection below."""
    zero_tilt, next_tilt = near_zero_tilts
    for material in materials:
        zero_total = sum(
            c.total_intensity
            for c in near_zero_candidates
            if c.material == material and c.tilt_deg == zero_tilt
        )
        next_total = sum(
            c.total_intensity
            for c in near_zero_candidates
            if c.material == material and c.tilt_deg == next_tilt
        )
        if zero_total <= 0.0 and next_total > 0.0:
            print(
                f"[warn] {material}: zero coherent-line intensity at "
                f"tilt={zero_tilt:g} deg but not at tilt={next_tilt:g} deg -- "
                "possible geometric degeneracy at exact zero tilt; tilt=0 "
                "remains a valid production sweep point regardless."
            )


def derive_bounds(
    materials,
    energies,
    top_k=TOP_K,
    coarse_ne=COARSE_NE,
    refine_ne=REFINE_NE,
    max_workers=None,
):
    reference_scan = CATALOG.material(materials[0]).scan
    near_zero_tilts = [float(reference_scan.tilt_deg[i]) for i in (0, 1)]
    spot_check_tilts = [float(reference_scan.tilt_deg[i]) for i in SPOT_CHECK_TILT_INDICES]
    azimuths = [float(a) for a in reference_scan.tilt_azim_deg]

    rows = []
    for energy_keV in energies:
        near_zero = _scan(materials, energy_keV, near_zero_tilts, azimuths, coarse_ne, max_workers)
        _warn_degenerate_zero_tilt(materials, near_zero_tilts, near_zero)
        spot_check = _scan(
            materials, energy_keV, spot_check_tilts, azimuths, coarse_ne, max_workers
        )

        top = sorted(near_zero, key=lambda c: c.coverage_energy_eV, reverse=True)[:top_k]
        best_spot = max(spot_check, key=lambda c: c.coverage_energy_eV)
        flagged = best_spot.coverage_energy_eV > top[0].coverage_energy_eV
        candidates = top + ([best_spot] if flagged else [])

        refined = _run_specs(
            [(c.material, c.tilt_deg, c.tilt_azim_deg) for c in candidates],
            energy_keV,
            refine_ne,
            max_workers,
        )
        driver = max(refined, key=lambda c: c.coverage_energy_eV)

        start_eV = float(reference_scan.E_grid_line_by_energy[energy_keV][0])
        stop_eV = margined_stop(driver.coverage_energy_eV, MARGIN, ROUND_TO_EV)
        num = spacing_num(start_eV, stop_eV, TARGET_SPACING_EV)
        rows.append(
            dict(
                energy_keV=energy_keV,
                raw_eV=driver.coverage_energy_eV,
                start_eV=start_eV,
                stop_eV=stop_eV,
                num=num,
                driver_material=driver.material,
                driver_tilt_deg=driver.tilt_deg,
                driver_azim_deg=driver.tilt_azim_deg,
                spot_check_flagged=flagged,
            )
        )
    return rows


def _print_report(rows):
    header = (
        f"{'energy':>8} {'raw_eV':>10} {'stop_eV':>9} {'num':>6} "
        f"{'driver':>14} {'tilt':>6} {'azim':>7} {'spot?':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['energy_keV']:>8g} {row['raw_eV']:>10.1f} {row['stop_eV']:>9.1f} "
            f"{row['num']:>6d} {row['driver_material']:>14} "
            f"{row['driver_tilt_deg']:>6.2f} {row['driver_azim_deg']:>7.2f} "
            f"{'yes' if row['spot_check_flagged'] else 'no':>6}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--materials",
        default=None,
        help="comma-separated catalog material keys (default: every standard-profile material)",
    )
    parser.add_argument(
        "--energies",
        default=None,
        help="comma-separated beam energies in keV (default: every standard beam energy)",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--coarse-ne", type=int, default=COARSE_NE)
    parser.add_argument("--refine-ne", type=int, default=REFINE_NE)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="worker processes for each run_cases batch (default: auto-sized, ~3/4 of CPUs)",
    )
    parser.add_argument("--json-out", default=None, help="optional path to write rows as JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    materials = args.materials.split(",") if args.materials else list(CATALOG.materials)
    if args.energies:
        energies = [float(e) for e in args.energies.split(",")]
    else:
        energies = [float(e) for e in CATALOG.material(materials[0]).scan.energy_keV]
    rows = derive_bounds(
        materials, energies, args.top_k, args.coarse_ne, args.refine_ne, args.max_workers
    )
    _print_report(rows)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[analyze_line_grid_bounds] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
