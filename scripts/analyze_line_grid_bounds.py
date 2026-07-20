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
energy at which 95% of coherent-line intensity is captured, refine the top
candidates at higher Ne, then report a +5%-margined ``stop`` and the ``num``
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
import os
import tempfile
import time
from dataclasses import asdict, dataclass

import numpy as np

from cxr_mc.config import material_sweep
from cxr_mc.line_grid_bounds import coverage_energy, margined_stop, spacing_num
from cxr_mc.materials import CATALOG
from cxr_mc.montecarlo.runner import run_cases
from cxr_mc.sweep import _quantized_angles, build_cases

COVERAGE = 0.95
MARGIN = 0.05
ROUND_TO_EV = 100.0
TARGET_SPACING_EV = 3.0
# Diagnostic histogram grid for measuring coverage. Its CEILING must sit well
# above the widest true 95% coverage energy at any beam energy, or coverage_energy
# silently truncates and reports a bound pinned near the ceiling (the failure the
# 2026-07-16 full scan hit at 150-300 keV). Overridable via --grid-stop/--grid-step;
# the spacing only needs to resolve the cumulative envelope, not each narrow line,
# since the final catalog `stop` is rounded to 100 eV regardless.
WIDE_GRID_START_EV = 10.0
# 30 keV matches line_grid_bounds_job.py's DEFAULT_GRID_STOP and the E_grid_brem
# ceiling, and clears the widest measured 95% line-coverage energy (~18.7 keV raw
# at the 300 keV beam) with headroom. A 10 keV default silently clipped the
# 200-300 keV bounds (the 2026-07-16 full-scan failure noted above), because
# coverage_energy pins to the grid ceiling instead of raising when the true
# envelope runs past it.
WIDE_GRID_STOP_EV = 30000.0
WIDE_GRID_STEP_EV = 5.0
WIDE_GRID_EV = np.arange(WIDE_GRID_START_EV, WIDE_GRID_STOP_EV, WIDE_GRID_STEP_EV)
COARSE_NE = 200
REFINE_NE = 2000
TOP_K = 5
CASE_BATCH_SIZE = 10
# Refine cases are spectrum-dominated and ran for about nine minutes each on
# qlmc at 250 keV.  Keep them individually checkpointable so the 10-minute
# soft slice budget can hand off well before the 30-minute SLURM backstop.
REFINE_BATCH_SIZE = 1
# At the 30 keV diagnostic ceiling, spectrum work dominates and the GPU path is
# about 6x faster for a measured heavy case. Both regimes therefore use auto.
COARSE_ENGINE = "auto"


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


def _run_specs(specs, energy_keV, n_electrons, max_workers, engine):
    """Build and run every (material, tilt_deg, tilt_azim_deg) geometry in
    ``specs`` as one run_cases batch, so the CPU worker pool stays saturated
    across the whole batch. run_cases returns results in the same order as
    ``specs`` (index-aligned, per its docstring), so zipping is safe.

    ``engine`` is threaded straight through to run_cases (see
    docs/superpowers/specs/2026-07-18-regime-split-scheduling-design.md):
    the CPU-bound coarse regime can force the full-case CPU pool even on a
    GPU box, while the refine regime keeps "auto" (GPU pipeline, unchanged)."""
    cases = [
        _build_case(material, energy_keV, tilt_deg, tilt_azim_deg, n_electrons)
        for material, tilt_deg, tilt_azim_deg in specs
    ]
    results = run_cases(cases, max_workers=max_workers, engine=engine)
    return [
        _candidate_from_result(material, tilt_deg, tilt_azim_deg, result)
        for (material, tilt_deg, tilt_azim_deg), result in zip(specs, results, strict=True)
    ]


def _scan_specs(specs, energy_keV, n_electrons, max_workers, engine):
    return _run_specs(specs, energy_keV, n_electrons, max_workers, engine)


def _resume_phase(
    key,
    specs,
    energy_keV,
    n_electrons,
    max_workers,
    engine,
    active,
    on_phase_progress,
    started,
    max_seconds,
    time_fn,
    runner,
):
    """Run one phase in resumable geometry batches.

    Cursor is persisted separately from candidate count because tests and future
    runners may filter results. Older sidecars lacking a cursor represent a
    completed phase, preserving compatibility with phase-level checkpoints.
    """
    values = [Candidate(**value) for value in active.get(key, [])]
    cursor_key = f"{key}_cursor"
    if key in active and cursor_key not in active:
        return values, False
    cursor = int(active.get(cursor_key, 0))
    batch_size = REFINE_BATCH_SIZE if key == "refined" else CASE_BATCH_SIZE
    while cursor < len(specs):
        end = min(cursor + batch_size, len(specs))
        values.extend(runner(specs[cursor:end], energy_keV, n_electrons, max_workers, engine))
        cursor = end
        active[key] = [asdict(candidate) for candidate in values]
        active[cursor_key] = cursor
        if on_phase_progress is not None:
            on_phase_progress(active)
        if max_seconds is not None and time_fn() - started >= max_seconds:
            return values, True
    return values, False


def _geometry_plan(materials, reference_scan):
    """Return reduced coarse geometry specs for the 95%-coverage search.

    Resonant energy is maximized near zero polar tilt. Keep one explicit 0deg
    geometry (azimuth is redundant there), then retain every standard azimuth
    at the next tilt (~9.89deg quantizes to 10deg), where all completed campaign
    rows found their driver. A minimal large-tilt guard samples 89deg at the
    azimuth endpoints and middle grid point. This retains the maximizing boundary
    and an explicit counterexample check while reducing 40 to 14 geometries per
    material.

    Both axes pass through ``_quantized_angles`` -- the same nearest-0.5deg
    rounding ``build_cases`` applies -- so the geometry recorded on each
    ``Candidate`` (and reported as the driver) is the geometry actually
    simulated, not the raw catalog linspace value.
    """
    tilts = [float(value) for value in _quantized_angles(reference_scan.tilt_deg)]
    azimuths = [float(value) for value in _quantized_angles(reference_scan.tilt_azim_deg)]
    near_zero_geometry = [(tilts[0], azimuths[0]), *[(tilts[1], azim) for azim in azimuths]]
    spot_azimuths = [azimuths[index] for index in (0, len(azimuths) // 2, len(azimuths) - 1)]
    spot_geometry = [(tilts[-1], azim) for azim in spot_azimuths]
    near_zero = [
        (material, tilt, azim) for material in materials for tilt, azim in near_zero_geometry
    ]
    spot_check = [(material, tilt, azim) for material in materials for tilt, azim in spot_geometry]
    return near_zero, spot_check


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


def _atomic_write_json(path, rows):
    """Write ``rows`` to ``path`` atomically (temp file in the same dir + rename)
    so a killed slice never leaves a half-written checkpoint that a resume would
    choke on."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def derive_bounds(
    materials,
    energies,
    top_k=TOP_K,
    coarse_ne=COARSE_NE,
    refine_ne=REFINE_NE,
    max_workers=None,
    existing_rows=None,
    on_progress=None,
    phase_state=None,
    on_phase_progress=None,
    coarse_engine=COARSE_ENGINE,
    max_seconds=None,
    time_fn=time.monotonic,
):
    reference_scan = CATALOG.material(materials[0]).scan
    # Quantized to match _geometry_plan / build_cases, so the degeneracy warning's
    # exact tilt comparison lines up with each Candidate's quantized tilt_deg.
    quantized_tilts = _quantized_angles(reference_scan.tilt_deg)
    near_zero_tilts = [float(quantized_tilts[i]) for i in (0, 1)]
    near_zero_specs, spot_check_specs = _geometry_plan(materials, reference_scan)

    existing = {float(r["energy_keV"]): r for r in (existing_rows or [])}
    rows_by_energy = dict(existing)
    started = time_fn()
    complete = True
    for energy_keV in energies:
        if float(energy_keV) in existing:
            print(
                f"[analyze_line_grid_bounds] energy {energy_keV:g} keV already in checkpoint; skipping"
            )
            continue
        if max_seconds is not None and time_fn() - started >= max_seconds:
            complete = False
            break
        active = phase_state if phase_state and phase_state.get("energy_keV") == energy_keV else {}
        if active.get("near_zero") is not None:
            print(f"[analyze_line_grid_bounds] resuming {energy_keV:g} keV after near-zero")
        else:
            active = {
                "energy_keV": energy_keV,
            }
        near_zero, timed_out = _resume_phase(
            "near_zero",
            near_zero_specs,
            energy_keV,
            coarse_ne,
            max_workers,
            coarse_engine,
            active,
            on_phase_progress,
            started,
            max_seconds,
            time_fn,
            _scan_specs,
        )
        if timed_out:
            complete = False
            break
        _warn_degenerate_zero_tilt(materials, near_zero_tilts, near_zero)
        if max_seconds is not None and time_fn() - started >= max_seconds:
            complete = False
            break

        if active.get("spot_check") is not None:
            print(f"[analyze_line_grid_bounds] resuming {energy_keV:g} keV after spot-check")
        spot_check, timed_out = _resume_phase(
            "spot_check",
            spot_check_specs,
            energy_keV,
            coarse_ne,
            max_workers,
            coarse_engine,
            active,
            on_phase_progress,
            started,
            max_seconds,
            time_fn,
            _scan_specs,
        )
        if timed_out:
            complete = False
            break

        top = sorted(near_zero, key=lambda c: c.coverage_energy_eV, reverse=True)[:top_k]
        best_spot = max(spot_check, key=lambda c: c.coverage_energy_eV)
        flagged = best_spot.coverage_energy_eV > top[0].coverage_energy_eV
        candidates = top + ([best_spot] if flagged else [])

        if max_seconds is not None and time_fn() - started >= max_seconds:
            complete = False
            break

        if active.get("refined") is not None:
            print(f"[analyze_line_grid_bounds] resuming {energy_keV:g} keV after refine")
        refined, timed_out = _resume_phase(
            "refined",
            [(c.material, c.tilt_deg, c.tilt_azim_deg) for c in candidates],
            energy_keV,
            refine_ne,
            max_workers,
            "auto",
            active,
            on_phase_progress,
            started,
            max_seconds,
            time_fn,
            _run_specs,
        )
        if timed_out:
            complete = False
            break
        driver = max(refined, key=lambda c: c.coverage_energy_eV)

        start_eV = float(reference_scan.E_grid_line_by_energy[energy_keV][0])
        stop_eV = margined_stop(driver.coverage_energy_eV, MARGIN, ROUND_TO_EV)
        num = spacing_num(start_eV, stop_eV, TARGET_SPACING_EV)
        rows_by_energy[float(energy_keV)] = dict(
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
        if on_progress is not None:
            on_progress([rows_by_energy[key] for key in sorted(rows_by_energy)])
        phase_state = None
        if on_phase_progress is not None:
            on_phase_progress(None)
    rows = [rows_by_energy[key] for key in sorted(rows_by_energy)]
    return rows, complete


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
    parser.add_argument(
        "--coarse-engine",
        choices=("cpu", "auto"),
        default=COARSE_ENGINE,
        help="run_cases engine for the coarse near-zero/spot-check scans (the CPU-bound "
        "regime); the refine scan always passes engine='auto' (default: %(default)s)",
    )
    parser.add_argument(
        "--grid-stop",
        type=float,
        default=WIDE_GRID_STOP_EV,
        help="ceiling (eV) of the diagnostic coverage grid; must clear the widest true "
        "95%% coverage energy at any beam energy (default: %(default)g)",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=WIDE_GRID_STEP_EV,
        help="spacing (eV) of the diagnostic coverage grid (default: %(default)g)",
    )
    parser.add_argument("--json-out", default=None, help="optional path to write rows as JSON")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="soft slice budget checked between completed phases; exit 75 when work remains",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    global WIDE_GRID_EV
    WIDE_GRID_EV = np.arange(WIDE_GRID_START_EV, args.grid_stop, args.grid_step)
    materials = args.materials.split(",") if args.materials else list(CATALOG.materials)
    if args.energies:
        energies = [float(e) for e in args.energies.split(",")]
    else:
        energies = [float(e) for e in CATALOG.material(materials[0]).scan.energy_keV]
    existing_rows = []
    phase_path = f"{args.json_out}.phase.json" if args.json_out else None
    phase_state = None
    if args.json_out and os.path.exists(args.json_out):
        with open(args.json_out) as f:
            existing_rows = json.load(f)
        done = sorted({float(r["energy_keV"]) for r in existing_rows})
        print(
            f"[analyze_line_grid_bounds] resuming from {args.json_out}; "
            f"{len(done)} energ(ies) already done: {done}"
        )
    if phase_path and os.path.exists(phase_path):
        with open(phase_path) as f:
            phase_state = json.load(f) or None

    def _save(rows):
        if args.json_out:
            _atomic_write_json(args.json_out, rows)

    def _save_phase(value):
        if phase_path:
            _atomic_write_json(phase_path, value or {})

    rows, complete = derive_bounds(
        materials,
        energies,
        args.top_k,
        args.coarse_ne,
        args.refine_ne,
        args.max_workers,
        existing_rows=existing_rows,
        on_progress=_save,
        phase_state=phase_state,
        on_phase_progress=_save_phase,
        coarse_engine=args.coarse_engine,
        max_seconds=None if args.max_minutes is None else args.max_minutes * 60.0,
    )
    _print_report(rows)
    if args.json_out:
        _atomic_write_json(args.json_out, rows)
        print(f"[analyze_line_grid_bounds] wrote {args.json_out}")
    if not complete:
        print("[analyze_line_grid_bounds] slice budget exhausted; work remains")
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
