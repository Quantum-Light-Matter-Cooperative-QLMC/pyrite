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

    python -m cxr_mc.energy_grid.derive
    python -m cxr_mc.energy_grid.derive --materials hopg,diamond --energies 30,50
    python -m cxr_mc.energy_grid.derive --json-out /tmp/line_grid_bounds.json
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
from cxr_mc.energy_grid import defaults as lg_defaults
from cxr_mc.energy_grid.bounds import coverage_energy, margined_stop, spacing_num
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
# 20 keV matches line_grid_bounds_job.py's DEFAULT_GRID_STOP and the E_grid_brem
# ceiling, and clears the widest measured 95% line-coverage energy (~18.7 keV raw
# at the 300 keV beam) with headroom.
WIDE_GRID_STOP_EV = 20000.0
WIDE_GRID_STEP_EV = 10.0
WIDE_GRID_EV = np.arange(WIDE_GRID_START_EV, WIDE_GRID_STOP_EV, WIDE_GRID_STEP_EV)
# Diagnostic brem grid for measuring the incoherent (brem) coverage. The
# production profile's E_grid_brem tops out at 30 keV, but a per-material bespoke
# brem stop must be measured on a grid that clears the widest true 95%
# brem-coverage energy, or coverage_energy clips it to the ceiling and refuses
# (CoverageGridTooNarrow). 40 keV carries headroom over the 30 keV production
# grid; overridable via --brem-grid-stop. Step matches the production E_grid_brem.
WIDE_BREM_STOP_EV = 40000.0
WIDE_BREM_STEP_EV = 25.0
WIDE_BREM_EV = np.arange(0.0, WIDE_BREM_STOP_EV, WIDE_BREM_STEP_EV)
COARSE_NE = 200
REFINE_NE = 2000
TOP_K = 3
CASE_BATCH_SIZE = 10
# Refine cases are spectrum-dominated and ran for about nine minutes each on
# qlmc at 250 keV.  Keep them individually checkpointable so the 10-minute
# soft slice budget can hand off well before the 30-minute SLURM backstop.
REFINE_BATCH_SIZE = 1
# At the 30 keV diagnostic ceiling, spectrum work dominates and the GPU path is
# about 6x faster for a measured heavy case. Both regimes therefore use auto.
COARSE_ENGINE = "auto"
# Single 1 mm crystal for every diagnostic geometry (issue_notes.md #1): thick
# crystals are the high-energy-dominated worst case, so pinning the thickest
# slab bounds the widest grid any thinner production crystal needs. No thickness
# scan here -- only tilt/azimuth vary.
DIAGNOSTIC_THICKNESS_ANG = 1.0e7


@dataclass(frozen=True)
class Candidate:
    material: str
    tilt_deg: float
    tilt_azim_deg: float
    # coherent = PXR/CBS line spectrum (r["spec"]); incoherent = bremsstrahlung
    # background (r["brem_wide"]). Both channels' 95% coverage is tracked so the
    # line grid and the brem grid are each bounded separately (issue_notes.md #1).
    coverage_energy_eV: float
    total_intensity: float
    incoherent_coverage_energy_eV: float
    incoherent_total_intensity: float
    # Crystal thickness the geometry was simulated at. Trails with a default so
    # the refine phase can reconstruct the exact geometry and so pre-thickness
    # checkpoints/callers stay valid (single-thickness scans use the diagnostic
    # slab).
    thickness_ang: float = DIAGNOSTIC_THICKNESS_ANG


def _build_case(material, energy_keV, tilt_deg, tilt_azim_deg, thickness_ang, n_electrons):
    """One run_case dict for a fixed material/energy/geometry, on the wide
    diagnostic grid. ``thickness_ang`` selects the crystal slab (default is the
    1 mm :data:`DIAGNOSTIC_THICKNESS_ANG` worst case); building is cheap and
    pure, running happens in a batch via _run_specs so the whole scan shares one
    worker pool instead of paying pool-startup and per-call overhead once per
    geometry."""
    sweep = material_sweep(
        material,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        E_grid_line=WIDE_GRID_EV,
        E_grid_line_by_energy=None,
        E_grid_brem=WIDE_BREM_EV,
    )
    return build_cases(sweep, n_electrons=n_electrons)[0]


def _candidate_from_result(material, tilt_deg, tilt_azim_deg, thickness_ang, result):
    E_grid, spec = result["E_grid"], result["spec"]
    E_brem, brem = result["E_grid_brem"], result["brem_wide"]
    return Candidate(
        material=material,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        coverage_energy_eV=coverage_energy(E_grid, spec, COVERAGE),
        total_intensity=float(np.trapezoid(spec, E_grid)),
        incoherent_coverage_energy_eV=coverage_energy(E_brem, brem, COVERAGE),
        incoherent_total_intensity=float(np.trapezoid(brem, E_brem)),
        thickness_ang=thickness_ang,
    )


def _run_specs(specs, energy_keV, n_electrons, max_workers, engine):
    """Build and run every (material, tilt_deg, tilt_azim_deg, thickness_ang)
    geometry in ``specs`` as one run_cases batch, so the CPU worker pool stays
    saturated across the whole batch. run_cases returns results in the same order
    as ``specs`` (index-aligned, per its docstring), so zipping is safe.

    ``engine`` is threaded straight through to run_cases (see
    docs/superpowers/specs/2026-07-18-regime-split-scheduling-design.md):
    the CPU-bound coarse regime can force the full-case CPU pool even on a
    GPU box, while the refine regime keeps "auto" (GPU pipeline, unchanged)."""
    cases = [
        _build_case(material, energy_keV, tilt_deg, tilt_azim_deg, thickness_ang, n_electrons)
        for material, tilt_deg, tilt_azim_deg, thickness_ang in specs
    ]
    results = run_cases(cases, max_workers=max_workers, engine=engine)
    return [
        _candidate_from_result(material, tilt_deg, tilt_azim_deg, thickness_ang, result)
        for (material, tilt_deg, tilt_azim_deg, thickness_ang), result in zip(
            specs, results, strict=True
        )
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


def _geometry_plan(materials, reference_scan, tilts=None, azimuths=None, thicknesses=None):
    """Return coarse geometry specs for the 95%-coverage search: the full
    quantized tilt x azimuth x thickness product per material, as 4-tuples
    ``(material, tilt_deg, tilt_azim_deg, thickness_ang)``.

    With no overrides the tilt/azimuth axes default to the reference material's
    profile angles and the thickness axis to a single :data:`DIAGNOSTIC_THICKNESS_ANG`
    slab -- exactly today's scan. ``tilts``/``azimuths``/``thicknesses`` (from
    ``--tilts``/``--azimuths``/``--thickness`` or persistent defaults) override
    each axis independently for diagnostic sweeps.

    The production grid is a small, deliberately curated set (issue_notes.md #1:
    tilt in {5, 45} deg, azimuth in {100, 140, 180} deg -- polar=0 and azim=90 are
    excluded and rejected at build_cases). That is only a handful of geometries, so
    every one is sampled instead of a near-zero/large-tilt subset. The 5 deg tilt
    maximizes resonant line energy (the widest line-grid driver); the coarse scan
    ranks the product and the refine phase re-measures the top few at higher Ne.

    Both angle axes pass through ``_quantized_angles`` -- the same nearest-0.5deg
    rounding ``build_cases`` applies -- so each ``Candidate``'s recorded geometry
    (reported as the driver) is the geometry actually simulated, not the raw
    catalog value. The second return value (a legacy large-tilt spot-check set)
    is empty: the full product already spans the largest tilt.
    """
    raw_tilts = tilts if tilts else reference_scan.tilt_deg
    raw_azims = azimuths if azimuths else reference_scan.tilt_azim_deg
    thicks = thicknesses if thicknesses else [DIAGNOSTIC_THICKNESS_ANG]
    tilt_vals = [float(value) for value in _quantized_angles(raw_tilts)]
    azim_vals = [float(value) for value in _quantized_angles(raw_azims)]
    coarse = [
        (material, tilt, azim, float(thick))
        for material in materials
        for tilt in tilt_vals
        for azim in azim_vals
        for thick in thicks
    ]
    return coarse, []


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
    tilts=None,
    azimuths=None,
    thicknesses=None,
):
    reference_scan = CATALOG.material(materials[0]).scan
    near_zero_specs, spot_check_specs = _geometry_plan(
        materials, reference_scan, tilts=tilts, azimuths=azimuths, thicknesses=thicknesses
    )

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
        # spot_check is empty under the curated grid (the coarse product already
        # spans the largest tilt); keep the flag machinery for legacy sidecars
        # that still carry a spot-check phase.
        best_spot = max(spot_check, key=lambda c: c.coverage_energy_eV) if spot_check else None
        flagged = best_spot is not None and best_spot.coverage_energy_eV > top[0].coverage_energy_eV
        extra = [best_spot] if best_spot is not None and flagged else []
        candidates = top + extra

        if max_seconds is not None and time_fn() - started >= max_seconds:
            complete = False
            break

        if active.get("refined") is not None:
            print(f"[analyze_line_grid_bounds] resuming {energy_keV:g} keV after refine")
        refined, timed_out = _resume_phase(
            "refined",
            [(c.material, c.tilt_deg, c.tilt_azim_deg, c.thickness_ang) for c in candidates],
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
        # Coherent driver sets the LINE grid; incoherent (brem) driver sets the
        # BREM grid. They are measured and bounded separately (issue_notes.md #1),
        # so the material/geometry that maximizes each channel's 95% energy can
        # differ.
        driver = max(refined, key=lambda c: c.coverage_energy_eV)
        brem_driver = max(refined, key=lambda c: c.incoherent_coverage_energy_eV)

        line_by_energy = reference_scan.E_grid_line_by_energy
        if line_by_energy is None:
            raise ValueError(
                f"reference material {materials[0]!r} has no E_grid_line_by_energy configured"
            )
        start_eV = float(line_by_energy[energy_keV][0])
        stop_eV = margined_stop(driver.coverage_energy_eV, MARGIN, ROUND_TO_EV)
        num = spacing_num(start_eV, stop_eV, TARGET_SPACING_EV)
        brem_stop_eV = margined_stop(brem_driver.incoherent_coverage_energy_eV, MARGIN, ROUND_TO_EV)
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
            brem_raw_eV=brem_driver.incoherent_coverage_energy_eV,
            brem_stop_eV=brem_stop_eV,
            brem_driver_material=brem_driver.material,
            brem_driver_tilt_deg=brem_driver.tilt_deg,
            brem_driver_azim_deg=brem_driver.tilt_azim_deg,
        )
        if on_progress is not None:
            on_progress([rows_by_energy[key] for key in sorted(rows_by_energy)])
        phase_state = None
        if on_phase_progress is not None:
            on_phase_progress(None)
    rows = [rows_by_energy[key] for key in sorted(rows_by_energy)]
    return rows, complete


def _brem_grid_for_rows(rows, step_eV=WIDE_BREM_STEP_EV):
    """Collapse a material's per-energy line-grid rows into ONE bespoke brem
    grid descriptor.

    ``E_grid_brem`` is a single grid per scan (not a per-energy table), so the
    material's brem ceiling must cover its widest-energy brem tail across every
    beam energy. Each row already carries a per-energy ``brem_stop_eV`` (the
    +5%-margined, 100 eV-rounded coverage energy) and the unmargined
    ``brem_raw_eV`` that produced it.

    ``step_eV`` controls downstream production-grid spacing; diagnostic
    bremsstrahlung sampling remains independently fixed by ``WIDE_BREM_STEP_EV``.
    """
    driver = max(rows, key=lambda row: row["brem_stop_eV"])
    return {
        "stop_eV": driver["brem_stop_eV"],
        "raw_eV": driver["brem_raw_eV"],
        "step_eV": float(step_eV),
    }


def _material_checkpoint_paths(json_out, material):
    """Per-material checkpoint + phase-sidecar paths, or ``(None, None)`` when no
    ``json_out`` is configured. Each material keeps today's flat single-material
    checkpoint format (``<json_out>.<material>.json`` + ``.phase.json``) so the
    proven resume/phase machinery is reused verbatim per material."""
    if not json_out:
        return None, None
    path = f"{json_out}.{material}.json"
    return path, f"{path}.phase.json"


def _run_one_material(
    material,
    energies,
    json_out,
    *,
    top_k,
    coarse_ne,
    refine_ne,
    max_workers,
    coarse_engine,
    max_seconds,
    time_fn,
    tilts=None,
    azimuths=None,
    thicknesses=None,
):
    """Derive one material's bespoke line rows against its OWN flat checkpoint
    and phase sidecar -- a thin per-material wrapper that reproduces ``main``'s
    resume-load / atomic-save wiring for a single material. Returns
    ``(rows, complete)`` from the underlying single-material ``derive_bounds``."""
    path, phase_path = _material_checkpoint_paths(json_out, material)
    existing_rows = []
    phase_state = None
    if path and os.path.exists(path):
        with open(path) as f:
            existing_rows = json.load(f)
    if phase_path and os.path.exists(phase_path):
        with open(phase_path) as f:
            phase_state = json.load(f) or None

    def _save(rows):
        if path:
            _atomic_write_json(path, rows)

    def _save_phase(value):
        if phase_path:
            _atomic_write_json(phase_path, value or {})

    rows, complete = derive_bounds(
        [material],
        energies,
        top_k,
        coarse_ne,
        refine_ne,
        max_workers,
        existing_rows=existing_rows,
        on_progress=_save,
        phase_state=phase_state,
        on_phase_progress=_save_phase,
        coarse_engine=coarse_engine,
        max_seconds=max_seconds,
        time_fn=time_fn,
        tilts=tilts,
        azimuths=azimuths,
        thicknesses=thicknesses,
    )
    if path and complete:
        _atomic_write_json(path, rows)
    return rows, complete


def derive_all_materials(
    materials,
    energies,
    top_k=TOP_K,
    coarse_ne=COARSE_NE,
    refine_ne=REFINE_NE,
    max_workers=None,
    json_out=None,
    coarse_engine=COARSE_ENGINE,
    max_seconds=None,
    time_fn=time.monotonic,
    tilts=None,
    azimuths=None,
    thicknesses=None,
    brem_step_eV=WIDE_BREM_STEP_EV,
):
    """Derive a bespoke per-material line grid + brem grid for each material,
    running the per-energy derivation independently per material (Approach A:
    each single-material ``derive_bounds`` call's driver is that material's own
    worst geometry).

    Returns ``(combined, complete)`` where ``combined`` maps each material to
    ``{"line_rows": [<rows>], "brem": {"stop_eV", "raw_eV", "step_eV"}}``. The
    soft ``max_seconds`` budget is threaded across the WHOLE material x energy
    set -- checked between materials and (inside ``derive_bounds``) between
    phases; when it is exhausted the function returns early with
    ``complete=False`` (the exit-75 "work remains" contract). The combined
    ``json_out`` is written only once every material is complete; per-material
    checkpoint files carry the resumable state in the meantime.
    """
    started = time_fn()
    combined = {}
    complete = True
    for material in materials:
        if max_seconds is not None and time_fn() - started >= max_seconds:
            complete = False
            break
        remaining = None if max_seconds is None else max_seconds - (time_fn() - started)
        rows, material_complete = _run_one_material(
            material,
            energies,
            json_out,
            top_k=top_k,
            coarse_ne=coarse_ne,
            refine_ne=refine_ne,
            max_workers=max_workers,
            coarse_engine=coarse_engine,
            max_seconds=remaining,
            time_fn=time_fn,
            tilts=tilts,
            azimuths=azimuths,
            thicknesses=thicknesses,
        )
        if rows:
            combined[material] = {
                "line_rows": rows,
                "brem": _brem_grid_for_rows(rows, brem_step_eV),
            }
        if not material_complete:
            complete = False
            break
    if complete and json_out:
        _atomic_write_json(json_out, combined)
    return combined, complete


def _print_report(rows):
    header = (
        f"{'energy':>8} {'raw_eV':>10} {'stop_eV':>9} {'num':>6} "
        f"{'driver':>14} {'tilt':>6} {'azim':>7} {'spot?':>6} "
        f"{'brem_raw':>10} {'brem_stop':>10} {'brem_drv':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        brem_raw = row.get("brem_raw_eV")
        brem_stop = row.get("brem_stop_eV")
        brem_drv = row.get("brem_driver_material", "-")
        # Legacy rows (older checkpoints) carry no brem channel; show placeholders.
        if brem_raw is None:
            brem_cols = f"{'-':>10} {'-':>10} {'-':>14}"
        else:
            brem_cols = f"{brem_raw:>10.1f} {brem_stop:>10.1f} {brem_drv:>14}"
        print(
            f"{row['energy_keV']:>8g} {row['raw_eV']:>10.1f} {row['stop_eV']:>9.1f} "
            f"{row['num']:>6d} {row['driver_material']:>14} "
            f"{row['driver_tilt_deg']:>6.2f} {row['driver_azim_deg']:>7.2f} "
            f"{'yes' if row['spot_check_flagged'] else 'no':>6} "
            f"{brem_cols}"
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
    parser.add_argument(
        "--brem-grid-stop",
        type=float,
        default=WIDE_BREM_STOP_EV,
        help="ceiling (eV) of the diagnostic brem coverage grid; must clear the widest "
        "true 95%% brem-coverage energy at any beam energy (default: %(default)g)",
    )
    parser.add_argument("--json-out", default=None, help="optional path to write rows as JSON")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="soft slice budget checked between completed phases; exit 75 when work remains",
    )
    parser.add_argument(
        "--tilts",
        default=None,
        help="comma polar tilts (deg); default: profile/persistent",
    )
    parser.add_argument(
        "--azimuths",
        default=None,
        help="comma azimuths (deg); default: profile/persistent",
    )
    parser.add_argument(
        "--thickness",
        default=None,
        help="comma crystal thicknesses (Angstrom); default: single 1 mm diagnostic slab",
    )
    parser.add_argument(
        "--brem-step",
        type=float,
        default=None,
        help="E_grid_brem step (eV) for downstream apply",
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="persist supplied geometry/energies/materials as new defaults",
    )
    return parser


def _floats(text):
    return [float(x) for x in text.split(",")] if text else None


def main(argv=None):
    args = build_parser().parse_args(argv)
    persisted = lg_defaults.load_defaults()
    tilts = _floats(args.tilts) or (persisted["tilts"] or None)
    azimuths = _floats(args.azimuths) or (persisted["azimuths"] or None)
    thicknesses = _floats(args.thickness) or (persisted["thickness_ang"] or None)
    brem_step_eV = (
        float(args.brem_step) if args.brem_step is not None else persisted["brem_step_ev"]
    )
    global WIDE_GRID_EV, WIDE_BREM_EV
    WIDE_GRID_EV = np.arange(WIDE_GRID_START_EV, args.grid_stop, args.grid_step)
    WIDE_BREM_EV = np.arange(0.0, args.brem_grid_stop, WIDE_BREM_STEP_EV)
    materials = (
        args.materials.split(",")
        if args.materials
        else list(persisted["materials"] or CATALOG.materials)
    )
    if args.energies:
        energies = [float(e) for e in args.energies.split(",")]
    elif persisted["energies"]:
        energies = [float(e) for e in persisted["energies"]]
    else:
        energies = [float(e) for e in CATALOG.material(materials[0]).scan.energy_keV]
    if args.set_default:
        lg_defaults.update_defaults(
            tilts=_floats(args.tilts),
            azimuths=_floats(args.azimuths),
            thickness_ang=_floats(args.thickness),
            brem_step_ev=args.brem_step,
            energies=energies if args.energies else None,
            materials=materials if args.materials else None,
        )

    combined, complete = derive_all_materials(
        materials,
        energies,
        args.top_k,
        args.coarse_ne,
        args.refine_ne,
        args.max_workers,
        json_out=args.json_out,
        coarse_engine=args.coarse_engine,
        max_seconds=None if args.max_minutes is None else args.max_minutes * 60.0,
        tilts=tilts,
        azimuths=azimuths,
        thicknesses=thicknesses,
        brem_step_eV=brem_step_eV,
    )
    for material in materials:
        entry = combined.get(material)
        if entry is None:
            continue
        print(f"=== {material} ===")
        _print_report(entry["line_rows"])
        brem = entry["brem"]
        print(
            f"brem grid: stop={brem['stop_eV']:g} eV  raw={brem['raw_eV']:g} eV  "
            f"step={brem['step_eV']:g} eV"
        )
    if args.json_out and complete:
        _atomic_write_json(args.json_out, combined)
        print(f"[analyze_line_grid_bounds] wrote {args.json_out}")
    if not complete:
        print("[analyze_line_grid_bounds] slice budget exhausted; work remains")
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
