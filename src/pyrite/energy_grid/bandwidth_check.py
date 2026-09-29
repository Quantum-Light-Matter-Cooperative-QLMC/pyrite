"""Measured line bandwidth against the kinematic-ceiling axis (issue #192).

Evidence for the ``resonance-population`` bandwidth policy. One transport per
configuration; the axis the policy measured and a reference axis on the same
nodes out to the closed-form ceiling are evaluated with the production line and
characteristic reductions (:class:`pyrite.energy_grid.convergence_case.CaseLadder`).

Under ``node`` quadrature a node's density does not depend on how far the axis
extends, so one reference evaluation gives the true truncated yield at any
``stop``: the measured edge, the edge with twice its margin, and the ceiling.
The production audit bound is reported beside the true fraction. Each
measured (and ``--compare-resolution``) axis also reports line shape and
detected counts against the reference (:func:`shape_and_counts`).

``reference`` runs under FP64 and pickles its transport; ``candidate`` reloads
it in a float32 process and evaluates only the measured axis, so ``compare``
separates bandwidth truncation from backend precision on identical segments.
With ``--compare-resolution`` the reference step additionally resolves and
evaluates the other resolution policy (uniform vs ``resonance-local``) on the
same pickled segments, so the two spacings compare without transport noise.
Heavy: remote only (``python -m pyrite.energy_grid.convergence_job
start-bandwidth``).

A measurement instrument only: no kernel, default, or policy changes here.
"""

import argparse
import dataclasses
import json
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .._line_grid_policy import RESONANCE_BANDWIDTH_POLICY

PAYLOAD_SCHEMA = 1
#: Largest axis piece evaluated at once; an 11 GiB device refuses 1.4 M nodes.
PIECE_POINTS = 200_000


def parse_configs(text: str) -> list[dict[str, float]]:
    """``tilt:azimuth:thickness_ang:ne`` items, comma separated."""
    configs = []
    for item in (value for value in text.split(",") if value):
        tilt, azimuth, thickness, ne = item.split(":")
        configs.append(
            {
                "tilt_deg": float(tilt),
                "tilt_azim_deg": float(azimuth),
                "thickness_ang": float(thickness),
                "n_electrons": int(ne),
            }
        )
    if not configs:
        raise ValueError("at least one configuration is required")
    return configs


def build_case(
    material: str, energy_keV: float, config, *, seed: int, resolution: str = "uniform"
) -> dict[str, Any]:
    """One production case under the ``resonance-population`` bandwidth."""
    from ..campaign.config import material_sweep
    from ..campaign.sweep import build_cases

    sweep = material_sweep(
        material,
        thickness_ang=config["thickness_ang"],
        energy_keV=energy_keV,
        tilt_deg=config["tilt_deg"],
        tilt_azim_deg=config["tilt_azim_deg"],
    )
    policy = {"bandwidth": RESONANCE_BANDWIDTH_POLICY}
    if resolution == "local":
        policy["resolution"] = "resonance-local"
        policy["quadrature"] = "bin-mean"
    elif resolution != "uniform":
        raise ValueError(f"unknown bandwidth resolution {resolution!r}")
    sweep = dataclasses.replace(sweep, line_grid_policy=policy)
    ne = int(config["n_electrons"])
    case = dict(build_cases(sweep, n_electrons=ne, n_electrons_brem=ne)[0])
    case["seed"] = int(seed)
    return case


def reference_axis(
    measured: np.ndarray, ceiling_eV: float, *, spacing_eV: float | None = None
) -> np.ndarray:
    """Uniform reference through the ceiling at the case's sinc spacing."""
    start, stop = float(measured[0]), float(measured[-1])
    step = float(spacing_eV) if spacing_eV is not None else (stop - start) / (measured.size - 1)
    count = int(np.floor((float(ceiling_eV) - start) / step)) + 1
    return start + step * np.arange(max(count, measured.size))


def truncated_fraction(E_grid: np.ndarray, density: np.ndarray, stop_eV: float) -> float:
    """Share of the trapezoid integral over ``E_grid`` lying above ``stop_eV``."""
    total = float(np.trapezoid(density, E_grid))
    if total <= 0.0:
        return 0.0
    kept = E_grid <= float(stop_eV)
    return 1.0 - float(np.trapezoid(density[kept], E_grid[kept])) / total


def _piecewise(evaluate, grid: np.ndarray, piece: int = PIECE_POINTS) -> np.ndarray:
    """``evaluate`` over ``grid`` in consecutive pieces, concatenated.

    A node's line density does not depend on the axis extent under ``node``
    quadrature, so pieces reproduce one evaluation while the device admits each.
    Characteristic bin masses use each node's bin, so the node next to a piece
    boundary takes a one-sided bin; that perturbs only densities at boundaries.
    """
    return np.concatenate(
        [np.asarray(evaluate(grid[i : i + piece]), dtype=float) for i in range(0, grid.size, piece)]
    )


def _yield_and_centroid(E_grid: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    total = float(np.trapezoid(density, E_grid))
    return total, float(np.trapezoid(density * E_grid, E_grid)) / total if total else float("nan")


#: Intrinsic line-shape comparison bin (eV): coarser than every axis spacing,
#: finer than any detector resolution.
SHAPE_BIN_EV = 100.0
#: Upper edge (eV) of the band scored through the Timepix3 response; its matrix
#: costs ~1 s per keV of band on the host, and silicon efficiency is small above.
DEFAULT_DETECTOR_MAX_EV = 60_000.0


def _bin_masses(
    E_grid: np.ndarray, density: np.ndarray, edges: np.ndarray, *, quadrature: str = "node"
) -> np.ndarray:
    """Mass of ``density`` in each ``edges`` bin, consistent with its quadrature.

    Under ``node`` the cumulative mass is the trapezoid at the nodes; under
    ``bin-mean`` each node carries its cell's mean over the ``bin_axis``
    midpoint cells, so the cumulative steps at those cell edges. Mass is linear
    within a cell, so a cell straddling a bin edge splits linearly.
    """
    if quadrature == "bin-mean":
        from ..montecarlo.spectrum.lines._bin_quadrature import bin_axis

        cells = np.asarray(_to_host(bin_axis(E_grid)[0]), dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(density * np.diff(cells))))
        return np.diff(np.interp(edges, cells, cumulative))
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(E_grid)))
    )
    return np.diff(np.interp(edges, E_grid, cumulative))


def _to_host(value):
    from .._backend import _to_cpu

    return _to_cpu(value)


def _histogram_deviation(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """L1 and peak-normalized worst-bin difference of two histograms."""
    total = float(np.abs(reference).sum())
    peak = float(np.abs(reference).max()) if reference.size else 0.0
    difference = np.abs(candidate - reference)
    return {
        "l1_rel": float(difference.sum()) / total if total > 0.0 else float("nan"),
        "max_bin_rel": float(difference.max()) / peak if peak > 0.0 else float("nan"),
    }


def _relative(candidate: float, reference: float) -> float:
    return (candidate - reference) / reference if reference else float("nan")


def _cells(E_grid: np.ndarray, quadrature: str) -> np.ndarray:
    """Cell edges the density's mass lives on: ``bin_axis`` midpoint cells under
    ``bin-mean``; the nodes themselves (trapezoid cells) under ``node``."""
    if quadrature != "bin-mean":
        return E_grid
    from ..montecarlo.spectrum.lines._bin_quadrature import bin_axis

    return np.asarray(_to_host(bin_axis(E_grid)[0]), dtype=float)


def _intrinsic_bins(
    ref_E: np.ndarray,
    ref: np.ndarray,
    cand_E: np.ndarray,
    cand: np.ndarray,
    *,
    bin_eV: float,
    quadrature: str,
) -> dict[str, float]:
    """Intrinsic bin masses on ~``bin_eV`` bins whose edges are the candidate's
    own cell edges, so its masses are exact.

    Neither spectrum resolves mass inside a cell. A bin edge inside a
    reference cell is split linearly, and ``reference_split_bound`` bounds what
    that split can move: twice the mass of the reference cells holding
    interior edges, over the total. An ``l1_rel`` below it cannot be told
    apart from the sub-cell ambiguity.
    """
    cand_cells = _cells(cand_E, quadrature)
    lo, hi = float(cand_E[0]), float(cand_E[-1])
    nominal = lo + bin_eV * np.arange(1, int(np.ceil((hi - lo) / bin_eV)))
    inner = cand_cells[(cand_cells > lo) & (cand_cells < hi)]
    snapped = np.unique(inner[np.clip(np.searchsorted(inner, nominal), 0, inner.size - 1)])
    edges = np.r_[lo, snapped, hi]
    ref_masses = _bin_masses(ref_E, ref, edges, quadrature=quadrature)
    cand_masses = _bin_masses(cand_E, cand, edges, quadrature=quadrature)
    ref_cells = _cells(ref_E, quadrature)
    if quadrature == "bin-mean":
        cell_mass = ref * np.diff(ref_cells)
    else:
        cell_mass = 0.5 * (ref[1:] + ref[:-1]) * np.diff(ref_cells)
    holding = np.clip(np.searchsorted(ref_cells, snapped, side="right") - 1, 0, cell_mass.size - 1)
    total = float(ref_masses.sum())
    return {
        "bin_eV": bin_eV,
        "n_bins": int(edges.size - 1),
        **_histogram_deviation(ref_masses, cand_masses),
        "reference_split_bound": 2.0 * float(cell_mass[holding].sum()) / total
        if total > 0.0
        else float("nan"),
    }


def _dominant_fwhm_eV(E_grid: np.ndarray, density: np.ndarray) -> float:
    """FWHM of the highest line: linear half-maximum crossings either side of
    the global peak. O(n), unlike prominence peak-finding, which is quadratic
    on multi-million-node reference axes."""
    peak = int(np.argmax(density))
    half = 0.5 * float(density[peak])
    if not half > 0.0:
        return float("nan")
    below = np.flatnonzero(density[:peak] < half)
    above = np.flatnonzero(density[peak:] < half)
    if below.size == 0 or above.size == 0:
        return float("nan")
    i, j = int(below[-1]), peak + int(above[0])

    def crossing(low: int, high: int) -> float:
        """Energy where the density passes ``half`` between node ``low``
        (below half) and its neighbour ``high`` (at or above half)."""
        rise = float(density[high] - density[low])
        return float(E_grid[low] + (half - density[low]) * (E_grid[high] - E_grid[low]) / rise)

    return crossing(j, j - 1) - crossing(i, i + 1)


def shape_and_counts(
    reference_E: np.ndarray,
    reference_density: np.ndarray,
    candidate_E: np.ndarray,
    candidate_density: np.ndarray,
    *,
    detector_max_eV: float = DEFAULT_DETECTOR_MAX_EV,
    shape_bin_eV: float = SHAPE_BIN_EV,
    quadrature: str = "node",
) -> dict[str, Any]:
    """Line shape and detected counts of a candidate axis against a reference.

    The two densities live on different axes (local vs full-ceiling uniform),
    so every comparison goes through a binning both share: ``shape_bin_eV``
    bins of intrinsic mass over the candidate's span; the highest line's FWHM
    (:func:`_dominant_fwhm_eV`, sensitive to local sampling of that one line); EagleXO
    counts (QE only) over each full axis; and Timepix3 detected events on its
    fixed native output bins, both spectra cut to ``[start, detector_max_eV]``
    so one response lattice serves both. Intrinsic mass above the cut is
    reported beside it; the detector can see at most that much more.
    """
    from ..detectors.spec import EagleXO, Timepix3

    ref_E = np.asarray(reference_E, dtype=float)
    ref = np.asarray(reference_density, dtype=float)
    cand_E = np.asarray(candidate_E, dtype=float)
    cand = np.asarray(candidate_density, dtype=float)
    stop = float(cand_E[-1])
    out: dict[str, Any] = {
        "intrinsic_bins": _intrinsic_bins(
            ref_E, ref, cand_E, cand, bin_eV=shape_bin_eV, quadrature=quadrature
        )
    }
    ref_fwhm, cand_fwhm = _dominant_fwhm_eV(ref_E, ref), _dominant_fwhm_eV(cand_E, cand)
    out["fwhm_rel"] = _relative(cand_fwhm, ref_fwhm)
    out["reference_fwhm_eV"] = ref_fwhm
    eaglexo = EagleXO()
    ref_counts = float(np.trapezoid(eaglexo.score(ref_E, ref, fwhm_eV=None, scale=1.0), ref_E))
    cand_counts = float(np.trapezoid(eaglexo.score(cand_E, cand, fwhm_eV=None, scale=1.0), cand_E))
    out["eaglexo_counts_rel"] = _relative(cand_counts, ref_counts)

    # Timepix3: exact (overlap-split) mass in the response's fixed input
    # channels, through one response matrix for both spectra, so the detected
    # difference is the line grid's alone. ``native_score`` (``apply_native``)
    # assigns each node's whole cell to the channel holding the node (#219);
    # that axis-dependent resampling term is reported separately.
    from ..detectors.timepix_response import get_response

    cut = min(float(detector_max_eV), stop)
    ref_band, cand_band = ref_E <= cut, cand_E <= cut
    detector = Timepix3()
    response = get_response(
        ref_E[ref_band],
        dE_mc=detector.dE_mc,
        dE_out=detector.dE_out,
        n_mc=detector.n_mc,
        seed=detector.seed,
        thickness_um=detector.thickness_um,
        bias_v=detector.bias_v,
    )
    channels = np.asarray(response.in_edges, dtype=float)
    ref_events = (
        _bin_masses(ref_E[ref_band], ref[ref_band], channels, quadrature=quadrature) @ response.R.T
    )
    cand_events = (
        _bin_masses(cand_E[cand_band], cand[cand_band], channels, quadrature=quadrature)
        @ response.R.T
    )
    ref_native = float(np.sum(detector.native_score(ref_E[ref_band], ref[ref_band]).events))
    cand_native = float(np.sum(detector.native_score(cand_E[cand_band], cand[cand_band]).events))
    total = float(np.trapezoid(ref, ref_E))
    above = float(np.trapezoid(ref[~ref_band], ref_E[~ref_band])) if (~ref_band).sum() > 1 else 0.0
    out["timepix3"] = {
        "band_max_eV": cut,
        "counts_rel": _relative(float(cand_events.sum()), float(ref_events.sum())),
        "reference_counts": float(ref_events.sum()),
        "intrinsic_fraction_above_band": above / total if total > 0.0 else 0.0,
        **_histogram_deviation(ref_events, cand_events),
        "node_rebin_counts_rel": _relative(cand_native, ref_native),
    }
    return out


def _peak_mib() -> float | None:
    from .._backend import BACKEND

    peak = BACKEND.allocator_stats().get("peak_mib")
    return None if peak is None else float(peak)


def _host_peak_mib() -> float:
    """Peak resident host memory of this process in MiB (Linux ``ru_maxrss``)."""
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _alternate_resolution(args, config, case, ladder, axis, reference_lines):
    """Resolve and evaluate the other resolution policy on the same segments.

    The alternate case shares the seed and every transport input, so its
    resolved grid is a function of the identical trajectories; only the line
    axis differs. Returns ``{"row": report fields, "payload": pickled axes}``.
    """
    from ..montecarlo.runner.line_grid import resolve_line_grid

    alt_case = build_case(
        args.material, args.energy, config, seed=args.seed, resolution=args.compare_resolution
    )
    grid, _record = resolve_line_grid(
        alt_case,
        ladder.transport["segs"],
        ladder.transport["n_hat"],
        int(ladder.transport["Ne_lines"]),
        np.asarray(ladder.transport["E_grid"], dtype=float),
        case.get("abs_layers"),
        ladder.transport.get("groove"),
    )
    grid = np.asarray(grid, dtype=float)
    started = time.perf_counter()
    lines = np.asarray(_piecewise(ladder.lines, grid), dtype=float)
    wall = time.perf_counter() - started
    reference = np.asarray(reference_lines, dtype=float)
    total, centroid = _yield_and_centroid(axis, reference)
    kept, kept_centroid = _yield_and_centroid(grid, lines)
    row = {
        "resolution": args.compare_resolution,
        "points": int(grid.size),
        "minimum_spacing_eV": float(np.diff(grid).min()),
        "maximum_spacing_eV": float(np.diff(grid).max()),
        "stop_eV": float(grid[-1]),
        "wall_s": wall,
        "yield_rel": (total - kept) / total if total else 0.0,
        "centroid_shift_eV": kept_centroid - centroid,
        "true_fraction_above_stop": truncated_fraction(axis, reference, float(grid[-1])),
        "shape": shape_and_counts(
            axis,
            reference,
            grid,
            lines,
            detector_max_eV=args.detector_max_eV,
            quadrature=case.get("line_quadrature", "node"),
        ),
    }
    return {"row": row, "payload": {"grid": grid, "lines_fp64": lines}}


def reference(args: argparse.Namespace) -> dict[str, Any]:
    """FP64: transport, evaluate measured and reference axes, pickle segments."""
    from .._backend import REAL
    from .convergence import segment_fingerprint
    from .convergence_case import CaseLadder

    rows, payload = [], []
    for config in parse_configs(args.configs):
        case = build_case(
            args.material, args.energy, config, seed=args.seed, resolution=args.resolution
        )
        ladder = CaseLadder(case)
        measured = np.asarray(ladder.transport["E_grid"], dtype=float)
        record = ladder.transport["diagnostic_grid"]
        ceiling = float(case["line_grid_policy"]["bandwidth"]["stop_eV"])
        axis = reference_axis(measured, ceiling, spacing_eV=record.get("feature_width_eV"))
        timings = {}
        spectra = {}
        for name, grid in (("measured", measured), ("reference", axis)):
            started = time.perf_counter()
            # Lines and characteristic only: the continuum is not on this axis.
            spectra[name] = {
                "lines": _piecewise(ladder.lines, grid),
                "characteristic": _piecewise(ladder._characteristic, grid),
            }
            timings[name] = time.perf_counter() - started
        stop = float(measured[-1])
        doubled = float(measured[0]) + 2.0 * (stop - float(measured[0]))
        row: dict[str, Any] = {
            **config,
            "real": np.dtype(REAL).name,
            "n_segments": int(np.asarray(ladder.segments["L_ang"]).size),
            "transport_wall_s": ladder.transport_wall_s,
            "measured_points": int(measured.size),
            "reference_points": int(axis.size),
            "minimum_spacing_eV": float(np.diff(measured).min()),
            "maximum_spacing_eV": float(np.diff(measured).max()),
            "reference_spacing_eV": float(axis[1] - axis[0]),
            "stop_eV": stop,
            "ceiling_eV": ceiling,
            "measured_bandwidth": record.get("measured_bandwidth"),
            "measured_wall_s": timings["measured"],
            "reference_wall_s": timings["reference"],
            "device_peak_mib": _peak_mib(),
        }
        for component in ("lines", "characteristic"):
            density = np.asarray(spectra["reference"][component], dtype=float)
            total, centroid = _yield_and_centroid(axis, density)
            kept, kept_centroid = _yield_and_centroid(
                measured, np.asarray(spectra["measured"][component], dtype=float)
            )
            component_row: dict[str, Any] = {
                "reference_yield": total,
                "measured_yield": kept,
                "yield_rel": (total - kept) / total if total else 0.0,
                "centroid_shift_eV": kept_centroid - centroid,
                "true_fraction_above_stop": truncated_fraction(axis, density, stop),
                "true_fraction_above_2x": truncated_fraction(axis, density, doubled),
            }
            row[component] = component_row
        row["lines"]["shape"] = shape_and_counts(
            axis,
            np.asarray(spectra["reference"]["lines"], dtype=float),
            measured,
            np.asarray(spectra["measured"]["lines"], dtype=float),
            detector_max_eV=args.detector_max_eV,
            quadrature=case.get("line_quadrature", "node"),
        )
        row["total_shape"] = shape_and_counts(
            axis,
            np.asarray(spectra["reference"]["lines"], dtype=float)
            + np.asarray(spectra["reference"]["characteristic"], dtype=float),
            measured,
            np.asarray(spectra["measured"]["lines"], dtype=float)
            + np.asarray(spectra["measured"]["characteristic"], dtype=float),
            detector_max_eV=args.detector_max_eV,
            quadrature=case.get("line_quadrature", "node"),
        )
        alternate = None
        if args.compare_resolution is not None:
            alternate = _alternate_resolution(
                args, config, case, ladder, axis, spectra["reference"]["lines"]
            )
            row["alternate"] = alternate["row"]
        row["host_peak_mib"] = _host_peak_mib()
        rows.append(row)
        payload.append(
            {
                "config": config,
                "case": case,
                "transport": ladder.transport,
                "fingerprint": segment_fingerprint(ladder.segments),
                "measured": measured,
                "lines_fp64": np.asarray(spectra["measured"]["lines"], dtype=float),
                "alternate": None
                if alternate is None
                else {"resolution": args.compare_resolution, **alternate["payload"]},
            }
        )
        print(json.dumps(row, default=str), flush=True)
    with Path(args.payload).open("wb") as stream:
        pickle.dump({"schema": PAYLOAD_SCHEMA, "items": payload}, stream, protocol=5)
    report = {"material": args.material, "energy_keV": args.energy, "rows": rows}
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return report


def production(args: argparse.Namespace) -> dict[str, Any]:
    """Production precision at production electron counts: axis, audit, cost.

    No reference axis and no pickle, so it scales to counts whose segments a
    full-ceiling evaluation could not afford. The audit is the one the spectrum
    phase gates on; a refusal is recorded, not raised.
    """
    from .._backend import BACKEND, REAL, _to_cpu
    from .._line_grid_policy import LineGridToleranceError
    from ..montecarlo import runner
    from ..montecarlo.runner.line_grid import (
        _device_mib,
        check_line_truncation,
        line_truncation_audit,
        line_yield_statistics,
    )
    from ..montecarlo.runner.oom import _ensure_pool_limit

    # Production caps the CuPy pool; without it an over-budget allocation can
    # spill into shared host memory under WSL instead of raising.
    _ensure_pool_limit()
    rows = []
    for config in parse_configs(args.configs):
        case = build_case(
            args.material, args.energy, config, seed=args.seed, resolution=args.resolution
        )
        case["_profile_line_grid_stages"] = True
        case["_min_line_electron_blocks"] = int(args.min_electron_blocks)
        row: dict[str, Any] = {**config, "real": np.dtype(REAL).name}
        started = time.perf_counter()
        try:
            transport = runner._transport_case(case, keep_segments_on_device=True)
        except LineGridToleranceError as error:
            row["refused"] = str(error)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            continue
        except Exception as error:
            if not runner._is_gpu_oom(error):
                raise
            row.update(device_oom="transport", error=str(error), device_at_oom=_device_mib())
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            BACKEND.release_memory()
            continue
        row["transport_wall_s"] = time.perf_counter() - started
        row["device_after_transport"] = _device_mib()
        row["segments_on_device"] = not isinstance(transport["segs"]["L_ang"], np.ndarray)
        print(json.dumps({"stage": "transport", **row}, default=str), flush=True)
        grid = np.asarray(transport["E_grid"], dtype=float)
        audit = line_truncation_audit(case, grid, n_electrons=transport["Ne_lines"])
        started = time.perf_counter()
        try:
            lines = runner._lines_for_segments(
                transport["segs"],
                grid,
                case,
                transport["n_hat"],
                case.get("abs_layers"),
                transport.get("groove"),
                coherent=False,
                Ne=transport["Ne_lines"],
                truncation_audit=audit,
            )
        except Exception as error:
            if not runner._is_gpu_oom(error):
                raise
            row.update(device_oom="lines", error=str(error), device_at_oom=_device_mib())
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
            del transport
            BACKEND.release_memory()
            continue
        row["lines_wall_s"] = time.perf_counter() - started
        row["device_after_lines"] = _device_mib()
        try:
            row["truncation_audit"] = check_line_truncation(case, audit)
        except LineGridToleranceError as error:
            row["refused"] = str(error)
            if "electron_mass" in audit:
                row["line_yield_statistics"] = line_yield_statistics(audit["electron_mass"])
        record = transport["diagnostic_grid"]
        density = np.asarray(lines, dtype=float)
        total, centroid = _yield_and_centroid(grid, density)
        row.update(
            n_segments=int(np.asarray(_to_cpu(transport["segs"]["L_ang"])).size),
            points=int(grid.size),
            minimum_spacing_eV=float(np.diff(grid).min()),
            maximum_spacing_eV=float(np.diff(grid).max()),
            stop_eV=float(grid[-1]),
            measured_bandwidth=record.get("measured_bandwidth"),
            line_grid_profile=transport["diagnostic_grid"].get("line_grid_profile"),
            line_yield=total,
            line_centroid_eV=centroid,
            device=BACKEND.device.name,
            device_peak_mib=_peak_mib(),
            host_peak_mib=_host_peak_mib(),
        )
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
        del transport, lines
        BACKEND.release_memory()
    report = {"material": args.material, "energy_keV": args.energy, "rows": rows}
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return report


def attribute(args: argparse.Namespace) -> dict[str, Any]:
    """Heaviest incoherent lines and per-electron line mass (issue #201).

    Transports each configuration as ``production`` does, then evaluates the
    production line kernel once on the two-node ``[start, ceiling]`` axis with
    an attribution audit. Reports the ``--top`` lines by mass with the factors
    that built them, and how the line mass spreads over electrons.
    """
    from .._backend import BACKEND, REAL, _to_cpu
    from ..montecarlo import runner
    from ..montecarlo.runner.oom import _ensure_pool_limit
    from ..montecarlo.spectrum.lines._attribution import merge_line_attribution

    _ensure_pool_limit()
    rows = []
    for config in parse_configs(args.configs):
        case = build_case(
            args.material, args.energy, config, seed=args.seed, resolution=args.resolution
        )
        row: dict[str, Any] = {**config, "real": np.dtype(REAL).name}
        started = time.perf_counter()
        transport = runner._transport_case(case, keep_segments_on_device=True)
        row["transport_wall_s"] = time.perf_counter() - started
        bandwidth = case["line_grid_policy"]["bandwidth"]
        start, ceiling = float(bandwidth["start_eV"]), float(bandwidth["stop_eV"])
        measured_stop = float(np.asarray(transport["E_grid"])[-1])
        audit = {
            "start_eV": start,
            "stop_eV": ceiling,
            "collect": [],
            "attribute": {"top": args.top, "stop_eV": measured_stop},
        }
        started = time.perf_counter()
        runner._lines_for_segments(
            transport["segs"],
            np.array([start, ceiling]),
            case,
            transport["n_hat"],
            case.get("abs_layers"),
            transport.get("groove"),
            coherent=False,
            Ne=transport["Ne_lines"],
            truncation_audit=audit,
        )
        row["attribution_wall_s"] = time.perf_counter() - started
        merged = merge_line_attribution(audit.get("attribution", []), args.top)
        ids, mass = merged["electron_ids"], merged["electron_mass"]
        tail = merged["electron_tail"]
        total = float(mass.sum())
        tail_total = float(tail.sum())
        ranked = np.sort(mass)[::-1]
        ne = int(transport["Ne_lines"])
        row.update(
            n_segments=int(np.asarray(_to_cpu(transport["segs"]["L_ang"])).size),
            n_lines=int(sum(chunk[0].size for chunk in audit["collect"])),
            ceiling_eV=ceiling,
            line_mass=total,
            line_mass_per_electron=total / max(ne, 1),
            top_electron_share=[
                float(ranked[:k].sum() / total) if total else 0.0 for k in (1, 10, 100)
            ],
            mass_by_electron_quarter=[
                float(mass[(ids >= q * ne // 4) & (ids < (q + 1) * ne // 4)].sum())
                for q in range(4)
            ],
            heaviest_electrons={int(ids[j]): float(mass[j]) for j in np.argsort(-mass)[:20]},
            measured_stop_eV=measured_stop,
            measured_bandwidth=transport["diagnostic_grid"].get("measured_bandwidth"),
            tail_fraction_at_stop=tail_total / total if total else 0.0,
            tail_by_electron={
                int(ids[j]): float(tail[j] / tail_total) for j in np.argsort(-tail)[:20]
            }
            if tail_total
            else {},
            lines_by_mass={k: v.tolist() for k, v in merged["lines_by_mass"].items()},
            lines_by_tail={k: v.tolist() for k, v in merged["lines_by_tail"].items()},
            device=BACKEND.device.name,
            host_peak_mib=_host_peak_mib(),
        )
        rows.append(row)
        summary = {k: v for k, v in row.items() if not k.startswith("lines")}
        print(json.dumps(summary, default=str), flush=True)
        del transport, audit
        BACKEND.release_memory()
    report = {"material": args.material, "energy_keV": args.energy, "seed": args.seed, "rows": rows}
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return report


def candidate(args: argparse.Namespace) -> dict[str, Any]:
    """float32: the measured axis on the pickled segments, against FP64."""
    from .._backend import BACKEND, REAL
    from .convergence import lineshape_deviation, segment_fingerprint
    from .convergence_case import CaseLadder

    if np.dtype(REAL) != np.dtype(np.float32):
        raise SystemExit(f"backend REAL is {np.dtype(REAL).name} on {BACKEND.name}, need float32")
    with Path(args.payload).open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - our own measurement artifact
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise SystemExit(f"{args.payload} is not a schema-{PAYLOAD_SCHEMA} bandwidth payload")
    report = json.loads(Path(args.json_out).read_text())
    for row, item in zip(report["rows"], payload["items"], strict=True):
        ladder = CaseLadder(item["case"], transport=item["transport"])
        if segment_fingerprint(ladder.segments) != item["fingerprint"]:
            raise SystemExit("pickled segments do not match their stored fingerprint")
        started = time.perf_counter()
        lines = np.asarray(ladder.lines(item["measured"]), dtype=float)
        wall = time.perf_counter() - started
        grid, reference = item["measured"], item["lines_fp64"]
        fp64_yield, fp64_centroid = _yield_and_centroid(grid, reference)
        f32_yield, f32_centroid = _yield_and_centroid(grid, lines)
        # Where the centroid moves: its per-node contribution difference.
        moment = (lines - reference) * (grid - fp64_centroid)
        worst = int(np.argmax(np.abs(moment)))
        row["float32"] = {
            "yield_rel": (f32_yield - fp64_yield) / fp64_yield if fp64_yield else 0.0,
            "centroid_shift_eV": f32_centroid - fp64_centroid,
            "fp64_centroid_eV": fp64_centroid,
            "deviation": lineshape_deviation(grid, reference, lines),
            "worst_moment_node_eV": float(grid[worst]),
            "moment_share_above_20keV": float(
                np.trapezoid(np.where(grid > 20_000.0, moment, 0.0), grid)
                / (np.trapezoid(moment, grid) or 1.0)
            ),
            "wall_s": wall,
            "device_peak_mib": _peak_mib(),
            "host_peak_mib": _host_peak_mib(),
        }
        alternate = item.get("alternate")
        if alternate is not None:
            alt_grid = np.asarray(alternate["grid"], dtype=float)
            alt_lines = _piecewise(ladder.lines, alt_grid)
            alt_reference = np.asarray(alternate["lines_fp64"], dtype=float)
            alt_fp64_yield, alt_fp64_centroid = _yield_and_centroid(alt_grid, alt_reference)
            alt_yield, alt_centroid = _yield_and_centroid(alt_grid, alt_lines)
            row["float32"]["alternate"] = {
                "resolution": alternate["resolution"],
                "yield_rel": (alt_yield - alt_fp64_yield) / alt_fp64_yield
                if alt_fp64_yield
                else 0.0,
                "centroid_shift_eV": alt_centroid - alt_fp64_centroid,
                "deviation": lineshape_deviation(alt_grid, alt_reference, alt_lines),
            }
        print(json.dumps(row["float32"]), flush=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference", help="FP64 transport and reference evaluation")
    ref.add_argument("--material", default="hbn")
    ref.add_argument("--energy", type=float, default=5000.0)
    ref.add_argument("--configs", required=True, help="tilt:azimuth:thickness_ang:ne,...")
    ref.add_argument("--seed", type=int, default=0)
    ref.add_argument("--resolution", choices=("uniform", "local"), default="uniform")
    ref.add_argument(
        "--compare-resolution",
        choices=("uniform", "local"),
        default=None,
        help="also resolve and evaluate this resolution on the same segments",
    )
    ref.add_argument(
        "--detector-max-eV",
        dest="detector_max_eV",
        type=float,
        default=DEFAULT_DETECTOR_MAX_EV,
        help="upper edge of the Timepix3-scored band",
    )
    ref.add_argument("--payload", required=True)
    ref.add_argument("--json-out", required=True)
    prod = commands.add_parser("production", help="production-precision axis, audit and cost")
    prod.add_argument("--material", default="hbn")
    prod.add_argument("--energy", type=float, default=5000.0)
    prod.add_argument("--configs", required=True, help="tilt:azimuth:thickness_ang:ne,...")
    prod.add_argument("--seed", type=int, default=0)
    prod.add_argument("--resolution", choices=("uniform", "local"), default="uniform")
    prod.add_argument("--json-out", required=True)
    prod.add_argument("--min-electron-blocks", type=int, default=1)
    attr = commands.add_parser("attribute", help="heaviest lines and per-electron line mass")
    attr.add_argument("--material", default="hbn")
    attr.add_argument("--energy", type=float, default=5000.0)
    attr.add_argument("--configs", required=True, help="tilt:azimuth:thickness_ang:ne,...")
    attr.add_argument("--seed", type=int, default=0)
    attr.add_argument("--resolution", choices=("uniform", "local"), default="local")
    attr.add_argument("--top", type=int, default=200)
    attr.add_argument("--json-out", required=True)
    cand = commands.add_parser("candidate", help="float32 measured-axis evaluation")
    cand.add_argument("--payload", required=True)
    cand.add_argument("--json-out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reference":
        reference(args)
    elif args.command == "production":
        production(args)
    elif args.command == "attribute":
        attribute(args)
    else:
        candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
