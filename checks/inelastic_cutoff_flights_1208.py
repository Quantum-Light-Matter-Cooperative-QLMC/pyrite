"""Remote-only diagnostic: why CXR line shape depends on the inelastic cutoff (#1208).

One HOPG transport per (arm, seed), with the #1142 frozen physics and only
``inelastic_cutoff_eV`` changed. The script reports:

* flight statistics on the line electrons: path per flight, rows per flight,
  hard-inelastic events per unit path, and the events that close flights;
* the dominant-line yield, centroid, peak and FWHM, scored twice on the same
  trajectories. ``default`` uses the production flight grouping, in which a
  hard inelastic collision closes the flight. ``merged`` relabels flight ids
  so that hard inelastic collisions do not close it. The rows on both sides
  of the collision then add as one complex field. Each row keeps its own
  energy, direction, position and age, so the merged field is the classical
  amplitude of the piecewise-straight track through the collision.

If the cutoff dependence disappears under ``merged``, it comes from the
emission-unit boundary at hard events (candidate 1 in #1208), not from the
transport itself. No default or physics verdict is made here.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ARMS = {"cut35": 35.0, "baseline": 50.0, "cut100": 100.0}


def build_case(arm, seed, histories, energy_keV, thickness_ang, straggling=False, aed="kawrakow"):
    """The #1142 HOPG t2 case with the arm's inelastic cutoff."""
    from pyrite.energy_grid.convergence_case import build_ladder_case

    case = build_ladder_case(
        "hopg", energy_keV, 45, 180, thickness_ang=thickness_ang, n_electrons=histories, seed=seed
    )
    from pyrite._energy_grid_encoding import encode_energy_grid

    # The #1142 node carrier: 3 eV nodes over 0.5-10 keV; flight-grouped rows
    # need node quadrature.
    case.pop("line_grid_policy", None)
    carrier = encode_energy_grid(np.linspace(500.0, 10000.0, 3167))
    case.update(
        E_grid=carrier,
        E_grid_line=carrier,
        line_quadrature="node",
        emission="incoherent",
        transport_mode="single-scattering",
        transport_precision="fp64",
        max_dE_frac=0.0,
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=ARMS[arm],
        straggling=straggling,
        elastic_model="elsepa",
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        atomic_electron_deflection=None if aed == "none" else aed,
    )
    return case


def host_segments(segments):
    from pyrite._backend import _to_cpu

    return {k: (_to_cpu(v) if hasattr(v, "shape") else v) for k, v in segments.items()}


def merge_hard_inelastic(segments):
    """Copy of ``segments`` whose flights continue through hard inelastic events.

    A flight closed by a ``HARD_INELASTIC`` row and the flight after it get
    one id. Ids stay ordered within each owner (track or electron).
    """
    from pyrite.montecarlo.transport.events import SegmentEvent

    owner = segments.get("track_id")
    owner = np.asarray(segments["elec_id"] if owner is None else owner, dtype=np.int64)
    flight = np.asarray(segments["flight_id"], dtype=np.int64)
    kind = np.asarray(segments["event_kind"])
    span = int(flight.max()) + 2 if flight.size else 1
    key = owner * span + flight
    closed = np.sort(key[kind == int(SegmentEvent.HARD_INELASTIC)])
    before = np.searchsorted(closed, key, side="left") - np.searchsorted(
        closed, owner * span, side="left"
    )
    out = dict(segments)
    out["flight_id"] = (flight - before).astype(np.asarray(segments["flight_id"]).dtype)
    return out


def flight_stats(segments, n_lines):
    """Flight/row/event statistics of the line electrons."""
    from pyrite.montecarlo.transport.events import FLIGHT_CONTINUING, SegmentEvent

    keep = np.asarray(segments["elec_id"]) < n_lines
    owner = segments.get("track_id")
    owner = np.asarray(segments["elec_id"] if owner is None else owner)[keep]
    flight = np.asarray(segments["flight_id"])[keep]
    length = np.asarray(segments["L_ang"], dtype=float)[keep]
    kind = np.asarray(segments["event_kind"])[keep]
    v = np.asarray(segments["v_hat"], dtype=float)[keep]
    _, group = np.unique(np.stack([owner, flight]), axis=1, return_inverse=True)
    group = group.ravel()
    path = np.bincount(group, weights=length)
    rows = np.bincount(group)
    closing = ~np.isin(kind, [int(k) for k in FLIGHT_CONTINUING])
    closers = {
        SegmentEvent(int(k)).name: int(n)
        for k, n in zip(*np.unique(kind[closing], return_counts=True), strict=True)
    }
    total = float(length.sum())
    cos_z = np.abs(v[:, 2])
    order = np.argsort(cos_z)
    cdf = np.cumsum(length[order]) / total
    quant = {
        f"q{int(100 * q)}": float(cos_z[order][np.searchsorted(cdf, q)]) for q in (0.1, 0.5, 0.9)
    }
    return dict(
        rows=int(keep.sum()),
        flights=int(path.size),
        path_ang_total=total,
        flight_path_ang_mean=float(path.mean()),
        flight_path_ang_quantiles={
            f"q{q}": float(np.quantile(path, q / 100)) for q in (10, 50, 90, 99)
        },
        flight_path_ang_weighted_mean=float((path**2).sum() / path.sum()),
        rows_per_flight_mean=float(rows.mean()),
        closing_events=closers,
        hard_inelastic_per_um=1e4 * closers.get("HARD_INELASTIC", 0) / total,
        elastic_per_um=1e4 * closers.get("ELASTIC", 0) / total,
        path_weighted_abs_cos_z=quant,
        deflections_per_um=deflection_rates(owner, flight, segments, keep, kind, v, total),
    )


#: Deflection-angle thresholds [deg] of the per-event rate table.
THRESHOLDS_DEG = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def deflection_rates(owner, flight, segments, keep, kind, v, total):
    """Events per micrometre deflecting the track by more than each threshold.

    Pairs consecutive rows of one owner; the angle between them is charged to
    the event that ends the first row.
    """
    from pyrite.montecarlo.transport.events import SegmentEvent

    sub = np.asarray(segments["substep_id"])[keep] if "substep_id" in segments else 0 * flight
    order = np.lexsort((sub, flight, owner))
    a, b = order[:-1], order[1:]
    same = owner[a] == owner[b]
    a, b = a[same], b[same]
    angle = np.degrees(np.arccos(np.clip(np.sum(v[a] * v[b], axis=1), -1.0, 1.0)))
    out = {}
    for k in (SegmentEvent.ELASTIC, SegmentEvent.HARD_INELASTIC, SegmentEvent.HARD_RADIATIVE):
        mine = angle[kind[a] == int(k)]
        out[k.name] = {f">{t}": 1e4 * float((mine > t).sum()) / total for t in THRESHOLDS_DEG}
    out["ALL"] = {f">{t}": 1e4 * float((angle > t).sum()) / total for t in THRESHOLDS_DEG}
    return out


def line_metrics(grid, density):
    from pyrite.energy_grid.bandwidth_check import _dominant_fwhm_eV

    total = float(np.trapezoid(density, grid))
    fwhm = float(_dominant_fwhm_eV(grid, density))
    return dict(
        yield_per_primary=total,
        centroid_eV=float(np.trapezoid(density * grid, grid) / total),
        peak_density_per_eV=float(density.max()),
        peak_eV=float(grid[int(np.argmax(density))]),
        dominant_fwhm_eV=fwhm if np.isfinite(fwhm) else None,
    )


def score(case, transport, window, spacing):
    """Dominant-line metrics on a node axis of ``spacing`` around the peak."""
    from pyrite.energy_grid.convergence_case import CaseLadder

    ladder = CaseLadder(case, transport=transport)
    coarse = np.asarray(transport["E_grid"], dtype=float)
    started = time.perf_counter()
    peak = float(coarse[int(np.argmax(ladder.lines(coarse)))])
    grid = np.arange(peak - window, peak + window + spacing / 2, spacing)
    density = ladder.lines(grid)
    out = line_metrics(grid, density)
    out["score_s"] = time.perf_counter() - started
    return out, grid, density


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=1208001)
    parser.add_argument("--histories", type=int, default=4096)
    parser.add_argument("--energy-keV", type=float, default=30.0)
    parser.add_argument("--thickness-ang", type=float, default=20000.0)
    parser.add_argument("--transport-core", default="auto")
    parser.add_argument("--window-eV", type=float, default=60.0)
    parser.add_argument("--spacing-eV", type=float, default=0.1875)
    parser.add_argument("--aed", choices=["kawrakow", "none"], default="kawrakow")
    parser.add_argument(
        "--straggling", action="store_true", help="restricted soft-loss straggling (#642 default)"
    )
    args = parser.parse_args()
    from pyrite.montecarlo import runner

    args.output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}{'-strag' if args.straggling else ''}-aed{args.aed}-s{args.seed}"
    case = build_case(
        args.arm,
        args.seed,
        args.histories,
        args.energy_keV,
        args.thickness_ang,
        args.straggling,
        args.aed,
    )
    started = time.perf_counter()
    transport = runner._transport_case(case, transport_core=args.transport_core)
    transport_s = time.perf_counter() - started
    segments = host_segments(transport["segs"])
    n_lines = int(transport["Ne_lines"])
    merged = merge_hard_inelastic(segments)
    record = dict(
        arm=args.arm,
        inelastic_cutoff_eV=ARMS[args.arm],
        straggling=args.straggling,
        atomic_electron_deflection=args.aed,
        seed=args.seed,
        histories=args.histories,
        transport_s=transport_s,
        stats=dict(default=flight_stats(segments, n_lines), merged=flight_stats(merged, n_lines)),
    )
    spectra = {}
    for name, segs in (("default", segments), ("merged", merged)):
        metrics, grid, density = score(
            case, dict(transport, segs=segs), args.window_eV, args.spacing_eV
        )
        record[name] = metrics
        spectra[f"{name}_grid"], spectra[f"{name}_density"] = grid, density
    (args.output / f"{tag}.json").write_text(json.dumps(record, indent=1))
    np.savez_compressed(args.output / f"{tag}.npz", **spectra)
    print(json.dumps({k: record[k] for k in ("arm", "seed", "default", "merged")}))


if __name__ == "__main__":
    main()
