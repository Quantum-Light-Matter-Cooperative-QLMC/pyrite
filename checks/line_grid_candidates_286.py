"""Fixed-track bin-mean line-grid coarsening study for #286 (remote GPU only).

Uniform cell centers tile one unchanged physical band. Two successively finer
references expose unresolved grid error; candidates never change transport or
bandwidth. Intrinsic integrals use cell masses, not trapezoids of bin means.
Detector comparisons rebin all candidates onto one fixed 25 eV source axis
below 100 keV, sharing seeded response calculations. Shape L1 is reported on
fixed 100 eV channels. This diagnostic does not change a numerical policy or
provide an independent physics-validation verdict.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from pyrite._energy_grid_encoding import decode_energy_grid
from pyrite.energy_grid.bandwidth_check import _bin_masses
from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths

BUDGETS = {"yield": 1e-3, "centroid_eV": 1e-3, "timepix3_counts": 1e-2, "eaglexo_counts": 1e-2}


def uniform_cells(start, stop, count):
    """Cell centers whose midpoint edges tile exactly the requested positive band."""
    if not 0 < start < stop or count < 2:
        raise ValueError("require 0 < start < stop and at least two cells")
    return start + (np.arange(count, dtype=float) + 0.5) * ((stop - start) / count)


def case_band(case):
    """Decode the carrier axis before reading its first and last coordinates."""
    grid = decode_energy_grid(case["E_grid"])
    return float(grid[0]), float(grid[-1])


def observables(grid, density, detector_grid, detectors):
    """Integrate bin means and compare detectors on an identical reporting axis."""
    edges, widths = _energy_bin_edges_and_widths(grid)
    total = float(np.sum(density * widths))
    # Cell-constant first moment: exact for the represented bin-mean spectrum.
    centers = 0.5 * (edges[:-1] + edges[1:])
    result = {"yield": total, "centroid_eV": float(np.sum(density * widths * centers) / total)}
    det_edges, det_widths = _energy_bin_edges_and_widths(detector_grid)
    masses = _bin_masses(grid, density, det_edges, quadrature="bin-mean")
    det_density = masses / det_widths
    for name, detector in detectors.items():
        if name == "timepix3_counts":
            result[name] = float(np.sum(detector.native_score(detector_grid, det_density).events))
        else:
            scored = detector.score(detector_grid, det_density, fwhm_eV=None, scale=1.0)
            result[name] = float(np.sum(np.asarray(scored) * det_widths))
    return result


def relative(values, reference):
    return {key: (values[key] - reference[key]) / reference[key] for key in BUDGETS}


def evaluate(ladder, grid, detector_grid, detectors, repeats):
    """Time first/warm calls separately; observations and binning are outside timing."""
    import cupy

    walls = []
    for _ in range(repeats + 1):
        cupy.cuda.Device().synchronize()
        start = time.perf_counter()
        density = ladder.lines(grid)
        cupy.cuda.Device().synchronize()
        walls.append(time.perf_counter() - start)
    record = {
        "nodes": int(grid.size),
        "spacing_eV": float(grid[1] - grid[0]),
        "first_s": walls[0],
        "warm_s": walls[1:],
        "warm_median_s": float(np.median(walls[1:])),
        "observables": observables(grid, density, detector_grid, detectors),
        "host_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "device_pool_reserved_mib": cupy.get_default_memory_pool().total_bytes() / 1024**2,
    }
    return record, density


def run_case(args, energy, seed, save):
    from pyrite.detectors.spec import EagleXO, Timepix3
    from pyrite.energy_grid.convergence_case import CaseLadder, build_ladder_case

    case = build_ladder_case(
        "hopg", energy, 45.0, 180.0, thickness_ang=1e7, n_electrons=args.electrons, seed=seed
    )
    start, stop = case_band(case)
    # An explicit axis skips the automatic policy's high-energy point-budget refusal.
    carrier_policy = case.pop("line_grid_policy", None)
    base_n = args.high_energy_nodes if energy > 1000 else int(np.ceil((stop - start) / 3.0))
    base_grid = uniform_cells(start, stop, base_n)
    case.update(E_grid=base_grid, E_grid_line=base_grid, line_quadrature="bin-mean")
    ladder = CaseLadder(case)
    detector_stop = min(stop, args.detector_max_ev)
    detector_grid = uniform_cells(start, detector_stop, int(np.ceil((detector_stop - start) / 25)))
    detectors = {"timepix3_counts": Timepix3(n_mc=32, seed=7), "eaglexo_counts": EagleXO()}
    shape_edges = np.linspace(start, stop, int(np.ceil((stop - start) / 100)) + 1)
    row = {
        "energy_keV": energy,
        "seed": seed,
        "electrons": args.electrons,
        "band_eV": [start, stop],
        "carrier_policy": carrier_policy,
        "fingerprint": ladder.fingerprint,
        "transport_s": ladder.transport_wall_s,
        "baseline_nodes": base_n,
        "grids": {},
    }
    # Interleaving baseline calls brackets candidates to expose warm timing drift.
    arms = [("reference4", base_n * 4), ("reference2", base_n * 2), ("baseline", base_n)]
    arms += [(f"coarse{factor}", max(2, base_n // factor)) for factor in args.factors]
    arms.append(("baseline_after", base_n))
    reference_shape = None
    for label, count in arms:
        grid = uniform_cells(start, stop, count)
        record, density = evaluate(ladder, grid, detector_grid, detectors, args.repeats)
        shape = _bin_masses(grid, density, shape_edges, quadrature="bin-mean")
        if reference_shape is None:
            reference_shape = shape
        ref = row["grids"].get("reference4", record)["observables"]
        record["relative_to_reference4"] = relative(record["observables"], ref)
        record["within_budget"] = all(
            abs(v) <= BUDGETS[k] for k, v in record["relative_to_reference4"].items()
        )
        record["shape_100eV_l1_relative"] = float(
            np.sum(abs(shape - reference_shape)) / np.sum(abs(reference_shape))
        )
        row["grids"][label] = record
        save(row)
        print(json.dumps({"energy_keV": energy, "seed": seed, "arm": label, **record}), flush=True)
    row["reference_refinement_within_budget"] = row["grids"]["reference2"]["within_budget"]
    save(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-kev", type=float, nargs="+", default=[300, 29999])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--electrons", type=int, default=100)
    parser.add_argument("--high-energy-nodes", type=int, default=600_000)
    parser.add_argument("--factors", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--detector-max-ev", type=float, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.electrons < 1 or min(args.factors) < 2:
        parser.error("require positive electrons/repeats and coarsening factors >= 2")
    from pyrite._backend import xp

    if getattr(xp, "__name__", "") != "cupy":
        parser.error("this measurement requires the CUDA backend on the remote GPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {"args": vars(args) | {"output": str(args.output)}, "cases": []}
    for energy in args.energy_kev:
        for seed in args.seeds:
            state["cases"].append({})

            def save(row):
                state["cases"][-1] = row
                pending = args.output.with_suffix(".partial")
                pending.write_text(json.dumps(state, indent=1) + "\n")
                pending.replace(args.output)

            run_case(args, energy, seed, save)


if __name__ == "__main__":
    main()
