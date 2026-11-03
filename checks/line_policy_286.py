"""Same-track production line-policy pilot for #286; CUDA/remote only.

Policy arms use the production resolver, including its refusal and truncation
audit. Uniform common-band controls isolate spacing from bandwidth. Explicit
full-band bin-mean references are refined twice; no policy is adopted here.
First calls are process-initial (shared compiler state), not isolated cold JIT.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import resource
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from checks.line_grid_candidates_286 import BUDGETS, observables, relative, uniform_cells
from pyrite.energy_grid.bandwidth_check import _bin_masses
from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths


def build_case(args, material, energy, seed, policy):
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases

    sweep = material_sweep(
        material,
        thickness_ang=args.thickness_ang,
        energy_keV=energy,
        tilt_deg=45.0,
        tilt_azim_deg=180.0,
    )
    sweep = dataclasses.replace(sweep, cxr_grid_policy=policy)
    case = dict(build_cases(sweep, n_electrons=args.electrons, n_electrons_brem=args.electrons)[0])
    case.update(seed=seed, _profile_line_grid_stages=True)
    return case


def policy_specs(spacings):
    yield "default_node", {"bandwidth": "kinematic-ceiling"}
    yield "full_bin", {"bandwidth": "kinematic-ceiling", "quadrature": "bin-mean"}
    for quadrature in ("node", "bin-mean"):
        yield (
            f"population_{quadrature}",
            {
                "bandwidth": "resonance-population",
                "quadrature": quadrature,
            },
        )
    for spacing in spacings:
        yield (
            f"adaptive_{spacing:g}",
            {
                "bandwidth": "resonance-population",
                "resolution": "resonance-local",
                "quadrature": "bin-mean",
                "max_spacing_eV": spacing,
            },
        )


def resolve(case, ladder, cache_root):
    """Real disk cache in a fresh private directory: miss then hit, same axis."""
    from pyrite.montecarlo.runner.cxr_grid import resolve_cxr_grid

    tp = ladder.transport
    records, grids, walls = [], [], []
    with patch(
        "pyrite._cxr_grid_policy._cache_path", side_effect=lambda key: cache_root / f"{key}.json"
    ):
        for _ in range(2):
            synchronize()
            started = time.perf_counter()
            grid, record = resolve_cxr_grid(
                case,
                ladder.segments,
                tp["n_hat"],
                tp["Ne_lines"],
                tp["E_grid"],
                case.get("abs_layers"),
                tp.get("groove"),
            )
            synchronize()
            walls.append(time.perf_counter() - started)
            records.append(record)
            grids.append(np.asarray(grid))
    if records[0]["cache"] != "miss" or records[1]["cache"] != "hit":
        raise RuntimeError("pilot must measure a real cache miss followed by a hit")
    np.testing.assert_array_equal(*grids)
    return grids[0], {"miss_s": walls[0], "hit_s": walls[1], "record": records[0]}


def synchronize():
    import cupy

    cupy.cuda.Device().synchronize()


def mass_bins(grid, density, edges, quadrature):
    """Exact integrals of cell-constant or piecewise-linear represented density."""
    if quadrature == "bin-mean":
        return _bin_masses(grid, density, edges, quadrature=quadrature)
    cumulative = np.r_[0.0, np.cumsum(np.diff(grid) * (density[:-1] + density[1:]) / 2)]
    points = np.clip(edges, grid[0], grid[-1])
    index = np.minimum(np.searchsorted(grid, points, side="right") - 1, grid.size - 2)
    offset = points - grid[index]
    slope = np.diff(density)[index] / np.diff(grid)[index]
    integral = cumulative[index] + density[index] * offset + slope * offset**2 / 2
    return np.diff(integral)


def score(ladder, case, grid, repeats):
    """Include the production bandwidth audit in every measured scoring pass."""
    from checks.line_cost_286 import Recorder
    from checks.line_cost_286 import summarize as reducer_summary
    from pyrite._backend import _to_cpu
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner.cxr_grid import check_cxr_truncation, cxr_truncation_audit

    tp, walls, profiles = ladder.transport, [], []
    for _ in range(repeats + 1):
        synchronize()
        started = time.perf_counter()
        audit = cxr_truncation_audit(case, grid, n_electrons=tp["Ne_lines"])
        audit_prepare_s = time.perf_counter() - started
        recorder = (
            Recorder(
                compare_routes=case.get("_compare_line_reducer_routes", False),
                capture_root=case.get("_reducer_capture_root"),
            )
            if case.get("_profile_line_reducers")
            else None
        )
        with recorder if recorder is not None else nullcontext():
            score_started = time.perf_counter()
            density = runner._cxr_for_segments(
                ladder._segments_device,
                grid,
                case,
                tp["n_hat"],
                case.get("abs_layers"),
                tp.get("groove"),
                coherent=False,
                Ne=tp["Ne_lines"],
                table_cache={},
                truncation_audit=audit,
            )
            if recorder is not None:
                synchronize()
            scoring_s = time.perf_counter() - score_started
            audit_started = time.perf_counter()
            audit_record = None if audit is None else check_cxr_truncation(case, audit)
            if recorder is not None:
                synchronize()
            audit_check_s = time.perf_counter() - audit_started
            density = np.asarray(_to_cpu(density), dtype=float)
            synchronize()
            walls.append(time.perf_counter() - started)
        if recorder is not None:
            profiles.append(
                {
                    **reducer_summary(recorder.flushes, scoring_s),
                    "audit_prepare_s": audit_prepare_s,
                    "audit_check_s": audit_check_s,
                    "flushes": recorder.flushes,
                }
            )
    timing = {"first_s": walls[0], "warm_s": walls[1:], "audit": audit_record}
    if profiles:
        timing["reducer_profiles"] = profiles
    return density, timing


def summarize(grid, density, quadrature, detector_grid, detectors, shape_edges):
    """Quadrature-aware integrals; shared detector bins plus native-axis events."""
    edges, widths = _energy_bin_edges_and_widths(detector_grid)
    masses = mass_bins(grid, density, edges, quadrature)
    # Reuse the controlled detector accounting on exactly its own source cells.
    result = observables(detector_grid, masses / widths, detector_grid, detectors)
    if quadrature == "bin-mean":
        cells, cell_widths = _energy_bin_edges_and_widths(grid)
        total = float(np.sum(density * cell_widths))
        moment = float(np.sum(density * cell_widths * (cells[:-1] + cells[1:]) / 2))
    else:
        total = float(np.trapezoid(density, grid))
        # Exact first moment of piecewise-linear density.
        moment = float(
            np.sum(
                np.diff(grid)
                / 6
                * (
                    (2 * grid[:-1] + grid[1:]) * density[:-1]
                    + (grid[:-1] + 2 * grid[1:]) * density[1:]
                )
            )
        )
    result.update(yield_=total, centroid_eV=moment / total if total else None)
    result["yield"] = result.pop("yield_")
    cut = grid <= edges[-1]
    native = detectors["timepix3_counts"].native_score(grid[cut], density[cut])
    native_edges, native_widths = _energy_bin_edges_and_widths(grid[cut])
    shape = mass_bins(grid, density, shape_edges, quadrature)
    return (
        result,
        shape,
        {
            "timepix3_counts": float(np.sum(native.events)),
            "quadrature": quadrature,
            "source_nodes": int(cut.sum()),
            "source_cell_band_eV": native_edges[[0, -1]].tolist(),
            "source_bin_mass": float(np.sum(density[cut] * native_widths)),
        },
    )


def run_case(args, material, energy, seed, save):
    import cupy

    from pyrite.detectors.spec import EagleXO, Timepix3
    from pyrite.energy_grid.convergence_case import CaseLadder

    base = build_case(args, material, energy, seed, {"bandwidth": "kinematic-ceiling"})
    payload = base.pop("line_grid_policy")
    start, stop = (payload["bandwidth"][k] for k in ("start_eV", "stop_eV"))
    carrier = uniform_cells(start, stop, 1024)
    base.update(E_grid=carrier, E_grid_line=carrier, line_quadrature="bin-mean")
    ladder = CaseLadder(base)
    detectors = {"timepix3_counts": Timepix3(n_mc=32, seed=7), "eaglexo_counts": EagleXO()}
    detector_grid = uniform_cells(
        start,
        min(stop, args.detector_max_ev),
        int(np.ceil((min(stop, args.detector_max_ev) - start) / 25)),
    )
    shape_edges = np.linspace(start, stop, int(np.ceil((stop - start) / 100)) + 1)
    row = {
        "material": material,
        "energy_keV": energy,
        "seed": seed,
        "fingerprint": ladder.fingerprint,
        "transport_s": ladder.transport_wall_s,
        "case": {k: v for k, v in base.items() if not k.startswith("E_grid")},
        "full_band_eV": [start, stop],
        "arms": {},
    }
    reference_shape = None

    def measure(label, case, grid, setup=None):
        nonlocal reference_shape
        compare = getattr(args, "compare_reducer_routes", False)
        if getattr(args, "profile_reducers", False) or compare:
            case = dict(
                case,
                _profile_line_reducers=True,
                _compare_line_reducer_routes=compare,
                _reducer_capture_root=str(args.output.with_suffix(".flushes")) if compare else None,
            )
        density, timing = score(ladder, case, grid, args.repeats)
        quadrature = case.get("line_quadrature", "node")
        values, shape, native = summarize(
            grid, density, quadrature, detector_grid, detectors, shape_edges
        )
        if reference_shape is None:
            reference_shape = shape
        reference = row["arms"].get("reference4", {"observables": values})["observables"]
        errors = relative(values, reference)
        spacing = np.diff(grid)
        artifact = args.output.parent / (
            f"{args.output.stem}-{material}-{energy:g}-{seed}-{label}.npz"
        )
        np.savez_compressed(artifact, grid_eV=grid, density=density)
        row["arms"][label] = {
            "nodes": int(grid.size),
            "axis_eV": [float(grid[0]), float(grid[-1])],
            "spacing_quantiles_eV": np.quantile(spacing, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "quadrature": quadrature,
            "axis_sha256": hashlib.sha256(np.asarray(grid, dtype="<f8").tobytes()).hexdigest(),
            "spectrum_artifact": str(artifact),
            "setup": setup,
            **timing,
            "observables": values,
            "relative_to_reference4": errors,
            "within_scalar_budgets": all(abs(v) <= BUDGETS[k] for k, v in errors.items()),
            "shape_100eV_l1": float(np.sum(abs(shape - reference_shape)) / np.sum(reference_shape)),
            "native": native,
            "host_process_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "device_pool_reserved_mib": cupy.get_default_memory_pool().total_bytes() / 1024**2,
        }
        save(row)
        print(json.dumps({"arm": label, **row["arms"][label]}, default=_json), flush=True)

    count = int(np.ceil((stop - start) / args.reference_spacing_ev))
    for factor in (4, 2):
        measure(f"reference{factor}", base, uniform_cells(start, stop, count * factor))
    for label, policy in policy_specs(args.spacings):
        case = build_case(args, material, energy, seed, policy)
        from pyrite._cxr_grid_policy import CxrGridToleranceError

        attempt_started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="policy286-") as directory:
                grid, setup = resolve(case, ladder, Path(directory))
            measure(label, case, grid, setup)
        except (CxrGridToleranceError, cupy.cuda.memory.OutOfMemoryError) as exc:
            row["arms"][label] = {
                "refused": str(exc),
                "exception_type": type(exc).__name__,
                "attempt_s": time.perf_counter() - attempt_started,
            }
            save(row)
        if "refused" in row["arms"][label]:
            # The exception traceback must be released before freeing temporaries.
            gc.collect()
            cupy.get_default_memory_pool().free_all_blocks()
            continue
        if label.startswith("adaptive_"):
            # Exactly the adaptive bin edges, holding the bandwidth fixed.
            edges = _energy_bin_edges_and_widths(grid)[0]
            for spacing in sorted({3.0, float(policy["max_spacing_eV"])}):
                control = dict(base)
                measure(
                    f"common_{label}_uniform_{spacing:g}",
                    control,
                    uniform_cells(
                        edges[0], edges[-1], int(np.ceil((edges[-1] - edges[0]) / spacing))
                    ),
                )
    row["complete"] = True
    save(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", nargs="+", default=["hopg"])
    parser.add_argument("--energy-kev", nargs="+", type=float, default=[300, 29999])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--electrons", type=int, default=20)
    parser.add_argument("--thickness-ang", type=float, default=1e7)
    parser.add_argument("--spacings", nargs="+", type=float, default=[3, 12, 25, 50])
    parser.add_argument("--reference-spacing-ev", type=float, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--detector-max-ev", type=float, default=100_000)
    parser.add_argument(
        "--profile-reducers",
        action="store_true",
        help="record synchronized per-flush reducer routes; adds profiling overhead",
    )
    parser.add_argument(
        "--compare-reducer-routes",
        action="store_true",
        help="replay pairs/tree and save flush inputs/model predictions; adds substantial overhead",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.spacings + [args.reference_spacing_ev, args.thickness_ang]) <= 0
        or min(args.electrons, args.repeats) < 1
    ):
        parser.error("require positive spacing, thickness, electrons and repeats")
    from pyrite._backend import xp

    if getattr(xp, "__name__", "") != "cupy":
        parser.error("remote CUDA required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {"args": vars(args) | {"output": str(args.output)}, "cases": []}
    for material in args.materials:
        for energy in args.energy_kev:
            for seed in args.seeds:
                state["cases"].append({})

                def save(row):
                    state["cases"][-1] = row
                    pending = args.output.with_suffix(".partial")
                    pending.write_text(json.dumps(state, indent=1, default=_json) + "\n")
                    pending.replace(args.output)

                run_case(args, material, energy, seed, save)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    main()
