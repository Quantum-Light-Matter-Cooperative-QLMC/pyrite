"""Uniform versus edge-geometric CXR line-table mesh on frozen transports (#1201).

Replays a trusted saved transport (the #642 Si 300 keV slab or MoS2-on-Si
layered scene) through :func:`select_case_reflections` on a declared band
axis or the case's automatic axis, once per arm:

- ``uniform``: the former line-table mesh, a 1 eV mesh from the padded axis
  start (``_line_mesh`` patched to ``np.arange(lo, hi + 1, 1)``);
- ``edge-geometric``: the production mesh, geometric at relative spacing
  ``_LINE_TAB_RATIO`` with 1 eV windows around edges.

Both arms add the same native Chantler and EPDL nodes. Arms run round-robin
after one untimed warm-up each. Each arm records ``n_tab`` (the largest line
table length of the selection), wall time, the device pool reservation after
the selection (its high-water mark, the pool being emptied first), the
selected-set digest, and the spectrum. Spectra are compared against the
``uniform`` arm: peak-normalized maximum difference, axis-weighted yield and
the worst relative difference over nodes above 1e-3 of the peak
(``max_rel_diff_strong``).

Remote only (frozen transports, multi-minute GPU selections). Measurement
record; the accuracy tolerance is set in the issue, not here.
"""

import argparse
import hashlib
import json
import pickle
import resource
import runpy
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
GATE = HERE / "defaults_spectral_gate_642.py"
PROFILE = HERE / "cxr_selection_profile_1200.py"
CRYSTAL = {"mos2-on-si": "mos2"}
ARMS = ("uniform", "edge-geometric")


@contextmanager
def mesh_arm(arm):
    from pyrite.montecarlo.spectrum.cxr import _tabulation_mesh as mesh

    original = mesh._line_mesh
    if arm == "uniform":
        mesh._line_mesh = lambda lo, hi, _edges: np.arange(lo, hi + 1.0, 1.0)
    try:
        yield
    finally:
        mesh._line_mesh = original


def compare(density, reference, weights):
    peak = float(np.abs(reference).max()) or 1.0
    strong = np.abs(reference) > 1e-3 * peak
    rel = np.abs(density - reference)[strong] / np.abs(reference)[strong]
    return dict(
        max_abs_diff_rel_peak=float(np.abs(density - reference).max()) / peak,
        yield_rel_diff=float((density - reference) @ weights) / float(reference @ weights),
        max_rel_diff_strong=float(rel.max()) if rel.size else 0.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", choices=["silicon", "mos2-on-si"])
    parser.add_argument("replay", type=Path, help="trusted same-source transport pickle")
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", choices=["band", "automatic"], default="band")
    parser.add_argument("--band", type=float, nargs=2, default=(500.0, 10000.0))
    parser.add_argument("--spacing", type=float, default=0.09375, help="band node spacing (eV)")
    parser.add_argument("--quadrature", choices=["node", "bin-mean"], default="node")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=2, help="round-robin passes per arm")
    args = parser.parse_args()

    from pyrite._backend import BACKEND
    from pyrite.energy_grid.convergence_case import CaseLadder
    from pyrite.montecarlo.runner.reflection_cutoff import (
        axis_yield_weights,
        select_case_reflections,
        selection_call_sizes,
    )
    from pyrite.montecarlo.spectrum.cxr._setup import SETUP_STATS

    gate = runpy.run_path(str(GATE))
    profile = runpy.run_path(str(PROFILE))
    sync, device, pool_bytes, _ = gate["runtime"]()
    release = BACKEND.release_memory if BACKEND.name != "cpu" else (lambda: None)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.replay.read_bytes()
    saved = pickle.loads(raw)  # noqa: S301 - explicit trusted study artifact
    case, transport = saved["case"], saved["transport"]
    if case["crystal"] != CRYSTAL.get(args.material, args.material):
        raise ValueError("replay material mismatch")
    report = dict(
        schema="pyrite.cxr-tab-grid.v1",
        material=args.material,
        source_revision=Path("SOURCE_SHA").read_text().strip()
        if Path("SOURCE_SHA").exists()
        else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        replay_sha256=hashlib.sha256(raw).hexdigest(),
        seed=case["seed"],
        histories=case["Ne"],
        backend=BACKEND.name,
        device=device,
        axis=args.axis,
        tolerance=args.tolerance,
        repeats=args.repeats,
        arms=[],
    )
    del raw

    def save():
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.axis == "automatic":
        case, grid, _resolve_s = profile["automatic_axis"](case, args.material, transport, gate)
        quadrature = case.get("line_quadrature", "node")
    else:
        quadrature = args.quadrature
        grid = gate["axis"](*args.band, args.spacing, quadrature)
        case = dict(case, line_quadrature=quadrature)
    report.update(quadrature=quadrature, nodes=int(grid.size), support_eV=[grid[0], grid[-1]])
    save()
    scorer = CaseLadder(case, transport=transport)
    weights = axis_yield_weights(grid, quadrature)

    def select():
        sizes = selection_call_sizes(case, grid)

        def line_density(pilot, axis_grid):
            scorer.case = pilot
            return scorer.lines(axis_grid)

        return select_case_reflections(
            case,
            line_density,
            args.tolerance,
            grid,
            weights,
            max_reflections=lambda layer, _radiator: sizes[layer],
        ), sizes

    spectra = {}
    for arm in ARMS:  # untimed warm-ups: kernels and module caches
        with mesh_arm(arm):
            sync()
            select()
            sync()
    for repeat in range(args.repeats):
        for arm in ARMS:
            release()
            SETUP_STATS.clear()
            with mesh_arm(arm):
                sync()
                begin = time.perf_counter()
                (pruned, outcomes, density), sizes = select()
                sync()
                elapsed = time.perf_counter() - begin
            density = np.asarray(density, dtype=float)
            spectra.setdefault(arm, density)
            if repeat == 0:
                np.savez_compressed(
                    args.output / f"spectrum-{arm}.npz", grid_eV=grid, density=density
                )
            record = dict(
                repeat=repeat,
                arm=arm,
                wall_s=elapsed,
                n_tab=int(SETUP_STATS.get("line_tab_points", 0)),
                line_table_mib=float(SETUP_STATS.get("line_table_mib", 0.0)),
                reflections_per_call={str(k): v for k, v in sizes.items()},
                device_pool_reserved_after_emptied_bytes=pool_bytes(),
                hkl_digest=profile["hkl_digest"](pruned),
                reflections=[len(o.hkl_list) for o in outcomes],
            )
            if "uniform" in spectra:
                record.update(compare(density, spectra["uniform"], weights))
            report["arms"].append(record)
            save()
            print(json.dumps({k: record[k] for k in ("arm", "repeat", "wall_s", "n_tab")}))


if __name__ == "__main__":
    main()
