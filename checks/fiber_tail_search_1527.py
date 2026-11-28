"""Frozen production-line benchmark for #1527 (all, coarse/fine, pilot).

Run from the repository root in the desired CPU/CUDA/ROCm environment.
No spectrum cache or adaptive population is used. This measures radiation
and tail reduction together, separately from the single frozen transport.
"""

import argparse
import json
from time import perf_counter

import numpy as np

from pyrite import Scene
from pyrite._cxr_grid_policy import BANDWIDTH_PROXY_SAFETY
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.campaign.lowering import lower_simulation
from pyrite.campaign.resolution import resolve_simulation
from pyrite.config import CxrReflections, FiberAverage, FixedSampling, Numerics, Physics
from pyrite.detectors import Detector
from pyrite.geometry import Footprint, Slab
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner._pilot_tail import pilot_block, resolve_pilot_tail
from pyrite.montecarlo.runner.cxr_grid import _direction_tail
from pyrite.montecarlo.spectrum._lattice_search import refine_tail
from pyrite.montecarlo.spectrum._lattice_tail import LatticeTail, stop_lattice
from pyrite.montecarlo.spectrum.cxr_seeds import _lattice_stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=512)
    parser.add_argument("--energy", type=float, default=300)
    parser.add_argument("--tilt", type=float, default=50)
    parser.add_argument("--cutoff", type=float, default=8)
    parser.add_argument("--rolls", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    scene = Scene(
        Beam(args.energy, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 5000.0, tilt_deg=args.tilt, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=args.histories, brem_histories=args.histories),
        cxr_reflections=CxrReflections(
            fiber=FiberAverage(cutoff_invang=args.cutoff, azimuth_count=args.rolls)
        ),
    )
    case = lower_simulation(
        resolve_simulation(
            scene,
            physics=Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled"),
            numerics=numerics,
        )
    )
    began = perf_counter()
    tp = runner._transport_case(case, transport_core="per-electron")
    segments = runner._segments_on_device(tp["segs"])
    print(
        json.dumps(
            {
                "transport_wall_s": perf_counter() - began,
                "histories": args.histories,
                "segments": int(segments["L_ang"].size),
            }
        ),
        flush=True,
    )
    band = case["line_grid_policy"]["bandwidth"]
    start, ceiling = float(band["start_eV"]), float(band["stop_eV"])
    limit = float(band["truncation_limit"]) / BANDWIDTH_PROXY_SAFETY
    full = stop_lattice(ceiling)
    results = {}
    for repeat in range(args.repeats):
        # Reverse order on alternate repetitions to expose warm-up effects.
        modes = ("one", "all", "coarse-fine", "pilot")
        if repeat % 2:
            modes = tuple(reversed(modes))
        for mode in modes:
            passes = []

            def collect(stops, part=segments, passes=passes):
                t = perf_counter()
                summary = _direction_tail(
                    case,
                    part,
                    tp["n_hat"],
                    int(tp["Ne_lines"]),
                    case.get("abs_layers"),
                    tp.get("groove"),
                    start,
                    ceiling,
                    stops=stops,
                )
                # Construct the host view to synchronize all queued reductions.
                tail = LatticeTail(stops, summary)
                passes.append(
                    {
                        "candidates": len(stops),
                        "wall_s": perf_counter() - t,
                        "raw_lines": tail.raw_lines,
                    }
                )
                return summary

            began = perf_counter()
            if mode == "one":
                collect(np.array([ceiling]))
                print(json.dumps({"repeat": repeat, "mode": mode, "passes": passes}), flush=True)
                continue
            certified = None
            pilot_rows = None
            if mode == "all":
                tail = LatticeTail(full, collect(full))
            elif mode == "coarse-fine":
                tail = refine_tail(ceiling, limit, collect)
            else:
                stops, summary, pilot_rows = pilot_block(
                    segments, ceiling, limit, lambda part, candidates: collect(candidates, part)
                )
                tail, certified = resolve_pilot_tail(ceiling, limit, stops, summary, collect)
            stop, record = _lattice_stop(tail, ceiling, limit, 100.0)
            result = {
                "repeat": repeat,
                "mode": mode,
                "stop_eV": stop,
                "wall_s": perf_counter() - began,
                "passes": passes,
                "fraction": record["proxy_truncated_fraction"],
            }
            if mode == "pilot":
                result.update(pilot_rows=pilot_rows, pilot_certified=certified)
            print(json.dumps(result), flush=True)
            results[mode] = result
        for mode in ("coarse-fine", "pilot"):
            assert results["all"]["stop_eV"] == results[mode]["stop_eV"]
            np.testing.assert_allclose(
                results["all"]["fraction"], results[mode]["fraction"], rtol=3e-7, atol=1e-15
            )


if __name__ == "__main__":
    main()
