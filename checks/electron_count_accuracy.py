"""Accuracy vs electron count for ``Ne`` (lines) and ``Ne_brem`` (#278 plan item 5).

For one production case (catalog default grids: the automatic line grid and
the 25 eV ``E_grid_brem``), run independently seeded replicas at each count
in ``--counts`` plus one higher-count reference, each a full production
transport and spectrum reduction (:class:`CaseLadder`). ``Ne == Ne_brem == N``
per run; line observables come from the line spectrum only and continuum
observables from the bremsstrahlung continuum only, so each depends on its own
population.

Observables (per incident electron per sr):

* ``Ne_brem``: intrinsic continuum yield and centroid, Timepix3/EagleXO
  detected continuum counts.
* ``Ne``: line yield, Timepix3/EagleXO detected line counts.

The JSON is checkpointed after every run and resumes; ``--summarize`` prints
per-count relative SE across replicas, bias against the reference, and the
fitted exponent of RSE ~ N^p.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pyrite._cxr_grid_policy import CxrGridToleranceError
from pyrite.detectors.spec import EagleXO, Timepix3
from pyrite.energy_grid import convergence_case as cc
from pyrite.energy_grid.convergence import spectrum_observables

LINE = ("yield", "timepix3_counts", "eaglexo_counts")
#: Energies whose automatic line grid was refused once; later runs go explicit.
REFUSED: set[float] = set()
CONTINUUM = (
    "continuum_yield",
    "continuum_centroid_eV",
    "timepix3_continuum_counts",
    "eaglexo_continuum_counts",
)


def explicit_line_grid(case, points):
    """Replace the automatic line-grid policy with ``points`` uniform nodes over its band.

    The policy can refuse a case outright (HOPG at 29,999 keV needs ~2.9 M
    nodes against the 600 k default budget); transport and the continuum do not
    depend on the line grid.
    """
    E = np.asarray(case["E_grid"], dtype=float)
    grid = np.linspace(E[0], E[-1], int(points))
    case = dict(case)
    case.pop("line_grid_policy", None)
    case["E_grid"] = case["E_grid_line"] = grid
    return case


def windowed_observables(E, lines, brem, detectors, args):
    """Yield/centroid over the whole band; detector counts at or below ``--detector-max-ev``.

    The read-time Timepix3 response is a dense (output x input) Monte Carlo
    matrix for keV lines; a 30 MeV band needs TiB and gives NaN above ~1 MeV.
    """
    out = spectrum_observables(E, lines, brem, detectors={})
    m = E <= args.detector_max_ev
    det = spectrum_observables(E[m], lines[m], brem[m], detectors=detectors)
    return out | {k: v for k, v in det.items() if k.endswith("counts")}


def one_run(args, energy_keV, n, seed):
    case = cc.build_ladder_case(
        args.material,
        energy_keV,
        args.tilt_deg,
        args.tilt_azim_deg,
        thickness_ang=args.thickness_ang,
        n_electrons=n,
        seed=seed,
    )
    t0 = time.perf_counter()
    line_grid = "automatic"
    ladder = None
    if energy_keV not in REFUSED:
        try:
            ladder = cc.CaseLadder(case)
        except CxrGridToleranceError:
            if not args.line_points_fallback:
                raise
            REFUSED.add(energy_keV)
    if ladder is None:
        line_grid = f"explicit-uniform-{args.line_points_fallback}"
        ladder = cc.CaseLadder(explicit_line_grid(case, args.line_points_fallback))
    tp = ladder.transport
    detectors = {"timepix3_counts": Timepix3(n_mc=32, seed=7), "eaglexo_counts": EagleXO()}
    t1 = time.perf_counter()
    E = np.asarray(tp["E_grid"], dtype=float)
    lines = ladder.lines(E)
    t2 = time.perf_counter()
    Eb = np.asarray(tp["E_brem"], dtype=float)
    brem = ladder.continuum(Eb)
    t3 = time.perf_counter()
    line_obs = windowed_observables(E, lines, np.zeros_like(lines), detectors, args)
    brem_obs = windowed_observables(Eb, np.zeros_like(brem), brem, detectors, args)
    return dict(
        energy_keV=energy_keV,
        n=n,
        seed=seed,
        line_grid=line_grid,
        line_nodes=int(E.size),
        brem_nodes=int(Eb.size),
        segments=int(ladder.fingerprint["n_segments"]),
        seconds=dict(
            transport=ladder.transport_wall_s,
            lines=t2 - t1,
            continuum=t3 - t2,
            observe=time.perf_counter() - t3,
            total=time.perf_counter() - t0,
        ),
        observables={k: line_obs[k] for k in LINE} | {k: brem_obs[k] for k in CONTINUUM},
    )


def plan(args):
    runs = []
    for energy in args.energy_kev:
        runs.append((energy, args.reference_count, args.reference_seed))
        for i, n in enumerate(args.counts):
            # Distinct seeds per (count, replica): no electron is shared between runs.
            runs += [(energy, n, args.seed0 + 1000 * i + r) for r in range(args.replicas)]
    return runs


def summarize(state):
    out = {}
    for energy in sorted({r["energy_keV"] for r in state["runs"]}):
        runs = [r for r in state["runs"] if r["energy_keV"] == energy]
        ref = [r for r in runs if r["n"] == state["reference_count"]]
        if not ref:
            continue
        ref = ref[0]["observables"]
        table = {}
        for name in LINE + CONTINUUM:
            rows = []
            for n in sorted({r["n"] for r in runs if r["n"] != state["reference_count"]}):
                vals = np.array([r["observables"][name] for r in runs if r["n"] == n])
                if vals.size < 2:
                    continue
                mean = float(vals.mean())
                rse = float(vals.std(ddof=1) / abs(mean)) if mean else float("nan")
                rows.append(
                    dict(
                        n=n,
                        replicas=int(vals.size),
                        mean=mean,
                        relative_se=rse,
                        bias_vs_reference=(mean - ref[name]) / ref[name] if ref[name] else None,
                    )
                )
            good = [(r["n"], r["relative_se"]) for r in rows if r["relative_se"] > 0]
            slope = (
                float(np.polyfit(np.log([g[0] for g in good]), np.log([g[1] for g in good]), 1)[0])
                if len(good) >= 2
                else None
            )
            table[name] = dict(
                population="Ne" if name in LINE else "Ne_brem",
                reference=ref[name],
                rows=rows,
                fitted_exponent=slope,
            )
        out[f"{energy:g}keV"] = table
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="hopg")
    parser.add_argument("--energy-kev", type=float, action="append", required=True)
    parser.add_argument("--tilt-deg", type=float, default=45.0)
    parser.add_argument("--tilt-azim-deg", type=float, default=180.0)
    parser.add_argument("--thickness-ang", type=float, default=1.0e7)
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 400, 1600])
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--reference-count", type=int, default=6400)
    parser.add_argument("--reference-seed", type=int, default=999)
    parser.add_argument(
        "--line-points-fallback",
        type=int,
        default=600_000,
        help="explicit uniform line-grid size when the automatic policy refuses a case "
        "(0: fail); recorded per run",
    )
    parser.add_argument("--detector-max-ev", type=float, default=100e3)
    parser.add_argument("--max-hours", type=float, default=math.inf)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else dict(reference_count=args.reference_count, runs=[])
    )
    if not args.summarize:
        start = time.perf_counter()
        done = {(r["energy_keV"], r["n"], r["seed"]) for r in state["runs"]}
        for key in plan(args):
            if key in done:
                continue
            if time.perf_counter() - start > 3600 * args.max_hours:
                print("out of time", flush=True)
                break
            run = one_run(args, *key)
            state["runs"].append(run)
            args.output.write_text(json.dumps(state, indent=1) + "\n")
            print(
                json.dumps({k: run[k] for k in ("energy_keV", "n", "seed", "seconds")}), flush=True
            )
    state["summary"] = summarize(state)
    args.output.write_text(json.dumps(state, indent=1) + "\n")
    print(json.dumps(state["summary"], indent=1))


if __name__ == "__main__":
    main()
