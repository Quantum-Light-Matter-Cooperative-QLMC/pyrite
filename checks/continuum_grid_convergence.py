"""Continuum photon-grid candidates on fixed tracks (#278 plan item 2).

One production transport per case (:class:`CaseLadder`), then the production
bremsstrahlung reduction (``CaseLadder.continuum``: attenuation through the
case's stack, BremsLib/EEDL route of the case) on every candidate grid, so each
difference is grid error alone. Observables are
:func:`pyrite.energy_grid.convergence.spectrum_observables`: intrinsic
continuum yield and centroid (budget 1e-3) and Timepix3/EagleXO detected
continuum counts (budget 1e-2), each relative to the reference.

Candidates:

``reference``
    Uniform ``--ref-step-ev`` from the medium's continuum floor to the case's
    kinematic endpoint; the production ``E_grid_brem`` step is 25 eV.
``geometric-<n>``
    :func:`refined_continuum_grid` (log-spaced, anchored at absorption edges and
    the endpoint), observables on its own nodes.
``geometric-<n>-interp``
    The same density linearly interpolated onto the reference grid: what a
    production policy that scores on few nodes but keeps the uniform output
    axis would report.
``gl-<order>x<ratio>``
    Gauss--Legendre collocation: panels split at the floor, every
    edge-bracket anchor and the endpoint, then subdivided geometrically so no
    panel spans more than ``ratio`` in energy; ``order`` GL nodes per panel in
    ``ln E``. The density is reconstructed on the reference grid by the
    panel's interpolating polynomial; inside an edge bracket each output node
    takes the nearer anchor's value (the bin boundary falls on the jump).

Run remotely; the JSON records per-grid wall time, node count and observables.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from pyrite.detectors.spec import EagleXO, Timepix3
from pyrite.energy_grid import convergence_case as cc
from pyrite.energy_grid.convergence import spectrum_observables
from pyrite.energy_grid.floor import photon_continuum_floor_eV
from pyrite.energy_grid.refine import (
    EDGE_KIND,
    case_endpoint_eV,
    continuum_refinement_marks,
    refined_continuum_grid,
)

BUDGET = {
    "continuum_yield": 1e-3,
    "continuum_centroid_eV": 1e-3,
    "timepix3_continuum_counts": 1e-2,
    "eaglexo_continuum_counts": 1e-2,
}


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


def gl_panels(material, floor_eV, endpoint_eV, ratio):
    """Smooth panels ``[(a, b)]`` and edge brackets ``[(below, above)]``.

    Breakpoints are the floor, both anchors of every absorption-edge mark the
    refined geometric grid uses, and the endpoint. Each smooth interval is cut
    into equal-``ln E`` panels no wider than ``ratio``.
    """
    marks, _summary = continuum_refinement_marks(material, floor_eV, endpoint_eV * (1 + 1e-9))
    brackets = sorted((min(m.anchors_eV), max(m.anchors_eV)) for m in marks if m.kind == EDGE_KIND)
    cuts = [floor_eV]
    for below, above in brackets:
        cuts += [below, above]
    cuts.append(endpoint_eV)
    smooth = []
    for i in range(0, len(cuts), 2):
        a, b = cuts[i], cuts[i + 1]
        pieces = max(1, math.ceil(math.log(b / a) / math.log(ratio)))
        bounds = a * (b / a) ** (np.arange(pieces + 1) / pieces)
        smooth += list(zip(bounds[:-1], bounds[1:], strict=True))
    return smooth, brackets


def gl_nodes(panels, brackets, order):
    """Strictly increasing nodes (GL in ``ln E`` per panel, plus bracket anchors).

    Also returns, per panel, the indices of its GL nodes in the node array.
    """
    x, _w = np.polynomial.legendre.leggauss(order)
    parts, owner = [], []
    for k, (a, b) in enumerate(panels):
        la, lb = math.log(a), math.log(b)
        parts.append(np.exp(0.5 * (lb - la) * x + 0.5 * (la + lb)))
        owner.append(np.full(order, k))
    anchors = np.asarray(brackets, dtype=float).reshape(-1)
    parts.append(anchors)
    owner.append(np.full(anchors.size, -1))
    nodes, owner = np.concatenate(parts), np.concatenate(owner)
    order_idx = np.argsort(nodes)
    nodes, owner = nodes[order_idx], owner[order_idx]
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("GL nodes are not strictly increasing")
    return nodes, [np.flatnonzero(owner == k) for k in range(len(panels))]


def gl_reconstruct(panels, brackets, order, nodes, members, density, grid):
    """Density on ``grid`` from GL-node values: per-panel polynomial in ``ln E``."""
    x, _w = np.polynomial.legendre.leggauss(order)
    out = np.zeros_like(grid)
    for (a, b), idx in zip(panels, members, strict=True):
        la, lb = math.log(a), math.log(b)
        coef = np.polynomial.legendre.legfit(x, density[idx], order - 1)
        sel = (grid >= a) & (grid <= b)
        t = (2.0 * np.log(grid[sel]) - la - lb) / (lb - la)
        out[sel] = np.polynomial.legendre.legval(t, coef)
    for below, above in brackets:
        sel = (grid > below) & (grid < above)
        lo = density[np.searchsorted(nodes, below)]
        hi = density[np.searchsorted(nodes, above)]
        out[sel] = np.where(grid[sel] < 0.5 * (below + above), lo, hi)
    return out


#: Detected counts use only the density at or below this energy. The read-time
#: Timepix3 response is a dense (output x input) Monte Carlo matrix for keV
#: lines: at a 30 MeV band it needs TiB and above ~1 MeV it returns NaN.
DETECTOR_MAX_EV = 100e3


def observe(grid, density, detectors):
    """Intrinsic observables over the whole band; detector counts below DETECTOR_MAX_EV."""
    zeros = np.zeros_like(density)
    out = spectrum_observables(grid, zeros, density, detectors={})
    m = grid <= DETECTOR_MAX_EV
    det = spectrum_observables(grid[m], zeros[m], density[m], detectors=detectors)
    return out | {k: v for k, v in det.items() if k.endswith("counts")}


def relative(candidate, reference):
    return {
        k: (abs(candidate[k] - reference[k]) / abs(reference[k]) if reference[k] else float("nan"))
        for k in BUDGET
    }


def within(rel):
    return all(rel[k] <= BUDGET[k] for k in BUDGET)


def timed(ladder, grid):
    t = time.perf_counter()
    density = ladder.continuum(grid)
    return density, time.perf_counter() - t


def run_case(args, energy_keV):
    case = cc.build_ladder_case(
        args.material,
        energy_keV,
        args.tilt_deg,
        args.tilt_azim_deg,
        thickness_ang=args.thickness_ang,
        n_electrons=args.n_electrons,
        seed=args.seed,
    )
    # Lines are never evaluated here; a small explicit line grid skips the policy.
    ladder = cc.CaseLadder(explicit_line_grid(case, 2000))
    endpoint = case_endpoint_eV(case)
    floor = photon_continuum_floor_eV(args.material)
    detectors = {"timepix3_counts": Timepix3(n_mc=32, seed=7), "eaglexo_counts": EagleXO()}
    # Production E_grid_brem runs past the endpoint (zero above it).
    ref_grid = np.arange(floor, endpoint + args.ref_step_ev, args.ref_step_ev)
    ref_density, ref_s = timed(ladder, ref_grid)
    t = time.perf_counter()
    ref_obs = observe(ref_grid, ref_density, detectors)
    observe_s = time.perf_counter() - t
    truth_grid = refined_continuum_grid(
        args.material, endpoint, args.truth_geometric_nodes, endpoint_eV=endpoint
    )
    truth_density, truth_s = timed(ladder, truth_grid)
    truth_obs = observe(truth_grid, truth_density, detectors)
    production_vs_truth = relative(ref_obs, truth_obs)
    print("production-vs-truth", {k: f"{v:.2e}" for k, v in production_vs_truth.items()})
    record = dict(
        energy_keV=energy_keV,
        endpoint_eV=endpoint,
        floor_eV=floor,
        transport_wall_s=ladder.transport_wall_s,
        fingerprint=ladder.fingerprint,
        reference=dict(
            step_eV=args.ref_step_ev,
            nodes=int(ref_grid.size),
            seconds=ref_s,
            observe_seconds=observe_s,
            observables=ref_obs,
            relative_to_truth=production_vs_truth,
            within_budget_of_truth=within(production_vs_truth),
        ),
        truth=dict(
            geometric_nodes=args.truth_geometric_nodes,
            nodes=int(truth_grid.size),
            seconds=truth_s,
            observables=truth_obs,
        ),
        candidates={},
    )

    def add(label, nodes, seconds, outputs, **extra):
        """``outputs``: axis name (``own``, ``truth``, ``production``) -> (grid, density).

        ``own`` and ``truth`` outputs are judged against the truth reference;
        the ``production`` output (on the 25 eV axis) against production.
        """
        entry = dict(nodes=int(nodes), seconds=seconds, speedup_vs_production=ref_s / seconds)
        line = [label, int(nodes), f"{seconds:.3g}s", f"x{ref_s / seconds:.0f}"]
        for axis, (grid, density) in outputs.items():
            obs = observe(grid, density, detectors)
            ref = ref_obs if axis == "production" else truth_obs
            rel = relative(obs, ref)
            entry[axis] = dict(
                against="production" if axis == "production" else "truth",
                observables=obs,
                relative=rel,
                within_budget=within(rel),
            )
            line += [axis, {k: f"{v:.1e}" for k, v in rel.items()}]
        record["candidates"][label] = entry | extra
        print(*line, flush=True)

    for n in args.geometric_nodes:
        grid = refined_continuum_grid(args.material, endpoint, n, endpoint_eV=endpoint)
        density, seconds = timed(ladder, grid)
        add(
            f"geometric-{n}",
            grid.size,
            seconds,
            {
                "own": (grid, density),
                "truth": (truth_grid, np.interp(truth_grid, grid, density, left=0.0, right=0.0)),
                "production": (ref_grid, np.interp(ref_grid, grid, density, left=0.0, right=0.0)),
            },
        )
    for ratio in args.gl_ratio:
        panels, brackets = gl_panels(args.material, floor, endpoint, ratio)
        for order in args.gl_order:
            nodes, members = gl_nodes(panels, brackets, order)
            density, seconds = timed(ladder, nodes)

            def on(axis, args=(panels, brackets, order, nodes, members, density)):
                return axis, gl_reconstruct(*args, axis)

            add(
                f"gl-{order}x{ratio:g}",
                nodes.size,
                seconds,
                {"truth": on(truth_grid), "production": on(ref_grid)},
                panels=len(panels),
                brackets=len(brackets),
            )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", default="hopg")
    parser.add_argument("--energy-kev", type=float, action="append", required=True)
    parser.add_argument("--tilt-deg", type=float, default=45.0)
    parser.add_argument("--tilt-azim-deg", type=float, default=180.0)
    parser.add_argument("--thickness-ang", type=float, default=1.0e7)
    parser.add_argument("--n-electrons", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ref-step-ev", type=float, default=25.0)
    parser.add_argument(
        "--truth-geometric-nodes",
        type=int,
        default=65537,
        help="dense refined geometric grid every candidate is also judged against",
    )
    parser.add_argument("--geometric-nodes", type=int, action="append", default=[])
    parser.add_argument("--gl-order", type=int, action="append", default=[])
    parser.add_argument("--gl-ratio", type=float, action="append", default=[])
    parser.add_argument(
        "--detector-max-ev",
        type=float,
        default=DETECTOR_MAX_EV,
        help="highest photon energy the detector-count observables include",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()["DETECTOR_MAX_EV"] = args.detector_max_ev
    cases = []
    for energy in args.energy_kev:
        cases.append(run_case(args, energy))
        args.output.write_text(
            json.dumps(
                dict(args=vars(args) | {"output": str(args.output)}, cases=cases),
                indent=1,
                default=float,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
