"""PyRITE CUDA arm of the matched-bounds Geant4 benchmark (#821).

Physics matches ``scenes.json`` bounds and the Geant4 app's electron-only
constructor: ELSEPA single scattering, Kawrakow atomic-electron deflection
(Geant4: DPWA scattering-power correction), SBETHE continuous collision
stopping without straggling or delta rays, coupled BremsLib soft/hard
radiation at 1 keV, 10 keV electron cutoff, first-exit termination. No
PXR/CBS, characteristic or secondary transport is requested.

The transport schedule is the production default (``auto``: bounded
continuation, which never replays a history). The full-history fallback is
also made replay-free: initial segment capacity exceeds the longest observed
track and the probe batch is disabled, so no capacity overflow recomputes a
history. Launched versus retained rows are recorded to prove it.

One process measures one (scene, seed): imports and tables, a complete
same-workload warmup at ``seed + 1000000`` (the first, cold request; full size
unless ``--warmup-histories`` shortens it, as time-to-precision runs do, to at
least one block) and the measured request. Each request runs fixed history blocks and reduces each block
to per-primary scores inside the timer, as the Geant4 app scores in its event
loop. CUDA is synchronized at both ends of every timed request.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import resource
import sys
import time
from pathlib import Path

START = time.perf_counter()
HERE = Path(__file__).parent
BANDS = (
    ("photons_1_10keV", 1e3, 1e4),
    ("photons_10_50keV", 1e4, 5e4),
    ("photons_50_100keV", 5e4, 1e5),
    ("photons_ge100keV", 1e5, math.inf),
)


CONE_BANDS = ("10_50kev", "50_100kev", "ge100kev")
# Only these CXR-package helpers may run: segment clipping/orientation geometry
# the bremsstrahlung estimator imports. They compute no PXR/CBS amplitude or line.
SHARED_GEOMETRY = {
    "pyrite.montecarlo.spectrum.cxr._kernels." + name
    for name in (
        "_clip_segments_to_cutoff",
        "_observation_direction",
        "_validate_groove_escape_direction",
    )
}


def forbidden(module, name):
    """True for a PXR/CBS or line-scorer computation (import scaffolding excluded)."""
    if name.startswith("<") or name[:1].isupper() or name in {"__annotate__", "__create_fn__"}:
        return False
    crystal = module.startswith(
        ("pyrite.montecarlo.spectrum.cxr", "pyrite.montecarlo.spectrum.lines")
    )
    return crystal and f"{module}.{name}" not in SHARED_GEOMETRY


def cone_axis(cone):
    import numpy as np

    c = math.radians(cone["center_deg"])
    return np.array([math.sin(c), 0.0, math.cos(c)])


def discrete_cone_scores(r, n, cone):
    """Sampled hard photons >= 10 keV entering the cone at creation, per primary."""
    import numpy as np

    from pyrite.perf.precision import grouped_scores

    ids = np.asarray(r["electron_id"], dtype=np.int64)
    k = np.asarray(r["hard_radiative_k_eV"], dtype=np.float64)
    hit = np.zeros(k.size, dtype=bool)
    m = k >= 1e4
    d = np.asarray(r["hard_radiative_direction"], dtype=np.float64)[m]
    hit[m] = d @ cone_axis(cone) >= math.cos(math.radians(cone["half_angle_deg"]))
    out = {}
    for name, lo, hi in zip(CONE_BANDS, (1e4, 5e4, 1e5), (5e4, 1e5, math.inf), strict=True):
        sel = hit & (k >= lo) & (k < hi)
        out[f"cone_source_photons_{name}"] = grouped_scores(ids[sel], np.ones(sel.sum()), n)
    out["cone_source_photons_ge10kev"] = sum(out[f"cone_source_photons_{b}"] for b in CONE_BANDS)
    return out


def continuous_cone_scores(r, n, cone, ctx):
    """Production weighted per-segment estimator integrated over the cone.

    Every value is :func:`mc_coupled_brem_spectrum` (BremsLib DDCS by track
    length, per-electron band weights) toward each quadrature direction, with
    unit transmission (photons at creation, as the discrete arms score them).
    ``ctx["mode"]``: ``adaptive`` refines the angular/energy order until the
    primary sums and sums of squares change by ``rtol``; ``fixed`` evaluates
    once at ``ctx["order"]``; ``point`` evaluates the cone axis only, times
    the cone solid angle (production point-detector convention).
    """
    import numpy as np

    from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum
    from pyrite.perf.comparison import cone_rule, cone_solid_angle, converged_cone_scores
    from pyrite.perf.detector_scoring import ENERGY_NODES_PER_ORDER, band_panels, band_rule

    comp, e0 = ctx["composition"], ctx["energy_keV"]
    layers = [(0.0, ctx["thickness_ang"], [(el, 0.0) for el, _ in comp])]
    bands = [
        band_panels(lo, min(hi, e0), ctx["edges"]) if lo < e0 else None
        for lo, hi in ((10.0, 50.0), (50.0, 100.0), (100.0, math.inf))
    ]

    def f(directions, order):
        out = np.zeros((directions.shape[0], n, len(bands)))
        rules = [None if p is None else band_rule(p, ENERGY_NODES_PER_ORDER * order) for p in bands]
        for i, n_hat in enumerate(directions):
            for b, rule in enumerate(rules):
                if rule is None:
                    continue
                v = mc_coupled_brem_spectrum(
                    r,
                    rule[0],
                    cutoff_eV=ctx["cutoff_eV"],
                    bremslib_tables=ctx["brem"],
                    composition=comp,
                    n_hat=n_hat,
                    layers=layers,
                    electron_band_weights=rule[1],
                    chunk=ctx["chunk"],
                )
                out[i, :, b] = v.get() if hasattr(v, "get") else np.asarray(v)
        return out

    axis = cone_axis(cone)
    half = math.radians(cone["half_angle_deg"])
    if ctx["mode"] == "adaptive":
        res = converged_cone_scores(
            f, axis, half, rtol=ctx["rtol"], exact_zero=[p is None for p in bands], max_order=32
        )
        if not res["converged"]:
            raise RuntimeError(f"cone quadrature unconverged at order {res['order']}")
        ctx["orders"].append(res["order"])
        values = res["scores"]
    elif ctx["mode"] == "fixed":
        dirs, weights = cone_rule(axis, half, ctx["order"], 2 * ctx["order"])
        values = np.tensordot(weights, f(dirs, ctx["order"]), axes=1)
    else:
        values = f(axis[None, :], ctx["order"])[0] * cone_solid_angle(half)
    out = {f"cone_source_photons_{b}": values[:, j] for j, b in enumerate(CONE_BANDS)}
    out["cone_source_photons_ge10kev"] = values.sum(axis=1)
    return out


def spectrum_grids(args, mat, composition, energy_keV):
    """Diagnostic continuum grids by label: uniform steps and refined geometric counts.

    ``uniform-<step>eV`` is ``step, 2 step, ... < E0`` (production ``E_grid_brem``
    uses 25 eV). ``geometric-<n>`` is :func:`refined_continuum_grid`: ``n``
    log-spaced nodes from the medium's continuum floor to ``E0``, plus the
    anchor pairs at absorption edges and the kinematic endpoint
    (``Validation: continuum-node-refinement``).
    """
    import numpy as np

    from pyrite.energy_grid.refine import refined_continuum_grid
    from pyrite.materials import MediumSpec

    e0 = 1e3 * energy_keV
    medium = MediumSpec(mat["catalog_key"], tuple(composition))
    grids = {f"uniform-{s:g}eV": np.arange(s, e0, s) for s in args.spectrum_step_ev}
    for n in args.spectrum_geometric_nodes:
        grids[f"geometric-{n}"] = np.asarray(
            refined_continuum_grid(medium, e0, n, endpoint_eV=e0), dtype=float
        )
    return grids


def grid_spectrum(r, grid, cone, ctx):
    """Production continuum toward the cone axis on ``grid`` [eV].

    Spectral-grid sensitivity diagnostic (#821 -> #278): the same
    :func:`mc_coupled_brem_spectrum` call and runner chunk policy a production
    case uses for ``E_grid_brem``, with unit transmission like the cone arms.
    Returns the density [photons / eV / sr / electron] and the segment chunk.
    """
    import numpy as np

    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.brem_events import mc_coupled_brem_spectrum

    dense = runner._EEDL_BREM_DENSE_INTERMEDIATES
    chunk = runner._admit_chunk(
        runner._RESOURCE_POLICY.brem_chunk
        or runner._adaptive_chunk(grid.size, intermediates=dense),
        grid.size,
        intermediates=dense,
    )
    v = mc_coupled_brem_spectrum(
        r,
        grid,
        cutoff_eV=ctx["cutoff_eV"],
        bremslib_tables=ctx["brem"],
        composition=ctx["composition"],
        n_hat=cone_axis(cone),
        layers=[(0.0, ctx["thickness_ang"], [(el, 0.0) for el, _ in ctx["composition"]])],
        chunk=chunk,
    )
    return (v.get() if hasattr(v, "get") else np.asarray(v)), chunk


def band_integral(nodes, density, lo, hi, weight=None):
    """Piecewise-linear integral of ``density`` (times ``weight``) over ``[lo, hi]``."""
    import numpy as np

    hi = min(hi, nodes[-1])
    if hi <= lo:
        return 0.0
    inner = (nodes > lo) & (nodes < hi)
    x = np.concatenate([[lo], nodes[inner], [hi]])
    y = np.interp(x, nodes, density)
    return float(np.trapezoid(y * (x if weight == "energy" else 1.0), x))


def grid_summary(g, nodes, histories, cone):
    """Time, size, cone-band yields and >=10 keV centroid (point convention)."""
    from pyrite.perf.comparison import cone_solid_angle

    density = g["density_sum"] / histories
    omega = cone_solid_angle(math.radians(cone["half_angle_deg"]))
    bands = {
        name: band_integral(nodes, density, lo, hi) * omega
        for name, lo, hi in (
            ("10_50kev", 1e4, 5e4),
            ("50_100kev", 5e4, 1e5),
            ("ge100kev", 1e5, math.inf),
        )
    }
    total = sum(bands.values())
    first = band_integral(nodes, density, 1e4, math.inf, weight="energy") * omega
    return dict(
        seconds=g["seconds"],
        nodes=int(nodes.size),
        chunks=sorted(g["chunks"]),
        cone_point_yield_per_primary=bands,
        cone_point_yield_ge10kev_per_primary=total,
        centroid_ge10kev_eV=first / total if total else None,
    )


def electron_groups(r, n, max_rows):
    """Split a block's rows into contiguous electron ranges of at most ``max_rows`` rows.

    Yields ``(first, count, segments)`` with ``elec_id`` rebased to the range and
    every per-row field restricted to it, so the continuous estimator's whole-block
    temporaries stay inside VRAM independently of the transport block size.
    """
    import numpy as np

    eid = np.asarray(r["elec_id"])
    rows = eid.size
    order = None if np.all(eid[1:] >= eid[:-1]) else np.argsort(eid, kind="stable")
    sorted_eid = eid if order is None else eid[order]
    starts = np.searchsorted(sorted_eid, np.arange(n + 1))
    first = 0
    while first < n:
        last = int(np.searchsorted(starts, starts[first] + max_rows, side="right")) - 1
        last = min(max(last, first + 1), n)
        lo, hi = int(starts[first]), int(starts[last])
        pick = slice(lo, hi) if order is None else order[lo:hi]
        sub = {
            k: (np.asarray(v)[pick] if getattr(v, "shape", ()) and v.shape[0] == rows else v)
            for k, v in r.items()
        }
        sub["elec_id"] = sorted_eid[lo:hi] - first
        sub["Ne"] = last - first
        yield first, last - first, sub
        first = last


def block_scores(r, n):
    """Per-primary observables and work counts for one block."""
    import numpy as np

    from pyrite.montecarlo.transport.events import SegmentEvent as EventKind
    from pyrite.perf.precision import grouped_scores

    ids = np.asarray(r["electron_id"], dtype=np.int64)
    ev = np.asarray(r["event_kind"])
    ones = np.ones(ids.size)
    out = {}
    for name, code in [
        ("backscattered", EventKind.EXIT_TOP),
        ("transmitted", EventKind.EXIT_BOTTOM),
        ("side_exited", EventKind.EXIT_SIDE),
        ("stopped", EventKind.CUTOFF),
    ]:
        m = ev == code
        out[name] = grouped_scores(ids[m], ones[m], n)
    terminal = out["backscattered"] + out["transmitted"] + out["side_exited"] + out["stopped"]
    if not np.all(terminal == 1.0):
        raise RuntimeError("every primary needs exactly one terminal row")
    m = ev == EventKind.EXIT_BOTTOM
    out["transmitted_energy_keV"] = grouped_scores(ids[m], np.asarray(r["E_end_keV"])[m], n)
    k = np.asarray(r["hard_radiative_k_eV"], dtype=np.float64)
    for name, lo, hi in BANDS:
        m = (k >= lo) & (k < hi)
        out[name] = grouped_scores(ids[m], ones[m], n)
    m = k >= 1e4
    out["photon_keV_ge10keV"] = grouped_scores(ids[m], k[m] / 1e3, n)
    out["path_um"] = grouped_scores(ids, np.asarray(r["L_ang"], dtype=np.float64) / 1e4, n)
    out["rows"] = grouped_scores(ids, ones, n)
    for name, code in [
        ("rows_elastic", EventKind.ELASTIC),
        ("rows_substep", EventKind.SUBSTEP),
        ("rows_hard_radiative", EventKind.HARD_RADIATIVE),
    ]:
        m = ev == code
        out[name] = grouped_scores(ids[m], ones[m], n)
    return out


class Messages(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--block", type=int, default=0, help="histories per block; 0: scene's")
    parser.add_argument("--rows-per-round", type=int, default=256)
    parser.add_argument("--device-staging", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--warmup-histories", type=int, default=0, help="cold warmup request size; 0: --histories"
    )
    parser.add_argument("--max-de-frac", type=float, default=0.05)
    parser.add_argument("--scratch-mib", type=int, default=4096)
    parser.add_argument("--seg-capacity", type=int, default=16384)
    parser.add_argument(
        "--probe-electrons",
        type=int,
        default=0,
        help="full-history first-batch probe; production default is 1024 (#825)",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0: scene's")
    parser.add_argument(
        "--estimator",
        choices=["discrete", "continuous"],
        default="discrete",
        help="cone yield from sampled hard photons, or the weighted per-segment estimator",
    )
    parser.add_argument("--cone-mode", choices=["adaptive", "fixed", "point"], default="adaptive")
    parser.add_argument("--cone-order", type=int, default=8)
    parser.add_argument("--cone-rtol", type=float, default=1e-3)
    parser.add_argument("--chunk", type=int, default=20000, help="segments per estimator chunk")
    parser.add_argument(
        "--score-rows",
        type=int,
        default=20_000_000,
        help="continuous estimator: max rows per electron group",
    )
    parser.add_argument(
        "--kernel",
        choices=["default", "specialized", "monolithic"],
        default="default",
        help="CUDA transport kernel selection where the build supports it (#827)",
    )
    parser.add_argument(
        "--pool-fraction", type=float, default=0.85, help="CuPy pool limit, fraction of VRAM"
    )
    parser.add_argument(
        "--spectrum-step-ev",
        type=float,
        action="append",
        default=[],
        help="also time the production continuum on this uniform grid step (repeatable; "
        "excluded from the request timings)",
    )
    parser.add_argument(
        "--spectrum-geometric-nodes",
        type=int,
        action="append",
        default=[],
        help="also time the production continuum on the refined geometric grid with this "
        "baseline node count (repeatable; excluded from the request timings)",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="skip the untimed PXR/CBS call-profiler request (timing sweeps only)",
    )
    parser.add_argument(
        "--cprofile",
        type=Path,
        default=None,
        help="write a cProfile dump of the warm request (#825 host-phase attribution)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((HERE / "scenes.json").read_text())
    scene = next(s for s in spec["scenes"] if s["id"] == args.scene)
    mat = spec["materials"][scene["material"]]
    bounds = spec["bounds"]
    args.block = args.block or scene["block"]
    args.max_steps = args.max_steps or scene["max_steps"]

    import cupy as cp

    # WSL2/WDDM pages device memory to host instead of failing an allocation
    # past physical VRAM, which silently slows scoring several-fold. Keep the
    # pool below VRAM so CuPy recycles its cache first.
    if args.kernel != "default":
        from pyrite.montecarlo.transport import _jit_launch

        if not hasattr(_jit_launch, "_SPECIALIZE_TRANSPORT_KERNEL"):
            raise RuntimeError("--kernel needs a build with per-configuration kernels (#827)")
        _jit_launch._SPECIALIZE_TRANSPORT_KERNEL = args.kernel == "specialized"
    pool = cp.get_default_memory_pool()
    pool.set_limit(fraction=args.pool_fraction)
    import numpy as np

    from pyrite.montecarlo.transport import TransportLUTConfig, simulate_trajectories
    from pyrite.montecarlo.transport.batching import (
        DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG,
        PerElectronTransportConfig,
    )
    from pyrite.perf.precision import primary_moments
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables
    from pyrite.xsgen.sbethe.catalog import resolve_composition_table

    comp = [(scene["material"], mat["number_density_ang3"])]
    entries = resolve_layer_tables(comp)
    stopping = resolve_composition_table(mat["catalog_key"], comp)
    brem = load_bremsstrahlung_tables([scene["material"]])
    rounds = {"launched_rows": 0, "launches": 0, "assembly": []}

    def on_round(info):
        if info.get("phase") == "round":
            rounds["launches"] += 1
            rounds["launched_rows"] += int(info.get("rows", 0))
        elif info.get("phase") == "assembly":
            rounds["assembly"].append(dict(info))

    default = DEFAULT_PER_ELECTRON_TRANSPORT_CONFIG
    from pyrite.montecarlo.transport.continuation import TransportContinuationConfig

    config = PerElectronTransportConfig(
        seg_capacity=args.seg_capacity,
        scratch_budget_bytes=args.scratch_mib * 1024**2,
        probe_electrons=args.probe_electrons,
        max_batch=default.max_batch,
        schedule=default.schedule or "auto",
        continuation=TransportContinuationConfig(
            rows_per_round=args.rows_per_round,
            device_staging=args.device_staging,
            on_round=on_round,
        ),
    )
    log = Messages()
    for name in ("pyrite.montecarlo.transport.continuation",):
        logging.getLogger(name).addHandler(log)
        logging.getLogger(name).setLevel(logging.INFO)
    kwargs = dict(
        E0_keV=float(scene["energy_keV"]),
        composition=comp,
        thickness_ang=scene["thickness_um"] * 1e4,
        crystal_width_mm=bounds["width_mm"],
        crystal_height_mm=bounds["width_mm"],
        beam_dir=np.array([0.0, 0.0, 1.0]),
        E_cut_keV=float(bounds["electron_cutoff_keV"]),
        energy_model="midpoint",
        max_dE_frac=args.max_de_frac,
        max_steps=args.max_steps,
        elastic_model="elsepa",
        elastic_tables=[[dict(e.arrays) for e in entries]],
        stopping_tables=[stopping.arrays()],
        atomic_electron_deflection="kawrakow",
        straggling=False,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=float(bounds["hard_photon_cutoff_eV"]),
        bremslib_tables=brem,
        radiative_moments="table",
        transport_core="cuda",
        per_electron_config=config,
        # ELSEPA bypasses the LUT anyway; explicit so `auto` cannot fall back on it.
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    from pyrite.perf.detector_scoring import absorption_edges_keV

    cone = spec["cone"]
    ctx = dict(
        mode=args.cone_mode,
        order=args.cone_order,
        rtol=args.cone_rtol,
        chunk=args.chunk,
        composition=comp,
        energy_keV=float(scene["energy_keV"]),
        thickness_ang=scene["thickness_um"] * 1e4,
        cutoff_eV=float(bounds["hard_photon_cutoff_eV"]),
        brem=brem,
        edges=absorption_edges_keV(comp),
        orders=[],
    )
    tables_s = time.perf_counter() - START

    last_grids = {}
    grid_nodes = spectrum_grids(args, mat, comp, float(scene["energy_keV"]))

    def request(seed, histories):
        scores = {}
        retained = 0
        rounds.update(launched_rows=0, launches=0, assembly=[])
        log.messages.clear()
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        transport_s = scoring_s = 0.0
        grids = {label: dict(seconds=0.0, density_sum=None, chunks=set()) for label in grid_nodes}
        for offset in range(0, histories, args.block):
            n = min(args.block, histories - offset)
            t1 = time.perf_counter()
            r = simulate_trajectories(**kwargs, Ne=n, seed=seed, _electron_start=offset)
            cp.cuda.Device().synchronize()
            transport_s += time.perf_counter() - t1
            retained += int(np.asarray(r["event_kind"]).size)
            t2 = time.perf_counter()
            pool.free_all_blocks()
            block = block_scores(r, n)
            if args.estimator == "continuous":
                parts = [
                    (first, continuous_cone_scores(sub, count, cone, ctx))
                    for first, count, sub in electron_groups(r, n, args.score_rows)
                ]
                for key in parts[0][1]:
                    v = np.empty(n)
                    for first, part in parts:
                        v[first : first + part[key].size] = part[key]
                    block[key] = v
            else:
                block |= discrete_cone_scores(r, n, cone)
            cp.cuda.Device().synchronize()
            scoring_s += time.perf_counter() - t2
            for label, g in grids.items():
                pool.free_all_blocks()
                t3 = time.perf_counter()
                density, chunk = grid_spectrum(r, grid_nodes[label], cone, ctx)
                cp.cuda.Device().synchronize()
                g["seconds"] += time.perf_counter() - t3
                weighted = n * density
                g["density_sum"] = (
                    weighted if g["density_sum"] is None else g["density_sum"] + weighted
                )
                g["chunks"].add(int(chunk))
            for key, v in block.items():
                scores.setdefault(key, np.empty(histories))[offset : offset + n] = v
            del r
        cp.cuda.Device().synchronize()
        grid_s = sum(g["seconds"] for g in grids.values())
        # The grid diagnostic stays outside the arm's request time.
        wall = time.perf_counter() - t0 - grid_s
        work = dict(
            rounds,
            retained_rows=retained,
            scoring_s=scoring_s,
            schedule=sorted({m for m in log.messages if "schedule" in m}),
        )
        if grids:
            work["grid_spectra"] = {
                label: grid_summary(g, grid_nodes[label], histories, cone)
                for label, g in grids.items()
            }
            last_grids.clear()
            for label, g in grids.items():
                last_grids[f"density_{label}"] = g["density_sum"] / histories
                last_grids[f"nodes_{label}"] = grid_nodes[label]
        return scores, wall, transport_s, work

    warmup = args.warmup_histories or args.histories
    _, cold_s, cold_transport_s, cold_work = request(args.seed + 1_000_000, warmup)
    # Untimed audit request: fail on any PXR/CBS or line-scorer computation.
    calls = set()

    def audit(frame, event, _arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "") or ""
            if forbidden(module, frame.f_code.co_name):
                calls.add(f"{module}.{frame.f_code.co_name}")

    if not args.no_audit:
        sys.setprofile(audit)
        try:
            request(args.seed + 2_000_000, min(args.histories, 5000))
        finally:
            sys.setprofile(None)
    if calls:
        raise RuntimeError(f"PXR/CBS or line computation executed: {sorted(calls)}")
    profiler = None
    if args.cprofile is not None:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    scores, warm_s, warm_transport_s, work = request(args.seed, args.histories)
    if profiler is not None:
        profiler.disable()
        args.cprofile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.cprofile)
    if work["launches"] and work["launched_rows"] != work["retained_rows"]:
        raise RuntimeError(f"replayed rows detected: {work}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz = args.output.with_suffix(".npz")
    np.savez_compressed(npz, **scores)
    if last_grids:
        np.savez_compressed(args.output.with_suffix(".grid.npz"), **last_grids)
    import cupy

    dev = cupy.cuda.runtime.getDeviceProperties(0)
    record = dict(
        schema="pyrite.geant4-performance.sample.v1",
        engine="pyrite-cuda",
        estimator=args.estimator,
        cone=cone,
        cone_scoring=dict(
            mode=args.cone_mode if args.estimator == "continuous" else "sampled-photons",
            order=args.cone_order,
            rtol=args.cone_rtol,
            chunk=args.chunk,
            adaptive_orders=ctx["orders"],
        ),
        audit=dict(
            forbidden="pyrite.montecarlo.spectrum.{cxr,lines} computations",
            calls=[],
            skipped=bool(args.no_audit),
        ),
        scene=scene,
        histories=args.histories,
        warmup_histories=warmup,
        block=args.block,
        seed=args.seed,
        controls=dict(
            max_dE_frac=args.max_de_frac,
            scratch_mib=args.scratch_mib,
            seg_capacity=args.seg_capacity,
            probe_electrons=args.probe_electrons,
            max_steps=args.max_steps,
            schedule=config.schedule,
            rows_per_round=args.rows_per_round,
            device_staging=args.device_staging,
            radiative_moments="table",
            pool_fraction=args.pool_fraction,
            score_rows=args.score_rows,
            kernel=args.kernel,
            spectrum_step_eV=args.spectrum_step_ev,
            spectrum_geometric_nodes=args.spectrum_geometric_nodes,
        ),
        timings_s=dict(
            imports_and_tables=tables_s,
            cold_request=cold_s,
            cold_transport=cold_transport_s,
            warm_request=warm_s,
            warm_transport=warm_transport_s,
            process_total=time.perf_counter() - START,
        ),
        work=dict(cold=cold_work, warm=work),
        tables=dict(
            sbethe=stopping.key,
            sbethe_digest=stopping.digest,
            bremslib=brem[scene["material"]].key,
            elsepa=[t.key for e in entries for t in e.tables],
        ),
        device=dev["name"].decode(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        environment={
            k: os.environ.get(k) for k in ("PYRITE_MC_BACKEND", "PYRITE_MC_TRANSPORT_SCHEDULE")
        },
        observables={k: primary_moments(v) for k, v in scores.items()},
        raw_scores_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "timings_s": record["timings_s"],
                "work": record["work"]["warm"],
            }
        )
    )


if __name__ == "__main__":
    main()
