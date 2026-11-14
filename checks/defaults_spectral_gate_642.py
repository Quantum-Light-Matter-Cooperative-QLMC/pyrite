"""Remote-only bounded transport/spectrum sensitivity pilot (#1142, #1143).

No default or physics verdict. Fixed 0.5--10 keV reporting band; reference
refinement and changed-transport statistical qualification remain separate.
Our own pickle is retained for trusted, same-source replay only.
"""

import argparse
import hashlib
import json
import os
import pickle
import resource
import runpy
import time
from pathlib import Path

import numpy as np


def axis(start, stop, spacing, quadrature):
    """Node support or midpoint-bin support exactly spans the same interval."""
    n = int(np.ceil((stop - start) / spacing))
    if quadrature == "node":
        return np.linspace(start, stop, n + 1)
    return start + (np.arange(n) + 0.5) * ((stop - start) / n)


def metrics(grid, density, quadrature, edges=None):
    from pyrite.energy_grid.bandwidth_check import _bin_masses, _dominant_fwhm_eV
    from pyrite.montecarlo.spectrum.characteristic import _energy_bin_edges_and_widths

    if quadrature == "node":
        total = float(np.trapezoid(density, grid))
        first = float(np.trapezoid(density * grid, grid))
    else:
        cells, widths = _energy_bin_edges_and_widths(grid)
        mass = density * widths
        total = float(mass.sum())
        first = float(np.sum(mass * (cells[:-1] + cells[1:]) / 2))
    if total <= 0 or not np.isfinite(density).all():
        raise ValueError("nonpositive band yield or nonfinite spectrum")
    fwhm = float(_dominant_fwhm_eV(grid, density))
    return dict(
        yield_per_primary=total,
        centroid_eV=first / total,
        peak_density_per_eV=float(density.max()),
        dominant_fwhm_eV=fwhm if np.isfinite(fwhm) else None,
        band_masses_100eV=_bin_masses(
            grid,
            density,
            np.linspace(500, 10000, 96) if edges is None else edges,
            quadrature=quadrature,
        ).tolist(),
    )


CRYSTAL = {"mos2-on-si": "mos2"}


def scene_sweep(material, energy_keV, thickness_ang):
    """The frozen scene's sweep: single slab at 45/180 deg, or the layered stack."""
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.geometry import Stack

    if material == "mos2-on-si":
        target = Stack.on_substrate(
            "mos2", 2000.0, "silicon", 2e5, tilt_deg=45, tilt_azim_deg=180, footprint=None
        )
        return material_sweep("mos2", target=target, energy_keV=30.0)
    return material_sweep(
        material,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        tilt_deg=45,
        tilt_azim_deg=180,
    )


def layered_case(histories, seed):
    """Supported two-crystal stack: 2000 A MoS2 on a 20 um Si substrate at 30 keV.

    Both layers have shell records, so shell soft/hard transport admits it; the
    substrate stops the beam (near-range/backscatter scene) and also radiates.
    """
    from pyrite.campaign.sweep import build_cases

    sweep = scene_sweep("mos2-on-si", 30.0, None)
    case = dict(build_cases(sweep, n_electrons=histories, n_electrons_brem=histories)[0])
    case["seed"] = int(seed)
    return case


def mosaic_mc_case(case, nodes):
    """Exact MC mosaic average at the crystal's catalog FWHM; analytic term off.

    The analytic route only broadens at detector convolution, so it leaves the
    intrinsic source spectrum unchanged; only this route is a source-level ladder.
    """
    from pyrite.materials import CATALOG

    fwhm_deg = CATALOG.crystal(case["crystal"]).mosaic_fwhm_deg
    if not fwhm_deg:
        raise ValueError(f"{case['crystal']!r} has no catalog mosaic FWHM")
    return dict(
        case,
        mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=float(np.deg2rad(fwhm_deg)),
        mosaic_mc_nodes=int(nodes),
    )


#: Production automatic line-grid policies: arm name -> profile selector table.
#: ``sinc-*`` keep the kinematic-ceiling bandwidth; ``measured*`` use the
#: ``resonance-population`` edge; ``measured-local`` adds ``resonance-local``.
POLICY_ARMS = {
    "sinc": {"resolution": "sinc-nyquist"},
    "sinc-bin": {"resolution": "sinc-nyquist", "quadrature": "bin-mean"},
    "measured": {"bandwidth": "resonance-population"},
    "measured-bin": {"bandwidth": "resonance-population", "quadrature": "bin-mean"},
    "measured-local": {
        "bandwidth": "resonance-population",
        "resolution": "resonance-local",
        "quadrature": "bin-mean",
    },
}
#: Predeclared same-track numerical margins (#1143); ``mass_l1`` is the L1
#: distance of the two normalized 100 eV source-band mass vectors.
MARGINS = dict(
    yield_per_primary=1e-3,
    centroid_eV=1e-3,
    peak_density_per_eV=1e-2,
    dominant_fwhm_eV=1e-2,
    mass_l1=1e-2,
)
BAND = (500.0, 10000.0)
#: Warm timings whose relative spread (max - min) / median exceeds this are
#: reported unresolved: no point estimate is claimed from them (#1143 Plan 5).
TIMING_SPREAD_LIMIT = 0.2


def timed_calls(call, grid, repeats, sync, probe=None):
    """First call, ``repeats - 1`` synchronized warm calls, and statistics.

    ``first_s - warm_median_s`` estimates compile/first-setup cost; it is
    labeled an estimate, not a measured compile time. Device peak memory comes
    from one extra untimed probe call. Returns ``(density, record)``.
    """
    times = []
    for _ in range(max(1, int(repeats))):
        sync()
        begin = time.perf_counter()
        density = call(grid)
        sync()
        times.append(time.perf_counter() - begin)
    warm = times[1:]
    record = dict(
        first_s=times[0],
        warm_s=warm,
        grid_sha256=hashlib.sha256(np.ascontiguousarray(grid, dtype=float).tobytes()).hexdigest(),
    )
    if len(warm) >= 3:
        median = float(np.median(warm))
        spread = (max(warm) - min(warm)) / median
        record.update(
            warm_median_s=median,
            warm_relative_spread=spread,
            first_minus_warm_s_estimate=times[0] - median,
            warm_resolved=bool(spread <= TIMING_SPREAD_LIMIT),
        )
    else:
        record.update(warm_median_s=None, warm_resolved=False)
    record["device_memory"] = None if probe is None else probe(call, grid)
    return density, record


def band_edges(start, stop):
    """100 eV source-band edges inside ``[start, stop]``, aligned to 100 eV."""
    return np.arange(np.ceil(start / 100) * 100, np.floor(stop / 100) * 100 + 1, 100.0)


def versus(candidate, reference):
    """Separate signed relative changes against a same-track reference.

    Each observable is judged on its own margin, so conserved mass cannot hide
    a shape failure. An undefined FWHM on either side stays ``None``.
    """
    out = {}
    for key in ("yield_per_primary", "centroid_eV", "peak_density_per_eV", "dominant_fwhm_eV"):
        c, r = candidate[key], reference[key]
        out[key] = None if c is None or r is None else (c - r) / r
    cm = np.asarray(candidate["band_masses_100eV"], dtype=float)
    rm = np.asarray(reference["band_masses_100eV"], dtype=float)
    out["mass_l1"] = float(np.abs(cm / cm.sum() - rm / rm.sum()).sum())
    out["within_margin"] = {
        key: None if out[key] is None else bool(abs(out[key]) <= margin)
        for key, margin in MARGINS.items()
    }
    return out


def _jsonable(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)


def policy_case(case, sweep, selectors):
    """``case`` carrying the production policy payload ``build_cases`` attaches.

    Only the policy and its quadrature come from the rebuilt case; transport
    inputs stay those of the (possibly replayed) frozen case.
    """
    from dataclasses import replace

    from pyrite.campaign.sweep import build_cases

    built = build_cases(
        replace(sweep, cxr_grid_policy=dict(selectors)),
        n_electrons=int(case["Ne"]),
        n_electrons_brem=int(case["Ne"]),
    )[0]
    return dict(
        case,
        line_grid_policy=built["line_grid_policy"],
        line_quadrature=built.get("line_quadrature", "node"),
        _profile_line_grid_stages=True,
    )


def matched_grid(grid, record, quadrature, backbone_eV):
    """The policy's spacing on the fixed band.

    Uniform policies: their resolved spacing on the band. ``resonance-local``:
    its own nodes inside the band, continued past a measured edge below the
    band top at the policy's backbone spacing (what it would place there).
    """
    lo, hi = BAND
    if record.get("resolution_policy") == "resonance-local" or "actual_spacing_eV" not in record:
        inner = grid[(grid > lo) & (grid < hi)]
        if grid[-1] < hi:
            tail = axis(float(grid[-1]), hi, backbone_eV, "node")[1:-1]
            inner = np.r_[inner, tail[tail > inner[-1]]]
        return np.r_[lo, inner, hi]
    return axis(lo, hi, float(record["actual_spacing_eV"]), quadrature)


def score_policies(case, sweep, ladder, names, reference_spacing, sync, repeats=1, probe=None):
    """Resolve each policy from the saved trajectories and score it twice.

    ``native`` is the production axis (own support) against a node reference
    at ``reference_spacing`` on that same support. ``matched`` puts the
    policy's spacing on the fixed band, against the band reference, isolating
    support from spacing/quadrature.
    """
    from pyrite.energy_grid.bandwidth_check import _piecewise
    from pyrite.energy_grid.convergence_case import CaseLadder
    from pyrite.montecarlo.runner.cxr_grid import resolve_cxr_grid

    tp = ladder.transport
    node = CaseLadder(dict(case, line_quadrature="node"), transport=tp)
    references = {}

    def reference(start, stop):
        if (start, stop) not in references:
            grid = axis(start, stop, reference_spacing, "node")
            density = _piecewise(node.lines, grid)
            edges = band_edges(start, stop)
            references[start, stop] = metrics(grid, density, "node", edges=edges)
        return references[start, stop]

    rows = []
    for name in names:
        pcase = policy_case(case, sweep, POLICY_ARMS[name])
        quadrature = pcase["line_quadrature"]
        row = dict(policy=name, selectors=POLICY_ARMS[name], quadrature=quadrature)
        try:
            sync()
            begin = time.perf_counter()
            grid, record = resolve_cxr_grid(
                pcase,
                tp["segs"],
                tp["n_hat"],
                int(tp["Ne_lines"]),
                np.asarray(tp["E_grid"], dtype=float),
                case.get("abs_layers"),
                tp.get("groove"),
            )
            row["resolution_s"] = time.perf_counter() - begin
        except ValueError as exc:  # CxrGridToleranceError and admission refusals
            rows.append(dict(row, refused=f"{type(exc).__name__}: {exc}"))
            continue
        grid = np.asarray(grid, dtype=float)
        row["record"] = json.loads(json.dumps(record, default=_jsonable))
        sync()
        begin = time.perf_counter()
        scorer = CaseLadder(pcase, transport=tp)
        sync()
        row["scorer_setup_s"] = time.perf_counter() - begin
        backbone = float(pcase["line_grid_policy"]["resolution"]["max_spacing_eV"])
        start, stop = float(grid[0]), float(grid[-1])
        for label, g, ref, edges in (
            ("native", grid, reference(start, stop), band_edges(start, stop)),
            (
                "matched",
                matched_grid(grid, record, quadrature, backbone),
                reference(*BAND),
                None,
            ),
        ):
            density, timing = timed_calls(scorer.lines, g, repeats, sync, probe)
            m = metrics(g, density, quadrature, edges=edges)
            row[label] = dict(
                nodes=int(g.size),
                support_eV=[float(g[0]), float(g[-1])],
                min_spacing_eV=float(np.diff(g).min()),
                max_spacing_eV=float(np.diff(g).max()),
                **timing,
                metrics=m,
                versus_reference=versus(m, ref),
            )
        rows.append(row)
    return dict(
        reference_spacing_eV=reference_spacing,
        references={f"{a:g}-{b:g}": m for (a, b), m in references.items()},
        margins=MARGINS,
        policies=rows,
    )


def families_case(case, sweep, n):
    """``case`` with the reflection set of ladder point ``n``; transport unchanged.

    Production crystals rebuild through ``build_cases`` at ``n_families=n`` (every
    layer radiator follows). HOPG's catalog pins (002)/(004) and texture rules
    out in-plane families, so its axis is the explicit (00l) harmonic order
    ``l = 2, 4, ..., 2n`` -- a study ladder, not a production selector.

    A string point ``"g<G>"`` instead keeps every nonforbidden reflection with
    ``|g| <= G`` 1/angstrom in every radiator (the family count is then not the
    truncation; the reciprocal cutoff is). Production enumerates to 8/angstrom.

    Returns the case, the production line-axis support (film crystal only) and
    the support every radiator's own reflections need under the same resolver.
    """
    from dataclasses import replace

    from pyrite.campaign.geometry import crystal_params
    from pyrite.campaign.sweep import _automatic_cxr_grid_policy, build_cases
    from pyrite.materials.crystal import dominant_reflections

    g_max = float(n[1:]) if isinstance(n, str) else None
    families = 4 if g_max is not None else int(n)
    built = build_cases(replace(sweep, n_families=families), n_electrons=1, n_electrons_brem=1)[0]
    out = dict(case, hkl_list=built["hkl_list"])
    if case.get("layer_radiators") is not None:
        out["layer_radiators"] = [None if r is None else dict(r) for r in built["layer_radiators"]]
    radiators = out.get("layer_radiators") or [out]
    radiators = [r for r in radiators if r is not None]
    if case["crystal"] == "hopg":
        if g_max is not None:
            raise ValueError("HOPG uses its (00l) order ladder, not a reciprocal cutoff")
        out["hkl_list"] = [(0, 0, s * 2 * k) for k in range(1, n + 1) for s in (1, -1)]
    elif g_max is not None:
        for r in radiators:
            r["hkl_list"] = dominant_reflections(
                r["crystal"], n_families=10**9, B_ang2=r["B_ang2"], g_max_invang=g_max
            )
        if case.get("layer_radiators") is not None:
            out["hkl_list"] = radiators[0]["hkl_list"]
    supports = [
        _automatic_cxr_grid_policy(
            sweep,
            dict(crystal_params(r["crystal"], families), hkl_list=r["hkl_list"]),
            float(case["E0_keV"]),
        )
        for r in radiators
    ]
    bandwidth = built["line_grid_policy"]["bandwidth"]
    production = (float(bandwidth["start_eV"]), float(bandwidth["stop_eV"]))
    if case["crystal"] == "hopg":
        production = (float(supports[0].start_eV), float(supports[0].stop_eV))
    needed = (min(p.start_eV for p in supports), max(p.stop_eV for p in supports))
    return out, production, tuple(float(x) for x in needed)


def reflection_chunks(case, max_reflections):
    """``case`` split into cases of at most ``max_reflections`` reflections per radiator.

    Incoherent line densities add over reflections, so the chunk spectra sum to
    the one-call spectrum on the same energy axis. Chunking reflections, not
    energy, keeps the scored window -- production keeps only lines resonant
    within the window +-20 % of its span and anchors its absorption tabulation
    to it, so an energy chunk drops neighbouring sinc tails. Production (#1167)
    batches the coupling-table transient but still holds the full
    (reflections x tabulation) tables and refuses lists past the device
    budget; tens of thousands of reflections over a wide support need this.
    """
    from math import ceil

    layered = case.get("layer_radiators") is not None
    radiators = case["layer_radiators"] if layered else [case]
    count = max(len(r["hkl_list"]) for r in radiators if r is not None)
    parts = max(1, ceil(count / max(1, int(max_reflections))))
    if parts == 1:
        return [case]
    width = ceil(count / parts)
    chunks = []
    for i in range(parts):
        sl = slice(i * width, (i + 1) * width)
        if not layered:
            chunks.append(dict(case, hkl_list=case["hkl_list"][sl]))
            continue
        layers = [
            None if r is None or not r["hkl_list"][sl] else dict(r, hkl_list=r["hkl_list"][sl])
            for r in radiators
        ]
        if any(r is not None for r in layers):
            first = next(r for r in layers if r is not None)
            chunks.append(dict(case, layer_radiators=layers, hkl_list=first["hkl_list"]))
    return chunks


def chunked_lines(scorer, chunks, grid):
    """Sum of ``scorer.lines(grid)`` over reflection-chunk cases; restores ``scorer.case``."""
    whole = scorer.case
    try:
        total = np.zeros(grid.size)
        for chunk in chunks:
            scorer.case = chunk
            total += scorer.lines(grid)
        return total
    finally:
        scorer.case = whole


def score_families(case, sweep, ladder, ns, reference_spacing, wide_spacing, sync, max_reflections):
    """Score each reflection set on the saved transport against the largest.

    Band (0.5--10 keV, node ``reference_spacing``): yield, centroid, peak, FWHM,
    100 eV mass shape. Wide (node ``wide_spacing`` over every radiator's own
    resolved support at the largest set): full-support yield, so lines leaving
    the band -- or a film-only production axis -- are not missed. The yield
    the production support omits is recorded per set. Each observable keeps
    its own margin.
    """
    from pyrite.energy_grid.convergence_case import CaseLadder

    tp = ladder.transport
    cases = {n: families_case(case, sweep, n) for n in ns}
    start = min(needed[0] for *_, needed in cases.values())
    stop = max(needed[1] for *_, needed in cases.values())
    wide = axis(start, stop, wide_spacing, "node")
    band = axis(*BAND, reference_spacing, "node")
    rows = []
    for n in ns:
        fcase, production, needed = cases[n]
        scorer = CaseLadder(fcase, transport=tp)  # one device copy of the segments
        chunks = reflection_chunks(fcase, max_reflections)
        row = dict(
            n=n,
            reflections=len(fcase["hkl_list"]),
            production_support_eV=production,
            all_radiator_support_eV=needed,
        )
        if fcase.get("layer_radiators") is not None:
            row["layer_reflections"] = [
                None if r is None else len(r["hkl_list"]) for r in fcase["layer_radiators"]
            ]
        for label, grid, edges in (("band", band, None), ("wide", wide, band_edges(start, stop))):
            sync()
            begin = time.perf_counter()
            density = chunked_lines(scorer, chunks, grid)
            sync()
            row[label] = dict(
                nodes=int(grid.size),
                reflection_chunks=len(chunks),
                first_s=time.perf_counter() - begin,
                metrics=metrics(grid, density, "node", edges=edges),
            )
            if label == "wide":
                lo, hi = production
                inside = window_mass(grid, density, lo, hi)
                row["outside_production_support_fraction"] = 1.0 - inside / float(
                    np.trapezoid(density, grid)
                )
        rows.append(row)
    top = rows[-1]
    for row in rows:
        for label in ("band", "wide"):
            row[label]["versus_largest"] = versus(row[label]["metrics"], top[label]["metrics"])
    return dict(
        axis="00l-orders" if case["crystal"] == "hopg" else "n_families",
        band_spacing_eV=reference_spacing,
        wide_spacing_eV=wide_spacing,
        wide_support_eV=[start, stop],
        max_reflections_per_call=max_reflections,
        margins=MARGINS,
        families=rows,
    )


#: Continuum-only margins (#1143 Plan 4): integral observables at 1e-3, local
#: (floor, edge, tip, log-band shape, line/background) at 1e-2. Numerical only.
CONTINUUM_MARGINS = dict(
    yield_per_primary=1e-3,
    centroid_eV=1e-3,
    log_band_l1=1e-2,
    floor_mass=1e-2,
    edge_mass=1e-2,
    edge_jump=1e-2,
    tip_mass=1e-2,
    line_background_ratio=1e-2,
)
#: Relative half-width of the edge windows, the floor window and the tip window.
EDGE_WINDOW, FLOOR_WINDOW, TIP_WINDOW = 0.01, 0.25, 0.05
LOG_BANDS = 32


def window_mass(grid, density, lo, hi):
    """Exact integral of the piecewise-linear node interpolant over ``[lo, hi]``.

    Zero outside the grid support, so a window past the tip carries no mass.
    """
    lo, hi = max(lo, float(grid[0])), min(hi, float(grid[-1]))
    if hi <= lo:
        return 0.0
    inner = grid[(grid > lo) & (grid < hi)]
    x = np.r_[lo, inner, hi]
    return float(np.trapezoid(np.interp(x, grid, density, left=0.0, right=0.0), x))


def continuum_features(case):
    """Floor, tip and located edges of every radiating/absorbing crystal in the scene.

    The production continuum refines only ``case['crystal']`` plus the detector
    path, so a layered substrate's own edges are listed here (flagged
    ``refined=False``) to measure what that omission costs.
    """
    from pyrite._continuum_grid import (
        DETECTOR_PATH_ELEMENTS,
        EDGE_KIND,
        continuum_refinement_marks,
    )
    from pyrite.energy_grid.floor import photon_continuum_floor_eV
    from pyrite.energy_grid.refine import case_endpoint_eV

    medium = str(case["crystal"])
    floor, tip = photon_continuum_floor_eV(medium), case_endpoint_eV(case)
    crystals = [medium] + [
        str(r["crystal"]) for r in case.get("layer_radiators") or () if r["crystal"] != medium
    ]
    edges = {}
    for index, crystal in enumerate(dict.fromkeys(crystals)):
        path = DETECTOR_PATH_ELEMENTS if index == 0 else ()
        marks, _ = continuum_refinement_marks(crystal, floor, tip, path_elements=path)
        for mark in marks:
            if mark.kind == EDGE_KIND:
                centre = 0.5 * (mark.anchors_eV[0] + mark.anchors_eV[-1])
                key = f"{mark.label}@{centre:.1f}"
                edges.setdefault(key, dict(edge_eV=centre, refined=index == 0))
    return dict(medium=medium, floor_eV=floor, tip_eV=tip, edges=edges)


def continuum_metrics(grid, density, features):
    """Intrinsic continuum observables on the node interpolant (no detector)."""
    if not np.isfinite(density).all() or np.trapezoid(density, grid) <= 0:
        raise ValueError("nonpositive continuum yield or nonfinite continuum")
    floor, tip = features["floor_eV"], features["tip_eV"]
    total = window_mass(grid, density, floor, tip)
    x = np.r_[floor, grid[(grid > floor) & (grid < tip)], tip]
    first = float(np.trapezoid(x * np.interp(x, grid, density, left=0, right=0), x))
    bands = np.geomspace(floor, tip, LOG_BANDS + 1)
    edges = {}
    for key, edge in features["edges"].items():
        e = edge["edge_eV"]
        below = window_mass(grid, density, e * (1 - EDGE_WINDOW), e)
        above = window_mass(grid, density, e, e * (1 + EDGE_WINDOW))
        edges[key] = dict(below=below, above=above, jump=above / below if below > 0 else None)
    return dict(
        yield_per_primary=total,
        centroid_eV=first / total,
        log_band_masses=[
            window_mass(grid, density, a, b) for a, b in zip(bands[:-1], bands[1:], strict=True)
        ],
        floor_mass=window_mass(grid, density, floor, floor * (1 + FLOOR_WINDOW)),
        tip_mass=window_mass(grid, density, tip * (1 - TIP_WINDOW), tip),
        edges=edges,
    )


def continuum_versus(candidate, reference):
    """Signed relative changes, one verdict per observable; edges report their worst."""

    def rel(c, r):
        return None if c is None or r is None or r == 0 else (c - r) / r

    out = {k: rel(candidate[k], reference[k]) for k in ("yield_per_primary", "centroid_eV")}
    out["floor_mass"] = rel(candidate["floor_mass"], reference["floor_mass"])
    out["tip_mass"] = rel(candidate["tip_mass"], reference["tip_mass"])
    cb, rb = (np.asarray(m["log_band_masses"], dtype=float) for m in (candidate, reference))
    out["log_band_l1"] = float(np.abs(cb / cb.sum() - rb / rb.sum()).sum())
    per_edge = {}
    for key, ref in reference["edges"].items():
        cand = candidate["edges"][key]
        per_edge[key] = dict(
            below=rel(cand["below"], ref["below"]),
            above=rel(cand["above"], ref["above"]),
            jump=rel(cand["jump"], ref["jump"]),
        )
    out["edges"] = per_edge

    def worst(values):
        values = [abs(v) for v in values if v is not None]
        return max(values) if values else None

    out["edge_mass"] = worst(v[s] for v in per_edge.values() for s in ("below", "above"))
    out["edge_jump"] = worst(v["jump"] for v in per_edge.values())
    out["within_margin"] = {
        key: None if out.get(key) is None else bool(abs(out[key]) <= margin)
        for key, margin in CONTINUUM_MARGINS.items()
        if key != "line_background_ratio"
    }
    return out


def continuum_grid(material, tip_eV, step, refined):
    """Production refined continuum at ``step`` in ln E, or its unrefined geometric twin.

    The twin spans the same floor to ``tip * exp(step)`` with the nearest node
    count, so edges and the tip fall wherever the lattice puts them.
    """
    from pyrite.energy_grid.floor import geometric_continuum_grid, photon_continuum_floor_eV
    from pyrite.energy_grid.refine import automatic_continuum_grid

    if refined:
        return automatic_continuum_grid(material, tip_eV, log_step=step)
    stop = tip_eV * np.exp(step)
    count = int(round(np.log(stop / photon_continuum_floor_eV(material)) / step)) + 1
    return geometric_continuum_grid(material, stop, count)


def score_continuum(case, ladder, steps, reference_divisors, line_spacing, sync):
    """Plan 4: continuum floor/edges/tip and line/background on one saved transport.

    Every candidate is scored against a refined reference at
    ``default_step / reference_divisors[1]``, whose own refinement pair is
    reported first. The dominant-line window is fixed from a node line reference
    on the band, so only the continuum changes between candidates; its truth is
    the continuum evaluated directly on that window, not interpolated.
    """
    from pyrite._continuum_grid import CONTINUUM_LOG_STEP
    from pyrite.energy_grid.bandwidth_check import _dominant_fwhm_eV
    from pyrite.energy_grid.convergence_case import CaseLadder

    features = continuum_features(case)
    medium, tip = features["medium"], features["tip_eV"]
    tp = ladder.transport
    node = CaseLadder(dict(case, line_quadrature="node"), transport=tp)

    def evaluate(grid):
        sync()
        begin = time.perf_counter()
        density = node.continuum(grid)
        sync()
        return density, time.perf_counter() - begin

    line_grid = axis(*BAND, line_spacing, "node")
    line = node.lines(line_grid)
    fwhm = float(_dominant_fwhm_eV(line_grid, line))
    window = None
    if np.isfinite(fwhm):
        peak = float(line_grid[np.argmax(line)])
        window = (max(BAND[0], peak - 3 * fwhm), min(BAND[1], peak + 3 * fwhm))
        inside = line_grid[(line_grid > window[0]) & (line_grid < window[1])]
        direct_grid = np.r_[window[0], inside, window[1]]
        direct, _ = evaluate(direct_grid)
        line_mass = window_mass(line_grid, line, *window)
        direct_ratio = line_mass / window_mass(direct_grid, direct, *window)

    def row(label, step, refined):
        grid = continuum_grid(medium, tip, step, refined)
        density, first_s = evaluate(grid)
        out = dict(
            label=label,
            log_step=step,
            refined=refined,
            nodes=int(grid.size),
            support_eV=[float(grid[0]), float(grid[-1])],
            first_s=first_s,
            metrics=continuum_metrics(grid, density, features),
        )
        if window is not None:
            ratio = line_mass / window_mass(grid, density, *window)
            out["line_background_ratio"] = ratio
            out["line_background_vs_direct"] = (ratio - direct_ratio) / direct_ratio
        return out

    ladder_rows = [
        row(f"refined/{d}", CONTINUUM_LOG_STEP / d, True) for d in (1, *reference_divisors)
    ]
    reference = ladder_rows[2]["metrics"]
    candidates = [
        row(f"{'refined' if refined else 'geometric'}-{step:g}", step, refined)
        for step in steps
        for refined in (True, False)
    ]
    for item in ladder_rows + candidates:
        item["versus_reference"] = continuum_versus(item["metrics"], reference)
        if "line_background_vs_direct" in item:
            item["versus_reference"]["within_margin"]["line_background_ratio"] = bool(
                abs(item["line_background_vs_direct"]) <= CONTINUUM_MARGINS["line_background_ratio"]
            )
    production = np.asarray(tp["E_brem"], dtype=float)
    default = continuum_grid(medium, tip, CONTINUUM_LOG_STEP, True)
    return dict(
        features=features,
        margins=CONTINUUM_MARGINS,
        windows=dict(edge=EDGE_WINDOW, floor=FLOOR_WINDOW, tip=TIP_WINDOW, log_bands=LOG_BANDS),
        reference=ladder_rows[2]["label"],
        refinement_check=dict(
            coarser_vs_reference=continuum_versus(ladder_rows[1]["metrics"], reference),
            finer_vs_reference=continuum_versus(ladder_rows[3]["metrics"], reference),
        ),
        line_window_eV=None if window is None else list(window),
        line_fwhm_eV=fwhm if np.isfinite(fwhm) else None,
        line_background_direct=None if window is None else direct_ratio,
        saved_transport_grid_is_default=bool(
            production.shape == default.shape and np.array_equal(production, default)
        ),
        ladder=ladder_rows,
        candidates=candidates,
    )


def runtime():
    """Host replay must not initialize a GPU or change saved transport."""
    from pyrite._backend import BACKEND

    if BACKEND.name == "cpu":
        return lambda: None, "cpu", lambda: 0, None
    import cupy

    return (
        cupy.cuda.Device().synchronize,
        str(cupy.cuda.runtime.getDeviceProperties(0)["name"]),
        lambda: cupy.get_default_memory_pool().total_bytes(),
        device_peak_probe(cupy),
    )


def device_peak_probe(cupy):
    """One untimed call: peak CuPy-pool bytes in use, and pool reservation.

    The pool is emptied first, so its reservation after the call is the
    call's reservation high-water. A hook tracks rounded in-use bytes; it
    slows malloc, so it never wraps a timed call.
    """
    pool = cupy.get_default_memory_pool()

    class Peak(cupy.cuda.MemoryHook):
        name = "Peak"

        def __init__(self):
            self.live = self.peak = 0

        def malloc_postprocess(self, **kw):
            if kw["mem_ptr"]:
                self.live += kw["mem_size"]
                self.peak = max(self.peak, self.live)

        def free_postprocess(self, **kw):
            self.live -= kw["mem_size"]

    def probe(call, *args):
        cupy.cuda.Device().synchronize()
        pool.free_all_blocks()
        base = pool.used_bytes()
        hook = Peak()
        with hook:
            call(*args)
            cupy.cuda.Device().synchronize()
        return dict(
            device_peak_in_use_bytes=int(base + hook.peak),
            device_pool_reserved_after_emptied_call_bytes=int(pool.total_bytes()),
            device_in_use_before_call_bytes=int(base),
        )

    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", choices=["hopg", "silicon", "mos2-on-si"])
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--arm",
        choices=[
            "baseline",
            "half-cap",
            "quarter-cap",
            "cut25",
            "cut35",
            "cut100",
            "mixed-precision",
            "mixed-elastic",
        ],
        default="baseline",
    )
    parser.add_argument("--seed", type=int, default=642501)
    parser.add_argument("--histories", type=int, default=64)
    parser.add_argument("--replay", type=Path, help="trusted same-source transport pickle")
    parser.add_argument("--transport-only", action="store_true")
    parser.add_argument(
        "--node-spacings",
        type=lambda text: [float(x) for x in text.split(",") if x],
        default=[0.75, 0.375, 0.1875],
        help="comma-separated node spacings in eV (reference-refinement ladder); empty for none",
    )
    parser.add_argument(
        "--bin-mean-spacings",
        type=lambda text: [float(x) for x in text.split(",") if x],
        default=None,
        help="comma-separated bin-mean spacings in eV; baseline default 0.375,3,12; empty for none",
    )
    parser.add_argument("--energy-keV", type=float, default=None, help="override beam energy")
    parser.add_argument(
        "--thickness-ang", type=float, default=None, help="override single-slab thickness"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="scoring calls per grid (first + warm); 1 for slow host-grouped replays",
    )
    parser.add_argument(
        "--mosaic-mc-nodes",
        type=int,
        default=None,
        help="score with the exact MC mosaic route at the catalog FWHM and N nodes per axis",
    )
    parser.add_argument(
        "--policy-arms",
        type=lambda text: [x for x in text.split(",") if x],
        default=[],
        help=f"comma-separated automatic grid policies: {','.join(POLICY_ARMS)}",
    )
    parser.add_argument(
        "--policy-reference-spacing",
        type=float,
        default=0.09375,
        help="node reference spacing (eV) for policy arms; 0.09375 passed refinement",
    )
    parser.add_argument(
        "--continuum",
        action="store_true",
        help="score continuum floor/edges/tip and line/background (refined vs geometric)",
    )
    parser.add_argument(
        "--continuum-steps",
        type=lambda text: [float(x) for x in text.split(",") if x],
        default=[0.01, 0.005, 0.0025],
        help="comma-separated ln-E candidate steps; each scored refined and unrefined",
    )
    parser.add_argument(
        "--continuum-reference-divisors",
        type=lambda text: [int(x) for x in text.split(",") if x],
        default=[4, 8, 16],
        help="three divisors of the default step: coarser check, reference, finer check",
    )
    parser.add_argument(
        "--families",
        type=lambda text: [x if x.startswith("g") else int(x) for x in text.split(",") if x],
        default=[],
        help="ascending reflection ladder: n_families counts (HOPG: (00l) orders) or g<G> "
        "reciprocal cutoffs in 1/angstrom; last is reference",
    )
    parser.add_argument(
        "--families-wide-spacing",
        type=float,
        default=0.375,
        help="node spacing (eV) of the full-support yield axis for --families",
    )
    parser.add_argument(
        "--families-max-reflections",
        type=int,
        default=1024,
        help="reflections per radiator per line call for --families (bounds device tables)",
    )
    args = parser.parse_args()
    if len(set(args.families)) != len(args.families):
        parser.error("--families points must be distinct")
    if len(args.continuum_reference_divisors) != 3:
        parser.error("--continuum-reference-divisors needs exactly three values")
    unknown = sorted(set(args.policy_arms) - set(POLICY_ARMS))
    if unknown:
        parser.error(f"unknown policy arms {unknown}")
    os.environ["PYRITE_MC_TRANSPORT_PRECISION"] = (
        "mixed" if args.arm == "mixed-precision" else "fp64"
    )
    from pyrite._backend import BACKEND
    from pyrite._energy_grid_encoding import encode_energy_grid
    from pyrite.energy_grid.convergence import segment_fingerprint
    from pyrite.energy_grid.convergence_case import CaseLadder, build_ladder_case

    if args.material == "mos2-on-si":
        if args.energy_keV is not None or args.thickness_ang is not None:
            parser.error("the layered scene is frozen; no energy/thickness overrides")
        case = layered_case(args.histories, args.seed)
    else:
        energy, thickness = (30.0, 20000.0) if args.material == "hopg" else (300.0, 100000.0)
        energy = energy if args.energy_keV is None else args.energy_keV
        thickness = thickness if args.thickness_ang is None else args.thickness_ang
        case = build_ladder_case(
            args.material,
            energy,
            45,
            180,
            thickness_ang=thickness,
            n_electrons=args.histories,
            seed=args.seed,
        )
    case.pop("line_grid_policy", None)
    carrier = encode_energy_grid(axis(500, 10000, 3, "node"))
    case.update(
        E_grid=carrier,
        E_grid_line=carrier,
        line_quadrature="node",
        emission="incoherent",
        transport_mode="single-scattering",
        transport_precision="fp64",
        max_dE_frac=0.0,
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        straggling=False,
        elastic_model="elsepa",
        bremsstrahlung_model="bremslib",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        atomic_electron_deflection="kawrakow",
    )
    if args.arm in ("half-cap", "quarter-cap"):
        case["max_dE_frac"] = 0.00025 if args.arm == "half-cap" else 0.000125
    elif args.arm in ("cut25", "cut35", "cut100"):
        case["inelastic_cutoff_eV"] = {"cut25": 25.0, "cut35": 35.0, "cut100": 100.0}[args.arm]
    elif args.arm == "mixed-precision":
        case["transport_precision"] = "mixed"
    elif args.arm == "mixed-elastic":
        # Use the model's own validated tuning defaults; admission may refuse.
        case.update(transport_mode="mixed", mixed_c1=0.05, mixed_c2=0.05)
    sync, device, pool_bytes, peak_probe = runtime()
    args.output.mkdir(parents=True, exist_ok=False)
    sync()
    start = time.perf_counter()
    if args.replay:
        saved = pickle.loads(args.replay.read_bytes())  # noqa: S301 - explicit trusted study artifact
        case = saved["case"]
        if (
            case["crystal"] != CRYSTAL.get(args.material, args.material)
            or case["seed"] != args.seed
            or case["Ne"] != args.histories
        ):
            raise ValueError("replay material/seed/population mismatch")
        ladder = CaseLadder(case, transport=saved["transport"])
    else:
        ladder = CaseLadder(case, transport_core="cuda")
    sync()
    transport_s = time.perf_counter() - start
    payload = args.output / "transport.pkl"
    payload.write_bytes(pickle.dumps(dict(case=case, transport=ladder.transport), protocol=5))
    score = runpy.run_path(str(Path(__file__).with_name("straggling_defaults") / "study.py"))[
        "primary_scores"
    ]
    primary, residual = score(ladder.segments, args.histories, float(case["E0_keV"]))
    np.savez_compressed(args.output / "primary.npz", **primary)
    records = []
    if args.mosaic_mc_nodes is not None:
        case = mosaic_mc_case(case, args.mosaic_mc_nodes)
    arms = [("node", h) for h in args.node_spacings]
    bin_mean = args.bin_mean_spacings
    if bin_mean is None:
        bin_mean = [0.375, 3.0, 12.0] if args.arm == "baseline" else []
    arms += [("bin-mean", h) for h in bin_mean]
    report = dict(
        schema="pyrite.defaults-spectral-pilot.v1",
        material=args.material,
        E0_keV=float(case["E0_keV"]),
        thickness_ang=float(case["thickness_ang"]),
        mosaic_mc_nodes=args.mosaic_mc_nodes,
        mosaic_mc_fwhm_rad=case.get("mosaic_mc_fwhm_rad"),
        arm=args.arm,
        seed=args.seed,
        histories=args.histories,
        backend=BACKEND.name,
        device=device,
        replay_transport_sha256=None
        if args.replay is None
        else hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        band_eV=[500, 10000],
        source_revision=Path("SOURCE_SHA").read_text().strip(),
        transport_s_including_first_setup=None if args.replay else transport_s,
        replay_adoption_s=transport_s if args.replay else None,
        fingerprint=ladder.fingerprint,
        transport_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
        qualification="sensitivity only; no accuracy or equal-precision verdict",
        diagnostics=dict(
            rows=len(ladder.segments["L_ang"]),
            energy_residual_keV=residual,
            primary_means={k: float(v.mean()) for k, v in primary.items()},
        ),
        spectra=records,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.transport_only:
        return
    for quadrature, spacing in arms:
        scoring_case = dict(case, line_quadrature=quadrature)
        sync()
        begin = time.perf_counter()
        scorer = CaseLadder(scoring_case, transport=ladder.transport)
        sync()
        setup_s = time.perf_counter() - begin
        grid = axis(500, 10000, spacing, quadrature)
        density, timing = timed_calls(scorer.lines, grid, args.repeats, sync, peak_probe)
        label = f"{quadrature}-{spacing}"
        np.savez_compressed(args.output / f"{label}.npz", grid_eV=grid, density=density)
        record = dict(
            quadrature=quadrature,
            spacing_eV=spacing,
            nodes=len(grid),
            scorer_setup_s=setup_s,
            **timing,
            metrics=metrics(grid, density, quadrature),
        )
        records.append(record)
        assert segment_fingerprint(ladder.segments) == ladder.fingerprint
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["device_pool_reserved_bytes_not_peak"] = pool_bytes()
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(dict(arm=args.arm, spectrum=label, metrics=record["metrics"])), flush=True)
    if args.continuum:
        report["continuum_arms"] = score_continuum(
            case,
            ladder,
            args.continuum_steps,
            args.continuum_reference_divisors,
            args.policy_reference_spacing,
            sync,
        )
        assert segment_fingerprint(ladder.segments) == ladder.fingerprint
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if args.families:
        sweep = scene_sweep(args.material, float(case["E0_keV"]), float(case["thickness_ang"]))
        report["family_arms"] = score_families(
            case,
            sweep,
            ladder,
            args.families,
            args.policy_reference_spacing,
            args.families_wide_spacing,
            sync,
            args.families_max_reflections,
        )
        assert segment_fingerprint(ladder.segments) == ladder.fingerprint
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if args.policy_arms:
        sweep = scene_sweep(args.material, float(case["E0_keV"]), float(case["thickness_ang"]))
        report["policy_arms"] = score_policies(
            case,
            sweep,
            ladder,
            args.policy_arms,
            args.policy_reference_spacing,
            sync,
            repeats=args.repeats,
            probe=peak_probe,
        )
        assert segment_fingerprint(ladder.segments) == ladder.fingerprint
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
