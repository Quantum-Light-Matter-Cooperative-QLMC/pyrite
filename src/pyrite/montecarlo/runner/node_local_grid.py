"""Node-quadrature local refinement of an over-budget ``sinc-oversampled`` axis (#1257).

The ``sinc-oversampled`` default places nodes ``W / K`` apart over the whole
bandwidth (``W`` the sinc-Nyquist step, ``K =`` :data:`LINE_NYQUIST_OVERSAMPLING`).
On wide axes whose lines occupy a small part of the band -- the Si 1 MeV
kinematic fallback spans 50 eV to 966 keV with every scored line between 5.6
and 6.8 keV -- that uniform axis exceeds the point budget. This module builds
the replacement the resolver takes instead of refusing: the ``W / K`` step
around every line narrower than ``K h_b``, and the maximum automatic spacing
``h_b`` (``max_spacing_eV``, 3 eV) everywhere else; a wider line already has
``K`` backbone nodes per first-zero width. ``W`` is a line-width quantile, so
nothing in its rationale applies where no line resonates, and a ``W``
backbone alone exceeds the budget on a 1 MeV kinematic band.

Node quadrature on a piecewise axis is not exact at a spacing join (#192). Each
window therefore extends a halo :func:`node_join_halo_eV` beyond its line, far
enough that the joins its union leaves cost each line at most
:data:`DEFAULT_NODE_JOIN_LIMIT` of its mass per side. The bound and its
derivation are in
``docs/validation/beam-transport/line-grid-node-local-refinement.md``.

Validation: line-grid-node-local-refinement
"""

import numpy as np

from ..._backend import REAL
from ..._cxr_grid_policy import OVERSAMPLED_RESOLUTION_POLICY, CxrGridToleranceError
from ..._cxr_windows import FeatureSeed, build_window_plan
from ..._grid_semantics import resolution_num, validate_backend_coordinates

__all__ = [
    "DEFAULT_NODE_JOIN_LIMIT",
    "NODE_LOCAL_BIN_EV",
    "NODE_LOCAL_REVISION",
    "NODE_LOCAL_ROUTE",
    "NODE_LOCAL_SOURCE",
    "SINCSQ_CURVATURE_INTEGRAL",
    "node_join_halo_eV",
    "node_local_axis",
    "node_local_seeds",
    "uniform_over_budget",
]

#: Grid-record route of the local axis, beside ``sinc-shape``'s ``shape_axis``.
NODE_LOCAL_ROUTE = "node-local"
NODE_LOCAL_SOURCE = "pxr-node-local"
#: Largest share of one line's mass the joins on one side of it may misplace
#: under node quadrature; twice this per line in total. Charged, like the
#: ``resonance-local`` halo share (``DEFAULT_LOCAL_HALO_LIMIT``), to the
#: feature-window row of ``tbl-line-budget-allocation``, which an axis without
#: feature windows does not otherwise spend.
DEFAULT_NODE_JOIN_LIMIT = 1.0e-4
#: Window edges round outward to this grid from the bandwidth start, so nearby
#: populations share an axis; rounding outward only lengthens a halo.
NODE_LOCAL_BIN_EV = 100.0
#: Bumped whenever the same population would give a different local axis; it
#: keys the speed cache of ``sinc-oversampled`` grids.
NODE_LOCAL_REVISION = 2


#: Upper bound on ``int_0^inf |g''(u)| du`` for ``g(u) = sin(u)**2 / u**2``
#: (numerically 1.4743), the Peano-kernel constant of a join on a line's centre.
SINCSQ_CURVATURE_INTEGRAL = 1.5


def node_join_halo_eV(width_eV, central_eV, backbone_eV: float, join_limit: float):
    """Join-free distance each side of a line that bounds its node join error.

    For a ``sinc**2`` line of first-zero width ``w`` and mass ``M``, sampled at
    spacing ``h_c <= w`` up to the nearest spacing join on one side, at distance
    ``d``, and at no spacing above ``h_b`` beyond it, the node-quadrature error
    of every join on that side is at most ``M`` times either

    * ``(w / (pi**2 d)) (1 + h_b / (2 d))`` -- the tail-mass bound, or
    * ``(h_b**2 + h_c**2) pi G(pi d / w) / (8 w**2)`` -- the Peano-kernel bound,
      ``G(u) = int_u^inf |g''|``, at most :data:`SINCSQ_CURVATURE_INTEGRAL`
      and at most ``2/u + 2/u**2 + 2/u**3``.

    The tail bound holds at ``d = w / (pi**2 eta) + h_b / 2``. With
    ``G* = 8 w**2 eta / (pi (h_b**2 + h_c**2))`` the Peano bound holds at every
    ``d`` once ``G* >=`` :data:`SINCSQ_CURVATURE_INTEGRAL`, at ``d = 3 w / pi``
    once ``G* >= 1``, and at ``d = 3 w / (pi G*)`` below. The smaller distance is
    returned. Limiting cases: a line much narrower than ``h_b`` takes the tail
    halo ``~ w / (pi**2 eta)`` of ``resonance-local``; a line wider than
    ``h_b sqrt(pi 1.5 / (8 eta))`` needs none.

    Validation: line-grid-node-local-refinement
    """
    width = np.asarray(width_eV, dtype=float)
    central = np.asarray(central_eV, dtype=float)
    eta = float(join_limit)
    tail = width / (np.pi**2 * eta) + 0.5 * float(backbone_eV)
    target = 8.0 * width**2 * eta / (np.pi * (float(backbone_eV) ** 2 + central**2))
    u0 = np.where(target >= SINCSQ_CURVATURE_INTEGRAL, 0.0, 3.0 / np.minimum(target, 1.0))
    return np.minimum(tail, u0 * width / np.pi)


def _line_intervals(populations, start, stop, fine, backbone, narrow_below, join_limit, bin_eV):
    """Per-line ``(lo, hi, fine?)`` arrays: fine windows and join-exclusion zones.

    A line narrower than ``narrow_below`` asks for ``fine`` over its halo,
    rounded out to ``bin_eV``; a wider one, already at least ``K`` backbone
    nodes per width, asks only that no join fall within its halo.
    """
    from ..spectrum.cxr_seeds import _population_blocks

    lows, highs, fines = [], [], []
    lines = 0
    for energy, weight, width in _population_blocks(populations):
        keep = np.isfinite(energy) & np.isfinite(width) & (width > 0.0) & (weight > 0.0)
        if not keep.any():
            continue
        energy, width = energy[keep], width[keep]
        narrow = width < narrow_below
        halo = node_join_halo_eV(width, np.where(narrow, fine, backbone), backbone, join_limit)
        lo, hi = energy - halo, energy + halo
        lo = np.where(narrow, start + np.floor((lo - start) / bin_eV) * bin_eV, lo)
        hi = np.where(narrow, start + np.ceil((hi - start) / bin_eV) * bin_eV, hi)
        inside = (hi > start) & (lo < stop) & (hi > lo)
        lows.append(np.maximum(lo[inside], start))
        highs.append(np.minimum(hi[inside], stop))
        fines.append(narrow[inside])
        lines += int(keep.sum())
    return lows, highs, fines, lines


def _fine_union(lows, highs, fines):
    """Disjoint fine ``(lo, hi)`` runs: overlapping intervals chained to a fine window.

    Intervals that overlap (not merely touch) form components; a component
    holding a fine window is refined whole, so every join lies outside every
    line's halo, and a component of exclusion zones alone holds no join.

    Validation: line-grid-node-local-refinement
    """
    if not lows:
        return []
    lo, hi, fine = np.concatenate(lows), np.concatenate(highs), np.concatenate(fines)
    if not fine.any():
        return []
    order = np.argsort(lo, kind="stable")
    lo, hi, fine = lo[order], hi[order], fine[order]
    reach = np.maximum.accumulate(hi)
    starts = np.flatnonzero(np.r_[True, lo[1:] >= reach[:-1]])
    ends = np.r_[starts[1:] - 1, lo.size - 1]
    has_fine = np.add.reduceat(fine.astype(np.int64), starts) > 0
    return [
        (float(lo[a]), float(reach[b]))
        for a, b, keep in zip(starts, ends, has_fine, strict=True)
        if keep
    ]


def node_local_seeds(
    direction_populations,
    *,
    start_eV: float,
    stop_eV: float,
    fine_eV: float,
    backbone_eV: float,
    oversampling: int,
    join_limit: float = DEFAULT_NODE_JOIN_LIMIT,
    bin_eV: float = NODE_LOCAL_BIN_EV,
):
    """Fine windows over every direction's lines, merged, with every join outside every halo.

    Each line with finite positive width and weight narrower than
    ``oversampling * backbone_eV`` asks for ``fine_eV`` over its resonance plus
    :func:`node_join_halo_eV` on each side; a wider line keeps at least
    ``oversampling`` backbone nodes per width and asks only that its halo hold
    no join, so its halo is refined only when it overlaps a fine window.
    Returns ``(seeds, summary)``.

    Validation: line-grid-node-local-refinement
    """
    start, stop = float(start_eV), float(stop_eV)
    fine, backbone = float(fine_eV), float(backbone_eV)
    if not 0.0 < fine <= backbone:
        raise ValueError("node-local spacing needs 0 < fine_eV <= backbone_eV")
    if not 0.0 < float(join_limit) < 1.0:
        raise ValueError("join_limit must lie in (0, 1)")
    narrow_below = int(oversampling) * backbone
    lows, highs, fines, lines = [], [], [], 0
    for populations in direction_populations:
        lo, hi, narrow, count = _line_intervals(
            populations, start, stop, fine, backbone, narrow_below, float(join_limit), float(bin_eV)
        )
        lows += lo
        highs += hi
        fines += narrow
        lines += count
    windows = _fine_union(lows, highs, fines)
    seeds = [
        FeatureSeed(
            source=NODE_LOCAL_SOURCE,
            label=f"{lo:.0f}-{hi:.0f} eV",
            centre_eV=lo,
            below_eV=0.0,
            above_eV=hi - lo,
            spacing_eV=fine,
        )
        for lo, hi in windows
    ]
    summary = {
        "fine_spacing_eV": fine,
        "backbone_spacing_eV": backbone,
        "narrow_below_eV": narrow_below,
        "join_limit": float(join_limit),
        "bin_eV": float(bin_eV),
        "lines": lines,
        "windows": len(seeds),
        "window_span_eV": float(sum(seed.above_eV for seed in seeds)),
    }
    return seeds, summary


def node_local_axis(payload, start, stop, target_step, direction_populations, uniform_num):
    """Local ``sinc-oversampled`` axis over ``[start, stop]``, or a refusal.

    ``target_step`` is the case's sinc-Nyquist feature width. The window
    spacing is ``min(target_step, max_spacing_eV)`` divided by the payload's
    oversampling, the step the uniform axis would have taken; the backbone is
    ``max_spacing_eV``. ``uniform_num`` is that uniform axis's
    over-budget count, recorded with the reason. Raises
    :class:`CxrGridToleranceError` when the local axis also exceeds the budget.
    Returns ``(grid, record)`` with the route under ``"local_axis"``.

    Validation: line-grid-node-local-refinement
    """
    resolution = payload["resolution"]
    oversampling = int(resolution["oversampling"])
    backbone = float(resolution["max_spacing_eV"])
    nyquist = min(float(target_step), backbone)
    fine = nyquist / oversampling
    max_points = int(resolution["max_points"])
    seeds, summary = node_local_seeds(
        direction_populations,
        start_eV=start,
        stop_eV=stop,
        fine_eV=fine,
        backbone_eV=backbone,
        oversampling=oversampling,
    )
    plan = build_window_plan(start, stop, backbone, seeds)
    rtol = dict(payload["rtol"])
    governing = min(rtol, key=lambda name: rtol[name])
    reason = (
        f"a uniform {fine:.4g} eV axis over [{start:g}, {stop:g}] eV needs {uniform_num} "
        f"coordinates, above the {max_points}-point budget"
    )
    if plan.num > max_points:
        raise CxrGridToleranceError(
            f"automatic line-grid resolution cannot meet rtol={rtol[governing]:g} for "
            f"{governing!r}: {reason}, and its node-local refinement ({summary['windows']} "
            f"windows, {summary['window_span_eV']:g} eV at {fine:.4g} eV on a {backbone:.4g} eV "
            f"backbone) needs {plan.num}. Raise the budget with PYRITE_ENERGY_GRID_MAX_POINTS, "
            "narrow the bandwidth, or supply an explicit grid. Or select the "
            f"{oversampling}x coarser sinc-nyquist resolution, which keeps line area but not "
            "peak height or FWHM: in Python, Numerics(line_resolution='sinc-nyquist'); in a "
            "profile, `pyrite profile numerics cxr-grid set NAME --resolution sinc-nyquist`. "
            "The grid is not coarsened automatically."
        )
    grid = plan.coordinates()
    try:
        smallest = validate_backend_coordinates(
            grid, dtype=REAL, safety_ulps=float(resolution["backend_safety_ulps"])
        )
    except ValueError as exc:
        raise CxrGridToleranceError(
            f"automatic line-grid resolution cannot meet rtol={rtol[governing]:g} for "
            f"{governing!r} in backend precision: {exc}"
        ) from None
    record = {
        "schema": int(payload.get("schema", 1)),
        "bandwidth_policy": str(payload["bandwidth"]["policy"]),
        "resolution_policy": str(resolution["policy"]),
        "start_eV": float(start),
        "stop_eV": float(stop),
        "num": int(plan.num),
        "oversampling": oversampling,
        "nyquist_spacing_eV": float(nyquist),
        "target_spacing_eV": float(fine),
        "backbone_spacing_eV": float(backbone),
        "min_spacing_eV": float(np.diff(grid).min()),
        "min_cast_spacing_eV": float(smallest),
        "rtol": rtol,
        "governing_observable": governing,
        "sources": dict(payload.get("sources", {})),
        "window_plan": plan.payload(),
        "local_axis": {
            "route": NODE_LOCAL_ROUTE,
            "reason": reason,
            "uniform_points": int(uniform_num),
            "max_points": max_points,
            "revision": NODE_LOCAL_REVISION,
            **summary,
        },
    }
    return grid, record


def uniform_over_budget(payload, start, stop, target_step) -> int | None:
    """Uniform ``sinc-oversampled`` count over ``[start, stop]`` when above budget.

    ``None`` for any other resolution, or when the uniform axis fits: the
    resolver then builds exactly the axis it built before #1257.
    """
    resolution = payload["resolution"]
    if resolution["policy"] != OVERSAMPLED_RESOLUTION_POLICY or payload.get("windows") is not None:
        return None
    step = min(float(target_step), float(resolution["max_spacing_eV"]))
    target = step / int(resolution["oversampling"])
    if not np.isfinite(target) or target <= 0.0:
        return None
    num = resolution_num(float(start), float(stop), target)
    return num if num > int(resolution["max_points"]) else None
