"""Local refinement of an over-budget ``sinc-oversampled`` axis (#1257).

The ``sinc-oversampled`` default places nodes ``W / K`` apart over the whole
bandwidth (``W`` the sinc-Nyquist step, ``K`` the quadrature's step divisor,
:func:`~pyrite._cxr_grid_policy.uniform_step_divisor`: 8 under ``node``,
about 10.5 under ``bin-mean`` since #1265).
On wide axes whose lines occupy a small part of the band -- the Si 1 MeV
kinematic fallback spans 50 eV to 966 keV with every scored line between 5.6
and 6.8 keV -- that uniform axis exceeds the point budget. This module builds
the replacement the resolver takes instead of refusing: the ``W / K`` step
around every line narrower than ``K h_b``, and the maximum automatic spacing
``h_b`` (``max_spacing_eV``, 3 eV) everywhere else; a wider line already has
``K`` backbone nodes per first-zero width. ``W`` is a line-width quantile, so
nothing in its rationale applies where no line resonates, and a ``W``
backbone alone exceeds the budget on a 1 MeV kinematic band.

Beyond every line's halo the backbone is graded (#1271): each step grows by
:data:`NODE_LOCAL_GROWTH` of its distance from the nearest protected edge (a
fine window, a wide line's halo, a DBR/Kossel Bragg window or a band edge), up
to :data:`NODE_LOCAL_GRADED_CAP_EV`. The halos widen to pay for it. Cases whose
line axis carries spectra the line population does not describe (channeling,
transition radiation, mixed transport) keep the ungraded backbone.

Node quadrature on a piecewise axis is not exact at a spacing join (#192). Each
window therefore extends a halo :func:`node_join_halo_eV` beyond its line, far
enough that the joins its union leaves cost each line at most
:data:`DEFAULT_NODE_JOIN_LIMIT` of its mass per side. The bound and its
derivation are in
``docs/validation/beam-transport/line-grid-node-local-refinement.md``. Bin means
telescope over contiguous edges (#192; to within the ``sinc-bin-far-envelope``
term), so under ``bin-mean`` the same halos are conservative.

Validation: line-grid-node-local-refinement
"""

import numpy as np

from ..._backend import REAL
from ..._cxr_grid_policy import (
    OVERSAMPLED_RESOLUTION_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    CxrGridToleranceError,
    uniform_step_divisor,
)
from ..._cxr_windows import FeatureSeed, build_window_plan
from ..._grid_semantics import resolution_num, validate_backend_coordinates

__all__ = [
    "DEFAULT_NODE_JOIN_LIMIT",
    "NODE_LOCAL_BIN_EV",
    "NODE_LOCAL_FAR_SHARE",
    "NODE_LOCAL_GRADED_CAP_EV",
    "NODE_LOCAL_GROWTH",
    "NODE_LOCAL_REVISION",
    "NODE_LOCAL_ROUTE",
    "NODE_LOCAL_SOURCE",
    "SINCSQ_CURVATURE_INTEGRAL",
    "graded_coordinates",
    "node_graded_halos_eV",
    "node_join_halo_eV",
    "node_local_axis",
    "node_local_seeds",
    "uniform_over_budget",
]

#: Grid-record route of the local axis.
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
NODE_LOCAL_REVISION = 4
#: Relative growth ``rho`` of a graded backbone step per unit distance from the
#: nearest protected edge: ``h <= h_b + rho s``, geometric ``h_b (1 + rho)**k``.
NODE_LOCAL_GROWTH = 0.1
#: Largest graded backbone step [eV]: the Peano join bound needs a step cap.
NODE_LOCAL_GRADED_CAP_EV = 10.0
#: Share of the per-side join limit a Peano-halo line spends on the
#: grading-free distance ``D``; the rest bounds its join-free distance ``d``.
NODE_LOCAL_FAR_SHARE = 0.5
#: The extra opt-out an over-budget refusal names on the measured bandwidth (#1265).
_LOCAL_HINT = "Or, with bin-mean, resonance-local refines only near resonances. "


#: Upper bound on ``int_0^inf |g''(u)| du`` for ``g(u) = sin(u)**2 / u**2``
#: (numerically 1.4743), the Peano-kernel constant of a join on a line's centre.
SINCSQ_CURVATURE_INTEGRAL = 1.5


def _peano_distance(width, step_sq, eta):
    """Distance ``d`` at which ``step_sq pi G(pi d / w) / (8 w**2) <= eta``."""
    with np.errstate(divide="ignore"):
        target = 8.0 * width**2 * eta / (np.pi * step_sq)
    u0 = np.where(target >= SINCSQ_CURVATURE_INTEGRAL, 0.0, 3.0 / np.minimum(target, 1.0))
    return u0 * width / np.pi


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
    return np.minimum(tail, _peano_distance(width, float(backbone_eV) ** 2 + central**2, eta))


def node_graded_halos_eV(
    width_eV,
    central_eV,
    backbone_eV: float,
    join_limit: float,
    growth: float,
    cap_eV: float,
    far_share: float = NODE_LOCAL_FAR_SHARE,
):
    """``(d, D)``: join-free and grading-free distances each side of a line.

    The line keeps spacing ``h_c`` to its nearest join at ``d``, then at most
    ``h_b`` to ``D >= d``, then intervals no longer than ``H`` (``cap_eV``) nor
    than ``h_b + rho (x - D)`` at their near end ``x`` (``rho`` = ``growth``).
    The join error on that side is then at most ``M`` times either

    * ``(w / pi**2) ((1 + h_b / (2 d)) / d + rho / (2 D))`` -- tail mass, or
    * ``pi ((h_b**2 + h_c**2) G(pi d / w) + (H**2 - h_b**2) G(pi D / w)) / (8 w**2)``
      -- Peano kernel, ``G`` as in :func:`node_join_halo_eV`.

    The tail option takes ``d = D = (1 + rho / 2) w / (pi**2 eta) + h_b / 2``.
    The Peano option spends ``(1 - far_share) eta`` on
    ``d`` (the Peano distance of :func:`node_join_halo_eV` at that limit) and
    ``far_share eta`` on ``D``. Each line takes the option with
    the smaller ``d``. Limiting case: ``H = h_b`` gives ``D = d``.

    Validation: line-grid-node-local-refinement
    """
    width = np.asarray(width_eV, dtype=float)
    central = np.asarray(central_eV, dtype=float)
    eta, backbone = float(join_limit), float(backbone_eV)
    tail = (1.0 + 0.5 * float(growth)) * width / (np.pi**2 * eta) + 0.5 * backbone
    near = _peano_distance(width, backbone**2 + central**2, (1.0 - float(far_share)) * eta)
    far_sq = max(float(cap_eV) ** 2 - backbone**2, 0.0)
    far = _peano_distance(width, far_sq, float(far_share) * eta) if far_sq else np.zeros_like(near)
    use_tail = tail <= near
    join = np.where(use_tail, tail, near)
    return join, np.where(use_tail, tail, np.maximum(near, far))


def _line_intervals(populations, start, stop, fine, backbone, narrow_below, bounds, bin_eV):
    """Per-line ``(lo, hi, fine?)`` arrays: fine windows and join-exclusion zones.

    A line narrower than ``narrow_below`` asks for ``fine`` over its halo,
    rounded out to ``bin_eV``; a wider one, already at least ``K`` backbone
    nodes per width, asks only that no join fall within its halo. ``bounds``
    holds the join limit, growth and cap; with a cap the halos are
    :func:`node_graded_halos_eV` and ``far`` collects each line's
    grading-free ``[E - D, E + D]``, else :func:`node_join_halo_eV`.
    """
    from ..spectrum.cxr_seeds import _population_blocks

    lows, highs, fines, far = [], [], [], []
    lines = 0
    eta, cap = bounds["join_limit"], bounds["cap_eV"]
    for energy, weight, width in _population_blocks(populations):
        keep = np.isfinite(energy) & np.isfinite(width) & (width > 0.0) & (weight > 0.0)
        if not keep.any():
            continue
        energy, width = energy[keep], width[keep]
        narrow = width < narrow_below
        central = np.where(narrow, fine, backbone)
        if cap is None:
            halo = node_join_halo_eV(width, central, backbone, eta)
        else:
            halo, reach = node_graded_halos_eV(width, central, backbone, eta, bounds["growth"], cap)
            outer = (reach > 0.0) & (energy + reach > start) & (energy - reach < stop)
            far += [
                (max(e - r, start), min(e + r, stop))
                for e, r in zip(energy[outer], reach[outer], strict=True)
            ]
        lo, hi = energy - halo, energy + halo
        lo = np.where(narrow, start + np.floor((lo - start) / bin_eV) * bin_eV, lo)
        hi = np.where(narrow, start + np.ceil((hi - start) / bin_eV) * bin_eV, hi)
        inside = (hi > start) & (lo < stop) & (hi > lo)
        lows.append(np.maximum(lo[inside], start))
        highs.append(np.minimum(hi[inside], stop))
        fines.append(narrow[inside])
        lines += int(keep.sum())
    return lows, highs, fines, far, lines


def _components(lows, highs, fines):
    """Disjoint ``(lo, hi, fine?)`` runs: overlapping intervals chained together.

    Intervals that overlap (not merely touch) form components; a component
    holding a fine window is refined whole, so every join lies outside every
    line's halo, and a component of exclusion zones alone holds no join: it
    stays on the ungraded backbone.

    Validation: line-grid-node-local-refinement
    """
    if not lows:
        return []
    lo, hi, fine = np.concatenate(lows), np.concatenate(highs), np.concatenate(fines)
    if not lo.size:
        return []
    order = np.argsort(lo, kind="stable")
    lo, hi, fine = lo[order], hi[order], fine[order]
    reach = np.maximum.accumulate(hi)
    starts = np.flatnonzero(np.r_[True, lo[1:] >= reach[:-1]])
    ends = np.r_[starts[1:] - 1, lo.size - 1]
    has_fine = np.add.reduceat(fine.astype(np.int64), starts) > 0
    return [
        (float(lo[a]), float(reach[b]), bool(keep))
        for a, b, keep in zip(starts, ends, has_fine, strict=True)
    ]


def node_local_seeds(
    direction_populations,
    *,
    start_eV: float,
    stop_eV: float,
    fine_eV: float,
    backbone_eV: float,
    oversampling: float,
    join_limit: float = DEFAULT_NODE_JOIN_LIMIT,
    bin_eV: float = NODE_LOCAL_BIN_EV,
    growth: float = 0.0,
    cap_eV: float | None = None,
):
    """Fine windows over every direction's lines, merged, with every join outside every halo.

    Each line with finite positive width and weight narrower than
    ``oversampling * backbone_eV`` asks for ``fine_eV`` over its resonance plus
    its join-free halo on each side (:func:`node_graded_halos_eV` with a
    ``cap_eV``, else :func:`node_join_halo_eV`); a
    wider line keeps at least ``oversampling`` backbone nodes per width and
    asks only that its halo hold no join, so its halo is refined only when it
    overlaps a fine window. Returns ``(seeds, summary)``; the summary's
    ``protected_eV`` lists the backbone-only halo runs a graded backbone must
    leave at ``backbone_eV``.

    Validation: line-grid-node-local-refinement
    """
    start, stop = float(start_eV), float(stop_eV)
    fine, backbone = float(fine_eV), float(backbone_eV)
    if not 0.0 < fine <= backbone:
        raise ValueError("node-local spacing needs 0 < fine_eV <= backbone_eV")
    if not 0.0 < float(join_limit) < 1.0:
        raise ValueError("join_limit must lie in (0, 1)")
    narrow_below = float(oversampling) * backbone
    bounds = {"join_limit": float(join_limit), "growth": float(growth), "cap_eV": cap_eV}
    lows, highs, fines, far, lines = [], [], [], [], 0
    for populations in direction_populations:
        lo, hi, narrow, reach, count = _line_intervals(
            populations, start, stop, fine, backbone, narrow_below, bounds, float(bin_eV)
        )
        lows += lo
        highs += hi
        fines += narrow
        far += reach
        lines += count
    components = _components(lows, highs, fines)
    seeds = [
        FeatureSeed(
            source=NODE_LOCAL_SOURCE,
            label=f"{lo:.0f}-{hi:.0f} eV",
            centre_eV=lo,
            below_eV=0.0,
            above_eV=hi - lo,
            spacing_eV=fine,
        )
        for lo, hi, has_fine in components
        if has_fine
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
        "protected_eV": _merged_runs(
            far + [(lo, hi) for lo, hi, has_fine in components if not has_fine]
        ),
    }
    return seeds, summary


def _merged_runs(intervals):
    """Sorted union of closed ``[lo, hi]`` runs; touching runs merge."""
    runs = []
    for lo, hi in sorted((float(lo), float(hi)) for lo, hi in intervals if hi > lo):
        if runs and lo <= runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], hi)
        else:
            runs.append([lo, hi])
    return runs


def _graded_gap(lo: float, hi: float, backbone: float, growth: float, cap: float):
    """Interior nodes of ``(lo, hi)``, each step at most ``min(cap, h_b + rho s)``.

    ``s`` is the distance of a step's near end from the nearer gap end. Steps
    grow geometrically from both ends; the middle is split evenly at the step
    its nearer end allows. Deterministic in its arguments.

    Validation: line-grid-node-local-refinement
    """

    def step(s):
        return min(cap, backbone + growth * s)

    left, right = [lo], [hi]
    while right[-1] - left[-1] > step(left[-1] - lo) + step(hi - right[-1]):
        if left[-1] - lo <= hi - right[-1]:
            left.append(left[-1] + step(left[-1] - lo))
        else:
            right.append(right[-1] - step(hi - right[-1]))
    a, b = left[-1], right[-1]
    count = max(1, int(np.ceil((b - a) / step(min(a - lo, hi - b)))))
    middle = np.linspace(a, b, count + 1)[1:-1]
    return np.concatenate([left[1:], middle, right[:0:-1]])


def _backbone_parts(lo: float, hi: float, runs, backbone: float):
    """``(lo, hi, graded?)`` parts of one backbone piece split by protected runs.

    A gap shorter than ``2 h_b`` stays protected and a protected part shorter
    than ``h_b`` widens to ``h_b`` (more protection never hurts a bound), so no
    part is a sliver; adjacent protected parts merge.

    Validation: line-grid-node-local-refinement
    """
    clipped = []
    for run_lo, run_hi in runs:
        a, b = max(run_lo, lo), min(run_hi, hi)
        if b < a:
            continue
        if b - a < backbone:
            middle = 0.5 * (a + b)
            a, b = max(lo, middle - 0.5 * backbone), min(hi, middle + 0.5 * backbone)
        clipped.append((a, b))
    edges = [lo]
    for a, b in _merged_runs(clipped) if clipped else []:
        edges += [a, b]
    edges.append(hi)
    parts = []
    for index in range(len(edges) - 1):
        a, b = edges[index], edges[index + 1]
        graded = index % 2 == 0 and b - a >= 2.0 * backbone
        if b <= a:
            continue
        if parts and not graded and not parts[-1][2]:
            parts[-1] = (parts[-1][0], b, False)
        else:
            parts.append((a, b, graded))
    return parts


def graded_coordinates(plan, grading) -> np.ndarray:
    """Node-local coordinates of ``plan`` with its backbone graded by ``grading``.

    ``grading`` (the record's ``local_axis["grading"]``) gives ``growth``,
    ``cap_eV`` and the ``protected_eV`` runs. Fine pieces and protected runs
    keep their plan spacing; every other stretch of backbone, the band ends
    included, steps out from ``h_b`` at its ends by at most
    ``min(cap_eV, h_b + growth s)``. ``None`` reproduces the plan.

    Validation: line-grid-node-local-refinement
    """
    if grading is None:
        return plan.coordinates()
    backbone = float(plan.backbone_spacing_eV)
    growth, cap = float(grading["growth"]), float(grading["cap_eV"])
    runs = _merged_runs(tuple(run) for run in grading["protected_eV"])
    parts = [np.array([plan.start_eV])]
    for lo, hi, spacing in plan.pieces:
        if spacing < backbone:
            parts.append(np.linspace(lo, hi, resolution_num(lo, hi, spacing))[1:])
            continue
        for a, b, graded in _backbone_parts(lo, hi, runs, backbone):
            if graded:
                parts.append(np.r_[_graded_gap(a, b, backbone, growth, cap), b])
            else:
                parts.append(np.linspace(a, b, resolution_num(a, b, backbone))[1:])
    return np.concatenate(parts)


def _local_candidate(direction_populations, band, protected_eV):
    """``(plan, summary, grading, grid)`` of one node-local axis.

    ``protected_eV`` ``None`` builds the ungraded #1257 axis; a sequence of
    extra protected runs (DBR/Kossel windows) builds the graded one (#1271).
    """
    start, stop, fine, backbone, oversampling = band
    graded = protected_eV is not None
    seeds, summary = node_local_seeds(
        direction_populations,
        start_eV=start,
        stop_eV=stop,
        fine_eV=fine,
        backbone_eV=backbone,
        oversampling=oversampling,
        growth=NODE_LOCAL_GROWTH if graded else 0.0,
        cap_eV=NODE_LOCAL_GRADED_CAP_EV if graded else None,
    )
    protected = summary.pop("protected_eV")
    grading = None
    if graded:
        grading = {
            "growth": NODE_LOCAL_GROWTH,
            "cap_eV": NODE_LOCAL_GRADED_CAP_EV,
            "far_share": NODE_LOCAL_FAR_SHARE,
            "protected_eV": _merged_runs([*protected, *protected_eV]),
        }
    plan = build_window_plan(start, stop, backbone, seeds)
    return plan, summary, grading, graded_coordinates(plan, grading)


def node_local_axis(
    payload,
    start,
    stop,
    target_step,
    direction_populations,
    uniform_num,
    *,
    graded: bool = True,
    protected_eV=(),
):
    """Local ``sinc-oversampled`` axis over ``[start, stop]``, or a refusal.

    ``target_step`` is the case's sinc-Nyquist feature width. The window
    spacing is ``min(target_step, max_spacing_eV)`` divided by the payload's
    oversampling, the step the uniform axis would have taken; the backbone is
    ``max_spacing_eV``. ``uniform_num`` is that uniform axis's
    over-budget count, recorded with the reason. With ``graded`` the backbone
    outside every halo, fine window and ``protected_eV`` run (DBR/Kossel Bragg
    windows) grows by :data:`NODE_LOCAL_GROWTH` up to
    :data:`NODE_LOCAL_GRADED_CAP_EV` (#1271), recorded under
    ``local_axis["grading"]``. Raises
    :class:`CxrGridToleranceError` when the local axis also exceeds the budget.
    Returns ``(grid, record)`` with the route under ``"local_axis"``.

    Validation: line-grid-node-local-refinement
    """
    resolution = payload["resolution"]
    oversampling = uniform_step_divisor(payload)
    backbone = float(resolution["max_spacing_eV"])
    nyquist = min(float(target_step), backbone)
    fine = nyquist / oversampling
    max_points = int(resolution["max_points"])
    band = (start, stop, fine, backbone, oversampling)
    plan, summary, grading, grid = _local_candidate(direction_populations, band, None)
    if graded:
        # The graded backbone widens fine halos; keep it only where it saves nodes.
        ungraded = int(grid.size)
        candidate = _local_candidate(direction_populations, band, protected_eV)
        if candidate[3].size < ungraded:
            plan, summary, grading, grid = candidate
        summary["ungraded_points"] = ungraded
    rtol = dict(payload["rtol"])
    governing = min(rtol, key=lambda name: rtol[name])
    measured = payload["bandwidth"]["policy"] == RESONANCE_BANDWIDTH_POLICY
    reason = (
        f"a uniform {fine:.4g} eV axis over [{start:g}, {stop:g}] eV needs {uniform_num} "
        f"coordinates, above the {max_points}-point budget"
    )
    if grid.size > max_points:
        raise CxrGridToleranceError(
            f"automatic line-grid resolution cannot meet rtol={rtol[governing]:g} for "
            f"{governing!r}: {reason}, and its node-local refinement ({summary['windows']} "
            f"windows, {summary['window_span_eV']:g} eV at {fine:.4g} eV on a {backbone:.4g} eV "
            f"{'graded ' if grading else ''}backbone) needs {grid.size}. Raise the budget "
            "with PYRITE_ENERGY_GRID_MAX_POINTS, "
            "narrow the bandwidth, or supply an explicit grid. Or select the "
            f"{oversampling:.3g}x coarser sinc-nyquist resolution, which keeps line area but not "
            "peak height or FWHM: in Python, Numerics(line_resolution='sinc-nyquist'); in a "
            "profile, `pyrite profile numerics cxr-grid set NAME --resolution sinc-nyquist`. "
            f"{_LOCAL_HINT if measured else ''}The grid is not coarsened automatically."
        )
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
        "num": int(grid.size),
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
            "grading": grading,
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
    target = step / uniform_step_divisor(payload)
    if not np.isfinite(target) or target <= 0.0:
        return None
    num = resolution_num(float(start), float(stop), target)
    return num if num > int(resolution["max_points"]) else None
