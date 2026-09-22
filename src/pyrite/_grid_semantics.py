"""Energy-grid semantics shared by spectrum and detector consumers.

A package-root leaf, like ``_energy_grid_encoding``: every physics-core package
(``detectors``, ``montecarlo``, ``results``) consumes it, so it must not sit
inside a domain package -- an edge from ``detectors`` into ``energy_grid`` would
put both inside the repository's driver import cycle.
:mod:`pyrite.energy_grid.semantics` is the public facade.

The project's photon-energy conventions are documented in
[Energy-grid semantics](../../docs/physics/radiation-physics/energy-grid-semantics.md).
This module carries the one piece of that contract the code has to enforce:
several consumers read a *single* spacing (``E[1] - E[0]``) and apply it across
the whole grid. That is correct only for a uniform grid, so those consumers call
:func:`require_uniform_grid` and refuse a nonuniform grid explicitly instead of
returning a silently wrong number.

Nothing here changes a uniform-grid result: :func:`require_uniform_grid` returns
exactly ``float(E[1] - E[0])``, the value its callers already used.
"""

import hashlib

import numpy as np

__all__ = [
    "NonuniformEnergyGridError",
    "grid_identity",
    "is_uniform_grid",
    "node_bin_edges_and_widths",
    "rebin_piecewise_constant_density",
    "resolution_num",
    "require_uniform_grid",
    "spacing_spread",
    "validate_backend_coordinates",
    "validate_backend_spacing",
    "zero_based_detector_edges",
]

#: Relative spread of adjacent spacings still accepted as "uniform", measured
#: after the representation floor below is subtracted. Any deliberately graded
#: or logarithmic grid is off by orders of magnitude, so the exact value is not
#: delicate.
DEFAULT_UNIFORM_RTOL = 1.0e-9

#: Adjacent nodes are only stored to the backend's working precision, so a grid
#: that IS uniform by construction still carries a few ulp of rounding jitter in
#: its differences -- and that jitter is large relative to the step whenever the
#: step is small compared with the absolute energy (a 2.5e-5 eV step at 1600 eV
#: is only ~1e8 float64 ulp wide, giving ~1e-8 relative jitter). The spread is
#: therefore credited this many ulp of the largest node before being compared
#: against ``rtol``.
_REPRESENTATION_ULPS = 16.0


class NonuniformEnergyGridError(ValueError):
    """A uniform-only consumer was handed a nonuniform energy grid.

    Raised rather than silently applying one spacing everywhere. Issue #98
    pinned the semantics and audited the consumers; widening a consumer to real
    nonuniform support is tracked separately (issues #99-#101).
    """


def node_bin_edges_and_widths(E_grid_eV: object) -> tuple[np.ndarray, np.ndarray]:
    """Return midpoint bin edges and local widths for evaluation nodes.

    The outer edges use one-sided half-widths.  This is the explicit
    node-to-histogram conversion defined by :eq:`eq-grid-midpoint-edges` in the
    energy-grid semantics documentation.  A density multiplied by the returned
    widths is therefore a per-bin mass, including on a nonuniform grid.
    """
    E = np.asarray(E_grid_eV, dtype=float)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("energy grid must be 1-D with at least two nodes")
    if not np.all(np.isfinite(E)) or np.any(np.diff(E) <= 0.0):
        raise ValueError("energy-grid nodes must be finite and strictly increasing")
    edges = np.empty(E.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (E[:-1] + E[1:])
    edges[0] = E[0] - 0.5 * (E[1] - E[0])
    edges[-1] = E[-1] + 0.5 * (E[-1] - E[-2])
    return edges, np.diff(edges)


def zero_based_detector_edges(E_grid_eV: object) -> tuple[np.ndarray, np.ndarray, bool]:
    """Midpoint source-bin edges anchored to a detector's explicit 0 eV boundary.

    A real instrument has a channel that starts at zero recorded energy -- the
    Timepix output histogram begins at 0 eV so charge-loss events below the
    incident energy are still scored -- and that boundary belongs to the
    detector, not to the source mesh. Once a continuum grid carries a positive
    floor (:mod:`pyrite.energy_grid.floor`), the two no longer coincide, and the
    detector must keep its own zero rather than inherit the floor.

    Returns ``(edges, widths, padded)`` where ``edges[0]`` is exactly ``0.0``.
    Two distinct cases, because they are physically different:

    * The midpoint reflection of the first node lands **below** zero. There is
      no such thing as a negative-energy half-bin, so the outer edge is clamped
      up to zero and the first node keeps a single, narrower bin. ``padded`` is
      ``False`` and the caller's density array is unchanged.
    * The reflection lands **above** zero, which is the ordinary case for a grid
      with a positive floor. The first node's own bin is correct as it stands
      and must not be widened -- stretching it down to zero would multiply that
      node's density by the extra width and invent photons. So a *separate*
      explicit channel ``[0, edges[0])`` is prepended instead. ``padded`` is
      ``True`` and the caller must prepend one zero to its density array to stay
      aligned. That channel carries no source mass: the continuum model has no
      support below its floor.

    The clamp branch is the behaviour ``TimepixResponse`` already relied on; the
    pad branch is what a positive continuum floor additionally requires.
    """
    edges, widths = node_bin_edges_and_widths(E_grid_eV)
    if edges[0] < 0.0:
        edges = edges.copy()
        edges[0] = 0.0
        return edges, np.diff(edges), False
    if edges[0] > 0.0:
        edges = np.concatenate(([0.0], edges))
        return edges, np.diff(edges), True
    return edges, widths, False


def rebin_piecewise_constant_density(
    source_edges_eV: object,
    source_density: object,
    target_edges_eV: object,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Integrate a histogram density onto new edges without losing mass.

    Returns target-bin masses plus masses below and above the target window.
    Density is assumed constant inside each source bin. The cumulative-integral
    implementation is O(N + M) in storage, rather than constructing a dense
    source-by-target overlap matrix.
    """
    source_edges = np.asarray(source_edges_eV, dtype=float)
    target_edges = np.asarray(target_edges_eV, dtype=float)
    density = np.asarray(source_density, dtype=float)
    for name, edges in (("source", source_edges), ("target", target_edges)):
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(f"{name} edges must be 1-D with at least two entries")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError(f"{name} edges must be finite and strictly increasing")
    if density.shape != (source_edges.size - 1,):
        raise ValueError(
            f"source density shape {density.shape} != source bins {(source_edges.size - 1,)}"
        )

    source_masses = density * np.diff(source_edges)
    cumulative = np.concatenate(([0.0], np.cumsum(source_masses)))

    def integral_to(points: np.ndarray) -> np.ndarray:
        clipped = np.clip(points, source_edges[0], source_edges[-1])
        bins = np.searchsorted(source_edges, clipped, side="right") - 1
        bins = np.clip(bins, 0, density.size - 1)
        values = cumulative[bins] + density[bins] * (clipped - source_edges[bins])
        values = np.where(points <= source_edges[0], 0.0, values)
        return np.where(points >= source_edges[-1], cumulative[-1], values)

    target_cumulative = integral_to(target_edges)
    below = float(integral_to(np.asarray([target_edges[0]]))[0])
    above = float(cumulative[-1] - integral_to(np.asarray([target_edges[-1]]))[0])
    return np.diff(target_cumulative), (below, above)


def spacing_spread(E_grid_eV) -> float:
    """Relative spread of the grid spacings, net of the representation floor.

    ``ptp(diff(E))`` is credited ``_REPRESENTATION_ULPS`` ulp of the largest node
    (in the input array's own floating dtype) and then divided by the median
    spacing, so a grid that is uniform by construction reports exactly ``0.0``
    whatever its step-to-energy ratio. Returns ``inf`` for a grid with no
    positive median spacing.
    """
    raw = np.asarray(E_grid_eV)
    dtype = raw.dtype if np.issubdtype(raw.dtype, np.floating) else np.dtype(float)
    E = raw.astype(float, copy=False)
    if E.ndim != 1 or E.size < 3:
        # Fewer than three nodes define at most one spacing, so nothing can
        # disagree with anything: treat it as uniform.
        return 0.0
    d = np.diff(E)
    reference = float(np.median(np.abs(d)))
    if not np.isfinite(reference) or reference <= 0.0:
        return float("inf")
    floor = _REPRESENTATION_ULPS * float(np.spacing(dtype.type(np.max(np.abs(E)))))
    return float(max(float(np.ptp(d)) - floor, 0.0) / reference)


def is_uniform_grid(E_grid_eV, *, rtol: float = DEFAULT_UNIFORM_RTOL) -> bool:
    """Whether every adjacent spacing agrees to within ``rtol``."""
    return spacing_spread(E_grid_eV) <= rtol


def require_uniform_grid(
    E_grid_eV,
    *,
    consumer: str,
    remedy: str | None = None,
    rtol: float = DEFAULT_UNIFORM_RTOL,
) -> float:
    """Return the single spacing ``float(E[1] - E[0])``, or refuse the grid.

    Parameters
    ----------
    E_grid_eV
        One-dimensional, strictly increasing photon-energy coordinate in eV.
    consumer
        Dotted name of the calling code, quoted in the error message.
    remedy
        Optional caller-specific sentence appended to the error.
    rtol
        Relative tolerance on the spread of adjacent spacings.

    Raises
    ------
    NonuniformEnergyGridError
        If the spacings disagree by more than ``rtol``.
    """
    if np.asarray(E_grid_eV).ndim != 1 or np.asarray(E_grid_eV).size < 2:
        raise ValueError(f"{consumer}: energy grid must be 1-D with at least two nodes")
    # spread is measured on the ORIGINAL dtype: the representation floor depends
    # on the precision the grid is actually stored (and indexed) in.
    spread = spacing_spread(E_grid_eV)
    E = np.asarray(E_grid_eV, dtype=float)
    if spread > rtol:
        d = np.diff(E)
        message = (
            f"{consumer} requires a UNIFORM energy grid: it applies the single "
            f"spacing E[1] - E[0] = {float(E[1] - E[0]):.6g} eV across the whole "
            f"grid, but the {d.size} spacings range over "
            f"[{float(d.min()):.6g}, {float(d.max()):.6g}] eV "
            f"(relative spread {spread:.3g} > rtol {rtol:.3g}). "
            "Nonuniform (for example logarithmic) photon grids are not yet "
            "supported by this consumer."
        )
        if remedy:
            message = f"{message} {remedy}"
        raise NonuniformEnergyGridError(message)
    return float(E[1] - E[0])


def grid_identity(E_grid_eV) -> tuple[int, float, float, str]:
    """Cache-key prefix that identifies the COMPLETE grid coordinates.

    ``(size, first, last)`` alone does not: two grids can share all three and
    still differ node by node (a linear and a logarithmic grid over the same
    interval, for instance), which silently returns one grid's cached detector
    response for the other. The trailing digest is over the exact float64 bytes,
    so distinct coordinates always produce distinct keys.
    """
    E = np.ascontiguousarray(np.asarray(E_grid_eV, dtype=float))
    digest = hashlib.blake2b(E.tobytes(), digest_size=16).hexdigest()
    return (E.size, round(float(E[0]), 6), round(float(E[-1]), 6), digest)


# Generated-spacing policy lives here, not in ``energy_grid.bounds``: the runner
# needs it while choosing a diagnostic line grid, and an edge from
# ``montecarlo`` into ``energy_grid`` would close an import cycle
# (``energy_grid.derive`` already imports ``montecarlo.runner``).
def resolution_num(start_eV: float, stop_eV: float, maximum_spacing_eV: float) -> int:
    """Endpoint-inclusive count whose actual spacing never exceeds the limit."""
    values = (float(start_eV), float(stop_eV), float(maximum_spacing_eV))
    if not all(np.isfinite(value) for value in values) or maximum_spacing_eV <= 0.0:
        raise ValueError("grid bounds and maximum spacing must be finite; spacing must be positive")
    if stop_eV <= start_eV:
        raise ValueError("grid stop must be greater than start")
    return int(np.ceil((stop_eV - start_eV) / maximum_spacing_eV)) + 1


def validate_backend_spacing(
    start_eV: float,
    stop_eV: float,
    num: int,
    *,
    dtype=np.float32,
    safety_ulps: float = 8.0,
) -> float:
    """Validate a generated linspace against backend coordinate precision."""
    if not np.isfinite(safety_ulps) or safety_ulps <= 0.0:
        raise ValueError("backend ULP safety factor must be finite and positive")
    if int(num) != num or num < 2:
        raise ValueError("grid num must be an integer of at least 2")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.kind != "f":
        raise ValueError("backend grid dtype must be floating point")
    spacing = (float(stop_eV) - float(start_eV)) / (int(num) - 1)
    largest = resolved_dtype.type(max(abs(float(start_eV)), abs(float(stop_eV))))
    minimum = float(safety_ulps) * abs(float(np.spacing(largest)))
    if spacing < minimum:
        raise ValueError(
            f"requested line-grid spacing {spacing:g} eV is below the {safety_ulps:g}-ULP "
            f"safety floor {minimum:g} eV for {resolved_dtype.name} at {float(largest):g} eV; "
            "run with PYRITE_FP64=1 or choose a physically justified coarser tolerance"
        )
    cast = np.linspace(start_eV, stop_eV, int(num), dtype=float).astype(resolved_dtype)
    cast_steps = np.diff(cast)
    collapsed = int(np.count_nonzero(cast_steps <= 0.0))
    if collapsed:
        raise ValueError(
            f"generated line grid is not strictly increasing after {resolved_dtype.name} cast "
            f"({collapsed} collapsed intervals); run with PYRITE_FP64=1 or choose a physically "
            "justified coarser tolerance"
        )
    minimum_cast_step = float(cast_steps.min())
    if minimum_cast_step < minimum:
        raise ValueError(
            f"generated line-grid intervals reach {minimum_cast_step:g} eV after "
            f"{resolved_dtype.name} cast, below the {safety_ulps:g}-ULP safety floor "
            f"{minimum:g} eV; run with PYRITE_FP64=1 or choose a physically justified "
            "coarser tolerance"
        )
    return spacing


def validate_backend_coordinates(
    E_grid_eV,
    *,
    dtype=np.float32,
    safety_ulps: float = 8.0,
) -> float:
    """Validate explicit coordinates against backend precision, interval by interval.

    The nonuniform counterpart of :func:`validate_backend_spacing`. Each
    interval must span ``safety_ulps`` ulp of ``dtype`` at its own larger node,
    before and after the cast: a fine window at 200 eV is judged by the
    precision at 200 eV, not at the top of the axis. On a uniform grid this is
    weaker than :func:`validate_backend_spacing`, which judges every interval at
    the largest node. Returns the smallest interval after the cast, in eV.
    """
    if not np.isfinite(safety_ulps) or safety_ulps <= 0.0:
        raise ValueError("backend ULP safety factor must be finite and positive")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.kind != "f":
        raise ValueError("backend grid dtype must be floating point")
    E = np.asarray(E_grid_eV, dtype=float)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("energy grid must be 1-D with at least two nodes")
    if not np.all(np.isfinite(E)) or np.any(np.diff(E) <= 0.0):
        raise ValueError("energy-grid nodes must be finite and strictly increasing")
    magnitude = np.maximum(np.abs(E[:-1]), np.abs(E[1:])).astype(resolved_dtype)
    floor = float(safety_ulps) * np.abs(np.spacing(magnitude)).astype(float)

    def _refuse(steps: np.ndarray, stage: str) -> None:
        short = steps < floor
        if not short.any():
            return
        first = int(np.argmax(short))
        raise ValueError(
            f"{int(short.sum())} line-grid intervals are below the {safety_ulps:g}-ULP "
            f"safety floor for {resolved_dtype.name} {stage}; the first is "
            f"{float(steps[first]):g} eV at {float(E[first + 1]):g} eV and needs at least "
            f"{float(floor[first]):g} eV; run with PYRITE_FP64=1 or choose a physically "
            "justified coarser tolerance"
        )

    _refuse(np.diff(E), "before the cast")
    cast_steps = np.diff(E.astype(resolved_dtype)).astype(float)
    collapsed = int(np.count_nonzero(cast_steps <= 0.0))
    if collapsed:
        raise ValueError(
            f"line grid is not strictly increasing after {resolved_dtype.name} cast "
            f"({collapsed} collapsed intervals); run with PYRITE_FP64=1 or choose a physically "
            "justified coarser tolerance"
        )
    _refuse(cast_steps, "after the cast")
    return float(cast_steps.min())
