"""Support-aware spectral integration and explicitly requested resampling.

A stored spectral component represents photon yield only over its *declared
support*: the closed node interval ``[E_0, E_{N-1}]`` of a node-sampled density,
or ``[edges[0], edges[N]]`` of a bin-mean density. Nothing here assumes the
emission is zero beyond that domain; window parts outside the support are
reported as ``uncovered`` rather than integrated as zero.

Three operations are deliberately separate, and each takes explicit target
coordinates or edges and a boundary policy:

* :func:`interpolate_nodes` evaluates a node-sampled density at other nodes.
  It promises neither yield conservation nor shape fidelity.
* :func:`nodes_to_bins` integrates the piecewise-linear interpolant of a
  node-sampled density exactly over target bins.
* :func:`rebin_bin_means` moves bin-mean (piecewise-constant) density onto new
  edges by exact overlap, through
  :func:`pyrite._grid_semantics.rebin_piecewise_constant_density`.

Both binning operations conserve represented mass: target-bin masses plus the
masses below and above the target window equal the source's total represented
yield. Bin means are never treated as node samples.

Validation: spectral-support-yield-conservation
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._grid_semantics import rebin_piecewise_constant_density

__all__ = [
    "BinnedSpectrum",
    "SpectralYield",
    "bin_mean_yield",
    "combine_yields",
    "interpolate_nodes",
    "node_yield",
    "nodes_to_bins",
    "rebin_bin_means",
]

Interval = tuple[float, float]
BoundaryPolicy = Literal["raise", "clip"]


@dataclass(frozen=True, slots=True)
class SpectralYield:
    """Represented yield of one or more components over a photon-energy window.

    Masses carry the density's units times eV: photons / incident electron /
    sr for a stored :class:`~pyrite.results.SpectralDensity`.

    inside
        Represented mass inside ``window_eV`` and the declared support.
    below, above
        Represented mass of the support that the window crops away, below
        and above it. ``inside + below + above`` is the full represented yield.
    support_eV
        Disjoint, sorted closed intervals of declared support.
    window_eV
        The requested window, or ``None`` for each component's own support.
    uncovered_eV
        Parts of the window outside the declared support. Emission there is
        unknown, not zero; ``complete`` is ``False`` when any exist.
    """

    inside: float
    below: float
    above: float
    support_eV: tuple[Interval, ...]
    window_eV: Interval | None
    uncovered_eV: tuple[Interval, ...]

    @property
    def total(self) -> float:
        """Full represented yield over the declared support."""
        return self.inside + self.below + self.above

    @property
    def complete(self) -> bool:
        """Whether the declared support covers the whole requested window."""
        return not self.uncovered_eV


@dataclass(frozen=True, slots=True, eq=False)
class BinnedSpectrum:
    """Represented mass on explicit bins, with cropped mass kept separately.

    edges_eV
        Strictly increasing target edges in eV, length ``M + 1``.
    mass
        Represented mass in each target bin, from its supported part only.
    supported_width_eV
        Width of each target bin covered by the source support.
    below, above
        Source mass below ``edges_eV[0]`` and above ``edges_eV[-1]``.
    """

    edges_eV: np.ndarray
    mass: np.ndarray
    supported_width_eV: np.ndarray
    below: float
    above: float

    def __post_init__(self) -> None:
        for name in ("edges_eV", "mass", "supported_width_eV"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def mean_density(self) -> np.ndarray:
        """Bin-mean density; ``NaN`` where the support covers a bin only partly."""
        widths = np.diff(self.edges_eV)
        full = self.supported_width_eV == widths
        return np.where(full, self.mass / widths, np.nan)

    @property
    def total(self) -> float:
        """Target-bin mass plus cropped mass: the source's represented yield."""
        return float(self.mass.sum()) + self.below + self.above


def _finite_array(name: str, value: object) -> np.ndarray:
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError(f"{name} has masked entries; masks are never filled with zero")
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has NaN or infinite entries; they are never zeroed")
    return array


def _ordered(name: str, value: object, *, minimum: int) -> np.ndarray:
    array = _finite_array(name, value)
    if array.size < minimum:
        raise ValueError(f"{name} needs at least {minimum} entries to declare an interval")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing (no duplicates or reordering)")
    return array


def _nodes(energy_eV: object, density: object) -> tuple[np.ndarray, np.ndarray]:
    energy = _ordered("node energy", energy_eV, minimum=2)
    values = _finite_array("density", density)
    if values.shape != energy.shape:
        raise ValueError("density must match its node coordinates")
    return energy, values


def _bins(edges_eV: object, means: object) -> tuple[np.ndarray, np.ndarray]:
    edges = _ordered("bin edges", edges_eV, minimum=2)
    values = _finite_array("bin-mean density", means)
    if values.shape != (edges.size - 1,):
        raise ValueError("bin-mean density needs exactly one value per bin")
    return edges, values


def _node_cumulative(energy: np.ndarray, density: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact integral of the linear interpolant from ``energy[0]`` to clipped points."""
    widths = np.diff(energy)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * widths * (density[:-1] + density[1:]))))
    x = np.clip(points, energy[0], energy[-1])
    k = np.clip(np.searchsorted(energy, x, side="right") - 1, 0, energy.size - 2)
    dx = x - energy[k]
    slope = (density[k + 1] - density[k]) / widths[k]
    return cumulative[k] + density[k] * dx + 0.5 * slope * dx * dx


def _bin_cumulative(edges: np.ndarray, means: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact integral of piecewise-constant density from ``edges[0]`` to clipped points."""
    cumulative = np.concatenate(([0.0], np.cumsum(means * np.diff(edges))))
    x = np.clip(points, edges[0], edges[-1])
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, means.size - 1)
    return cumulative[k] + means[k] * (x - edges[k])


def _window(window_eV: Interval | None, support: Interval) -> tuple[Interval, tuple[Interval, ...]]:
    if window_eV is None:
        return support, ()
    lower, upper = (float(value) for value in window_eV)
    if np.isnan(lower) or np.isnan(upper) or lower > upper:
        raise ValueError("energy window must be an ordered (lower, upper) pair")
    uncovered = []
    if lower < support[0]:
        uncovered.append((lower, min(upper, support[0])))
    if upper > support[1]:
        uncovered.append((max(lower, support[1]), upper))
    return (lower, upper), tuple(part for part in uncovered if part[1] > part[0])


def _yield(cumulative, support: Interval, window_eV: Interval | None) -> SpectralYield:
    window, uncovered = _window(window_eV, support)
    lower, upper = np.clip(window, support[0], support[1])
    at_lower, at_upper, total = cumulative(np.array([lower, upper, support[1]]))
    return SpectralYield(
        inside=float(at_upper - at_lower),
        below=float(at_lower),
        above=float(total - at_upper),
        support_eV=(support,),
        window_eV=None if window_eV is None else window,
        uncovered_eV=uncovered,
    )


def node_yield(
    energy_eV: object, density: object, window_eV: Interval | None = None
) -> SpectralYield:
    """Integrate a node-sampled density as a piecewise-linear function.

    The declared support is ``[energy_eV[0], energy_eV[-1]]``; ``window_eV``
    (default: that support) is clipped to it, and any window part beyond it
    is reported in ``uncovered_eV``. Coordinates must be finite and strictly
    increasing with at least two nodes; a single node declares no interval.

    Validation: spectral-support-yield-conservation
    """
    energy, values = _nodes(energy_eV, density)
    support = (float(energy[0]), float(energy[-1]))
    return _yield(lambda x: _node_cumulative(energy, values, x), support, window_eV)


def bin_mean_yield(
    edges_eV: object, means: object, window_eV: Interval | None = None
) -> SpectralYield:
    """Integrate a bin-mean density as piecewise-constant over its native edges.

    The declared support is ``[edges_eV[0], edges_eV[-1]]``. A window inside
    a bin takes its overlap fraction, the exact mass under the bin-mean
    assumption (uniform density within each bin).

    Validation: spectral-support-yield-conservation
    """
    edges, values = _bins(edges_eV, means)
    support = (float(edges[0]), float(edges[-1]))
    return _yield(lambda x: _bin_cumulative(edges, values, x), support, window_eV)


def _merge(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    merged: list[Interval] = []
    for lower, upper in sorted(intervals):
        if merged and lower <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return tuple(merged)


def _subtract(window: Interval, support: tuple[Interval, ...]) -> tuple[Interval, ...]:
    gaps, cursor = [], window[0]
    for lower, upper in support:
        if lower > cursor:
            gaps.append((cursor, min(lower, window[1])))
        cursor = max(cursor, upper)
    if cursor < window[1]:
        gaps.append((cursor, window[1]))
    return tuple(gap for gap in gaps if gap[1] > gap[0])


def combine_yields(parts: Iterable[SpectralYield]) -> SpectralYield:
    """Sum component yields integrated on their own native supports.

    All parts must share one window. The combined support is the union of
    component supports. For an explicit window, ``uncovered_eV`` lists window
    parts that *no* component supports; a part some component does not cover
    is part of a component's own ``uncovered_eV``, not of the union.

    Validation: spectral-support-yield-conservation
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("combine_yields needs at least one component yield")
    windows = {part.window_eV for part in parts}
    if len(windows) != 1:
        raise ValueError("component yields must share one energy window")
    (window,) = windows
    support = _merge(interval for part in parts for interval in part.support_eV)
    return SpectralYield(
        inside=float(sum(part.inside for part in parts)),
        below=float(sum(part.below for part in parts)),
        above=float(sum(part.above for part in parts)),
        support_eV=support,
        window_eV=window,
        uncovered_eV=() if window is None else _subtract(window, support),
    )


def _target_edges(edges_eV: object, support: Interval, boundary: BoundaryPolicy) -> np.ndarray:
    if boundary not in ("raise", "clip"):
        raise ValueError("boundary must be 'raise' or 'clip'")
    edges = _ordered("target edges", edges_eV, minimum=2)
    if boundary == "raise" and (edges[0] < support[0] or edges[-1] > support[1]):
        raise ValueError(
            f"target edges [{edges[0]:g}, {edges[-1]:g}] eV extend beyond the declared "
            f"support [{support[0]:g}, {support[1]:g}] eV; pass boundary='clip' to bin only "
            "the supported part (unsupported widths are reported, never zero-filled)"
        )
    return edges


def _binned(edges: np.ndarray, cumulative, support: Interval) -> BinnedSpectrum:
    at_edges = cumulative(edges)
    total = float(cumulative(np.array([support[1]]))[0])
    return BinnedSpectrum(
        edges_eV=edges,
        mass=np.diff(at_edges),
        supported_width_eV=np.diff(np.clip(edges, support[0], support[1])),
        below=float(at_edges[0]),
        above=total - float(at_edges[-1]),
    )


def nodes_to_bins(
    energy_eV: object, density: object, edges_eV: object, *, boundary: BoundaryPolicy
) -> BinnedSpectrum:
    """Integrate a node-sampled density's linear interpolant exactly over bins.

    ``boundary='raise'`` refuses target edges beyond the declared node
    support; ``'clip'`` integrates only the supported part of each bin and
    reports the covered width. Source mass outside the target edges is
    returned as ``below``/``above``.

    Validation: spectral-support-yield-conservation
    """
    energy, values = _nodes(energy_eV, density)
    support = (float(energy[0]), float(energy[-1]))
    edges = _target_edges(edges_eV, support, boundary)
    return _binned(edges, lambda x: _node_cumulative(energy, values, x), support)


def rebin_bin_means(
    edges_eV: object, means: object, target_edges_eV: object, *, boundary: BoundaryPolicy
) -> BinnedSpectrum:
    """Move bin-mean density onto new edges by exact overlap (piecewise-constant).

    Conservative: target masses plus cropped ``below``/``above`` mass equal
    the source mass. Finer target bins inherit their parent's mean; this does
    not recover structure the source bins did not resolve.

    Validation: spectral-support-yield-conservation
    """
    edges, values = _bins(edges_eV, means)
    support = (float(edges[0]), float(edges[-1]))
    target = _target_edges(target_edges_eV, support, boundary)
    mass, (below, above) = rebin_piecewise_constant_density(edges, values, target)
    return BinnedSpectrum(
        edges_eV=target,
        mass=mass,
        supported_width_eV=np.diff(np.clip(target, support[0], support[1])),
        below=float(below),
        above=float(above),
    )


def interpolate_nodes(
    energy_eV: object,
    density: object,
    target_eV: object,
    *,
    outside: Literal["raise", "nan"],
) -> np.ndarray:
    """Linearly interpolate a node-sampled density at explicit target nodes.

    A pointwise estimate only: coarse targets promise neither yield
    conservation nor line-shape fidelity, and the result is not a bin mean.
    Targets beyond the declared support raise (``'raise'``) or are ``NaN``
    (``'nan'``); they are never zero.
    """
    if outside not in ("raise", "nan"):
        raise ValueError("outside must be 'raise' or 'nan'")
    energy, values = _nodes(energy_eV, density)
    target = np.asarray(target_eV, dtype=float)
    if np.any(np.isnan(target)):
        raise ValueError("target nodes must not be NaN")
    beyond = (target < energy[0]) | (target > energy[-1])
    if outside == "raise" and beyond.any():
        raise ValueError(
            f"target nodes extend beyond the declared support "
            f"[{energy[0]:g}, {energy[-1]:g}] eV; pass outside='nan' to mark them"
        )
    return np.interp(target, energy, values, left=np.nan, right=np.nan)
