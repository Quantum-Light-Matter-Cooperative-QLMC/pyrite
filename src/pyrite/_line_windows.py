"""Piecewise line-axis window plans: a uniform backbone plus fine local windows.

Issue #101. A uniform line grid fine enough for the narrowest shape-bearing
feature spends most of its nodes on the continuum between features, and a
global logarithmic axis grades the wrong way: the finite-time sinc width
``pi / a_w`` carries no photon-energy dependence. The plan built here keeps a
uniform backbone over ``[start, stop]`` and refines only uniform windows around
deterministically seeded features.

Seeds come from closed-form or tabulated sources -- kinematic resonance
energies, absorption edges, characteristic-line centres -- never from a pilot
mesh, which can step over a narrow resonance. Any emission component may
contribute a :class:`FeatureSeed`; the planner has no component-specific cases.

This is numerical grid policy and certifies no radiation equation. A window's
spacing is a starting heuristic that the refinement ladder has to confirm.

A package-root leaf (``numpy`` and ``_grid_semantics`` only), importable from
every layer that resolves a line grid, like ``_line_grid_policy``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from ._grid_semantics import resolution_num

__all__ = [
    "LINE_WINDOW_PLAN_SCHEMA",
    "FeatureSeed",
    "WindowPlan",
    "build_window_plan",
    "window_plan_from_payload",
]

#: Plan payload version. The plan is hashed into case identity, so bump this
#: whenever the same seeds would produce different coordinates.
LINE_WINDOW_PLAN_SCHEMA = 1


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class FeatureSeed:
    """One narrow spectral feature that needs local resolution.

    Attributes
    ----------
    source
        Contributing component, such as ``"pxr-kinematic"`` or
        ``"absorption-edge"``.
    label
        Feature name within its source, such as ``"(002)"`` or ``"C K"``.
    centre_eV
        Feature position in eV.
    below_eV, above_eV
        Window extent on each side of the centre in eV. Either may be zero,
        which makes the window one-sided.
    spacing_eV
        Largest node spacing admitted inside the window.
    anchor
        Place a node exactly at ``centre_eV``. For one-sided information -- an
        absorption edge or a kinematic endpoint -- the discontinuity must be a
        node rather than fall between two.
    """

    source: str
    label: str
    centre_eV: float
    below_eV: float
    above_eV: float
    spacing_eV: float
    anchor: bool = False

    def __post_init__(self) -> None:
        if not str(self.source) or not str(self.label):
            raise ValueError("a feature seed needs a non-empty source and label")
        centre = _finite(self.centre_eV, "seed centre_eV")
        below = _finite(self.below_eV, "seed below_eV")
        above = _finite(self.above_eV, "seed above_eV")
        spacing = _finite(self.spacing_eV, "seed spacing_eV")
        if below < 0.0 or above < 0.0:
            raise ValueError(f"seed {self.source}:{self.label} has a negative window extent")
        if spacing <= 0.0:
            raise ValueError(f"seed {self.source}:{self.label} needs a positive spacing_eV")
        if below == 0.0 and above == 0.0 and not self.anchor:
            raise ValueError(f"seed {self.source}:{self.label} has an empty window and no anchor")
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "centre_eV", centre)
        object.__setattr__(self, "below_eV", below)
        object.__setattr__(self, "above_eV", above)
        object.__setattr__(self, "spacing_eV", spacing)
        object.__setattr__(self, "anchor", bool(self.anchor))

    def payload(self) -> dict[str, Any]:
        """Canonical JSON-able form."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _seed_key(seed: FeatureSeed) -> tuple:
    return (
        seed.centre_eV,
        seed.below_eV,
        seed.above_eV,
        seed.spacing_eV,
        seed.anchor,
        seed.source,
        seed.label,
    )


Piece = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class WindowPlan:
    """Resolved piecewise-uniform line axis.

    ``pieces`` are contiguous ``(lo, hi, spacing)`` intervals covering
    ``[start_eV, stop_eV]``; each is sampled endpoint-inclusively at the
    smallest node count whose spacing does not exceed ``spacing``, and shared
    endpoints appear once. ``anchors`` are energies guaranteed to be nodes.
    ``dropped`` are seeds whose window and anchor both lie outside the
    bandwidth; they are kept for provenance, not silently discarded.
    """

    start_eV: float
    stop_eV: float
    backbone_spacing_eV: float
    seeds: tuple[FeatureSeed, ...]
    pieces: tuple[Piece, ...]
    anchors: tuple[float, ...]
    dropped: tuple[FeatureSeed, ...]

    @property
    def windowed(self) -> bool:
        """True when any piece is finer than the backbone."""
        return any(spacing < self.backbone_spacing_eV for _, _, spacing in self.pieces)

    @property
    def num(self) -> int:
        """Number of coordinates the plan produces."""
        return 1 + sum(resolution_num(lo, hi, spacing) - 1 for lo, hi, spacing in self.pieces)

    def coordinates(self) -> np.ndarray:
        """Strictly increasing float64 line-axis coordinates in eV."""
        parts = []
        for index, (lo, hi, spacing) in enumerate(self.pieces):
            nodes = np.linspace(lo, hi, resolution_num(lo, hi, spacing))
            parts.append(nodes if index == 0 else nodes[1:])
        return np.concatenate(parts)

    def payload(self) -> dict[str, Any]:
        """Canonical JSON-able plan carried in policy provenance and identity."""
        return {
            "schema": LINE_WINDOW_PLAN_SCHEMA,
            "start_eV": self.start_eV,
            "stop_eV": self.stop_eV,
            "backbone_spacing_eV": self.backbone_spacing_eV,
            "seeds": [seed.payload() for seed in self.seeds],
            "pieces": [list(piece) for piece in self.pieces],
            "anchors": list(self.anchors),
            "dropped": [f"{seed.source}:{seed.label}" for seed in self.dropped],
            "num": self.num,
        }


def build_window_plan(
    start_eV: float,
    stop_eV: float,
    backbone_spacing_eV: float,
    seeds: Iterable[FeatureSeed] = (),
) -> WindowPlan:
    """Merge feature windows onto a uniform backbone, deterministically.

    Every elementary interval between window ends and anchors takes the finest
    spacing of the windows covering it, else the backbone spacing; a window
    coarser than the backbone refines nothing. Windows are clipped to the
    bandwidth. The result depends on the set of seeds, not their order.

    Two clean-ups keep overlapping windows from placing nodes closer than any
    window asks. A non-anchor breakpoint that leaves a sliver shorter than half
    its own spacing is removed, the sliver joining the neighbour with the finer
    spacing at that neighbour's spacing; a locally finest sliver therefore
    shrinks its window by less than half of one of its own steps. Adjacent
    pieces at one spacing with no anchor between them then become one piece.
    """
    start = _finite(start_eV, "start_eV")
    stop = _finite(stop_eV, "stop_eV")
    backbone = _finite(backbone_spacing_eV, "backbone_spacing_eV")
    if stop <= start:
        raise ValueError("window plan stop must be greater than start")
    if backbone <= 0.0:
        raise ValueError("backbone_spacing_eV must be positive")

    ordered = tuple(sorted(set(seeds), key=_seed_key))
    for seed in ordered:
        if not isinstance(seed, FeatureSeed):
            raise TypeError(f"window seeds must be FeatureSeed instances, got {seed!r}")
    anchors = {start, stop}
    intervals: list[Piece] = []
    dropped: list[FeatureSeed] = []
    for seed in ordered:
        lo = max(seed.centre_eV - seed.below_eV, start)
        hi = min(seed.centre_eV + seed.above_eV, stop)
        anchored = seed.anchor and start <= seed.centre_eV <= stop
        if anchored:
            anchors.add(seed.centre_eV)
        if hi > lo:
            intervals.append((lo, hi, min(seed.spacing_eV, backbone)))
        elif not anchored:
            dropped.append(seed)

    breakpoints = sorted(anchors.union(value for lo, hi, _ in intervals for value in (lo, hi)))
    pieces = []
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        spacing = backbone
        for window_lo, window_hi, window_spacing in intervals:
            if window_lo <= lo and hi <= window_hi:
                spacing = min(spacing, window_spacing)
        pieces.append((lo, hi, spacing))

    return WindowPlan(
        start_eV=start,
        stop_eV=stop,
        backbone_spacing_eV=backbone,
        seeds=ordered,
        pieces=_coalesce(pieces, anchors),
        anchors=tuple(sorted(anchors)),
        dropped=tuple(dropped),
    )


def _coalesce(pieces: list[Piece], anchors: set[float]) -> tuple[Piece, ...]:
    work = [list(piece) for piece in pieces]
    merged = True
    while merged:
        merged = False
        for index, (lo, hi, spacing) in enumerate(work):
            if hi - lo >= 0.5 * spacing:
                continue
            neighbours = []
            if index > 0 and lo not in anchors:
                neighbours.append(index - 1)
            if index + 1 < len(work) and hi not in anchors:
                neighbours.append(index + 1)
            if not neighbours:
                continue
            target = min(neighbours, key=lambda k: (work[k][2], k))
            first, last = sorted((index, target))
            work[first] = [work[first][0], work[last][1], work[target][2]]
            del work[last]
            merged = True
            break

    joined = [work[0]]
    for lo, hi, spacing in work[1:]:
        if joined[-1][2] == spacing and lo not in anchors:
            joined[-1][1] = hi
        else:
            joined.append([lo, hi, spacing])
    return tuple((float(lo), float(hi), float(spacing)) for lo, hi, spacing in joined)


def window_plan_from_payload(payload: Mapping[str, Any]) -> WindowPlan:
    """Rebuild a plan from its payload and refuse one this planner would not make.

    The stored pieces are derived data. Rebuilding them from the seeds and
    comparing catches a planner change that forgot to bump
    :data:`LINE_WINDOW_PLAN_SCHEMA`, instead of reusing coordinates the stored
    identity no longer describes.
    """
    schema = payload.get("schema")
    if schema != LINE_WINDOW_PLAN_SCHEMA:
        raise ValueError(
            f"line window plan schema {schema!r} is not supported "
            f"(expected {LINE_WINDOW_PLAN_SCHEMA})"
        )
    seeds = [FeatureSeed(**dict(entry)) for entry in payload["seeds"]]
    plan = build_window_plan(
        payload["start_eV"], payload["stop_eV"], payload["backbone_spacing_eV"], seeds
    )
    stored = [tuple(float(value) for value in piece) for piece in payload["pieces"]]
    if stored != list(plan.pieces):
        raise ValueError(
            "stored line window plan pieces do not match the pieces rebuilt from its "
            "seeds; the planner changed without a schema bump"
        )
    return plan
