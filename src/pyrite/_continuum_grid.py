"""Photon-continuum node placement: geometric baseline, feature refinement, default policy.

A package-root leaf, like :mod:`pyrite._photon_continuum_floor`: ``campaign``
resolves every case's default continuum grid here, and importing the
``energy_grid`` driver package from ``campaign`` would close a package cycle
(``energy_grid`` already imports ``campaign``). :mod:`pyrite.energy_grid.floor`
and :mod:`pyrite.energy_grid.refine` are the public facades.

:func:`photon_continuum_floor_eV` fixes where a continuum grid *starts*. This
module fixes where its nodes *go* between that floor and the ceiling, and
:func:`automatic_continuum_grid` is the default every case uses when no
continuum grid is declared (#278).

The geometric baseline is the right default and the wrong default in exactly two
places. A grid uniform in ``u = ln E`` equidistributes relative quadrature error
for an integrand that is locally a power law, which the escaping bremsstrahlung
continuum is over most of the band. It says nothing about an integrand that is
*not* smooth on the scale of its own local step, and the modelled continuum
contains two such features:

* **absorption edges** of every element in the modelled path -- every medium of
  the target (the film and, in a layered target, each layer beneath it) and the
  detector's silicon sensor -- where the escape factor ``exp(-mu l)`` steps
  by a finite ratio over an energy interval far narrower than the local
  geometric step; and
* **kinematic endpoints** -- the bremsstrahlung tip at the highest instantaneous
  electron energy, above which the source term is identically zero.

Both turn the quadrature error in the straddling interval from second order in
the step to first order, and no amount of globally finer geometric spacing
changes that order. The rule below removes the first-order term at each feature
and refines no further than the resolution at which the model itself represents
the feature. See
[Energy-grid semantics](../../docs/physics/radiation-physics/energy-grid-semantics.md)
for the derivation, the assumptions, and the limiting case.

Narrow *line* windows and adaptive line-axis refinement are a different problem
on a different axis and are owned by issue #101
(:mod:`pyrite._cxr_windows`); nothing here plans a line grid.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

import numpy as np

from ._photon_continuum_floor import photon_continuum_floor_eV
from .materials import MediumSpec

__all__ = [
    "CONTINUUM_GRID_MODEL",
    "STACK_CONTINUUM_GRID_MODEL",
    "CONTINUUM_LOG_STEP",
    "DETECTOR_PATH_ELEMENTS",
    "EDGE_KIND",
    "ENDPOINT_KIND",
    "RefinementMark",
    "automatic_continuum_grid",
    "case_endpoint_eV",
    "continuum_refinement_marks",
    "geometric_continuum_grid",
    "refine_continuum_nodes",
    "refined_continuum_grid",
]

EDGE_KIND = "absorption-edge"
ENDPOINT_KIND = "kinematic-endpoint"

#: Elements of the modelled detection path beyond the medium itself. Both
#: detector forward models in this repository are silicon sensors
#: (``detectors/_si_sensor.py`` is shared by ``timepix_response.py`` and
#: ``eaglexo_response.py``), and the Si K edge at ~1.84 keV is carried
#: explicitly by the digitized Eagle XO quantum-efficiency curve as its notch.
#: It is inside every continuum band this repository models, so it is refined by
#: default. There is no window or filter material: the Eagle XO front end is
#: open (windowless), so nothing else stands in the path.
DETECTOR_PATH_ELEMENTS: tuple[str, ...] = ("Si",)

#: Two refinement coordinates closer than this fraction of the local target
#: spacing are the same coordinate as far as the model is concerned, and are
#: merged. The model represents no structure at that separation, so keeping both
#: buys no accuracy and costs a near-degenerate interval that the backend
#: coordinate-precision guard (``_grid_semantics.validate_backend_coordinates``)
#: would refuse. A quarter step is far below the native tabulation and far above
#: float64 drift in a sample lattice.
MERGE_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class RefinementMark:
    """One feature the geometric baseline cannot resolve, and the nodes it needs.

    ``anchors_eV`` are coordinates that must appear in the refined grid exactly:
    they are what removes the first-order error term. ``first_eV``/``last_eV``
    bound the neighbourhood sampled at ``spacing_eV``; for a mark that needs
    anchors only, the two coincide with the anchor span.

    ``centred`` marks a pair whose feature sits exactly at its midpoint (an
    EPDL edge, the kinematic endpoint), so what the pair must deliver is a bin
    boundary at that midpoint rather than the two coordinates themselves. A
    Chantler pair is two native table nodes with the jump somewhere between
    them, and is not centred.
    """

    kind: str
    label: str
    anchors_eV: tuple[float, ...]
    first_eV: float
    last_eV: float
    spacing_eV: float
    centred: bool = False

    def nodes(self) -> np.ndarray:
        """Anchors plus the neighbourhood samples, sorted and merged.

        Anchors always survive: they are the derived coordinates, and they are
        what removes the first-order error term. A sample within
        :data:`MERGE_FRACTION` of a step of an anchor (or of a sample already
        kept) is *dropped* rather than kept beside it. Without that, ordinary
        float drift in the sample lattice reproduces an anchor a few ulp away
        and opens an interval many orders of magnitude below anything the model
        resolves -- which the backend-precision guard then rightly refuses.
        """
        anchors = np.unique(np.asarray(self.anchors_eV, dtype=float))
        if not (self.last_eV > self.first_eV and self.spacing_eV > 0.0):
            return anchors
        count = int(np.floor((self.last_eV - self.first_eV) / self.spacing_eV)) + 1
        samples = self.first_eV + self.spacing_eV * np.arange(max(count, 1), dtype=float)
        tolerance = MERGE_FRACTION * self.spacing_eV
        kept = list(anchors)
        for sample in samples:
            if min(abs(sample - value) for value in kept) > tolerance:
                kept.append(float(sample))
        return np.unique(np.asarray(kept, dtype=float))


def case_endpoint_eV(case: Mapping[str, Any]) -> float:
    """The case's bremsstrahlung kinematic endpoint, in eV.

    The highest instantaneous electron kinetic energy in the run is the incident
    energy: electrons only lose energy, so no segment radiates above it and the
    escaping continuum is identically zero there. Transport imposes no *harder*
    cutoff below it -- the low-energy electron cutoff bounds the tip from below,
    not from above -- so this single coordinate is the continuum's only hard
    upper discontinuity.

    Reads the case's own incident energy (``E0_keV``, or ``energy_keV`` in the
    sweep-level spelling), so it cannot drift from the run it describes.
    """
    for key in ("E0_keV", "energy_keV"):
        if key in case:
            return float(case[key]) * 1.0e3
    raise KeyError("case carries no incident electron energy (E0_keV / energy_keV)")


def _medium_elements(material: str | MediumSpec) -> set[str]:
    """Every element the medium's own catalog composition contains."""
    from .materials.attenuation import _resolve_composition

    return {str(element) for element, _density in _resolve_composition(material)}


def _stack_floor_eV(material: str | MediumSpec, stack_media: Sequence[str | MediumSpec]) -> float:
    """Highest :func:`photon_continuum_floor_eV` over the film and its stack media.

    Every layer both radiates the continuum and lies on some escape path, so a
    node below any layer's floor sits where that layer's emission or escape
    model carries no validity. The stack floor is therefore the maximum, and a
    bare slab (no stack media) keeps its own floor exactly.
    """
    return max(photon_continuum_floor_eV(medium) for medium in (material, *stack_media))


def continuum_refinement_marks(
    material: str | MediumSpec,
    floor_eV: float,
    stop_eV: float,
    *,
    endpoint_eV: float | None = None,
    path_elements: Iterable[str] = DETECTOR_PATH_ELEMENTS,
    local_spacing_eV: float | None = None,
    stack_media: Iterable[str | MediumSpec] = (),
) -> tuple[list[RefinementMark], dict[str, Any]]:
    """Features of the modelled continuum the geometric baseline under-resolves.

    Edge positions are *located*, never listed: they come from
    :func:`~pyrite.montecarlo.spectrum.cxr_seeds.absorption_edge_brackets`,
    in the table each element is actually read from -- the medium's EPDL2025
    photoionization edges, which its escape ``mu`` contains as exact
    discontinuities, and the detection path's Chantler ``f2`` jumps, which the
    Si sensor response reads. An element whose edges all fall outside
    ``[floor_eV, stop_eV]`` contributes nothing and is not an error.

    Parameters
    ----------
    material
        Catalog crystal/media key or explicit homogeneous medium. Its own
        composition supplies the medium elements.
    floor_eV, stop_eV
        The modelled band. Features outside it are dropped.
    endpoint_eV
        Kinematic endpoint (see :func:`case_endpoint_eV`). ``None`` means the
        case has none inside the band, and no endpoint mark is produced.
    path_elements
        Elements of the detection path beyond the medium.
    local_spacing_eV
        Grid spacing at the endpoint, which sets the endpoint anchor pair's
        separation. Required whenever ``endpoint_eV`` is given: the anchors
        match the resolution the grid already has there rather than imposing one
        of their own, and that is a property of the grid, not of this band.
    stack_media
        Media of the layers beneath the film in a layered target. Each radiates
        and absorbs, and its escape ``mu`` is EPDL like the film's, so its
        elements join the medium elements (#1164). Empty for a bare slab.

    Returns
    -------
    tuple
        ``(marks, summary)``; the summary records located, kept, and dropped
        features so a grid can report what it refined and what it did not.

    Validation: continuum-node-refinement
    """
    from .montecarlo.spectrum.cxr_seeds import absorption_edge_brackets

    floor, stop = float(floor_eV), float(stop_eV)
    if not np.isfinite(floor) or not np.isfinite(stop) or stop <= floor:
        raise ValueError("refinement band must be finite with stop_eV above floor_eV")

    # The medium's escape mu is EPDL; the detection path's Si sensor response
    # reads Chantler f2. Each contributes the edges of the table it reads.
    medium = set().union(*(_medium_elements(m) for m in (material, *stack_media)))
    path = {str(value) for value in path_elements}
    elements = medium | path
    brackets, edge_summary = absorption_edge_brackets(medium, floor, stop, tables=("epdl",))
    path_brackets, path_summary = absorption_edge_brackets(path, floor, stop, tables=("chantler",))
    centred_count = len(brackets)
    brackets = brackets + path_brackets
    edge_summary = {"skipped": [*edge_summary["skipped"], *path_summary["skipped"]]}

    marks: list[RefinementMark] = []
    outside: list[str] = []
    for index, bracket in enumerate(brackets):
        # Degrade gracefully at the band edges: an edge whose jump itself lies
        # outside the modelled band places no requirement on a grid over it, and
        # a neighbourhood that only partly fits is clipped rather than refused.
        if bracket.below_eV <= floor or bracket.above_eV >= stop:
            outside.append(bracket.label)
            continue
        first = max(bracket.first_eV, floor)
        last = min(bracket.last_eV, stop)
        marks.append(
            RefinementMark(
                kind=EDGE_KIND,
                label=bracket.label,
                anchors_eV=(bracket.below_eV, bracket.above_eV),
                first_eV=first,
                last_eV=last,
                spacing_eV=bracket.spacing_eV,
                # EPDL pairs straddle the edge symmetrically; Chantler pairs
                # are native table nodes.
                centred=index < centred_count,
            )
        )

    summary: dict[str, Any] = {
        "elements": sorted(elements),
        "edges_located": len(brackets),
        "edges_refined": len(marks),
        "edges_outside_band": sorted(outside),
        "edges_skipped": list(edge_summary.get("skipped", ())),
        "endpoint_eV": None,
    }

    if endpoint_eV is not None:
        if local_spacing_eV is None:
            raise ValueError(
                "an endpoint mark needs local_spacing_eV: its anchor pair matches the "
                "resolution the grid already carries at the endpoint rather than imposing "
                "one, and that spacing is a property of the grid, not of this band"
            )
        endpoint = float(endpoint_eV)
        if np.isfinite(endpoint) and floor < endpoint < stop:
            spacing = float(local_spacing_eV)
            half = 0.5 * spacing
            if floor < endpoint - half and endpoint + half < stop:
                # The pair straddles the cutoff so the midpoint bin EDGE between
                # them falls exactly on it: the bin below carries the tip, the
                # bin above is exactly empty.
                marks.append(
                    RefinementMark(
                        kind=ENDPOINT_KIND,
                        label="bremsstrahlung tip",
                        anchors_eV=(endpoint - half, endpoint + half),
                        first_eV=endpoint - half,
                        last_eV=endpoint - half,
                        spacing_eV=spacing,
                        centred=True,
                    )
                )
                summary["endpoint_eV"] = endpoint
    return marks, summary


def _join_overlapping_pairs(
    marks: Sequence[RefinementMark], lo: float, hi: float
) -> tuple[list[RefinementMark], int]:
    """Re-anchor centred pairs whose spans overlap so every feature keeps its boundary.

    Two features closer than their pairs are wide (Pt M4 four eV below Au M5 in
    PtTe2 on gold) cannot both keep their own pair as adjacent nodes. What a
    centred pair delivers is a bin boundary on its feature, and a chain of
    nodes ``x_0 < x_1 < ...`` with ``x_k + x_(k+1) = 2 E_k`` delivers it for
    every ``E_k`` at once (bin edges are node midpoints). The chain starts from
    a shared node at the midpoint between the first two features, which makes
    the two joined intervals equal, and continues while the next pair still
    overlaps and every interval stays wider than the merge tolerance. Features
    within the merge tolerance of each other are one feature: the later takes
    the earlier's pair.

    Returns the marks (re-anchored where joined) and how many were re-anchored.
    """
    result = list(marks)
    centred = sorted(
        (sum(mark.anchors_eV) / 2.0, index)
        for index, mark in enumerate(marks)
        if mark.centred
        and len(mark.anchors_eV) == 2
        and lo < min(mark.anchors_eV)
        and max(mark.anchors_eV) < hi
    )
    joined = 0
    nodes: list[float] = []
    members: list[int] = []
    # Same-feature marks, each with the chain position whose pair it shares.
    shared_with: list[tuple[int, int]] = []

    def flush() -> int:
        for index, k in shared_with:
            result[index] = replace(marks[index], anchors_eV=(nodes[k], nodes[k + 1]))
        if len(members) < 2:
            return len(shared_with)
        for k, member in enumerate(members):
            result[member] = replace(marks[member], anchors_eV=(nodes[k], nodes[k + 1]))
        return len(members) + len(shared_with)

    for feature, index in centred:
        below, above = sorted(marks[index].anchors_eV)
        if members:
            last = members[-1]
            prior = 0.5 * (nodes[-2] + nodes[-1])
            tolerance = MERGE_FRACTION * min(marks[index].spacing_eV, marks[last].spacing_eV)
            if below - nodes[-1] <= tolerance:
                if feature - prior <= tolerance:
                    shared_with.append((index, len(members) - 1))
                    continue
                if len(members) == 1:
                    shared = 0.5 * (prior + feature)
                    candidate = [2.0 * prior - shared, shared]
                else:
                    candidate = list(nodes)
                following = 2.0 * feature - candidate[-1]
                if candidate[-1] < feature - tolerance and following > feature + tolerance:
                    nodes, members = [*candidate, following], [*members, index]
                    continue
        joined += flush()
        nodes, members, shared_with = [below, above], [index], []
    joined += flush()
    return result, joined


def _split_mark_nodes(
    marks: Sequence[RefinementMark], lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interior anchors, samples, their half-spacing guards, and anchor pairs.

    Anchors and samples come back sorted, each with the half spacing of the mark
    that produced it; an anchor several marks share keeps the finest guard.
    ``pairs`` is ``(n, 2)``: the span of every mark with two or more interior
    anchors, the interval the mark needs to stay a single grid interval.
    """
    anchor_parts: list[tuple[float, float]] = []
    sample_parts: list[tuple[float, float]] = []
    pairs: list[tuple[float, float]] = []
    for mark in marks:
        half = 0.5 * mark.spacing_eV
        own = np.unique(np.asarray(mark.anchors_eV, dtype=float))
        own = own[(own > lo) & (own < hi)]
        anchor_parts.extend((float(value), half) for value in own)
        if own.size >= 2:
            pairs.append((float(own[0]), float(own[-1])))
        nodes = mark.nodes()
        nodes = nodes[(nodes > lo) & (nodes < hi) & ~np.isin(nodes, own)]
        sample_parts.extend((float(value), half) for value in nodes)

    def _sorted(parts: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        if not parts:
            return np.empty(0), np.empty(0)
        table = np.asarray(sorted(parts), dtype=float)
        return table[:, 0], table[:, 1]

    anchors, anchor_guard = _sorted(anchor_parts)
    if anchors.size:
        # Sorted by (value, guard), so the first of each run of equal values
        # carries the finest guard.
        first = np.concatenate(([True], np.diff(anchors) > 0.0))
        anchors, anchor_guard = anchors[first], anchor_guard[first]
    samples, sample_guard = _sorted(sample_parts)
    return (
        anchors,
        anchor_guard,
        samples,
        sample_guard,
        np.asarray(pairs, dtype=float).reshape(-1, 2),
    )


def _strictly_inside_pairs(values: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Whether each value lies strictly between the two anchors of some pair."""
    if values.size == 0 or pairs.size == 0:
        return np.zeros(values.size, dtype=bool)
    return np.any(
        (values[:, None] > pairs[None, :, 0]) & (values[:, None] < pairs[None, :, 1]), axis=1
    )


def refine_continuum_nodes(
    baseline_eV: object,
    marks: Sequence[RefinementMark],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge refinement marks into a baseline grid, keeping the band fixed.

    A baseline node within half a mark's own spacing of one of that mark's nodes
    is *replaced* by it rather than kept alongside: the mark's coordinate is the
    derived one, and keeping both would open an interval far finer than anything
    the model resolves there, for no accuracy. The band endpoints are never
    dropped, so the floor and ceiling of the returned grid are exactly the
    baseline's.

    Every anchor survives (#1171): no sample or baseline node is kept strictly
    inside any mark's anchor pair, and centred pairs whose spans overlap are
    re-anchored as one chain (:func:`_join_overlapping_pairs`) so each feature
    still sits on a bin boundary.

    Returns ``(nodes, summary)`` with the node-count cost of the refinement,
    the number of re-anchored pairs, and the labels of any pair left split
    (none for a centred overlap the chain can resolve).

    Validation: continuum-node-refinement
    """
    baseline = np.asarray(baseline_eV, dtype=float)
    if baseline.ndim != 1 or baseline.size < 2:
        raise ValueError("baseline grid must be 1-D with at least two nodes")
    if not np.all(np.isfinite(baseline)) or np.any(np.diff(baseline) <= 0.0):
        raise ValueError("baseline nodes must be finite and strictly increasing")
    if not marks:
        return baseline.copy(), {"baseline_nodes": int(baseline.size), "added_nodes": 0}

    lo, hi = float(baseline[0]), float(baseline[-1])
    marks, joined = _join_overlapping_pairs(marks, lo, hi)
    anchors, anchor_guard, samples, sample_guard, pairs = _split_mark_nodes(marks, lo, hi)
    if anchors.size == 0 and samples.size == 0:
        return baseline.copy(), {"baseline_nodes": int(baseline.size), "added_nodes": 0}

    # Marks can overlap -- a secondary shell beside a stronger one (Se L2 beside
    # L3), or another layer's edge a few eV away in a stack, seeds a
    # neighbourhood inside its neighbour's. Anchors are the derived coordinates
    # that put a bin boundary on each feature, so every one survives the merge
    # (#1171): a sample strictly inside any mark's anchor pair is rejected,
    # since it would split the pair and move the boundary off the feature, and
    # the remaining samples merge against every kept node on the in-mark rule,
    # judged by the finer of the two spacings, so an overlap cannot open a
    # near-degenerate interval.
    inside = _strictly_inside_pairs(samples, pairs)
    samples, sample_guard = samples[~inside], sample_guard[~inside]
    keep = np.ones(samples.size, dtype=bool)
    last_kept = -1
    for index, sample in enumerate(samples):
        right = int(np.searchsorted(anchors, sample))
        near = [(anchors[i], anchor_guard[i]) for i in (right - 1, right) if 0 <= i < anchors.size]
        if last_kept >= 0:
            near.append((samples[last_kept], sample_guard[last_kept]))
        guard_here = sample_guard[index]
        if any(
            abs(sample - value) <= MERGE_FRACTION * 2.0 * min(guard_here, other)
            for value, other in near
        ):
            keep[index] = False
        else:
            last_kept = index
    new_nodes = np.concatenate((anchors, samples[keep]))
    guard = np.concatenate((anchor_guard, sample_guard[keep]))
    order = np.argsort(new_nodes, kind="stable")
    new_nodes, guard = new_nodes[order], guard[order]

    # Drop a baseline node when a mark node stands in for it, or when it lies
    # strictly inside an anchor pair (including exactly at the pair's
    # midpoint, which the half-spacing test alone keeps); the band endpoints
    # always survive.
    nearest = np.searchsorted(new_nodes, baseline)
    left = np.clip(nearest - 1, 0, new_nodes.size - 1)
    right = np.clip(nearest, 0, new_nodes.size - 1)
    displaced = (
        (np.abs(baseline - new_nodes[left]) < guard[left])
        | (np.abs(baseline - new_nodes[right]) < guard[right])
        | _strictly_inside_pairs(baseline, pairs)
    )
    displaced[0] = displaced[-1] = False

    merged = np.unique(np.concatenate((baseline[~displaced], new_nodes)))
    # A pair left split is one whose features could not be chained (a
    # non-centred Chantler pair interleaved with another): reported, not hidden.
    split = [
        mark.label
        for mark in marks
        if len(mark.anchors_eV) >= 2
        and lo < min(mark.anchors_eV)
        and max(mark.anchors_eV) < hi
        and np.any((merged > min(mark.anchors_eV)) & (merged < max(mark.anchors_eV)))
    ]
    return merged, {
        "baseline_nodes": int(baseline.size),
        "added_nodes": int(merged.size - baseline.size),
        "displaced_baseline_nodes": int(np.count_nonzero(displaced)),
        "mark_nodes": int(new_nodes.size),
        "joined_anchor_pairs": joined,
        "split_anchor_pairs": split,
    }


@overload
def refined_continuum_grid(
    material: str | MediumSpec,
    stop_eV: float,
    num: int,
    *,
    floor_eV: float | None = ...,
    endpoint_eV: float | None = ...,
    path_elements: Iterable[str] = ...,
    return_summary: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def refined_continuum_grid(
    material: str | MediumSpec,
    stop_eV: float,
    num: int,
    *,
    floor_eV: float | None = ...,
    endpoint_eV: float | None = ...,
    path_elements: Iterable[str] = ...,
    return_summary: Literal[True],
) -> tuple[np.ndarray, dict[str, Any]]: ...


def refined_continuum_grid(
    material: str | MediumSpec,
    stop_eV: float,
    num: int,
    *,
    floor_eV: float | None = None,
    endpoint_eV: float | None = None,
    path_elements: Iterable[str] = DETECTOR_PATH_ELEMENTS,
    return_summary: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """The geometric continuum baseline, refined at edges and kinematic endpoints.

    The baseline is :func:`~pyrite.energy_grid.floor.geometric_continuum_grid`
    with the same ``num``, so the floor, the ceiling, and the resolution
    everywhere away from a feature are unchanged; refinement only adds nodes
    where a feature needs them.

    Limiting case: a band containing no located edge and no kinematic endpoint
    returns the geometric baseline *identically*, node for node.

    Validation: continuum-node-refinement
    """
    baseline = geometric_continuum_grid(material, stop_eV, num, floor_eV=floor_eV)
    # The baseline's own log step is what a geometric grid would place at the
    # endpoint, so the endpoint anchors match the local resolution there.
    log_step = np.log(float(baseline[-1]) / float(baseline[0])) / (baseline.size - 1)
    nodes, summary = _refine_baseline(
        material, baseline, log_step, endpoint_eV=endpoint_eV, path_elements=path_elements
    )
    return (nodes, summary) if return_summary else nodes


def _refine_baseline(
    material: str | MediumSpec,
    baseline: np.ndarray,
    log_step: float,
    *,
    endpoint_eV: float | None,
    path_elements: Iterable[str],
    stack_media: Sequence[str | MediumSpec] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    floor, stop = float(baseline[0]), float(baseline[-1])
    marks, summary = continuum_refinement_marks(
        material,
        floor,
        stop,
        endpoint_eV=endpoint_eV,
        path_elements=path_elements,
        local_spacing_eV=(
            float(endpoint_eV) * log_step
            if endpoint_eV is not None and np.isfinite(endpoint_eV)
            else None
        ),
        stack_media=stack_media,
    )
    nodes, cost = refine_continuum_nodes(baseline, marks)
    return nodes, {**summary, **cost, "marks": len(marks)}


def geometric_continuum_grid(
    material: str | MediumSpec,
    stop_eV: float,
    num: int,
    *,
    floor_eV: float | None = None,
) -> np.ndarray:
    """Geometric continuum nodes from the derived floor to ``stop_eV``.

    The geometric baseline only -- no refinement near absorption edges or
    kinematic endpoints, which is a separate concern.

    Parameters
    ----------
    material
        Catalog crystal/media key or explicit homogeneous medium.
    stop_eV
        Highest node energy, strictly above the floor.
    num
        Endpoint-inclusive node count, at least 2.
    floor_eV
        Override for the lowest node. Refused below
        :func:`photon_continuum_floor_eV`, because a lower bound would place
        nodes where the emission and escape models carry no validity; passing a
        *higher* value is an ordinary bandwidth choice and is allowed.

    Returns
    -------
    numpy.ndarray
        Strictly increasing nodes in eV, ``E[0]`` exactly the resolved floor.

    Validation: photon-continuum-floor
    """
    derived = photon_continuum_floor_eV(material)
    if floor_eV is None:
        floor = derived
    else:
        floor = float(floor_eV)
        if not np.isfinite(floor):
            raise ValueError("floor_eV must be finite")
        if floor < derived:
            raise ValueError(
                f"floor_eV {floor:.6g} eV is below the derived continuum floor "
                f"{derived:.6g} eV for this medium; the continuum and photon-escape "
                "models carry no validity there, so widening the band downward needs a "
                "physics decision, not a smaller number"
            )
    stop = float(stop_eV)
    if not np.isfinite(stop) or stop <= floor:
        raise ValueError(f"stop_eV must be finite and above the floor {floor:.6g} eV")
    if int(num) != num or int(num) < 2:
        raise ValueError("num must be an integer of at least 2")
    return np.geomspace(floor, stop, int(num))


#: Baseline spacing in ``u = ln E`` of the default continuum grid (#278). On
#: fixed tracks against a dense refined truth (HOPG 1 mm at 300 keV and
#: 29,999 keV, W 100 um at 1 MeV), intrinsic yield and centroid converge at
#: second order and stay below 2e-6 at this step. Timepix3 counts read the
#: source axis piecewise-constant and converge at first order, reaching 2e-4 to
#: 9e-4 here against their 1e-2 budget; that is what sets the step.
CONTINUUM_LOG_STEP = 0.0025


def _exp_series(x: float, terms: int = 14) -> float:
    """``exp(x)`` by a fixed-length Taylor sum in correctly rounded IEEE basic ops.

    Libm and NumPy's CPU-dispatched SIMD transcendentals differ by an ulp or two
    between machines; ``+``, ``*`` and ``/`` on float64 do not. The automatic
    continuum grid is part of case identity, so it may use nothing else.
    """
    term = total = 1.0
    for k in range(1, terms + 1):
        term = term * x / k
        total = total + term
    return total


#: Per-node ratio of the default continuum grid, ``exp(CONTINUUM_LOG_STEP)``
#: evaluated once by :func:`_exp_series`, so every machine builds the same bits.
CONTINUUM_RATIO = _exp_series(CONTINUUM_LOG_STEP)

#: Identity of the default continuum-grid policy. It joins a sweep's hashed
#: payload whenever a case's continuum grid comes from
#: :func:`automatic_continuum_grid`, so results on the earlier uniform 25 eV
#: default never share a dataset with it. ``v2`` builds the nodes from
#: correctly rounded basic operations only (no ``log``/``exp``/``geomspace``),
#: so node bits, and therefore case content keys, are machine-independent.
#: ``v3`` keeps every anchor through the cross-mark merge and joins overlapping
#: centred pairs, so each feature sits on a bin boundary even where marks
#: overlap (#1171).
CONTINUUM_GRID_MODEL = f"refined-geometric-v3/dlnE={CONTINUUM_LOG_STEP:g}"

#: Identity of the default continuum grid over a layered target, whose floor and
#: edge anchors span every layer's medium (#1164). Only stack sweeps carry it: a
#: bare slab's grid is unchanged node for node, so its datasets keep
#: :data:`CONTINUUM_GRID_MODEL`, while a stack's earlier film-only grid never
#: shares a dataset with this one.
STACK_CONTINUUM_GRID_MODEL = f"{CONTINUUM_GRID_MODEL}+stack-media-v1"


def _deterministic_geometric_nodes(floor: float, stop: float, ratio: float) -> np.ndarray:
    """``floor * ratio**k`` by repeated multiplication, closed exactly at ``stop``.

    ``E[0]`` is exactly ``floor`` and the last node exactly ``stop``. A final
    interval narrower than half a step is absorbed by dropping the last lattice
    node, so no interval is finer than half the baseline spacing.
    """
    nodes = [floor]
    value = floor
    while True:
        value = value * ratio
        if value >= stop:
            break
        nodes.append(value)
    if len(nodes) > 1 and stop <= nodes[-1] * (1.0 + 0.5 * (ratio - 1.0)):
        nodes.pop()
    nodes.append(stop)
    return np.asarray(nodes, dtype=float)


def automatic_continuum_grid(
    material: str | MediumSpec,
    endpoint_eV: float,
    *,
    floor_eV: float | None = None,
    log_step: float = CONTINUUM_LOG_STEP,
    stack_media: Sequence[str | MediumSpec] = (),
) -> np.ndarray:
    """Default continuum grid for one case: refined geometric, floor to past the endpoint.

    Nodes are :func:`refined_continuum_grid` with a baseline spacing of
    ``log_step`` in ``ln E`` from the medium's floor (or a higher declared
    ``floor_eV``) to one step above ``endpoint_eV``, so the kinematic endpoint is
    interior and gets its anchor pair, and the escaping continuum's absorption
    edges get theirs. Above the endpoint the continuum is identically zero, as
    the uniform band it replaces also extended one step past ``E0``.

    ``stack_media`` names the media beneath the film of a layered target. Their
    absorption edges are anchored like the film's, and the floor is the highest
    floor over the film and every layer. Empty (a bare slab) leaves the grid
    exactly as the film alone defines it.

    Validation: continuum-node-refinement
    """
    endpoint = float(endpoint_eV)
    step = float(log_step)
    if not np.isfinite(endpoint) or endpoint <= 0.0:
        raise ValueError("endpoint_eV must be a finite positive energy")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("log_step must be a finite positive number")
    stack_media = tuple(stack_media)
    floor = _stack_floor_eV(material, stack_media) if floor_eV is None else float(floor_eV)
    ratio = CONTINUUM_RATIO if step == CONTINUUM_LOG_STEP else _exp_series(step)
    stop = endpoint * ratio
    if stop <= floor:
        raise ValueError(
            f"endpoint {endpoint:.6g} eV is not above the continuum floor {floor:.6g} eV"
        )
    baseline = _deterministic_geometric_nodes(floor, stop, ratio)
    nodes, _ = _refine_baseline(
        material,
        baseline,
        step,
        endpoint_eV=endpoint,
        path_elements=DETECTOR_PATH_ELEMENTS,
        stack_media=stack_media,
    )
    return nodes
