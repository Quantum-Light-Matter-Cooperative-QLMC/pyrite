"""Node refinement of a photon-continuum grid near the structure it carries.

:mod:`pyrite.energy_grid.floor` fixes where a continuum grid *starts*. This
module fixes where its nodes *go* between that floor and the ceiling.

The geometric baseline is the right default and the wrong default in exactly two
places. A grid uniform in ``u = ln E`` equidistributes relative quadrature error
for an integrand that is locally a power law, which the escaping bremsstrahlung
continuum is over most of the band. It says nothing about an integrand that is
*not* smooth on the scale of its own local step, and the modelled continuum
contains two such features:

* **absorption edges** of every element in the modelled path -- the medium and
  the detector's silicon sensor -- where the escape factor ``exp(-mu l)`` steps
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
(:mod:`pyrite._line_windows`); nothing here plans a line grid.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np

from ..materials import MediumSpec
from .floor import geometric_continuum_grid

__all__ = [
    "DETECTOR_PATH_ELEMENTS",
    "EDGE_KIND",
    "ENDPOINT_KIND",
    "RefinementMark",
    "case_endpoint_eV",
    "continuum_refinement_marks",
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
    """

    kind: str
    label: str
    anchors_eV: tuple[float, ...]
    first_eV: float
    last_eV: float
    spacing_eV: float

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
    from ..materials.attenuation import _resolve_composition

    return {str(element) for element, _density in _resolve_composition(material)}


def continuum_refinement_marks(
    material: str | MediumSpec,
    floor_eV: float,
    stop_eV: float,
    *,
    endpoint_eV: float | None = None,
    path_elements: Iterable[str] = DETECTOR_PATH_ELEMENTS,
    local_spacing_eV: float | None = None,
) -> tuple[list[RefinementMark], dict[str, Any]]:
    """Features of the modelled continuum the geometric baseline under-resolves.

    Edge positions are *located*, never listed: they come from
    :func:`~pyrite.montecarlo.spectrum.line_seeds.absorption_edge_brackets`,
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

    Returns
    -------
    tuple
        ``(marks, summary)``; the summary records located, kept, and dropped
        features so a grid can report what it refined and what it did not.

    Validation: continuum-node-refinement
    """
    from ..montecarlo.spectrum.line_seeds import absorption_edge_brackets

    floor, stop = float(floor_eV), float(stop_eV)
    if not np.isfinite(floor) or not np.isfinite(stop) or stop <= floor:
        raise ValueError("refinement band must be finite with stop_eV above floor_eV")

    # The medium's escape mu is EPDL; the detection path's Si sensor response
    # reads Chantler f2. Each contributes the edges of the table it reads.
    medium = _medium_elements(material)
    path = {str(value) for value in path_elements}
    elements = medium | path
    brackets, edge_summary = absorption_edge_brackets(medium, floor, stop, tables=("epdl",))
    path_brackets, path_summary = absorption_edge_brackets(path, floor, stop, tables=("chantler",))
    brackets = brackets + path_brackets
    edge_summary = {"skipped": [*edge_summary["skipped"], *path_summary["skipped"]]}

    marks: list[RefinementMark] = []
    outside: list[str] = []
    for bracket in brackets:
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
                    )
                )
                summary["endpoint_eV"] = endpoint
    return marks, summary


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

    Returns ``(nodes, summary)`` with the node-count cost of the refinement.
    """
    baseline = np.asarray(baseline_eV, dtype=float)
    if baseline.ndim != 1 or baseline.size < 2:
        raise ValueError("baseline grid must be 1-D with at least two nodes")
    if not np.all(np.isfinite(baseline)) or np.any(np.diff(baseline) <= 0.0):
        raise ValueError("baseline nodes must be finite and strictly increasing")
    if not marks:
        return baseline.copy(), {"baseline_nodes": int(baseline.size), "added_nodes": 0}

    lo, hi = float(baseline[0]), float(baseline[-1])
    inserted: list[np.ndarray] = []
    guarded: list[np.ndarray] = []
    for mark in marks:
        nodes = mark.nodes()
        nodes = nodes[(nodes > lo) & (nodes < hi)]
        if nodes.size == 0:
            continue
        inserted.append(nodes)
        guarded.append(np.asarray([0.5 * mark.spacing_eV], dtype=float).repeat(nodes.size))
    if not inserted:
        return baseline.copy(), {"baseline_nodes": int(baseline.size), "added_nodes": 0}

    new_nodes = np.concatenate(inserted)
    guard = np.concatenate(guarded)
    order = np.argsort(new_nodes, kind="stable")
    new_nodes, guard = new_nodes[order], guard[order]

    # Marks can overlap -- a secondary shell beside a stronger one (Se L2 beside
    # L3) seeds a neighbourhood inside its neighbour's. Merge across marks on
    # the same rule used inside one, judged by the finer of the two spacings, so
    # an overlap cannot open a near-degenerate interval.
    keep = np.ones(new_nodes.size, dtype=bool)
    previous = 0
    for index in range(1, new_nodes.size):
        tolerance = MERGE_FRACTION * 2.0 * min(guard[index], guard[previous])
        if new_nodes[index] - new_nodes[previous] <= tolerance:
            keep[index] = False
        else:
            previous = index
    new_nodes, guard = new_nodes[keep], guard[keep]

    # Drop a baseline node only when a mark node stands in for it; the band
    # endpoints always survive.
    nearest = np.searchsorted(new_nodes, baseline)
    left = np.clip(nearest - 1, 0, new_nodes.size - 1)
    right = np.clip(nearest, 0, new_nodes.size - 1)
    displaced = (np.abs(baseline - new_nodes[left]) < guard[left]) | (
        np.abs(baseline - new_nodes[right]) < guard[right]
    )
    displaced[0] = displaced[-1] = False

    merged = np.unique(np.concatenate((baseline[~displaced], new_nodes)))
    return merged, {
        "baseline_nodes": int(baseline.size),
        "added_nodes": int(merged.size - baseline.size),
        "displaced_baseline_nodes": int(np.count_nonzero(displaced)),
        "mark_nodes": int(new_nodes.size),
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
    floor, stop = float(baseline[0]), float(baseline[-1])
    # The baseline's own log step is what a geometric grid would place at the
    # endpoint, so the endpoint anchors match the local resolution there.
    log_step = np.log(stop / floor) / (baseline.size - 1)
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
    )
    nodes, cost = refine_continuum_nodes(baseline, marks)
    if return_summary:
        return nodes, {**summary, **cost, "marks": len(marks)}
    return nodes
