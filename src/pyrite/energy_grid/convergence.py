"""Refinement-ladder convergence harness for line-grid resolution (issue #109).

The harness answers one question: *on fixed trajectories, which line-grid
spacing has converged?* It evaluates only the spectrum phase on one set of
transport segments across a ladder of photon-energy grids, so every difference
between rungs is pure grid (quadrature) error. It never compares separate Monte
Carlo runs; a resumed ladder that must re-run transport proves the segments are
identical through :func:`segment_fingerprint` before any rung is compared.

The module is grid-agnostic. A rung is any strictly increasing coordinate array
(uniform, graded, or case-local), labelled by its largest adjacent spacing.
:func:`nested_uniform_ladder` is one convenience generator; #101 (local windows)
and #117 (coherent-route analysis) supply their own coordinates to the same
:func:`evaluate_ladder` / :func:`richardson_acceptance` pair.

Gated observables and their tolerance classes (``GATED_OBSERVABLES``) reuse the
per-observable tolerances of the automatic line-grid policy
(:data:`pyrite._line_grid_policy.DEFAULT_RTOL`, unchanged): yield and centroid
as intrinsic-source quantities at ``1e-3``, detector-counted quantities at
``1e-2``. Dominant-line FWHM and line/background form a harness-only ``shape``
class gated at ``1e-2`` and are also reported, ungated, at ``1e-3``: automatic
``sinc-nyquist`` spacing certifies yield and centroid, not shape at ``1e-3``,
which belongs to #101's local windows. Peak height is deliberately not an
observable -- a sampled density maximum scales with the local spacing.
Dominant-line FWHM and the line/background ratio come from
:func:`pyrite.results.metrics.line_metrics`, whose widths are physical-energy
half-maximum crossings (#110), not sample counts times one spacing.
"""

from __future__ import annotations

import hashlib
import math
import resource
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .._grid_semantics import resolution_num
from .._line_grid_policy import DEFAULT_RTOL
from ..detectors.spec import EagleXO, Timepix3
from ..results.metrics import line_metrics

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_YIELD_FLOOR",
    "DIAGNOSTIC_OBSERVABLES",
    "GATED_OBSERVABLES",
    "HARNESS_RTOL",
    "NOISE_FRACTION",
    "SHAPE_DIAGNOSTIC_RTOL",
    "SHAPE_RTOL",
    "LadderReport",
    "ObservableVerdict",
    "Rung",
    "SegmentMismatchError",
    "SpectrumSample",
    "TripleVerdict",
    "evaluate_ladder",
    "lineshape_deviation",
    "nested_uniform_ladder",
    "representative_spacing",
    "require_identical_segments",
    "richardson_acceptance",
    "segment_fingerprint",
    "spectrum_observables",
]

#: Harness-level tolerance for sampled line *shape* (dominant-line FWHM and local
#: line/background). Validation only: automatic ``sinc-nyquist`` spacing
#: certifies yield and centroid, not shape at the intrinsic-source tolerance;
#: shape accuracy at that level belongs to #101's local windows.
SHAPE_RTOL = 1.0e-2
#: Shape observables are still judged at the intrinsic-source tolerance, as
#: ungated diagnostics, so a report shows how far 1e-3 shape accuracy lies.
SHAPE_DIAGNOSTIC_RTOL = 1.0e-3
#: Changes at or below this fraction of the tolerance are numerical noise
#: (round-off, detector-response resampling): monotonicity is not required.
#: The resampling half of that is the measured Timepix3 drift of up to 1e-3 per
#: halving, which sits inside the interpolation row of
#: ``tbl-line-budget-allocation`` (2e-3 of the detected-counts share).
NOISE_FRACTION = 0.1

#: Observable name -> tolerance class. Order is report order.
GATED_OBSERVABLES: Mapping[str, str] = {
    "yield": "intrinsic_source",
    "centroid_eV": "intrinsic_source",
    "fwhm_eV": "shape",
    "line_background_ratio": "shape",
    "timepix3_counts": "detected_counts",
    "eaglexo_counts": "detected_counts",
}

#: Relative tolerance per class: the production policy's ``DEFAULT_RTOL``
#: (unchanged) plus the harness-only ``shape`` class. Each value is a column
#: total of ``tbl-line-budget-allocation`` in
#: ``docs/validation/beam-transport/line-spectrum-error-budget.md``, which
#: allocates it among bandwidth, backbone quadrature, window quadrature,
#: interpolation and backend precision. The ladder gates that grid share only:
#: transport and Monte Carlo statistics are held identical across rungs by the
#: segment fingerprint and cancel, rather than being bounded here.
HARNESS_RTOL: Mapping[str, float] = {**DEFAULT_RTOL, "shape": SHAPE_RTOL}

#: Ungated diagnostic observables and the relative tolerance they are judged at.
DIAGNOSTIC_OBSERVABLES: Mapping[str, float] = {
    "fwhm_eV": SHAPE_DIAGNOSTIC_RTOL,
    "line_background_ratio": SHAPE_DIAGNOSTIC_RTOL,
}

#: Near-zero spectrum floor on the integrated line yield, in photons per sr per
#: incident electron. At the default 1 nA average current this is about 0.02
#: photons/sr per hour: nothing below it is measurable, so a spectrum whose
#: finest-rung yield sits under it has nothing to resolve and every observable
#: of that triple is accepted as ``near-zero``.
DEFAULT_YIELD_FLOOR = 1.0e-15

#: Absolute change floors, in each observable's own units. A change at or below
#: its floor counts as converged even when the relative scale is ~0, which is
#: what keeps an (almost) empty spectrum from dividing by zero.
DEFAULT_ATOL: Mapping[str, float] = {
    "yield": DEFAULT_YIELD_FLOOR,
    "centroid_eV": 0.0,
    "fwhm_eV": 0.0,
    "line_background_ratio": 0.0,
    "timepix3_counts": DEFAULT_YIELD_FLOOR,
    "eaglexo_counts": DEFAULT_YIELD_FLOOR,
}

_SEGMENT_FIELDS = ("E_keV", "L_ang", "v_hat", "r_mid", "elec_id", "layer")


class SegmentMismatchError(RuntimeError):
    """Two ladder rungs were evaluated on different transport segments."""


@dataclass(frozen=True)
class SpectrumSample:
    """One spectrum evaluation on one grid.

    ``line`` and ``background`` are densities on the rung's coordinates, in the
    same per-eV units. ``informational`` carries ungated scalars (for example a
    grid-exact characteristic yield) that a report keeps beside the gated set.
    """

    line: np.ndarray
    background: np.ndarray
    informational: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Rung:
    """Observables, cost, and grid description of one evaluated rung."""

    spacing_eV: float
    n_points: int
    start_eV: float
    stop_eV: float
    observables: dict[str, float]
    informational: dict[str, float]
    wall_s: float
    host_peak_rss_mib: float
    device_peak_mib: float | None


@dataclass(frozen=True)
class ObservableVerdict:
    """Richardson verdict for one observable over one ``(h, h/2, h/4)`` triple."""

    name: str
    coarse_change: float
    fine_change: float
    tolerance: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class TripleVerdict:
    """Verdict for the coarsest spacing of one consecutive rung triple."""

    spacing_eV: float
    accepted: bool
    observables: tuple[ObservableVerdict, ...]


@dataclass(frozen=True)
class LadderReport:
    """Rungs, per-triple verdicts, and the accepted spacing (``None`` if none)."""

    rungs: tuple[Rung, ...]
    triples: tuple[TripleVerdict, ...]
    accepted_spacing_eV: float | None
    rtol: dict[str, float]
    #: Accepted spacing of each gated observable on its own.
    observable_accepted_spacing_eV: dict[str, float | None] = field(default_factory=dict)
    #: Accepted spacing of each ungated diagnostic observable at its diagnostic rtol.
    diagnostic_accepted_spacing_eV: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able representation (NaN survives as ``float('nan')``)."""
        return asdict(self)


def representative_spacing(E_grid_eV: object) -> float:
    """Largest adjacent spacing of a strictly increasing grid, in eV.

    The largest spacing is the one that aliases first, so it labels a rung on a
    nonuniform grid as conservatively as the single spacing of a uniform one.
    """
    E = np.asarray(E_grid_eV, dtype=float)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("ladder grid must be 1-D with at least two nodes")
    steps = np.diff(E)
    if not np.all(np.isfinite(E)) or np.any(steps <= 0.0):
        raise ValueError("ladder grid must be finite and strictly increasing")
    return float(steps.max())


def nested_uniform_ladder(
    start_eV: float, stop_eV: float, coarsest_spacing_eV: float, n_rungs: int
) -> list[np.ndarray]:
    """Uniform grids over one fixed bandwidth, each exactly half the previous step.

    The coarsest count comes from :func:`pyrite._grid_semantics.resolution_num`,
    so its actual spacing never exceeds ``coarsest_spacing_eV``; every finer rung
    doubles the interval count, so each grid's nodes are a subset of the next
    and the spacings halve exactly.
    """
    if int(n_rungs) != n_rungs or n_rungs < 1:
        raise ValueError("n_rungs must be a positive integer")
    intervals = resolution_num(start_eV, stop_eV, coarsest_spacing_eV) - 1
    return [
        np.linspace(float(start_eV), float(stop_eV), intervals * 2**rung + 1)
        for rung in range(int(n_rungs))
    ]


def _host(array: object) -> np.ndarray:
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def segment_fingerprint(segments: Mapping[str, Any]) -> dict[str, Any]:
    """Row count plus a digest over the fields the spectrum kernels read.

    Two fingerprints are equal only for bit-identical kinematics, so a ladder
    resumed from a fresh fixed-seed transport can prove it is still comparing
    rungs on the same trajectories.
    """
    digest = hashlib.blake2b(digest_size=16)
    n_segments = int(np.asarray(_host(segments["L_ang"])).size)
    for name in _SEGMENT_FIELDS:
        value = segments.get(name)
        if value is None:
            continue
        array = np.ascontiguousarray(_host(value))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return {"n_segments": n_segments, "digest": digest.hexdigest()}


def require_identical_segments(
    reference: Mapping[str, Any], segments_or_fingerprint: Mapping[str, Any]
) -> None:
    """Raise :class:`SegmentMismatchError` unless two segment sets are identical."""
    candidate = (
        segments_or_fingerprint
        if "digest" in segments_or_fingerprint
        else segment_fingerprint(segments_or_fingerprint)
    )
    if int(candidate["n_segments"]) != int(reference["n_segments"]):
        raise SegmentMismatchError(
            f"ladder rungs use different transport: n_segments {candidate['n_segments']} "
            f"!= {reference['n_segments']}; rungs are only comparable on identical trajectories"
        )
    if candidate["digest"] != reference["digest"]:
        raise SegmentMismatchError(
            "ladder rungs use different transport: equal n_segments but different "
            "segment kinematics; rungs are only comparable on identical trajectories"
        )


def spectrum_observables(
    E_grid_eV: object,
    line_density: object,
    background_density: object,
    *,
    rel_prominence: float = 0.03,
    detectors: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Grid-aware scalar observables of one line spectrum.

    ``yield`` and ``centroid_eV`` are trapezoids over the rung's own
    coordinates. ``fwhm_eV`` and ``line_background_ratio`` are the dominant
    (largest-prominence) line's physical-energy FWHM and its local line/
    background ratio from :func:`pyrite.results.metrics.line_metrics`.
    Detected counts integrate each read-time detector response of the line
    density (default: :class:`~pyrite.detectors.spec.Timepix3` and
    :class:`~pyrite.detectors.spec.EagleXO` at their defaults). Every value is
    per incident electron per sr (or eV for positions and widths).
    """
    E = np.asarray(E_grid_eV, dtype=float)
    line = np.asarray(_host(line_density), dtype=float)
    background = np.asarray(_host(background_density), dtype=float)
    if line.shape != E.shape or background.shape != E.shape:
        raise ValueError("line and background densities must match the grid shape")
    total = float(np.trapezoid(line, E))
    centroid = float(np.trapezoid(E * line, E) / total) if total > 0.0 else float("nan")
    metrics = line_metrics(
        {"E_grid": E, "spec": line, "brem": background, "scale": 1.0},
        None,
        rel_prominence=rel_prominence,
        metric="prominence",
    )
    fwhm = float(metrics["fwhm_eV"]) if total > 0.0 else float("nan")
    ratio = float(metrics["line_brem_ratio"]) if total > 0.0 else float("nan")
    out = {
        "yield": total,
        "centroid_eV": centroid,
        "fwhm_eV": fwhm if fwhm > 0.0 else float("nan"),
        "line_background_ratio": ratio,
    }
    responses = (
        {"timepix3_counts": Timepix3(), "eaglexo_counts": EagleXO()}
        if detectors is None
        else detectors
    )
    for name, response in responses.items():
        detected = response.score(E, line, fwhm_eV=None, scale=1.0)
        out[name] = float(np.trapezoid(np.asarray(detected, dtype=float), E))
    return out


def _host_peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB. It is the process high-water mark, so it
    # is monotone across rungs: a rung's figure bounds every rung before it.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def evaluate_ladder(
    grids: Iterable[object],
    evaluate: Callable[[np.ndarray], SpectrumSample],
    *,
    segments: Mapping[str, Any] | None = None,
    observables: Callable[..., dict[str, float]] = spectrum_observables,
    spacings: Sequence[float] | None = None,
    device_peak_mib: Callable[[], float | None] | None = None,
    on_rung: Callable[[Rung], None] | None = None,
) -> list[Rung]:
    """Evaluate ``evaluate`` on every grid, coarse to fine, on one segment set.

    ``evaluate`` must be a pure spectrum reduction over already-transported
    segments. When ``segments`` is given, the harness fingerprints them before
    the first rung and re-checks after every rung, so an evaluator that mutates
    or replaces the trajectories fails loudly instead of producing a ladder of
    incomparable runs.

    ``spacings`` labels each rung with the quantity actually being refined
    instead of the grid's own largest spacing. A window ladder (#101) refines
    local windows on a deliberately fixed backbone, so its largest spacing never
    moves; the labels must still decrease strictly, and every grid is still
    checked for finite, strictly increasing coordinates.
    """
    reference = None if segments is None else segment_fingerprint(segments)
    rungs: list[Rung] = []
    previous = math.inf
    grids = list(grids)
    if spacings is not None:
        labels = [float(value) for value in spacings]
        if len(labels) != len(grids):
            raise ValueError("ladder spacings must label every grid")
    else:
        labels = None
    for index, grid in enumerate(grids):
        E = np.asarray(grid, dtype=float)
        spacing = representative_spacing(E) if labels is None else labels[index]
        if labels is not None:
            representative_spacing(E)
        if spacing >= previous:
            raise ValueError("ladder grids must refine strictly: each spacing below the previous")
        previous = spacing
        started = time.perf_counter()
        sample = evaluate(E)
        line = np.asarray(_host(sample.line), dtype=float)
        background = np.asarray(_host(sample.background), dtype=float)
        wall = time.perf_counter() - started
        if reference is not None and segments is not None:
            require_identical_segments(reference, segments)
        rung = Rung(
            spacing_eV=spacing,
            n_points=int(E.size),
            start_eV=float(E[0]),
            stop_eV=float(E[-1]),
            observables=observables(E, line, background),
            informational={key: float(value) for key, value in sample.informational.items()},
            wall_s=float(wall),
            host_peak_rss_mib=_host_peak_rss_mib(),
            device_peak_mib=None if device_peak_mib is None else device_peak_mib(),
        )
        rungs.append(rung)
        if on_rung is not None:
            on_rung(rung)
    return rungs


def _observable_verdict(
    name: str,
    values: Sequence[float],
    *,
    rtol: float,
    atol: float,
    near_zero: bool,
) -> ObservableVerdict:
    coarse, middle, fine = (float(value) for value in values)
    if near_zero:
        return ObservableVerdict(name, 0.0, 0.0, 0.0, True, "near-zero")
    if not all(math.isfinite(value) for value in (coarse, middle, fine)):
        return ObservableVerdict(name, float("nan"), float("nan"), float("nan"), False, "undefined")
    coarse_change = abs(coarse - middle)
    fine_change = abs(middle - fine)
    tolerance = rtol * abs(fine) + atol
    # A zero change passes even against a zero tolerance (an exactly constant
    # observable); otherwise the first halving must land strictly inside.
    if not (coarse_change < tolerance or coarse_change == 0.0):
        return ObservableVerdict(
            name, coarse_change, fine_change, tolerance, False, "h->h/2 above tolerance"
        )
    # One lucky pair is not convergence: the next halving may not grow the
    # change, unless both changes are noise far inside the tolerance.
    if fine_change <= coarse_change:
        return ObservableVerdict(name, coarse_change, fine_change, tolerance, True, "converged")
    if max(coarse_change, fine_change) <= NOISE_FRACTION * tolerance:
        return ObservableVerdict(
            name, coarse_change, fine_change, tolerance, True, "converged below noise fraction"
        )
    return ObservableVerdict(name, coarse_change, fine_change, tolerance, False, "h/2->h/4 larger")


def _coarsest_consistent(spacings: Sequence[float], passed: Sequence[bool]) -> float | None:
    """Coarsest spacing whose triple and every finer triple passed."""
    accepted = None
    for spacing, ok in zip(reversed(spacings), reversed(passed), strict=True):
        if not ok:
            break
        accepted = spacing
    return accepted


def richardson_acceptance(
    rungs: Sequence[Rung],
    *,
    rtol: Mapping[str, float] | None = None,
    atol: Mapping[str, float] | None = None,
    gated: Mapping[str, str] = GATED_OBSERVABLES,
    diagnostics: Mapping[str, float] = DIAGNOSTIC_OBSERVABLES,
    yield_floor: float = DEFAULT_YIELD_FLOOR,
) -> LadderReport:
    """Richardson acceptance over consecutive rung triples.

    For the triple ``(h, h/2, h/4)`` (any strictly refining spacings), with
    ``tol = rtol * |q(h/4)| + atol``, ``d1 = |q(h) - q(h/2)|`` and
    ``d2 = |q(h/2) - q(h/4)|``, an observable passes when ``d1 < tol`` (or
    ``d1 == 0`` for an exactly constant observable) **and** either ``d2 <= d1``
    or ``max(d1, d2) <= NOISE_FRACTION * tol``. The second clause still rejects
    one lucky pair but accepts converged values whose residual changes are
    noise. A triple whose finest-rung yield is at or below ``yield_floor`` is
    accepted as near-zero.

    ``rtol`` maps a tolerance class to its relative tolerance and defaults to
    :data:`HARNESS_RTOL`. A triple passes when every gated observable passes.
    The accepted spacing is the coarsest ``h`` whose triple and every finer
    triple pass; the two finest rungs cannot be judged (no ``h/4``). The report
    also carries each gated observable's own accepted spacing, and each
    ``diagnostics`` observable's accepted spacing at its diagnostic rtol.
    """
    tolerances = dict(HARNESS_RTOL)
    tolerances.update(dict(rtol or {}))
    floors = dict(DEFAULT_ATOL)
    floors.update(dict(atol or {}))
    unknown = sorted(set(gated.values()) - set(tolerances))
    if unknown:
        raise ValueError(f"no relative tolerance for observable classes {unknown}")
    triples: list[TripleVerdict] = []
    for index in range(len(rungs) - 2):
        window = rungs[index : index + 3]
        near_zero = float(window[2].observables.get("yield", math.inf)) <= yield_floor
        verdicts = tuple(
            _observable_verdict(
                name,
                [rung.observables[name] for rung in window],
                rtol=float(tolerances[category]),
                atol=float(floors.get(name, 0.0)),
                near_zero=near_zero,
            )
            for name, category in gated.items()
        )
        triples.append(
            TripleVerdict(
                spacing_eV=window[0].spacing_eV,
                accepted=all(verdict.accepted for verdict in verdicts),
                observables=verdicts,
            )
        )
    spacings = [triple.spacing_eV for triple in triples]
    per_observable = {
        name: _coarsest_consistent(
            spacings, [triple.observables[index].accepted for triple in triples]
        )
        for index, name in enumerate(gated)
    }
    diagnostic: dict[str, float | None] = {}
    for name, diagnostic_rtol in diagnostics.items():
        if not all(name in rung.observables for rung in rungs):
            continue  # a producer that never measured this observable
        passed = []
        for index in range(len(rungs) - 2):
            window = rungs[index : index + 3]
            passed.append(
                _observable_verdict(
                    name,
                    [rung.observables[name] for rung in window],
                    rtol=float(diagnostic_rtol),
                    atol=float(floors.get(name, 0.0)),
                    near_zero=float(window[2].observables.get("yield", math.inf)) <= yield_floor,
                ).accepted
            )
        diagnostic[name] = _coarsest_consistent(spacings, passed)
    return LadderReport(
        rungs=tuple(rungs),
        triples=tuple(triples),
        accepted_spacing_eV=_coarsest_consistent(spacings, [t.accepted for t in triples]),
        rtol={category: float(tolerances[category]) for category in sorted(set(gated.values()))},
        observable_accepted_spacing_eV=per_observable,
        diagnostic_accepted_spacing_eV=diagnostic,
    )


def _relative(candidate: float, reference: float) -> float:
    if not (math.isfinite(candidate) and math.isfinite(reference)) or reference == 0.0:
        return float("nan")
    return abs(candidate - reference) / abs(reference)


def lineshape_deviation(
    E_grid_eV: object,
    reference_line: object,
    candidate_line: object,
    *,
    candidate_E_grid_eV: object | None = None,
    rel_prominence: float = 0.03,
) -> dict[str, float]:
    """Deviation of one line density from a reference on the same nodes.

    Unlike a ladder, the two densities share their node count, so the pointwise
    lineshape difference is defined. ``candidate_E_grid_eV`` optionally gives
    the candidate's integration coordinates (for example a grid after a backend
    cast) while the reference keeps ``E_grid_eV``; pointwise comparison is by
    node index either way. Relative quantities are NaN where the reference is
    zero or undefined.
    """
    E = np.asarray(E_grid_eV, dtype=float)
    E_candidate = E if candidate_E_grid_eV is None else np.asarray(candidate_E_grid_eV, dtype=float)
    reference = np.asarray(_host(reference_line), dtype=float)
    candidate = np.asarray(_host(candidate_line), dtype=float)
    if reference.shape != E.shape or candidate.shape != E.shape or E_candidate.shape != E.shape:
        raise ValueError("reference, candidate, and both grids must share one node count")
    zeros = np.zeros_like(E)
    ref = spectrum_observables(E, reference, zeros, rel_prominence=rel_prominence, detectors={})
    cand = spectrum_observables(
        E_candidate, candidate, zeros, rel_prominence=rel_prominence, detectors={}
    )
    peak = float(np.abs(reference).max()) if reference.size else 0.0
    return {
        "yield_rel": _relative(cand["yield"], ref["yield"]),
        "centroid_shift_eV": float(cand["centroid_eV"] - ref["centroid_eV"]),
        "fwhm_rel": _relative(cand["fwhm_eV"], ref["fwhm_eV"]),
        "max_pointwise_rel": (
            float(np.abs(candidate - reference).max() / peak) if peak > 0.0 else float("nan")
        ),
    }
