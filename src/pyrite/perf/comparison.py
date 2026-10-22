"""Common per-primary comparison schema for opt-in cross-engine benchmarks.

Every engine adapter reduces its native output to one score per incident
primary and observable before anything is compared. This module checks the
units and invariants that make those arrays comparable, records failed or
unmet outcomes explicitly, and integrates point-direction estimators over a
finite cone acceptance with a quadrature convergence check.

Three record kinds share the schema: ``source`` (creation-level photon
counts), ``transport`` (terminal fractions and transmitted energy only, no
radiation scores) and ``detector`` (transport plus weighted photon yields in a
cone, at creation and after uncollided escape).
"""

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .benchmark_matrix import SOURCE_BANDS_KEV, TERMINAL_FRACTIONS
from .precision import primary_moments

SCHEMA = "pyrite.equal-precision.comparison.v1"
STATUSES = (
    "ok",
    "failed",
    "timeout",
    "oom",
    "precision-unmet",
    "coverage-gap",
    "unsupported-physics",
)
BAND_FIELDS = tuple(name for name, _low, _high in SOURCE_BANDS_KEV) + ("photons_ge100kev",)
KINDS = ("source", "transport", "detector")
TRANSPORT_FIELDS = (*TERMINAL_FRACTIONS, "transmitted_energy_keV_per_primary")
# Weighted cone-detector bands [keV]: same edges as the creation-level bands.
DETECTOR_BANDS_KEV = (
    ("10_50kev", 10.0, 50.0),
    ("50_100kev", 50.0, 100.0),
    ("ge100kev", 100.0, math.inf),
)
DETECTOR_STAGES = ("source", "escape")


def detector_field(stage: str, band: str) -> str:
    """Per-primary cone yield field, e.g. ``cone_escape_photons_10_50kev``."""
    return f"cone_{stage}_photons_{band}"


DETECTOR_BAND_FIELDS = tuple(
    detector_field(stage, band) for stage in DETECTOR_STAGES for band, _l, _h in DETECTOR_BANDS_KEV
)
DETECTOR_TOTAL_FIELDS = tuple(detector_field(stage, "ge10kev") for stage in DETECTOR_STAGES)


def validate_primary_scores(
    scores: Mapping[str, Any], n_primaries: int, energy_keV: float, *, kind: str = "source"
) -> dict[str, NDArray[np.float64]]:
    """Return validated float arrays, or raise on any broken invariant.

    Each primary has exactly one terminal indicator; transmitted energy lies in
    ``[0, E0]`` and is zero for every non-transmitted primary. ``source``
    photon counts are nonnegative integers whose fixed bands partition the
    10 keV total. ``transport`` records carry no photon or cone field at all.
    ``detector`` cone yields are nonnegative weights whose bands sum to the
    10 keV total, and uncollided escape never exceeds the yield at creation.
    In every kind, no band starting at or above the endpoint scores anything.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown record kind {kind!r}")
    if n_primaries < 2:
        raise ValueError("need at least two incident primaries")
    out: dict[str, NDArray[np.float64]] = {}
    for name, value in scores.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (n_primaries,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} needs one finite score per incident primary")
        out[name] = array
    required = {
        "source": (*TERMINAL_FRACTIONS, "photons_ge10kev", *BAND_FIELDS),
        "transport": TRANSPORT_FIELDS,
        "detector": (*TRANSPORT_FIELDS, *DETECTOR_BAND_FIELDS, *DETECTOR_TOTAL_FIELDS),
    }[kind]
    missing = [n for n in required if n not in out]
    if missing:
        raise ValueError(f"missing observables: {', '.join(missing)}")
    _check_terminals(out, energy_keV)
    if kind == "source":
        _check_photon_counts(out, energy_keV)
    elif kind == "transport":
        radiative = [n for n in out if n.startswith(("photons_", "cone_"))]
        if radiative:
            raise ValueError(f"transport-only records carry no radiation scores: {radiative}")
    else:
        _check_cone_yields(out, energy_keV)
    return out


def _check_terminals(out: Mapping[str, NDArray[np.float64]], energy_keV: float) -> None:
    terminals = np.stack([out[name] for name in TERMINAL_FRACTIONS])
    if not np.all(np.isin(terminals, (0.0, 1.0))) or not np.all(terminals.sum(axis=0) == 1.0):
        raise ValueError("every primary needs exactly one terminal state")
    if "transmitted_energy_keV_per_primary" in out:
        energy = out["transmitted_energy_keV_per_primary"]
        if np.any(energy < 0.0) or np.any(energy > energy_keV):
            raise ValueError("transmitted energy outside [0, E0]")
        if np.any(energy[out["transmitted"] == 0.0] != 0.0):
            raise ValueError("energy scored for a non-transmitted primary")


def _check_photon_counts(out: Mapping[str, NDArray[np.float64]], energy_keV: float) -> None:
    for name in ("photons_ge10kev", *BAND_FIELDS):
        if np.any(out[name] < 0.0) or np.any(out[name] != np.round(out[name])):
            raise ValueError(f"{name} must hold nonnegative integer counts")
    if not np.array_equal(sum(out[name] for name in BAND_FIELDS), out["photons_ge10kev"]):
        raise ValueError("fixed photon bands do not partition the 10 keV total")
    lows = {name: low for name, low, _high in SOURCE_BANDS_KEV} | {"photons_ge100kev": 100.0}
    for name, low in lows.items():
        if low >= energy_keV and np.any(out[name] != 0.0):
            raise ValueError(f"{name} scored above the beam endpoint")


def _check_cone_yields(out: Mapping[str, NDArray[np.float64]], energy_keV: float) -> None:
    for stage in DETECTOR_STAGES:
        bands = [out[detector_field(stage, band)] for band, _l, _h in DETECTOR_BANDS_KEV]
        total = out[detector_field(stage, "ge10kev")]
        if any(np.any(values < 0.0) for values in (*bands, total)):
            raise ValueError(f"{stage} cone yields must be nonnegative")
        if not np.allclose(sum(bands), total, rtol=1e-12, atol=0.0):
            raise ValueError(f"{stage} cone bands do not partition the 10 keV total")
        for (band, low, _high), values in zip(DETECTOR_BANDS_KEV, bands, strict=True):
            if low >= energy_keV and np.any(values != 0.0):
                raise ValueError(f"{detector_field(stage, band)} scored above the beam endpoint")
    for band, _low, _high in (*DETECTOR_BANDS_KEV, ("ge10kev", 0.0, 0.0)):
        source = out[detector_field("source", band)]
        escape = out[detector_field("escape", band)]
        # Beer-Lambert transmission is at most one on every segment.
        if np.any(escape > source * (1.0 + 1e-12)):
            raise ValueError("uncollided escape exceeds the yield at creation")


def transmitted_energy_moments(
    transmitted: Any, energy_keV: Any
) -> dict[str, int | float | str | None]:
    """Moments of transmitted energy over transmitted primaries [keV, keV^2].

    The conditional mean is the ratio of two per-primary means, so its SE uses
    the delta method on ``E_i - r T_i`` over all incident primaries.
    """
    t = np.asarray(transmitted, dtype=np.float64)
    e = np.asarray(energy_keV, dtype=np.float64)
    n, n_t = t.size, int(t.sum())
    if n_t == 0:
        return {"n_transmitted": 0, "status": "unresolved"}
    mean = float(e.sum() / n_t)
    second = float(np.dot(e, e) / n_t)
    se = float(np.std(e - mean * t, ddof=1) / (math.sqrt(n) * (n_t / n))) if n_t > 1 else None
    return {
        "n_transmitted": n_t,
        "mean_keV": mean,
        "second_moment_keV2": second,
        "variance_keV2": float(np.var(e[t == 1.0], ddof=1)) if n_t > 1 else None,
        "mean_se_keV": se,
        "status": "estimable" if se else "unresolved",
    }


def comparison_record(
    *,
    case_id: str,
    engine: str,
    arm: str,
    status: str,
    n_primaries: int,
    energy_keV: float,
    seed: int | None,
    scores: Mapping[str, Any] | None = None,
    reason: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    kind: str = "source",
) -> dict[str, Any]:
    """Build one schema record; unsuccessful outcomes keep a reason, never scores.

    Successful records carry primary moments for every observable and, for
    photon kinds, the covariance of the band means, so integrated yields keep
    their within-primary correlation. Every kind also carries the conditional
    moments of transmitted energy.
    """
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if kind not in KINDS:
        raise ValueError(f"unknown record kind {kind!r}")
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": kind,
        "case_id": case_id,
        "engine": engine,
        "arm": arm,
        "status": status,
        "n_primaries": int(n_primaries),
        "energy_keV": float(energy_keV),
        "seed": seed,
        "normalization": "per incident primary, including zero-score primaries",
        "provenance": dict(provenance or {}),
    }
    if status != "ok":
        if not reason or scores is not None:
            raise ValueError("unsuccessful outcomes need a reason and carry no scores")
        record["reason"] = reason
        return record
    if scores is None:
        raise ValueError("a successful outcome needs per-primary scores")
    arrays = validate_primary_scores(scores, n_primaries, energy_keV, kind=kind)
    record["moments"] = {name: primary_moments(values) for name, values in arrays.items()}
    if "transmitted_energy_keV_per_primary" in arrays:
        record["transmitted_energy"] = transmitted_energy_moments(
            arrays["transmitted"], arrays["transmitted_energy_keV_per_primary"]
        )
    fields = {"source": BAND_FIELDS, "detector": DETECTOR_BAND_FIELDS}.get(kind)
    if fields:
        bands = np.stack([arrays[name] for name in fields])
        # Covariance of the band means: sample covariance over primaries divided by n.
        record["band_mean_covariance"] = {
            "fields": list(fields),
            "matrix": (np.cov(bands, ddof=1) / n_primaries).tolist(),
        }
    return record


def cone_solid_angle(half_angle_rad: float) -> float:
    """Solid angle of a circular cone, ``2 pi (1 - cos alpha) = 4 pi sin^2(alpha/2)``."""
    return 4.0 * math.pi * math.sin(0.5 * half_angle_rad) ** 2


def integrate_cone(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    axis: Any,
    half_angle_rad: float,
    n_theta: int,
    n_phi: int,
) -> float:
    """Integrate ``f(directions)`` over a cone about ``axis`` [sr * f].

    Gauss-Legendre nodes in ``cos(theta)`` on ``[cos(alpha), 1]`` and a
    uniform periodic rule in azimuth; ``f`` receives an ``(N, 3)`` array of
    unit vectors and returns ``N`` values.
    """
    dirs, weights = cone_rule(axis, half_angle_rad, n_theta, n_phi)
    values = np.asarray(f(dirs), dtype=float).reshape(dirs.shape[0])
    return float(weights @ values)


def cone_rule(
    axis: Any, half_angle_rad: float, n_theta: int, n_phi: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit directions ``(N, 3)`` and solid-angle weights ``(N,)`` [sr] of the cone rule."""
    if not 0.0 < half_angle_rad <= math.pi or n_theta < 1 or n_phi < 1:
        raise ValueError("invalid cone or quadrature order")
    w = np.asarray(axis, dtype=float)
    w = w / np.linalg.norm(w)
    helper = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, helper)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    c0 = math.cos(half_angle_rad)
    mu = 0.5 * (1.0 - c0) * x + 0.5 * (1.0 + c0)
    w_mu = 0.5 * (1.0 - c0) * wx
    phi = (np.arange(n_phi) + 0.5) * (2.0 * math.pi / n_phi)
    s = np.sqrt(np.clip(1.0 - mu * mu, 0.0, None))
    dirs = (
        mu[:, None, None] * w
        + (s[:, None] * np.cos(phi))[:, :, None] * u
        + (s[:, None] * np.sin(phi))[:, :, None] * v
    ).reshape(-1, 3)
    weights = np.repeat(w_mu * (2.0 * math.pi / n_phi), n_phi)
    return dirs, weights


def converged_cone_integral(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    axis: Any,
    half_angle_rad: float,
    *,
    rtol: float,
    start: int = 4,
    max_order: int = 256,
) -> dict[str, Any]:
    """Double both quadrature orders until successive results agree to ``rtol``.

    Returns the finer estimate with its last change; an unconverged result is
    reported with ``converged = False`` rather than raised, so it stays evidence.
    Repeated zero estimates are unresolved, like zero sample variance.
    """
    if start < 1 or 2 * start > max_order:
        raise ValueError("need room for at least one order doubling")
    n = start
    change = math.inf
    previous = integrate_cone(f, axis, half_angle_rad, n, 2 * n)
    while 2 * n <= max_order:
        n *= 2
        current = integrate_cone(f, axis, half_angle_rad, n, 2 * n)
        change = abs(current - previous)
        # Two zero estimates can both miss a narrow feature; never call that converged.
        if current != 0.0 and change <= rtol * abs(current):
            return {"value": current, "change": change, "order": n, "converged": True}
        previous = current
    return {"value": previous, "change": change, "order": n, "converged": False}


def converged_cone_scores(
    f: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    axis: Any,
    half_angle_rad: float,
    *,
    rtol: float,
    exact_zero: Any = (),
    start: int = 2,
    max_order: int = 32,
) -> dict[str, Any]:
    """Cone-integrate per-primary scores, refining until their moments converge.

    ``f(directions, order)`` returns ``(N, n_primaries, n_fields)`` values at
    ``N`` unit directions; it may refine any inner rule (for example photon
    energy) with ``order`` so the refinement is joint. The angular rule is
    :func:`cone_rule` with ``order`` polar and ``2 * order`` azimuthal nodes,
    doubled until, for every field, the primary sum and the primary sum of
    squares (the inputs of the mean and its variance) each change by at most
    ``rtol``. A field whose sums stay zero is unresolved unless listed in
    ``exact_zero`` (structurally zero, e.g. a band above the endpoint), which
    must then be zero at every order. Unconverged results are returned with
    ``converged = False``, never raised.
    """
    if start < 1 or 2 * start > max_order or not rtol > 0.0:
        raise ValueError("need a positive rtol and room for one order doubling")
    exact = np.asarray(exact_zero, dtype=bool)

    def evaluate(order: int) -> NDArray[np.float64]:
        dirs, weights = cone_rule(axis, half_angle_rad, order, 2 * order)
        values = np.asarray(f(dirs, order), dtype=np.float64)
        if values.ndim != 3 or values.shape[0] != dirs.shape[0]:
            raise ValueError("f must return (directions, primaries, fields) values")
        return np.tensordot(weights, values, axes=1)

    order = start
    previous = evaluate(order)
    fields: list[dict[str, Any]] = []
    converged = False
    while 2 * order <= max_order and not converged:
        order *= 2
        current = evaluate(order)
        fields = _moment_changes(previous, current, exact, rtol)
        converged = all(field["status"] in {"converged", "exact-zero"} for field in fields)
        previous = current
    return {"scores": previous, "order": order, "converged": converged, "fields": fields}


def _moment_changes(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    exact: NDArray[np.bool_],
    rtol: float,
) -> list[dict[str, Any]]:
    out = []
    for j in range(current.shape[1]):
        sums = [float(previous[:, j].sum()), float(current[:, j].sum())]
        squares = [float(np.dot(a[:, j], a[:, j])) for a in (previous, current)]
        change = abs(sums[1] - sums[0])
        change_sq = abs(squares[1] - squares[0])
        if exact.size and exact[j]:
            zero = not np.any(previous[:, j]) and not np.any(current[:, j])
            status = "exact-zero" if zero else "nonzero-exact-zero-field"
        elif sums[1] == 0.0:
            status = "unresolved-zero"
        elif change <= rtol * abs(sums[1]) and change_sq <= rtol * squares[1]:
            status = "converged"
        else:
            status = "unconverged"
        out.append(
            {
                "sum": sums[1],
                "sum_change": change,
                "sum_squares": squares[1],
                "sum_squares_change": change_sq,
                "status": status,
            }
        )
    return out
