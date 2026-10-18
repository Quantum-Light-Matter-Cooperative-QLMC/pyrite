"""Common per-primary comparison schema for opt-in cross-engine benchmarks.

Every engine adapter reduces its native output to one score per incident
primary and observable before anything is compared. This module checks the
units and invariants that make those arrays comparable, records failed or
unmet outcomes explicitly, and integrates point-direction estimators over a
finite cone acceptance with a quadrature convergence check.
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


def validate_primary_scores(
    scores: Mapping[str, Any], n_primaries: int, energy_keV: float
) -> dict[str, NDArray[np.float64]]:
    """Return validated float arrays, or raise on any broken invariant.

    Each primary has exactly one terminal indicator; transmitted energy lies in
    ``[0, E0]`` and is zero for every non-transmitted primary; photon counts
    are nonnegative integers whose fixed bands partition the 10 keV total, and
    no band starting at or above the endpoint scores anything.
    """
    if n_primaries < 2:
        raise ValueError("need at least two incident primaries")
    out: dict[str, NDArray[np.float64]] = {}
    for name, value in scores.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (n_primaries,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} needs one finite score per incident primary")
        out[name] = array
    missing = [n for n in (*TERMINAL_FRACTIONS, "photons_ge10kev", *BAND_FIELDS) if n not in out]
    if missing:
        raise ValueError(f"missing observables: {', '.join(missing)}")
    _check_terminals(out, energy_keV)
    _check_photon_counts(out, energy_keV)
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
) -> dict[str, Any]:
    """Build one schema record; unsuccessful outcomes keep a reason, never scores.

    Successful records carry primary moments for every observable and the
    covariance of the fixed photon bands, so integrated yields keep their
    within-primary correlation.
    """
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    record: dict[str, Any] = {
        "schema": SCHEMA,
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
    arrays = validate_primary_scores(scores, n_primaries, energy_keV)
    record["moments"] = {name: primary_moments(values) for name, values in arrays.items()}
    bands = np.stack([arrays[name] for name in BAND_FIELDS])
    # Covariance of the band means: sample covariance over primaries divided by n.
    record["band_mean_covariance"] = {
        "fields": list(BAND_FIELDS),
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
    values = np.asarray(f(dirs), dtype=float).reshape(n_theta, n_phi)
    return float(np.sum(w_mu[:, None] * values) * (2.0 * math.pi / n_phi))


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
