"""Scalar, sweep and index validation for public target geometry values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pint import Quantity

ScalarOrSeq = float | Sequence[float] | np.ndarray
Miller = tuple[int, int, int]


def _seq(x):
    """Normalize a scalar-or-sequence into a 1-D float array, order preserved."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def _unique_angles(values: ScalarOrSeq) -> np.ndarray:
    """Preserve degree values and order, removing only exact duplicates."""
    return np.asarray(list(dict.fromkeys(float(value) for value in _seq(values))), dtype=float)


def _reject_banned_angles(
    tilts: np.ndarray, azimuths: np.ndarray, *, allow_normal_incidence: bool = False
) -> None:
    """Refuse the degenerate geometries no emission sweep may use (issue_notes.md #1).

    polar tilt == 0 deg radiates zero coherent-line intensity (the tilt=0
    degeneracy the line-grid coverage search flags), and azimuth == 90 deg sits
    on a symmetry axis that mis-ranked the coverage search (the azim-90 bug).
    Guarding at construction makes every downstream path -- production sweeps,
    catalog grids, the analyze_line_grid_bounds diagnostic -- structurally
    incapable of silently sampling them; callers must pick a small nonzero tilt
    (e.g. 5 deg) and an azimuth off 90 (e.g. 100-180 deg) instead.

    ``allow_normal_incidence`` is the transport-only escape hatch: the
    penetration-depth study is not an emission sweep and uses tilt=0 as its
    physical baseline, so it opts out of the tilt check only. The azimuth==90 ban
    is unconditional -- it has no legitimate use.
    """
    if not allow_normal_incidence and np.any(np.isclose(tilts, 0.0)):
        raise ValueError(
            "polar tilt_deg == 0 is disallowed for emission sweeps (degenerate "
            "zero coherent-line geometry); use a small nonzero tilt such as 5 deg, "
            "or set allow_normal_incidence=True for a transport-only study"
        )
    if np.any(np.isclose(azimuths, 90.0)):
        raise ValueError(
            "tilt_azim_deg == 90 is disallowed (degenerate symmetry axis, the "
            "azim-90 ranking bug); use an azimuth off 90 such as 100-180 deg"
        )


def _positive_values(name: str, values: ScalarOrSeq) -> np.ndarray:
    """One vocabulary for every swept positive-length geometry field."""
    try:
        numbers = _seq(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number or a sequence of numbers") from exc
    if numbers.size == 0:
        raise ValueError(f"{name} must supply at least one value")
    if not np.all(np.isfinite(numbers) & (numbers > 0.0)):
        raise ValueError(f"{name} must be finite and positive")
    return numbers


def _quantity_geometry(value: ScalarOrSeq, unit: str) -> Quantity:
    """Return detached, registry-owned read-back, preserving sweep dimensionality."""
    from ..units import Q_

    magnitude = np.asarray(value)
    return Q_(float(magnitude) if magnitude.ndim == 0 else magnitude.copy(), unit)


def _as_miller(value: Any) -> Miller | None:
    if value is None:
        return None
    h, k, l = (int(i) for i in value)  # noqa: E741
    return (h, k, l)


def _miller(name: str, value: Any) -> Miller | None:
    if value is None:
        return None
    indices = tuple(value) if isinstance(value, (Sequence, np.ndarray)) else None
    if (
        indices is None
        or len(indices) != 3
        or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices)
        or not any(indices)
    ):
        raise ValueError(f"CrystalCut.{name} must be three integer indices, not all zero")
    return _as_miller(indices)
