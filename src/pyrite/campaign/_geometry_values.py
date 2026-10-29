"""Scalar, sweep and index validation for public target geometry values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._units import _is_pint_quantity, normalize_unit

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


def _angle_quantity(value: object, *, field: str) -> float | np.ndarray:
    """Explicit finite plane angle(s) in degrees, scalar or one-dimensional sweep.

    Values keep their sign, turn count and order; no wrapping is applied.
    """
    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires a Pint quantity with angle units")
    original = np.asarray(cast(Any, value).magnitude)
    if original.ndim > 1 or original.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real scalar or one-dimensional angle quantity")
    magnitude = np.asarray(
        normalize_unit(value, "degree", field=field, expected="angle", kind="angle")
    )
    if magnitude.size == 0:
        raise ValueError(f"{field} must supply at least one angle")
    with np.errstate(over="ignore", under="ignore"):
        stored = magnitude.astype(float)
    if not np.all(np.isfinite(stored)) or np.any((magnitude != 0) & (stored == 0)):
        raise ValueError(f"{field} must be finite and representable in numeric degree storage")
    return float(stored) if stored.ndim == 0 else stored.copy()


def _scalar_angle_quantity(value: object, *, field: str) -> float:
    """Explicit finite plane angle in degrees, without wrapping or sign changes."""
    if _is_pint_quantity(value) and np.ndim(cast(Any, value).magnitude) != 0:
        raise TypeError(f"{field} must be a real scalar angle quantity")
    return cast(float, _angle_quantity(value, field=field))


def _angle_spelling(
    owner: str,
    numeric_name: str,
    numeric: Any,
    semantic_name: str,
    semantic: object,
    *,
    unset: object,
    default: Any,
    scalar: bool = False,
) -> Any:
    """Resolve one numeric-degree field against its keyword-only semantic angle.

    Supplying both spellings conflicts even at the default; omitting both keeps
    the numeric default.
    """
    if semantic is not unset:
        if numeric is not unset:
            raise TypeError(f"{owner}.{numeric_name} and {semantic_name} are mutually exclusive")
        convert = _scalar_angle_quantity if scalar else _angle_quantity
        return convert(semantic, field=f"{owner}.{semantic_name}")
    return default if numeric is unset else numeric


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
