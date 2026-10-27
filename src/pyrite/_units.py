"""Lazy unit-normalization helpers for public configuration boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np

_ANGLE_MARKERS = ("rad", "degree", "deg", "arcminute", "arcsecond", "turn", "grad")
_SOLID_ANGLE_MARKERS = ("steradian", "sr")


def _is_pint_quantity(value: object) -> bool:
    """Recognize Pint quantities without importing Pint on numeric paths."""
    return (
        type(value).__module__.split(".", maxsplit=1)[0] == "pint"
        and hasattr(value, "magnitude")
        and hasattr(value, "units")
        and hasattr(value, "to")
    )


def normalize_unit(
    value: Any,
    target_unit: str,
    *,
    field: str,
    expected: str,
    kind: str | None = None,
) -> Any:
    """Convert Pint values at an input boundary, leaving legacy numbers alone.

    Tuples and lists are traversed so callers may mix ordinary components with
    scalar quantities, e.g. ``(0, 0, 10 * ureg.cm)``. A Pint quantity whose
    magnitude is an ndarray is converted as one vector operation.
    """
    if _is_pint_quantity(value):
        received = str(value.units)
        if kind == "angle" and not any(marker in received.lower() for marker in _ANGLE_MARKERS):
            raise ValueError(
                f"{field} received unit {received!r}; expected an angle convertible to "
                f"{target_unit!r}"
            )
        if kind == "solid_angle" and not any(
            marker in received.lower() for marker in _SOLID_ANGLE_MARKERS
        ):
            raise ValueError(
                f"{field} received unit {received!r}; expected a solid angle convertible to "
                f"{target_unit!r}"
            )
        try:
            converted = value.to(target_unit)
        except Exception as exc:
            # Pint uses its own DimensionalityError and UndefinedUnitError types;
            # keep Pint out of the module-level imports for numeric paths.
            raise ValueError(
                f"{field} received unit {received!r}; expected {expected} convertible to "
                f"{target_unit!r}"
            ) from exc
        magnitude = converted.magnitude
        dtype = getattr(magnitude, "dtype", None)
        if isinstance(magnitude, complex) or getattr(dtype, "kind", None) == "c":
            raise TypeError(f"{field} must be real; received complex {expected}")
        return magnitude
    if isinstance(value, tuple):
        return tuple(
            normalize_unit(item, target_unit, field=field, expected=expected, kind=kind)
            for item in value
        )
    if isinstance(value, list):
        return [
            normalize_unit(item, target_unit, field=field, expected=expected, kind=kind)
            for item in value
        ]
    return value


def _positive_length_quantity(
    value: object, *, unit: str, field: str, shape: tuple[int, ...] | None = None
) -> float | np.ndarray:
    """Normalize an explicit length, retaining scalar/sweep shape and float range."""
    import numpy as np

    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires a Pint quantity with length units")
    original = np.asarray(cast(Any, value).magnitude)
    if original.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real scalar or one-dimensional length quantity")
    magnitude = normalize_unit(value, unit, field=field, expected="length")
    source = np.asarray(magnitude)
    if source.dtype.kind not in "iuf" or source.ndim > 1:
        raise TypeError(f"{field} must be a real scalar or one-dimensional length quantity")
    if shape is not None and source.shape != shape:
        raise TypeError(f"{field} must have shape {shape}; received {source.shape}")
    if source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError(f"{field} must contain nonempty finite lengths")
    with np.errstate(over="ignore", under="ignore"):
        stored = source.astype(float)
    if not np.all(np.isfinite(stored)) or np.any((source != 0) & (stored == 0)):
        raise ValueError(f"{field} is not representable in numeric {unit} storage")
    if not np.all(stored > 0.0):
        raise ValueError(f"{field} must be finite and positive")
    return float(stored) if stored.ndim == 0 else stored.copy()


def parse_unit_string(
    value: str,
    target_unit: str,
    *,
    field: str,
    expected: str,
    kind: str | None = None,
) -> Any:
    """Parse a user-facing quantity string through the shared project registry."""
    from .units import ureg

    try:
        quantity = ureg.Quantity(value)
    except Exception as exc:
        raise ValueError(
            f"{field} could not parse {value!r}; expected a {expected} quantity "
            f"compatible with {target_unit!r}"
        ) from exc
    return normalize_unit(quantity, target_unit, field=field, expected=expected, kind=kind)


def normalize_unit_input(
    value: Any,
    target_unit: str,
    *,
    field: str,
    expected: str,
    kind: str | None = None,
) -> Any:
    """Normalize a CLI/profile value, accepting explicit unit strings."""
    if isinstance(value, str):
        return parse_unit_string(value, target_unit, field=field, expected=expected, kind=kind)
    if isinstance(value, tuple):
        return tuple(
            normalize_unit_input(item, target_unit, field=field, expected=expected, kind=kind)
            for item in value
        )
    if isinstance(value, list):
        return [
            normalize_unit_input(item, target_unit, field=field, expected=expected, kind=kind)
            for item in value
        ]
    return normalize_unit(value, target_unit, field=field, expected=expected, kind=kind)


__all__ = ["normalize_unit", "normalize_unit_input", "parse_unit_string"]
