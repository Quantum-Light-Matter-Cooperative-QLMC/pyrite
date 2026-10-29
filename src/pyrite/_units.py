"""Lazy unit-normalization helpers for public configuration boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np


def _is_pint_quantity(value: object) -> bool:
    """Recognize Pint quantities without importing Pint on numeric paths."""
    return (
        type(value).__module__.split(".", maxsplit=1)[0] == "pint"
        and hasattr(value, "magnitude")
        and hasattr(value, "units")
        and hasattr(value, "to")
    )


def _root_unit_powers(value: Any) -> dict[str, float]:
    """Root-unit exponents, e.g. ``{"radian": 1}`` for any angle unit.

    Pint keeps radians as a dimensionless base unit, so angles, solid angles
    and bare ratios are told apart by root units rather than dimensionality
    or unit-name substrings (``"steradian"`` contains ``"rad"``).
    """
    _, root = value._REGISTRY.get_root_units(value.units)
    return dict(root._units)


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
        if kind == "angle" and _root_unit_powers(value) != {"radian": 1}:
            raise ValueError(
                f"{field} received unit {received!r}; expected an angle convertible to "
                f"{target_unit!r}"
            )
        if kind == "temperature" and any(name.startswith("delta_") for name in value._units):
            raise ValueError(
                f"{field} received temperature difference {received!r}; expected an "
                f"absolute temperature convertible to {target_unit!r} (e.g. K or degC)"
            )
        if kind == "solid_angle" and _root_unit_powers(value) != {"radian": 2}:
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


def reject_quantity(value: Any, *, field: str, unit: str) -> None:
    """Refuse Pint quantities at a numeric-only boundary.

    Numeric-only boundaries store plain numbers in a fixed unit and do not
    convert. A quantity would otherwise be stripped to its magnitude or kept
    unconverted, so it is refused with the expected numeric unit instead.
    Tuples and lists are traversed; plain numbers and arrays pass unchanged.
    """
    if _is_pint_quantity(value):
        raise TypeError(
            f"{field} is numeric-only and does not accept Pint quantities "
            f"(received unit {str(value.units)!r}); pass plain numbers in {unit!r}"
        )
    if isinstance(value, (tuple, list)):
        for item in value:
            reject_quantity(item, field=field, unit=unit)


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


def _is_pint_unit(value: object) -> bool:
    """Recognize Pint unit objects without importing Pint."""
    return (
        type(value).__module__.split(".", maxsplit=1)[0] == "pint"
        and hasattr(value, "dimensionality")
        and not hasattr(value, "magnitude")
    )


def parse_unit_expression(value: Any, *, field: str) -> str:
    """Validate a declared unit, returning its identity-preserving text.

    Strings are parsed through the shared registry and returned unchanged, so
    persisted identities built from the text stay byte-identical; the empty
    string means dimensionless and does not import Pint. Pint ``Unit`` objects
    are rendered with ``str``. Anything unparseable is refused.
    """
    if _is_pint_unit(value):
        return str(value)
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a unit string or Pint unit (empty means dimensionless)")
    if not value.strip():
        return value
    from .units import ureg

    try:
        ureg.Unit(value)
    except Exception as exc:
        raise ValueError(
            f"{field} could not parse unit {value!r}; expected a Pint unit expression "
            "(empty means dimensionless)"
        ) from exc
    return value


def unit_conversion_factor(source: str, target: str) -> float | None:
    """Return the factor from ``source`` to ``target`` units, or ``None``.

    Units are compatible only when their Pint *root* units agree, so radians
    and steradians are not interchangeable with bare ratios even though Pint
    treats all three as dimensionless.
    """
    from .units import ureg

    source_factor, source_root = ureg.get_root_units(ureg.Unit(source.strip() or "dimensionless"))
    target_factor, target_root = ureg.get_root_units(ureg.Unit(target.strip() or "dimensionless"))
    if source_root != target_root:
        return None
    return float(cast(Any, source_factor)) / float(cast(Any, target_factor))


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


__all__ = [
    "normalize_unit",
    "normalize_unit_input",
    "parse_unit_expression",
    "parse_unit_string",
    "reject_quantity",
    "unit_conversion_factor",
]
