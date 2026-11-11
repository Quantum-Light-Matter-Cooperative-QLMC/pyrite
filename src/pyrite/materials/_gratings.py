"""Parsing for named ``[gratings.NAME]`` catalog objects.

A grating object records a diffraction grating and its spectrometer mount as
declarative data: groove density, diffraction order and its sign convention,
coating, mount geometry, blank size, and an optional efficiency curve
reference (resolved by the catalog loader). No grating physics runs here.
"""

from collections.abc import Mapping
from types import MappingProxyType

from ._beam_detector_parse import _pair, _unit_number
from ._catalog_decode import _Errors, _table

#: ``m lambda = d (sin alpha + sin beta)`` ("sum") or
#: ``m lambda = d (sin alpha - sin beta)`` ("difference"), with ``d`` the
#: groove spacing and angles measured from the grating normal.
ORDER_CONVENTIONS = frozenset({"sum", "difference"})
_GRATING_KEYS = frozenset(
    {
        "label",
        "description",
        "groove_density_per_mm",
        "order",
        "order_convention",
        "coating",
        "blank_mm",
        "efficiency_curve",
        "mount",
    }
)
_MOUNT_KEYS = frozenset(
    {
        "r_mm",
        "alpha_deg",
        "band_nm",
        "r_prime_mm",
        "beta_deg",
        "beta_H_deg",
        "r_H_prime_mm",
        "focal_plane_length_mm",
    }
)


def _optional_text(row: Mapping[str, object], key: str, path: str, errors: _Errors) -> None:
    value = row.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        errors.add(f"{path}.{key}", "must be a nonempty string")


def _positive(
    row: Mapping[str, object], key: str, path: str, errors: _Errors, unit: str, expected: str
) -> float | None:
    value = _unit_number(row.get(key), f"{path}.{key}", errors, unit, expected=expected)
    if value is None or value <= 0:
        if not errors.has(f"{path}.{key}"):
            errors.add(f"{path}.{key}", f"must be a finite positive number in {unit}")
        return None
    return value


def _angle(row: Mapping[str, object], key: str, path: str, errors: _Errors) -> float | None:
    value = _unit_number(
        row.get(key), f"{path}.{key}", errors, "degree", expected="angle", kind="angle"
    )
    if value is None or not -90.0 < value < 90.0:
        if not errors.has(f"{path}.{key}"):
            errors.add(f"{path}.{key}", "must lie strictly between -90 and 90 degrees")
        return None
    return value


def _band_end_pair(
    row: Mapping[str, object], key: str, path: str, errors: _Errors
) -> tuple[float, float] | None:
    """Decode an ``r_prime_mm`` or ``beta_deg`` pair given at the two band ends."""
    angle = key == "beta_deg"
    unit, expected = ("degree", "angle") if angle else ("mm", "length")
    pair = _pair(row[key], f"{path}.{key}", errors, unit=unit, expected=expected)
    if pair is None:
        return None
    valid = all(-90.0 < item < 90.0 if angle else item > 0 for item in pair)
    if not valid:
        message = "must lie strictly between -90 and 90 degrees" if angle else "must be positive"
        errors.add(f"{path}.{key}", message)
        return None
    return float(pair[0]), float(pair[1])


def _parse_mount(raw: object, path: str, errors: _Errors) -> Mapping[str, object] | None:
    row = _table(raw, path, errors)
    if row is None:
        return None
    errors.keys(row, path, set(_MOUNT_KEYS))
    for key in ("r_mm", "alpha_deg"):
        if key not in row:
            errors.add(f"{path}.{key}", "missing required key")
    out: dict[str, object] = {}
    for key in ("r_mm", "r_H_prime_mm", "focal_plane_length_mm"):
        if key in row:
            out[key] = _positive(row, key, path, errors, "mm", "length")
    for key in ("alpha_deg", "beta_H_deg"):
        if key in row:
            out[key] = _angle(row, key, path, errors)
    if "band_nm" in row:
        band = _pair(row["band_nm"], f"{path}.band_nm", errors, unit="nm", expected="length")
        if band is not None and not 0 < band[0] < band[1]:
            errors.add(f"{path}.band_nm", "must be two increasing positive wavelengths")
        else:
            out["band_nm"] = band
    for key in ("r_prime_mm", "beta_deg"):
        if key not in row:
            continue
        if "band_nm" not in row:
            errors.add(f"{path}.{key}", "requires band_nm (values at the band ends)")
        else:
            out[key] = _band_end_pair(row, key, path, errors)
    return MappingProxyType({key: value for key, value in out.items() if value is not None})


def _parse_grating(row: Mapping[str, object], path: str, errors: _Errors) -> Mapping[str, object]:
    errors.keys(row, path, set(_GRATING_KEYS))
    for key in ("groove_density_per_mm", "order", "order_convention", "mount"):
        if key not in row:
            errors.add(f"{path}.{key}", "missing required key")
    out: dict[str, object] = {}
    if "groove_density_per_mm" in row:
        out["groove_density_per_mm"] = _positive(
            row, "groove_density_per_mm", path, errors, "1/mm", "an inverse length"
        )
    order = row.get("order")
    if "order" in row:
        if type(order) is not int or order == 0:
            errors.add(f"{path}.order", "must be a nonzero integer")
        else:
            out["order"] = order
    convention = row.get("order_convention")
    if "order_convention" in row:
        if convention not in ORDER_CONVENTIONS:
            errors.add(f"{path}.order_convention", f"must be one of {sorted(ORDER_CONVENTIONS)}")
        else:
            out["order_convention"] = convention
    for key in ("label", "description", "coating", "efficiency_curve"):
        _optional_text(row, key, path, errors)
        if isinstance(row.get(key), str):
            out[key] = str(row[key]).strip()
    if "blank_mm" in row:
        blank = _pair(row["blank_mm"], f"{path}.blank_mm", errors, unit="mm", expected="length")
        if blank is not None and any(item <= 0 for item in blank):
            errors.add(f"{path}.blank_mm", "must contain positive lengths")
        elif blank is not None:
            out["blank_mm"] = blank
    if "mount" in row:
        out["mount"] = _parse_mount(row["mount"], f"{path}.mount", errors)
    return MappingProxyType({key: value for key, value in out.items() if value is not None})


def parse_gratings(raw: object, errors: _Errors) -> dict[str, Mapping[str, object]]:
    """Validate named ``[gratings.NAME]`` objects; curve names stay unresolved."""
    table = _table(raw, "gratings", errors)
    if table is None:
        return {}
    gratings: dict[str, Mapping[str, object]] = {}
    for key, value in table.items():
        path = f"gratings.{key}"
        row = _table(value, path, errors)
        if row is not None:
            gratings[key] = _parse_grating(row, path, errors)
    return gratings


__all__ = ["ORDER_CONVENTIONS", "parse_gratings"]
