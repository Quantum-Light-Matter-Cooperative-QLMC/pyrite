"""Tabulated instrument curves stored under a catalog's ``curves/`` directory.

A curve is a dimensionless response sampled on an energy or wavelength
abscissa: detector quantum efficiency, grating efficiency, filter
transmission, mirror reflectance. Catalog objects reference a curve by name
(``qe_curve = "NAME"``, ``efficiency_curve = "NAME"``); the name resolves to
exactly one of ``curves/NAME.csv`` or ``curves/NAME.toml`` in the referencing
object's catalog root. Paths must stay inside ``curves/`` after symlink
resolution, as for ``cifs/``.

CSV form: ``#`` comment lines hold TOML provenance keys, then one header row
names the abscissa and its unit and the optional value unit, then numeric
rows::

    # source = "Vendor datasheet DS-1"
    # revision = "B"
    # digitization = "WebPlotDigitizer, 1 nm steps"
    # confidential = true
    # quantity = "quantum_efficiency"
    wavelength [nm], value [percent]
    10.0, 31.0
    20.0, 45.5

TOML form: the same provenance keys at top level plus ``abscissa``
(``"energy"`` or ``"wavelength"``), ``abscissa_unit``, optional ``value_unit``,
and equal-length ``x`` / ``y`` arrays.

The abscissa must be finite and strictly increasing in file order. Units are
converted through the shared Pint registry: energies to eV, wavelengths to
energy in eV through ``E = h c / lambda``, and values to a dimensionless
fraction (``percent`` is accepted). Every supported quantity is a probability
or fraction, so converted values must lie in ``[0, 1]``.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .._catalog_layout import _STEM_RE, CURVE_DIR
from .._units import unit_conversion_factor
from ._catalog_decode import _Errors

CURVE_QUANTITIES = frozenset({"efficiency", "quantum_efficiency", "transmission", "reflectance"})
_ABSCISSAE = {"energy": "eV", "wavelength": "nm"}
_PROVENANCE_KEYS = frozenset({"source", "revision", "digitization", "confidential", "quantity"})
_TOML_KEYS = _PROVENANCE_KEYS | {"abscissa", "abscissa_unit", "value_unit", "x", "y", "label"}
_CSV_HEADER_RE = re.compile(
    r"^\s*(?P<abscissa>[A-Za-z_]+)\s*\[(?P<unit>[^\]]+)\]\s*,"
    r"\s*value\s*(?:\[(?P<value_unit>[^\]]*)\])?\s*$"
)


class CurveError(ValueError):
    """A curve file is missing, unsafe, or invalid."""


@dataclass(frozen=True, slots=True)
class TabulatedCurve:
    """One validated, dimensionless curve on an ascending energy grid.

    ``energy_eV`` is strictly increasing; ``value`` is the matching
    dimensionless fraction in ``[0, 1]``. ``abscissa`` and ``abscissa_unit``
    record the file's original sampling for display only.
    """

    name: str
    path: Path
    quantity: str
    energy_eV: tuple[float, ...]
    value: tuple[float, ...]
    abscissa: str
    abscissa_unit: str
    source: str
    digitization: str
    confidential: bool
    revision: str | None = None
    label: str | None = None


def _text(meta: Mapping[str, object], key: str, *, required: bool) -> str | None:
    value = meta.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CurveError(f"{key} must be a nonempty string")
    return value.strip()


def _floats(values: object, field: str) -> list[float]:
    if not isinstance(values, list) or len(values) < 2:
        raise CurveError(f"{field} must be an array of at least two numbers")
    out = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CurveError(f"{field}[{index}] must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise CurveError(f"{field}[{index}] must be finite")
        out.append(number)
    return out


def _read_csv(text: str) -> tuple[dict[str, object], list[float], list[float]]:
    meta_lines: list[str] = []
    header: re.Match[str] | None = None
    x: list[float] = []
    y: list[float] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if header is not None:
                raise CurveError(f"line {number}: provenance comments must precede the header")
            meta_lines.append(stripped[1:].strip())
            continue
        if header is None:
            header = _CSV_HEADER_RE.match(stripped)
            if header is None:
                raise CurveError(
                    f"line {number}: expected a header like 'wavelength [nm], value' "
                    "or 'energy [eV], value [percent]'"
                )
            continue
        cells = [cell.strip() for cell in stripped.split(",")]
        if len(cells) != 2:
            raise CurveError(f"line {number}: expected two comma-separated numbers")
        try:
            pair = [float(cell) for cell in cells]
        except ValueError:
            raise CurveError(f"line {number}: expected two numbers") from None
        if not all(math.isfinite(item) for item in pair):
            raise CurveError(f"line {number}: values must be finite")
        x.append(pair[0])
        y.append(pair[1])
    if header is None:
        raise CurveError("missing header row")
    try:
        meta: dict[str, object] = tomllib.loads("\n".join(meta_lines))
    except tomllib.TOMLDecodeError as exc:
        raise CurveError(f"provenance comments are not TOML key = value lines ({exc})") from None
    unknown = sorted(set(meta) - _PROVENANCE_KEYS - {"label"})
    if unknown:
        raise CurveError(f"unknown provenance keys {unknown}")
    meta["abscissa"] = header["abscissa"]
    meta["abscissa_unit"] = header["unit"].strip()
    meta["value_unit"] = (header["value_unit"] or "").strip()
    if len(x) < 2:
        raise CurveError("needs at least two data rows")
    return meta, x, y


def _read_toml(text: str) -> tuple[dict[str, object], list[float], list[float]]:
    try:
        meta: dict[str, object] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CurveError(str(exc)) from None
    unknown = sorted(set(meta) - _TOML_KEYS)
    if unknown:
        raise CurveError(f"unknown keys {unknown}")
    x = _floats(meta.get("x"), "x")
    y = _floats(meta.get("y"), "y")
    if len(x) != len(y):
        raise CurveError(f"x and y must have equal lengths ({len(x)} != {len(y)})")
    return meta, x, y


def _to_energy_eV(abscissa: str, unit: str, x: list[float]) -> list[float]:
    target = _ABSCISSAE[abscissa]
    try:
        factor = unit_conversion_factor(unit, target)
    except Exception:
        factor = None
    if factor is None:
        kind = "an energy" if abscissa == "energy" else "a length"
        raise CurveError(f"abscissa unit {unit!r} is not {kind} unit convertible to {target!r}")
    converted = [item * factor for item in x]
    if abscissa == "energy":
        if converted[0] < 0:
            raise CurveError("energies must be nonnegative")
        return converted
    if converted[0] <= 0:
        raise CurveError("wavelengths must be positive")
    from ..units import ureg

    energies = ureg.Quantity(converted, "nm").to("eV", "sp").magnitude
    return [float(item) for item in reversed(energies)]


def parse_curve(name: str, path: Path) -> TabulatedCurve:
    """Read and validate one curve file; raise :class:`CurveError` on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CurveError(f"cannot read ({exc})") from None
    meta, x, y = _read_csv(text) if path.suffix == ".csv" else _read_toml(text)
    abscissa = meta.get("abscissa")
    if abscissa not in _ABSCISSAE:
        raise CurveError(f"abscissa must be one of {sorted(_ABSCISSAE)}; got {abscissa!r}")
    unit = meta.get("abscissa_unit")
    if not isinstance(unit, str) or not unit.strip():
        raise CurveError("abscissa_unit must be a nonempty unit string")
    value_unit = meta.get("value_unit", "")
    if not isinstance(value_unit, str):
        raise CurveError("value_unit must be a unit string (empty means dimensionless)")
    quantity = meta.get("quantity")
    if quantity not in CURVE_QUANTITIES:
        raise CurveError(f"quantity must be one of {sorted(CURVE_QUANTITIES)}; got {quantity!r}")
    confidential = meta.get("confidential")
    if not isinstance(confidential, bool):
        raise CurveError("confidential must be true or false")
    source = _text(meta, "source", required=True)
    digitization = _text(meta, "digitization", required=True)
    revision = _text(meta, "revision", required=False)
    label = _text(meta, "label", required=False)
    for index in range(1, len(x)):
        if not x[index] > x[index - 1]:
            raise CurveError(
                f"abscissa must be strictly increasing; row {index + 1} "
                f"({x[index]:g}) does not exceed row {index} ({x[index - 1]:g})"
            )
    try:
        scale = unit_conversion_factor(value_unit, "")
    except Exception:
        scale = None
    if scale is None:
        raise CurveError(f"value unit {value_unit!r} is not dimensionless")
    values = [item * scale for item in y]
    for index, item in enumerate(values):
        if not 0.0 <= item <= 1.0:
            raise CurveError(f"{quantity} value at row {index + 1} is {item:g}; must lie in [0, 1]")
    energies = _to_energy_eV(str(abscissa), unit.strip(), x)
    if abscissa == "wavelength":
        values.reverse()
    return TabulatedCurve(
        name=name,
        path=path,
        quantity=str(quantity),
        energy_eV=tuple(energies),
        value=tuple(values),
        abscissa=str(abscissa),
        abscissa_unit=unit.strip(),
        source=str(source),
        digitization=str(digitization),
        confidential=confidential,
        revision=revision,
        label=label,
    )


def curve_path(name: str, root: Path) -> Path | None:
    """Return the contained file for curve ``name`` under ``root``, or ``None``.

    Raises :class:`CurveError` for an invalid name, an unsafe symlink, a path
    that escapes ``curves/`` after resolution, or both CSV and TOML files.
    """
    if not _STEM_RE.fullmatch(name):
        raise CurveError(
            "curve name must use letters, digits, '.', '_', '-' and start with a letter or digit"
        )
    directory = root / CURVE_DIR
    found = []
    for suffix in (".csv", ".toml"):
        unresolved = directory / f"{name}{suffix}"
        if not unresolved.exists() and not unresolved.is_symlink():
            continue
        if unresolved.is_symlink() or directory.is_symlink():
            raise CurveError(f"{unresolved}: unsafe symlink in curve path")
        base = root.resolve()
        curve_root = directory.resolve()
        candidate = unresolved.resolve()
        if not curve_root.is_relative_to(base) or not candidate.is_relative_to(curve_root):
            raise CurveError(f"{unresolved}: must stay inside {CURVE_DIR}/")
        if not candidate.is_file():
            raise CurveError(f"{unresolved}: not a regular file")
        found.append(candidate)
    if len(found) > 1:
        raise CurveError(f"curve {name!r} is defined by both {found[0].name} and {found[1].name}")
    return found[0] if found else None


def resolve_curve(
    name: object,
    path: str,
    errors: _Errors,
    roots: tuple[Path, ...],
    cache: dict[str, TabulatedCurve],
) -> TabulatedCurve | None:
    """Resolve a curve reference against ``roots`` in order, recording errors at ``path``."""
    if not isinstance(name, str) or not name:
        errors.add(path, "must be a nonempty curve name")
        return None
    if name in cache:
        return cache[name]
    for root in roots:
        try:
            file = curve_path(name, root)
        except CurveError as exc:
            errors.add(path, f"curve {name!r}: {exc}")
            return None
        if file is None:
            continue
        try:
            curve = parse_curve(name, file)
        except CurveError as exc:
            errors.add(path, f"curve {name!r} ({file}): {exc}")
            return None
        cache[name] = curve
        return curve
    searched = ", ".join(str(root / CURVE_DIR) for root in roots)
    errors.add(path, f"unknown curve {name!r}; expected {name}.csv or {name}.toml in {searched}")
    return None


def _owner_roots(source: Path) -> dict[tuple[str, str], Path]:
    """Map ``(table, name)`` to the catalog root that owns that object's file."""
    if not source.is_dir():
        return {}
    from .._catalog_layout import source_files

    owners = {}
    for relative, file in source_files(source):
        if "/" in relative:
            table, filename = relative.split("/", 1)
            owners[table, Path(filename).stem] = file.parent.parent
    return owners


def resolve_object_curves(
    source: Path,
    detectors: Mapping[str, Mapping[str, object]],
    profile_detector_sets: Mapping[str, Mapping[str, Mapping[str, object]]],
    gratings: Mapping[str, Mapping[str, object]],
    errors: _Errors,
) -> dict[str, TabulatedCurve]:
    """Resolve every detector ``qe_curve`` and grating ``efficiency_curve``.

    Each reference resolves under the ``curves/`` directory of the catalog
    root that owns the referencing object file (the user layer for user
    objects), so a user curve can never shadow a curve of a bundled object.
    Returns the loaded curves keyed by name.
    """
    from .._catalog_layout import catalog_root

    owners = _owner_roots(source)
    default = catalog_root(source)
    loaded: dict[str, TabulatedCurve] = {}
    caches: dict[Path, dict[str, TabulatedCurve]] = {}

    def resolve(name: object, path: str, root: Path) -> None:
        curve = resolve_curve(name, path, errors, (root,), caches.setdefault(root, {}))
        if curve is None:
            return
        previous = loaded.setdefault(curve.name, curve)
        if previous.path != curve.path:
            errors.add(
                path, f"curve {curve.name!r} resolves to both {previous.path} and {curve.path}"
            )

    named = {id(spec) for spec in detectors.values()}
    for name, spec in detectors.items():
        response = spec.get("response")
        if isinstance(response, Mapping) and "qe_curve" in response:
            root = owners.get(("detectors", name), default)
            resolve(response["qe_curve"], f"detectors.{name}.response.qe_curve", root)
    for profile, detector_set in profile_detector_sets.items():
        for detector_id, spec in detector_set.items():
            response = spec.get("response")
            if id(spec) in named or not isinstance(response, Mapping) or "qe_curve" not in response:
                continue
            root = owners.get(("profiles", profile), default)
            location = f"profiles.{profile}.detectors.{detector_id}.response.qe_curve"
            resolve(response["qe_curve"], location, root)
    for name, grating in gratings.items():
        if "efficiency_curve" in grating:
            root = owners.get(("gratings", name), default)
            resolve(grating["efficiency_curve"], f"gratings.{name}.efficiency_curve", root)
    return loaded


__all__ = [
    "CURVE_QUANTITIES",
    "CurveError",
    "TabulatedCurve",
    "curve_path",
    "parse_curve",
    "resolve_curve",
    "resolve_object_curves",
]
