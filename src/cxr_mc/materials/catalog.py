"""Immutable material catalog loaded from the packaged TOML schema.

The module owns configuration parsing only. It deliberately sits below the
scan and Monte Carlo drivers.
"""

from __future__ import annotations

import logging
import math
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import cast

import numpy as np

from .. import DATA_DIR
from ._cif import load_crystal_from_cif
from ._transport_data import TRANSPORT_ELEMENTS

logger = logging.getLogger(__name__)

GridValue = int | float | Mapping[str, object]
_SCAN_KEYS = (
    "thickness_ang",
    "thickness_layers",
    "energy_keV",
    "tilt_deg",
    "tilt_azim_deg",
    "E_grid_line",
    "E_grid_brem",
)
_GRID_KINDS = frozenset({"values", "arange", "linspace", "logspace"})


class MaterialConfigError(ValueError):
    """One or more path-qualified material catalog errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("invalid material catalog:\n" + "\n".join(f"- {e}" for e in self.errors))


@dataclass(frozen=True)
class CrystalInfo:
    """CIF-derived crystallographic data in cxr-mc units."""

    lattice: Mapping[str, str | float]
    basis: tuple[tuple[str, np.ndarray], ...]
    V_cell: float
    composition: tuple[tuple[str, float], ...]
    mosaic_fwhm_deg: float | None


@dataclass(frozen=True)
class CrystalSpec:
    """One configured crystal and its CIF-derived structure."""

    key: str
    cif: Path
    validation_id: str
    B_ang2: float
    beam_uvw: tuple[int, int, int] | None
    E_grid: np.ndarray | None
    hkl_families: tuple[tuple[int, int, int], ...]
    hkl_reason: str | None
    layers_per_cell: int | None
    info: CrystalInfo
    surface_hkl: tuple[int, int, int] | None

    @property
    def hkl_list(self) -> tuple[tuple[int, int, int], ...]:
        """Pinned representatives expanded to both reciprocal directions."""
        return tuple(hkl for family in self.hkl_families for hkl in (family, _negative(family)))

    @property
    def lattice(self) -> Mapping[str, str | float]:
        """CIF-derived lattice parameters."""
        return self.info.lattice

    @property
    def basis(self) -> tuple[tuple[str, np.ndarray], ...]:
        """CIF-derived expanded fractional basis."""
        return self.info.basis

    @property
    def V_cell(self) -> float:
        """Unit-cell volume in Angstrom cubed."""
        return self.info.V_cell

    @property
    def composition(self) -> tuple[tuple[str, float], ...]:
        """Element number densities in atoms per cubic Angstrom."""
        return self.info.composition

    @property
    def mosaic_fwhm_deg(self) -> float | None:
        """Configured c-axis mosaic FWHM, if present."""
        return self.info.mosaic_fwhm_deg


@dataclass(frozen=True)
class MediumSpec:
    """An amorphous medium represented by element number densities."""

    key: str
    composition: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ScanSpec:
    """Resolved, read-only one-dimensional scan grids."""

    thickness_ang: np.ndarray
    energy_keV: np.ndarray
    tilt_deg: np.ndarray
    tilt_azim_deg: np.ndarray
    E_grid_line: np.ndarray
    E_grid_brem: np.ndarray
    thickness_layers: np.ndarray | None = None


@dataclass(frozen=True)
class LayerSpec:
    """One fixed substrate-side layer in beam-entrance order."""

    material: str
    thickness_ang: float
    beam_uvw: tuple[int, int, int] | None = None
    azimuth_deg: float = 0.0


@dataclass(frozen=True)
class MaterialSpec:
    """A runnable scan target."""

    key: str
    label: str
    profile: str
    crystal_key: str
    scan: ScanSpec
    substrate: str | None = None
    stack: tuple[LayerSpec, ...] = ()

    @property
    def crystal(self) -> str:
        """Compatibility spelling for the film crystal key."""
        return self.crystal_key


@dataclass(frozen=True)
class MaterialCatalog:
    """Deeply immutable crystal, medium, and material registries."""

    schema_version: int
    crystals: Mapping[str, CrystalSpec]
    media: Mapping[str, MediumSpec]
    materials: Mapping[str, MaterialSpec]
    material_keys: tuple[str, ...]

    def crystal(self, key: str) -> CrystalSpec:
        """Return a crystal by key."""
        try:
            return self.crystals[key]
        except KeyError:
            raise KeyError(f"unknown crystal {key!r}; have {list(self.crystals)}") from None

    def material(self, key: str) -> MaterialSpec:
        """Return a runnable material by key."""
        try:
            return self.materials[key]
        except KeyError:
            raise KeyError(f"unknown material {key!r}; have {list(self.materials)}") from None

    def resolve_stack(
        self, key: str, film_thickness_ang: float
    ) -> tuple[Mapping[str, object], ...]:
        """Resolve a film and its explicit stack into ordered physical layers.

        Boundaries are cumulative from the beam-entrance face. Composition is
        resolved from the catalog's CIF-derived crystal density or named
        medium number density. A crystalline layer inherits its configured
        zone axis unless the layer overrides it; amorphous media have no
        default orientation.
        """
        thickness = _number(film_thickness_ang)
        if thickness is None or thickness <= 0:
            raise ValueError("film_thickness_ang must be finite and positive")
        material = self.material(key)
        film = self.crystal(material.crystal_key)
        physical: list[Mapping[str, object]] = []
        z_top = 0.0

        def append_layer(
            layer_key: str,
            layer_thickness: float,
            composition: tuple[tuple[str, float], ...],
            beam_uvw: tuple[int, int, int] | None,
            azimuth_deg: float,
        ) -> None:
            nonlocal z_top
            z_bottom = z_top + layer_thickness
            physical.append(
                MappingProxyType(
                    {
                        "key": layer_key,
                        "z_top_ang": z_top,
                        "z_bottom_ang": z_bottom,
                        "composition": composition,
                        "beam_uvw": beam_uvw,
                        "azimuth_deg": azimuth_deg,
                    }
                )
            )
            z_top = z_bottom

        append_layer(film.key, thickness, film.composition, film.beam_uvw, 0.0)
        for layer in material.stack:
            crystal = self.crystals.get(layer.material)
            if crystal is not None:
                composition = crystal.composition
                beam_uvw = layer.beam_uvw or crystal.beam_uvw
            else:
                composition = self.media[layer.material].composition
                beam_uvw = layer.beam_uvw
            append_layer(
                layer.material,
                layer.thickness_ang,
                composition,
                beam_uvw,
                layer.azimuth_deg,
            )
        return tuple(physical)


class _Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.items.append(f"{path}: {message}")

    def keys(self, raw: Mapping[str, object], path: str, allowed: set[str]) -> None:
        for key in raw:
            if key not in allowed:
                self.add(f"{path}.{key}", "unknown key")


def _readonly(values: object) -> np.ndarray:
    contiguous = np.asarray(values, dtype=np.float64).reshape(-1)
    # ``bytes`` owns immutable storage, unlike ``flags.writeable = False`` on
    # an owning ndarray, whose caller can simply re-enable writes.
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64)


def _negative(hkl: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(-value for value in hkl)  # type: ignore[return-value]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _descriptor_number(payload: Mapping[str, object], key: str) -> float:
    value = _number(payload.get(key))
    if value is None:
        raise ValueError(f"{key} must be a finite number (not bool)")
    return value


def _descriptor_num(payload: Mapping[str, object]) -> int:
    value = payload.get("num")
    if type(value) is not int or value <= 0:
        raise ValueError("num must be a positive integer")
    return value


def _descriptor_endpoint(payload: Mapping[str, object]) -> bool:
    value = payload.get("endpoint", True)
    if type(value) is not bool:
        raise ValueError("endpoint must be a boolean")
    return value


def _direction(value: object, path: str, errors: _Errors) -> tuple[int, int, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(v, int) or isinstance(v, bool) for v in value)
    ):
        errors.add(path, "must be three integer components")
        return None
    out = tuple(value)
    if out == (0, 0, 0):
        errors.add(path, "must be nonzero")
        return None
    return out  # type: ignore[return-value]


def _grid(value: object, path: str, errors: _Errors) -> np.ndarray | None:
    scalar = _number(value)
    if scalar is not None:
        out = _readonly([scalar])
    elif isinstance(value, Mapping):
        kinds = [key for key in value if key in _GRID_KINDS]
        unknown = [key for key in value if key not in _GRID_KINDS]
        for key in unknown:
            errors.add(f"{path}.{key}", "unknown grid descriptor")
        if len(kinds) != 1 or len(value) != 1:
            errors.add(path, "must contain exactly one of values/arange/linspace/logspace")
            return None
        kind = kinds[0]
        payload = value[kind]
        try:
            if kind == "values":
                if not isinstance(payload, list):
                    raise ValueError("values must be an array")
                numeric = [_number(item) for item in payload]
                if any(item is None for item in numeric):
                    raise ValueError("values entries must be finite numbers (not bool)")
                out = _readonly(numeric)
            else:
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{kind} must be a table")
                allowed = {
                    "arange": {"start", "stop", "step"},
                    "linspace": {"start", "stop", "num", "endpoint"},
                    "logspace": {"start", "stop", "num", "endpoint", "base"},
                }[kind]
                extra = set(payload) - allowed
                if extra:
                    raise ValueError(f"unknown keys {sorted(extra)}")
                if kind == "arange":
                    if set(payload) != {"start", "stop", "step"}:
                        raise ValueError("arange requires start, stop, and step")
                    start = _descriptor_number(payload, "start")
                    stop = _descriptor_number(payload, "stop")
                    step = _descriptor_number(payload, "step")
                    if step == 0:
                        raise ValueError("step must be nonzero")
                    out = _readonly(np.arange(start, stop, step))
                else:
                    required = {"start", "stop", "num"}
                    if not required <= set(payload):
                        raise ValueError(f"{kind} requires start, stop, and num")
                    start = _descriptor_number(payload, "start")
                    stop = _descriptor_number(payload, "stop")
                    num = _descriptor_num(payload)
                    endpoint = _descriptor_endpoint(payload)
                    if kind == "logspace":
                        base = _descriptor_number(payload, "base") if "base" in payload else 10.0
                        generated = np.logspace(start, stop, num, endpoint=endpoint, base=base)
                    else:
                        generated = np.linspace(start, stop, num, endpoint=endpoint)
                    out = _readonly(generated)
        except (FloatingPointError, OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.add(path, f"invalid {kind} descriptor ({exc})")
            return None
    else:
        errors.add(path, "must be a scalar or one grid descriptor")
        return None
    if out.size == 0:
        errors.add(path, "grid must be nonempty")
        return None
    if not np.all(np.isfinite(out)):
        errors.add(path, "grid values must be finite")
        return None
    return out


def _table(value: object, path: str, errors: _Errors) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        errors.add(path, "must be a table")
        return None
    return value


def _cif_path(value: object, path: str, errors: _Errors) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.add(path, "must be a nonempty relative path")
        return None
    candidate = (DATA_DIR / value).resolve()
    cif_root = (DATA_DIR / "cifs").resolve()
    if not candidate.is_relative_to(cif_root):
        errors.add(path, "must stay inside packaged data/cifs")
        return None
    if not candidate.is_file():
        errors.add(path, f"file does not exist ({value})")
        return None
    return candidate


def _parse_info(
    key: str, cif: Path, mosaic: float | None, path: str, errors: _Errors
) -> CrystalInfo | None:
    try:
        raw = load_crystal_from_cif(cif, mosaic_fwhm_deg=mosaic)
        lattice_raw = cast(Mapping[str, str | float], raw["lattice"])
        lattice = MappingProxyType(dict(lattice_raw))
        basis_items = []
        for element, position in cast(Sequence[tuple[str, object]], raw["basis"]):
            pos = _readonly(position)
            basis_items.append((str(element), pos))
        basis = tuple(basis_items)
        volume = float(cast(int | float, raw["V_cell"]))
        counts = Counter(element for element, _ in basis)
        composition = tuple((element, count / volume) for element, count in counts.items())
        return CrystalInfo(lattice, basis, volume, composition, mosaic)
    except ModuleNotFoundError as exc:
        if exc.name == "crystals":
            raise MaterialConfigError(
                (
                    "required dependency 'crystals' is not installed; "
                    "install the project environment with `uv sync`",
                )
            ) from None
        raise
    except Exception as exc:  # external CIF parser normalizes several exception types
        errors.add(path, f"could not load crystal {key!r} ({exc})")
        return None


def _parse_crystals(raw: object, errors: _Errors) -> dict[str, CrystalSpec]:
    table = _table(raw, "crystals", errors)
    if table is None:
        return {}
    out: dict[str, CrystalSpec] = {}
    allowed = {
        "cif",
        "validation_id",
        "B_ang2",
        "beam_uvw",
        "surface_hkl",
        "mosaic_fwhm_deg",
        "E_grid",
        "hkl_families",
        "hkl_reason",
        "layers_per_cell",
    }
    required = {"cif", "validation_id", "B_ang2"}
    for key, value in table.items():
        path = f"crystals.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(row, path, allowed)
        for name in sorted(required - set(row)):
            errors.add(f"{path}.{name}", "missing required key")
        cif = _cif_path(row.get("cif"), f"{path}.cif", errors)
        validation_id = row.get("validation_id")
        if not isinstance(validation_id, str) or not validation_id.strip():
            errors.add(f"{path}.validation_id", "must be a nonempty string")
            validation_id = ""
        B = _number(row.get("B_ang2"))
        if B is None or B < 0:
            errors.add(f"{path}.B_ang2", "must be finite and nonnegative")
            B = 0.0
        has_beam = "beam_uvw" in row
        has_surface = "surface_hkl" in row
        if has_beam == has_surface:
            errors.add(path, "requires exactly one of beam_uvw or surface_hkl")
        beam = _direction(row["beam_uvw"], f"{path}.beam_uvw", errors) if has_beam else None
        surface = (
            _direction(row["surface_hkl"], f"{path}.surface_hkl", errors) if has_surface else None
        )
        mosaic_raw = row.get("mosaic_fwhm_deg")
        mosaic = None
        if mosaic_raw is not None:
            mosaic_value = _number(mosaic_raw)
            if mosaic_value is None or mosaic_value <= 0:
                errors.add(f"{path}.mosaic_fwhm_deg", "must be finite and positive")
            else:
                mosaic = mosaic_value
        e_grid = None
        if "E_grid" in row:
            e_grid = _grid(row["E_grid"], f"{path}.E_grid", errors)
            if e_grid is not None and np.any(e_grid <= 0):
                errors.add(f"{path}.E_grid", "values must be positive")
                e_grid = None
        families: list[tuple[int, int, int]] = []
        hkl_raw = row.get("hkl_families")
        reason_raw = row.get("hkl_reason")
        if hkl_raw is not None:
            if not isinstance(hkl_raw, list) or not hkl_raw:
                errors.add(f"{path}.hkl_families", "must be a nonempty array")
            else:
                for index, item in enumerate(hkl_raw):
                    hkl = _direction(item, f"{path}.hkl_families[{index}]", errors)
                    if hkl is not None:
                        first = next(component for component in hkl if component)
                        if first < 0:
                            errors.add(
                                f"{path}.hkl_families[{index}]",
                                "must use the positive representative",
                            )
                        elif hkl in families:
                            errors.add(f"{path}.hkl_families[{index}]", "duplicate family")
                        else:
                            families.append(hkl)
            if not isinstance(reason_raw, str) or not reason_raw.strip():
                errors.add(f"{path}.hkl_reason", "is required when hkl_families is pinned")
        elif reason_raw is not None:
            errors.add(f"{path}.hkl_reason", "requires hkl_families")
        layers_raw = row.get("layers_per_cell")
        layers = None
        if layers_raw is not None:
            if not isinstance(layers_raw, int) or isinstance(layers_raw, bool) or layers_raw <= 0:
                errors.add(f"{path}.layers_per_cell", "must be a positive integer")
            else:
                layers = layers_raw
        info = _parse_info(key, cif, mosaic, f"{path}.cif", errors) if cif else None
        if cif is not None and info is not None and (beam is not None or surface is not None):
            out[key] = CrystalSpec(
                key=key,
                cif=cif,
                validation_id=str(validation_id),
                B_ang2=B,
                beam_uvw=beam,
                surface_hkl=surface,
                E_grid=e_grid,
                hkl_families=tuple(families),
                hkl_reason=reason_raw.strip() if isinstance(reason_raw, str) else None,
                layers_per_cell=layers,
                info=info,
            )
    return out


def _parse_media(raw: object, errors: _Errors) -> dict[str, MediumSpec]:
    table = _table(raw, "media", errors)
    if table is None:
        return {}
    out: dict[str, MediumSpec] = {}
    for key, value in table.items():
        path = f"media.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(row, path, {"composition"})
        comp_raw = _table(row.get("composition"), f"{path}.composition", errors)
        composition: list[tuple[str, float]] = []
        if comp_raw is not None:
            if not comp_raw:
                errors.add(f"{path}.composition", "must be nonempty")
            for element, density in comp_raw.items():
                density_value = _number(density)
                if density_value is None or density_value <= 0:
                    errors.add(f"{path}.composition.{element}", "must be finite and positive")
                else:
                    composition.append((element, density_value))
        if composition:
            out[key] = MediumSpec(key, tuple(composition))
    return out


def _validate_angle_grid(name: str, grid: np.ndarray, path: str, errors: _Errors) -> bool:
    if name == "tilt_deg" and (np.any(grid < 0) or np.any(grid >= 90)):
        errors.add(path, "values must satisfy 0 <= tilt_deg < 90")
        return False
    if name == "tilt_azim_deg" and (np.any(grid < 0) or np.any(grid > 360)):
        errors.add(path, "values must satisfy 0 <= tilt_azim_deg <= 360")
        return False
    return True


def _scan(
    values: Mapping[str, object],
    path: str,
    crystal: CrystalSpec | None,
    errors: _Errors,
) -> ScanSpec | None:
    has_ang = "thickness_ang" in values
    has_layers = "thickness_layers" in values
    if has_ang == has_layers:
        errors.add(path, "requires exactly one of thickness_ang or thickness_layers")
    grids: dict[str, np.ndarray | None] = {}
    for key in _SCAN_KEYS:
        if key in ("thickness_ang", "thickness_layers"):
            continue
        if key not in values:
            errors.add(f"{path}.{key}", "missing required key")
            grids[key] = None
        else:
            grids[key] = _grid(values[key], f"{path}.{key}", errors)
    thickness = None
    layer_grid = None
    if has_ang:
        thickness = _grid(values["thickness_ang"], f"{path}.thickness_ang", errors)
    elif has_layers:
        layer_grid = _grid(values["thickness_layers"], f"{path}.thickness_layers", errors)
        if layer_grid is not None:
            if np.any(layer_grid <= 0) or np.any(layer_grid != np.floor(layer_grid)):
                errors.add(f"{path}.thickness_layers", "values must be positive integers")
            elif crystal is None or crystal.layers_per_cell is None:
                errors.add(
                    f"{path}.thickness_layers",
                    "requires the referenced crystal to define layers_per_cell",
                )
            else:
                c_ang = float(crystal.lattice["c"])
                thickness = _readonly(layer_grid * c_ang / crystal.layers_per_cell)
    if thickness is not None and np.any(thickness <= 0):
        errors.add(f"{path}.thickness_ang", "values must be positive")
        thickness = None
    for key in ("energy_keV", "E_grid_line"):
        grid = grids.get(key)
        if grid is not None and np.any(grid <= 0):
            errors.add(f"{path}.{key}", "values must be positive")
            grids[key] = None
    brem = grids.get("E_grid_brem")
    if brem is not None and np.any(brem < 0):
        errors.add(f"{path}.E_grid_brem", "values must be nonnegative")
        grids["E_grid_brem"] = None
    for key in ("tilt_deg", "tilt_azim_deg"):
        grid = grids.get(key)
        if grid is not None and not _validate_angle_grid(key, grid, f"{path}.{key}", errors):
            grids[key] = None
    if thickness is None or any(grids.get(key) is None for key in _SCAN_KEYS[2:]):
        return None
    return ScanSpec(
        thickness_ang=thickness,
        thickness_layers=layer_grid,
        energy_keV=grids["energy_keV"],  # type: ignore[arg-type]
        tilt_deg=grids["tilt_deg"],  # type: ignore[arg-type]
        tilt_azim_deg=grids["tilt_azim_deg"],  # type: ignore[arg-type]
        E_grid_line=grids["E_grid_line"],  # type: ignore[arg-type]
        E_grid_brem=grids["E_grid_brem"],  # type: ignore[arg-type]
    )


def _parse_profiles(raw: object, errors: _Errors) -> dict[str, Mapping[str, object]]:
    table = _table(raw, "profiles", errors)
    if table is None:
        return {}
    out = {}
    for key, value in table.items():
        path = f"profiles.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(row, path, set(_SCAN_KEYS))
        has_ang = "thickness_ang" in row
        has_layers = "thickness_layers" in row
        if has_ang == has_layers:
            errors.add(path, "requires exactly one of thickness_ang or thickness_layers")
        for name in _SCAN_KEYS:
            if name in ("thickness_ang", "thickness_layers"):
                continue
            if name not in row:
                errors.add(f"{path}.{name}", "missing required key")
        for name in _SCAN_KEYS:
            if name in row:
                grid = _grid(row[name], f"{path}.{name}", errors)
                if grid is not None:
                    _validate_angle_grid(name, grid, f"{path}.{name}", errors)
        out[key] = row
    return out


def _parse_layer(
    value: object,
    path: str,
    crystals: Mapping[str, CrystalSpec],
    media: Mapping[str, MediumSpec],
    errors: _Errors,
) -> LayerSpec | None:
    row = _table(value, path, errors)
    if row is None:
        return None
    errors.keys(row, path, {"material", "thickness_ang", "beam_uvw", "azimuth_deg"})
    material = row.get("material")
    if not isinstance(material, str) or material not in crystals and material not in media:
        errors.add(f"{path}.material", "must reference a crystal or medium")
        material = ""
    thickness = _number(row.get("thickness_ang"))
    if thickness is None or thickness <= 0:
        errors.add(f"{path}.thickness_ang", "must be finite and positive")
        thickness = 0.0
    beam = None
    if "beam_uvw" in row:
        beam = _direction(row["beam_uvw"], f"{path}.beam_uvw", errors)
    azimuth = _number(row.get("azimuth_deg", 0.0))
    if azimuth is None:
        errors.add(f"{path}.azimuth_deg", "must be finite")
        azimuth = 0.0
    if material and thickness > 0:
        return LayerSpec(material, thickness, beam, azimuth)
    return None


def _material_elements(
    material: MaterialSpec,
    crystals: Mapping[str, CrystalSpec],
    media: Mapping[str, MediumSpec],
) -> set[str]:
    elements = {element for element, _ in crystals[material.crystal_key].composition}
    refs = ([material.substrate] if material.substrate else []) + [
        layer.material for layer in material.stack
    ]
    for ref in refs:
        if ref in crystals:
            elements.update(element for element, _ in crystals[ref].composition)
        elif ref in media:
            elements.update(element for element, _ in media[ref].composition)
    return elements


def _parse_materials(
    raw: object,
    profiles: Mapping[str, Mapping[str, object]],
    crystals: Mapping[str, CrystalSpec],
    media: Mapping[str, MediumSpec],
    errors: _Errors,
) -> dict[str, MaterialSpec]:
    table = _table(raw, "materials", errors)
    if table is None:
        return {}
    out: dict[str, MaterialSpec] = {}
    allowed = {"label", "profile", "crystal", "substrate", "stack", *_SCAN_KEYS}
    for key, value in table.items():
        path = f"materials.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(row, path, allowed)
        label = row.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.add(f"{path}.label", "must be a nonempty string")
            label = ""
        profile = row.get("profile")
        if not isinstance(profile, str) or profile not in profiles:
            errors.add(f"{path}.profile", "must reference a profile")
            profile = ""
        crystal_key = row.get("crystal", key)
        if not isinstance(crystal_key, str) or crystal_key not in crystals:
            errors.add(f"{path}.crystal", "must reference a crystal")
            crystal_key = ""
        values = dict(profiles.get(str(profile), {}))
        if "thickness_ang" in row or "thickness_layers" in row:
            values.pop("thickness_ang", None)
            values.pop("thickness_layers", None)
        values.update({name: row[name] for name in _SCAN_KEYS if name in row})
        scan = _scan(values, f"{path}.scan", crystals.get(str(crystal_key)), errors)
        substrate = row.get("substrate")
        if substrate is not None and (
            not isinstance(substrate, str) or substrate not in crystals and substrate not in media
        ):
            errors.add(f"{path}.substrate", "must reference a crystal or medium")
            substrate = None
        stack_raw = row.get("stack", [])
        layers: list[LayerSpec] = []
        if "stack" in row:
            if not isinstance(stack_raw, list) or not stack_raw:
                errors.add(f"{path}.stack", "must be a nonempty array of layers")
            else:
                for index, item in enumerate(stack_raw):
                    layer = _parse_layer(item, f"{path}.stack[{index}]", crystals, media, errors)
                    if layer is not None:
                        layers.append(layer)
        if substrate is not None and "stack" in row:
            errors.add(path, "cannot define both substrate and stack")
        if label and profile and crystal_key and scan is not None:
            out[key] = MaterialSpec(
                key, label, profile, crystal_key, scan, substrate, tuple(layers)
            )
    for key, material in out.items():
        unsupported = sorted(
            _material_elements(material, crystals, media) - set(TRANSPORT_ELEMENTS)
        )
        if unsupported:
            errors.add(
                f"materials.{key}",
                f"runnable composition has unsupported transport elements {unsupported}",
            )
    return out


def _warn_missing_mott(materials: Mapping[str, MaterialSpec], crystals, media) -> None:
    elements = set()
    for material in materials.values():
        elements.update(_material_elements(material, crystals, media))
    mott_dir = DATA_DIR / "mott_transport_cross_sections"
    for element in sorted(elements):
        path = mott_dir / f"DisplayCalcTCSTableFor{element}.csv"
        if not path.exists():
            logger.warning(
                "material catalog: no Mott transport table for %s; transport will use the analytic fallback",
                element,
            )


def load_material_catalog(path: Path | None = None) -> MaterialCatalog:
    """Load, validate, and deeply freeze a schema-version-1 material catalog."""
    source = DATA_DIR / "materials.toml" if path is None else Path(path)
    errors = _Errors()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MaterialConfigError((f"{source}: {exc}",)) from exc
    if not isinstance(raw, Mapping):
        raise MaterialConfigError((f"{source}: root must be a table",))
    errors.keys(raw, "catalog", {"schema_version", "profiles", "crystals", "media", "materials"})
    version = raw.get("schema_version")
    if type(version) is not int or version != 1:
        errors.add("schema_version", "must equal 1")
    for key in ("profiles", "crystals", "media", "materials"):
        if key not in raw:
            errors.add(key, "missing required table")
    profiles = _parse_profiles(raw.get("profiles"), errors)
    crystals = _parse_crystals(raw.get("crystals"), errors)
    media = _parse_media(raw.get("media"), errors)
    materials = _parse_materials(raw.get("materials"), profiles, crystals, media, errors)
    if errors.items:
        raise MaterialConfigError(errors.items)
    _warn_missing_mott(materials, crystals, media)
    return MaterialCatalog(
        schema_version=1,
        crystals=MappingProxyType(crystals),
        media=MappingProxyType(media),
        materials=MappingProxyType(materials),
        material_keys=tuple(materials),
    )


_DEFAULT_CATALOG: MaterialCatalog | None = None
_DEFAULT_CATALOG_LOCK = Lock()


def _get_default_catalog() -> MaterialCatalog:
    """Load the bundled catalog once, on first package-level access."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        with _DEFAULT_CATALOG_LOCK:
            if _DEFAULT_CATALOG is None:
                _DEFAULT_CATALOG = load_material_catalog()
    return _DEFAULT_CATALOG

__all__ = [
    "CrystalInfo",
    "CrystalSpec",
    "LayerSpec",
    "MaterialCatalog",
    "MaterialConfigError",
    "MaterialSpec",
    "MediumSpec",
    "ScanSpec",
    "load_material_catalog",
]
