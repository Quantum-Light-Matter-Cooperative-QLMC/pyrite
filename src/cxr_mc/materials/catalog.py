"""Immutable material catalog loaded from the packaged TOML schema.

The module owns configuration parsing only. It deliberately sits below the
scan and Monte Carlo drivers.
"""

from __future__ import annotations

import functools
import logging
import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .. import DATA_DIR
from ..detectors.spec import DetectorSpec
from ._catalog_decode import (
    LineGridByEnergy,
    _direction,
    _energy_grid_rows,
    _Errors,
    _grid,
    _number,
    _readonly,
    _table,
)
from ._cif import load_crystal_from_cif
from ._transport_data import TRANSPORT_ELEMENTS

logger = logging.getLogger(__name__)

_SCAN_KEYS = (
    "thickness_ang",
    "thickness_layers",
    "energy_keV",
    "tilt_deg",
    "tilt_azim_deg",
    "E_grid_line",
    "E_grid_brem",
    "n_electrons",
    "n_electrons_brem",
)

#: Valid ``[profiles.NAME].emission`` values. A plain ``str`` type (not
#: ``results.store.EmissionMode``): importing ``results`` from ``materials``
#: would cycle back through ``results/tables.py``'s ``from ..materials import
#: CATALOG``.
_EMISSION_VALUES = ("incoherent", "coherent", "both")


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
    full_name: str | None
    phase: str | None
    cod_id: int | None
    mp_id: str | None
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
    E_grid_line: np.ndarray | None
    E_grid_line_by_energy: LineGridByEnergy | None
    E_grid_brem: np.ndarray
    thickness_layers: np.ndarray | None = None
    #: Optional electron-count grids (profile settings): None -> the runner's
    #: settings-level counts (fidelity policy) apply. Single values typical;
    #: multiple values sweep transport statistics (see ``Sweep.n_electrons``).
    n_electrons: np.ndarray | None = None
    n_electrons_brem: np.ndarray | None = None


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
    #: Name of the ``[profiles.*]`` campaign row this material resolved its
    #: scan defaults from (plus that profile's per-material override, if any).
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
    #: Every ``[profiles.*]`` name defined by the source TOML.
    profile_names: tuple[str, ...] = ()
    #: Explicit ``profiles.NAME.materials`` membership lists, keyed by profile;
    #: profiles without a membership row are absent (all materials allowed).
    profile_memberships: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    #: Decoded ``[profiles.NAME.beam]`` distribution blocks (transverse size,
    #: bunch length/shape/offsets, rep-rate, charge, reserved divergence/spread),
    #: keyed by profile; profiles with no beam block are absent. Energy is NOT
    #: here -- it stays the per-material ``ScanSpec.energy_keV`` scan grid.
    profile_beams: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    #: Explicit profile detector blocks. Missing selected-profile blocks inherit
    #: ``standard``; missing standard falls back to :class:`DetectorSpec`.
    profile_detectors: Mapping[str, DetectorSpec] = MappingProxyType({})
    #: Explicit ``profiles.NAME.emission`` overrides ("incoherent"/"coherent"/
    #: "both"), keyed by profile; profiles with no emission key are absent (the
    #: active fidelity preset's emission stands unmodified).
    profile_emissions: Mapping[str, str] = MappingProxyType({})

    def profile_beam(self, name: str) -> Mapping[str, object] | None:
        """Decoded ``[profiles.NAME.beam]`` distribution overrides, or ``None``
        when the profile carries no beam block (the ``standard`` beam default
        applies). Consumed by :func:`config.material_sweep` via ``beam_replace``."""
        return self.profile_beams.get(name)

    def profile_emission(self, name: str) -> str | None:
        """Explicit ``profiles.NAME.emission`` override, or ``None`` when the
        profile carries no emission key. Consumed by :func:`scan._resolved_run`
        to override the active fidelity preset's emission."""
        return self.profile_emissions.get(name)

    def profile_detector(self, name: str) -> DetectorSpec:
        """Resolved detector for ``name`` with standard then legacy fallback."""
        return self.profile_detectors.get(
            name, self.profile_detectors.get("standard", DetectorSpec())
        )

    def profile_materials(self, name: str) -> tuple[str, ...] | None:
        """Explicit ``profiles.NAME.materials`` membership, or ``None`` when the
        profile has no membership row (every in-use material is allowed)."""
        if name not in self.profile_names:
            raise KeyError(f"unknown profile {name!r}; have {list(self.profile_names)}")
        return self.profile_memberships.get(name)

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
        direct-axis or reciprocal-surface orientation unless the layer supplies
        a direct-axis override; amorphous media have no default orientation.
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
            surface_hkl: tuple[int, int, int] | None,
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
                        "surface_hkl": surface_hkl,
                        "azimuth_deg": azimuth_deg,
                    }
                )
            )
            z_top = z_bottom

        append_layer(
            film.key,
            thickness,
            film.composition,
            film.beam_uvw,
            film.surface_hkl,
            0.0,
        )
        for layer in material.stack:
            crystal = self.crystals.get(layer.material)
            if crystal is not None:
                composition = crystal.composition
                if layer.beam_uvw is None:
                    beam_uvw = crystal.beam_uvw
                    surface_hkl = crystal.surface_hkl
                else:
                    beam_uvw = layer.beam_uvw
                    surface_hkl = None
            else:
                composition = self.media[layer.material].composition
                beam_uvw = layer.beam_uvw
                surface_hkl = None
            append_layer(
                layer.material,
                layer.thickness_ang,
                composition,
                beam_uvw,
                surface_hkl,
                layer.azimuth_deg,
            )
        return tuple(physical)


def _negative(hkl: tuple[int, int, int]) -> tuple[int, int, int]:
    return cast(tuple[int, int, int], tuple(-value for value in hkl))


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


_MP_ID_RE = re.compile(r"^mp-\d+$")


def _optional_text(value: object, path: str, errors: _Errors) -> str | None:
    """Validate an optional metadata string: absent is fine, present must be nonempty."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.add(path, "must be a nonempty string")
        return None
    return value.strip()


def _parse_crystals(raw: object, errors: _Errors) -> dict[str, CrystalSpec]:
    table = _table(raw, "crystals", errors)
    if table is None:
        return {}
    out: dict[str, CrystalSpec] = {}
    allowed = {
        "cif",
        "validation_id",
        "full_name",
        "phase",
        "cod_id",
        "mp_id",
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
        full_name = _optional_text(row.get("full_name"), f"{path}.full_name", errors)
        phase = _optional_text(row.get("phase"), f"{path}.phase", errors)
        cod_id = row.get("cod_id")
        if cod_id is not None and (
            not isinstance(cod_id, int) or isinstance(cod_id, bool) or cod_id <= 0
        ):
            errors.add(f"{path}.cod_id", "must be a positive integer")
            cod_id = None
        mp_id = row.get("mp_id")
        if mp_id is not None and (not isinstance(mp_id, str) or not _MP_ID_RE.match(mp_id)):
            errors.add(f"{path}.mp_id", "must match 'mp-<digits>'")
            mp_id = None
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
                full_name=full_name,
                phase=phase,
                cod_id=cod_id if isinstance(cod_id, int) and not isinstance(cod_id, bool) else None,
                mp_id=mp_id if isinstance(mp_id, str) else None,
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
    *,
    line_grid_store: LineGridByEnergy | None = None,
) -> ScanSpec | None:
    has_ang = "thickness_ang" in values
    has_layers = "thickness_layers" in values
    if has_ang == has_layers:
        errors.add(path, "requires exactly one of thickness_ang or thickness_layers")
    has_line = "E_grid_line" in values
    grids: dict[str, np.ndarray | None] = {}
    for key in ("energy_keV", "tilt_deg", "tilt_azim_deg", "E_grid_brem"):
        if key not in values:
            errors.add(f"{path}.{key}", "missing required key")
            grids[key] = None
        else:
            grids[key] = _grid(values[key], f"{path}.{key}", errors)
    if has_line:
        grids["E_grid_line"] = _grid(values["E_grid_line"], f"{path}.E_grid_line", errors)
    else:
        grids["E_grid_line"] = None
    line_grids = None
    if not has_line:
        energy_grid = grids.get("energy_keV")
        if energy_grid is not None:
            # dict preserves energy_keV's declared order (e.g. 30,40,...,300),
            # not set-hash order: the golden snapshot and E_grid_line_by_energy
            # consumers rely on that ordering.
            configured = list(dict.fromkeys(float(value) for value in energy_grid))
            available = set(line_grid_store) if line_grid_store else set()
            missing = sorted(set(configured) - available)
            if missing:
                errors.add(
                    path,
                    "requires E_grid_line, or an energy_grids store entry covering "
                    f"beam energies {missing}",
                )
            elif line_grid_store:
                line_grids = MappingProxyType(
                    {energy: line_grid_store[energy] for energy in configured}
                )
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
    electron_grids: dict[str, np.ndarray | None] = {}
    for key in ("n_electrons", "n_electrons_brem"):
        grid = None
        if key in values:
            grid = _grid(values[key], f"{path}.{key}", errors)
            if grid is not None and (np.any(grid <= 0) or np.any(grid != np.floor(grid))):
                errors.add(f"{path}.{key}", "values must be positive integers")
                grid = None
        electron_grids[key] = grid
    ordinary_required = ("energy_keV", "tilt_deg", "tilt_azim_deg", "E_grid_brem")
    line_valid = grids["E_grid_line"] is not None or line_grids is not None
    if (
        thickness is None
        or any(grids.get(key) is None for key in ordinary_required)
        or not line_valid
    ):
        return None
    energy_keV = grids["energy_keV"]
    tilt_deg = grids["tilt_deg"]
    tilt_azim_deg = grids["tilt_azim_deg"]
    E_grid_brem = grids["E_grid_brem"]
    assert energy_keV is not None
    assert tilt_deg is not None
    assert tilt_azim_deg is not None
    assert E_grid_brem is not None
    return ScanSpec(
        thickness_ang=thickness,
        thickness_layers=layer_grid,
        energy_keV=energy_keV,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
        E_grid_line=grids["E_grid_line"],
        E_grid_line_by_energy=line_grids,
        E_grid_brem=E_grid_brem,
        n_electrons=electron_grids["n_electrons"],
        n_electrons_brem=electron_grids["n_electrons_brem"],
    )


_OVERRIDABLE_KEYS = frozenset(_SCAN_KEYS)


def _parse_profile_overrides(raw: object, path: str, errors: _Errors) -> None:
    """Structurally validate ``[profiles.NAME.overrides.MATERIAL]`` tables.

    Only well-formedness is checked here (decodable grids, mutual exclusion
    of thickness alternatives). Full semantic validation -- merged against
    the profile's own defaults, including line-grid energy coverage against
    the shared ``[energy_grids.*]`` store -- happens per material in
    ``_parse_materials`` via ``_scan``.
    """
    table = _table(raw, path, errors)
    if table is None:
        return
    for material_key, value in table.items():
        material_path = f"{path}.{material_key}"
        row = _table(value, material_path, errors)
        if row is None:
            continue
        errors.keys(row, material_path, set(_OVERRIDABLE_KEYS))
        if "thickness_ang" in row and "thickness_layers" in row:
            errors.add(material_path, "cannot set both thickness_ang and thickness_layers")
        for name in _OVERRIDABLE_KEYS:
            if name not in row:
                continue
            grid = _grid(row[name], f"{material_path}.{name}", errors)
            if grid is not None:
                _validate_angle_grid(name, grid, f"{material_path}.{name}", errors)


# Beam *distribution* fields settable in a ``[profiles.NAME.beam]`` block. These
# mirror the non-energy fields of ``sweep.BeamSpec`` (energy stays the per-material
# ``ScanSpec.energy_keV`` scan grid -- decision 2); the isotropic aliases
# ``transverse_fwhm_mm`` / ``beam_fwhm_mm`` route onto BOTH transverse planes via
# ``sweep.beam_replace`` when the block is applied in ``config.material_sweep``.
# Kept as local literals here to avoid a ``materials -> sweep`` import cycle.
_BEAM_POSITIVE_KEYS = frozenset(
    {
        "transverse_fwhm_x_mm",
        "transverse_fwhm_y_mm",
        "transverse_fwhm_mm",
        "beam_fwhm_mm",
        "bunch_length_fs",
        "rep_rate_hz",
        "bunch_charge_pc",
        "divergence_mrad",
        "energy_spread_frac",
    }
)
_BEAM_LONG_SHAPES = frozenset({"gaussian", "uniform"})
_BEAM_KEYS = _BEAM_POSITIVE_KEYS | {"long_shape", "long_offsets_fs", "longitudinal"}
_DETECTOR_KEYS = frozenset(
    {
        "observation_angle_deg",
        "polar_acceptance_deg",
        "solid_angle_sr",
        "response_model",
        "qe_curve",
        "pixel_pitch_um",
        "sensor_thickness_um",
        "distance_mm",
        "threshold_eV",
    }
)
_LONGITUDINAL_KINDS = frozenset({"gaussian", "microtrain", "compressed"})
_LONGITUDINAL_KEYS = frozenset(
    {
        "kind",
        "envelope_rms_fs",
        "retained_coherence",
        "target_reflection",
        "spacing_periods",
        "modulation_depth",
        "timing_jitter_fs",
    }
)


def _parse_longitudinal_policy(raw: object, path: str, errors: _Errors) -> dict[str, object] | None:
    """Validate one declarative longitudinal distribution policy."""
    table = _table(raw, path, errors)
    if table is None:
        return None
    errors.keys(table, path, set(_LONGITUDINAL_KEYS))
    kind = table.get("kind")
    if not isinstance(kind, str) or kind not in _LONGITUDINAL_KINDS:
        errors.add(f"{path}.kind", f"must be one of {sorted(_LONGITUDINAL_KINDS)}")
        return None

    out: dict[str, object] = {"kind": kind}
    envelope = table.get("envelope_rms_fs")
    if kind in {"gaussian", "microtrain"}:
        number = _number(envelope)
        if number is None or number <= 0:
            errors.add(f"{path}.envelope_rms_fs", "must be a finite positive number")
        else:
            out["envelope_rms_fs"] = number
    elif envelope is not None:
        errors.add(f"{path}.envelope_rms_fs", "must be omitted for compressed")

    eta = table.get("retained_coherence", 0.9)
    eta_number = _number(eta)
    if eta_number is None or not 0 < eta_number <= 1:
        errors.add(f"{path}.retained_coherence", "must satisfy 0 < eta <= 1")
    elif "retained_coherence" in table:
        out["retained_coherence"] = eta_number

    spacing = table.get("spacing_periods", 1)
    if type(spacing) is not int or spacing < 1:
        errors.add(f"{path}.spacing_periods", "must be a positive integer")
    elif "spacing_periods" in table:
        out["spacing_periods"] = spacing

    modulation = table.get("modulation_depth", 1.0)
    modulation_number = _number(modulation)
    if modulation_number is None or not 0 <= modulation_number <= 1:
        errors.add(f"{path}.modulation_depth", "must satisfy 0 <= depth <= 1")
    elif "modulation_depth" in table:
        out["modulation_depth"] = modulation_number

    jitter = table.get("timing_jitter_fs", 0.0)
    jitter_number = _number(jitter)
    if jitter_number is None or jitter_number < 0:
        errors.add(f"{path}.timing_jitter_fs", "must be finite and non-negative")
    elif "timing_jitter_fs" in table:
        out["timing_jitter_fs"] = jitter_number

    reflection = table.get("target_reflection")
    if reflection is not None:
        valid = (
            isinstance(reflection, list)
            and len(reflection) == 3
            and all(type(item) is int for item in reflection)
            and any(reflection)
        )
        if not valid:
            errors.add(f"{path}.target_reflection", "must be a nonzero integer triple")
        else:
            out["target_reflection"] = tuple(reflection)

    targeted = (
        reflection is not None
        or eta_number != 0.9
        or spacing != 1
        or modulation_number != 1.0
        or jitter_number != 0.0
    )
    if kind == "gaussian" and targeted:
        errors.add(path, "gaussian does not accept target-line or modulation controls")
    return out


def _parse_profile_beam(raw: object, path: str, errors: _Errors) -> dict[str, object] | None:
    """Structurally validate a ``[profiles.NAME.beam]`` distribution block.

    Distribution fields only (transverse size, bunch length/shape/offsets,
    rep-rate, charge, reserved divergence/spread) -- ``energy_keV`` is NOT
    accepted (it stays the per-material scan grid, decision 2). Positive
    magnitudes must be finite and ``> 0``; ``long_shape`` is one of
    :data:`_BEAM_LONG_SHAPES`; ``long_offsets_fs`` is an array of finite numbers
    (any sign) coerced to a tuple. Returns the cleaned mapping, or ``None`` when
    the block is empty or fully rejected.
    """
    table = _table(raw, path, errors)
    if table is None:
        return None
    errors.keys(table, path, set(_BEAM_KEYS))
    out: dict[str, object] = {}
    for key, value in table.items():
        if key == "long_shape":
            if not isinstance(value, str) or value not in _BEAM_LONG_SHAPES:
                errors.add(f"{path}.long_shape", f"must be one of {sorted(_BEAM_LONG_SHAPES)}")
            else:
                out[key] = value
        elif key == "long_offsets_fs":
            offsets = [_number(item) for item in value] if isinstance(value, list) else None
            if not offsets or any(item is None for item in offsets):
                errors.add(f"{path}.long_offsets_fs", "must be a non-empty array of finite numbers")
            else:
                out[key] = tuple(offsets)
        elif key == "longitudinal":
            policy = _parse_longitudinal_policy(value, f"{path}.longitudinal", errors)
            if policy is not None:
                out[key] = MappingProxyType(policy)
        elif key in _BEAM_POSITIVE_KEYS:
            number = _number(value)
            if number is None or number <= 0:
                errors.add(f"{path}.{key}", "must be a finite positive number")
            else:
                out[key] = number
    return out or None


def _parse_profile_detector(raw: object, path: str, errors: _Errors) -> DetectorSpec | None:
    """Validate one portable ``[profiles.NAME.detector]`` block."""
    table = _table(raw, path, errors)
    if table is None:
        return None
    errors.keys(table, path, set(_DETECTOR_KEYS))
    known = {key: value for key, value in table.items() if key in _DETECTOR_KEYS}
    try:
        return DetectorSpec(**cast("dict[str, Any]", known))
    except (TypeError, ValueError) as exc:
        errors.add(path, str(exc))
        return None


def _parse_profiles(raw: object, errors: _Errors) -> dict[str, Mapping[str, object]]:
    """Parse ``[profiles.*]`` campaign rows.

    Schema inversion (docs/cli-energy-grid-sweep-rework-plan.md decision 2):
    a profile carries scan defaults plus an optional ``materials`` list
    (absent means all in-use materials) and an optional ``overrides`` table
    keyed by material, holding per-material deltas on the same scan keys.
    """
    table = _table(raw, "profiles", errors)
    if table is None:
        return {}
    out = {}
    for key, value in table.items():
        path = f"profiles.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(
            row, path, set(_SCAN_KEYS) | {"materials", "overrides", "beam", "detector", "emission"}
        )
        has_ang = "thickness_ang" in row
        has_layers = "thickness_layers" in row
        if has_ang == has_layers:
            errors.add(path, "requires exactly one of thickness_ang or thickness_layers")
        for name in ("energy_keV", "tilt_deg", "tilt_azim_deg", "E_grid_brem"):
            if name not in row:
                errors.add(f"{path}.{name}", "missing required key")
        for name in _SCAN_KEYS:
            if name in row:
                grid = _grid(row[name], f"{path}.{name}", errors)
                if grid is not None:
                    _validate_angle_grid(name, grid, f"{path}.{name}", errors)
        materials_list = row.get("materials")
        if materials_list is not None and (
            not isinstance(materials_list, list)
            or not all(isinstance(item, str) for item in materials_list)
        ):
            errors.add(f"{path}.materials", "must be an array of material keys")
        emission = row.get("emission")
        if emission is not None and emission not in _EMISSION_VALUES:
            errors.add(f"{path}.emission", f"must be one of {_EMISSION_VALUES}")
        if "overrides" in row:
            _parse_profile_overrides(row["overrides"], f"{path}.overrides", errors)
        row_out = dict(row)
        if "beam" in row_out:
            beam = _parse_profile_beam(row_out["beam"], f"{path}.beam", errors)
            if beam is not None:
                row_out["beam"] = beam
            else:
                del row_out["beam"]
        if "detector" in row_out:
            detector = _parse_profile_detector(row_out["detector"], f"{path}.detector", errors)
            if detector is not None:
                row_out["detector"] = detector
            else:
                del row_out["detector"]
        out[key] = row_out
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
    crystals: Mapping[str, CrystalSpec],
    media: Mapping[str, MediumSpec],
    energy_grids: Mapping[str, LineGridByEnergy],
    errors: _Errors,
    profiles: Mapping[str, Mapping[str, object]],
    profile_name: str = "standard",
) -> dict[str, MaterialSpec]:
    table = _table(raw, "materials", errors)
    if table is None:
        return {}
    out: dict[str, MaterialSpec] = {}

    selected_profile = profiles.get(profile_name)
    if not isinstance(selected_profile, Mapping):
        errors.add(
            f"profiles.{profile_name}",
            "must be defined; materials resolve scan defaults from it",
        )
        selected_profile = {}

    overrides_raw = selected_profile.get("overrides")
    overrides = overrides_raw if isinstance(overrides_raw, Mapping) else {}

    # A material without its own energy_grids.<key> row falls back to the
    # store entry keyed by its resolving profile's name, or "standard".
    default_line_grids = energy_grids.get(profile_name, energy_grids.get("standard"))

    allowed = {"label", "crystal", "substrate", "stack"}
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
        crystal_key = row.get("crystal", key)
        if not isinstance(crystal_key, str) or crystal_key not in crystals:
            errors.add(f"{path}.crystal", "must reference a crystal")
            crystal_key = ""

        override_raw = overrides.get(key)
        override: Mapping[str, object] = (
            cast(Mapping[str, object], override_raw) if isinstance(override_raw, Mapping) else {}
        )
        values = {name: selected_profile[name] for name in _SCAN_KEYS if name in selected_profile}
        if "thickness_ang" in override or "thickness_layers" in override:
            values.pop("thickness_ang", None)
            values.pop("thickness_layers", None)
        values.update({name: override[name] for name in _SCAN_KEYS if name in override})

        scan = _scan(
            values,
            f"{path}.scan",
            crystals.get(str(crystal_key)),
            errors,
            line_grid_store=energy_grids.get(key, default_line_grids),
        )
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
        if label and crystal_key and scan is not None:
            out[key] = MaterialSpec(
                key, label, profile_name, crystal_key, scan, substrate, tuple(layers)
            )

    for profile_key, profile_row in profiles.items():
        materials_list = profile_row.get("materials")
        if isinstance(materials_list, list):
            for material_key in materials_list:
                if isinstance(material_key, str) and material_key not in table:
                    errors.add(
                        f"profiles.{profile_key}.materials", f"unknown material {material_key!r}"
                    )
        profile_overrides = profile_row.get("overrides")
        if isinstance(profile_overrides, Mapping):
            for material_key in profile_overrides:
                if material_key not in table:
                    errors.add(
                        f"profiles.{profile_key}.overrides.{material_key}", "unknown material"
                    )
    for material_key in energy_grids:
        if material_key not in table and material_key not in profiles:
            errors.add(f"energy_grids.{material_key}", "unknown material")
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


def _parse_energy_grids(raw: object, errors: _Errors) -> dict[str, LineGridByEnergy]:
    """Parse the shared per-material derived-grid store: ``[energy_grids.*]``.

    Decision 3 (docs/cli-energy-grid-sweep-rework-plan.md): line-grid bounds
    live here, keyed by material, independent of any profile -- so profile
    edits can never delete expensive Monte-Carlo-derived bounds; only an
    explicit ``cxr energy-grid line delete`` can. Absent entirely means no
    material has a store entry (materials must then set ``E_grid_line``).
    """
    table = _table(raw, "energy_grids", errors)
    if table is None:
        return {}
    out: dict[str, LineGridByEnergy] = {}
    for material_key, value in table.items():
        path = f"energy_grids.{material_key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(row, path, {"line_by_energy"})
        if "line_by_energy" not in row:
            errors.add(f"{path}.line_by_energy", "missing required key")
            continue
        rows = _energy_grid_rows(row["line_by_energy"], f"{path}.line_by_energy", errors)
        if rows is not None:
            out[material_key] = rows
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


def load_material_catalog(
    path: Path | None = None,
    *,
    profile: str = "standard",
) -> MaterialCatalog:
    """Load, validate, and deeply freeze a schema-version-1 material catalog."""
    source = Path(DATA_DIR) / "materials.toml" if path is None else Path(path)
    # Cache key includes the file's mtime/size so rewriting the same path
    # (tests do this) is never served a stale catalog.
    try:
        stat = source.stat()
    except OSError as exc:
        raise MaterialConfigError((f"{source}: {exc}",)) from exc
    return _load_material_catalog_cached(source, stat.st_mtime_ns, stat.st_size, profile)


@functools.lru_cache(maxsize=32)
def _load_material_catalog_cached(
    source: Path,
    _mtime_ns: int,
    _size: int,
    profile: str,
) -> MaterialCatalog:
    errors = _Errors()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MaterialConfigError((f"{source}: {exc}",)) from exc
    if not isinstance(raw, Mapping):
        raise MaterialConfigError((f"{source}: root must be a table",))
    errors.keys(
        raw,
        "catalog",
        {"schema_version", "profiles", "crystals", "media", "materials", "energy_grids"},
    )
    version = raw.get("schema_version")
    if type(version) is not int or version != 1:
        errors.add("schema_version", "must equal 1")
    for key in ("profiles", "crystals", "media", "materials"):
        if key not in raw:
            errors.add(key, "missing required table")
    profiles = _parse_profiles(raw.get("profiles"), errors)
    crystals = _parse_crystals(raw.get("crystals"), errors)
    media = _parse_media(raw.get("media"), errors)
    energy_grids = _parse_energy_grids(raw.get("energy_grids", {}), errors)
    materials = _parse_materials(
        raw.get("materials"),
        crystals,
        media,
        energy_grids,
        errors,
        profiles,
        profile_name=profile,
    )
    if errors.items:
        raise MaterialConfigError(errors.items)
    _warn_missing_mott(materials, crystals, media)
    profile_memberships = {
        name: tuple(cast("list[str]", row["materials"]))
        for name, row in profiles.items()
        if isinstance(row.get("materials"), list)
    }
    profile_beams = {
        name: MappingProxyType(dict(cast("Mapping[str, object]", row["beam"])))
        for name, row in profiles.items()
        if isinstance(row.get("beam"), Mapping)
    }
    profile_detectors = {
        name: cast("DetectorSpec", row["detector"])
        for name, row in profiles.items()
        if isinstance(row.get("detector"), DetectorSpec)
    }
    profile_emissions = {
        name: cast(str, row["emission"])
        for name, row in profiles.items()
        if isinstance(row.get("emission"), str)
    }
    return MaterialCatalog(
        schema_version=1,
        crystals=MappingProxyType(crystals),
        media=MappingProxyType(media),
        materials=MappingProxyType(materials),
        material_keys=tuple(materials),
        profile_names=tuple(profiles),
        profile_memberships=MappingProxyType(profile_memberships),
        profile_beams=MappingProxyType(profile_beams),
        profile_detectors=MappingProxyType(profile_detectors),
        profile_emissions=MappingProxyType(profile_emissions),
    )


_DEFAULT_CATALOG: MaterialCatalog | None = None
_DEFAULT_CATALOG_LOCK = Lock()


# Test hook: clear the parse cache (e.g. after monkeypatching CIF loaders).
load_material_catalog.cache_clear = _load_material_catalog_cached.cache_clear  # ty: ignore[unresolved-attribute]


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
    "LineGridByEnergy",
    "MaterialCatalog",
    "MaterialConfigError",
    "MaterialSpec",
    "MediumSpec",
    "ScanSpec",
    "load_material_catalog",
]
