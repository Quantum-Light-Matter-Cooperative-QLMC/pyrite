"""Parsing and validation for schema-version-1 material catalogs."""

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np

from .. import DATA_DIR
from .._catalog_layout import ARTIFACT_DIR, catalog_root
from .._energy_grid_artifacts import ArtifactError, load_artifact
from .._line_grid_policy import BANDWIDTH_POLICIES, LINE_QUADRATURES, RESOLUTION_POLICIES
from .._numerics import validate_profile_numerics
from ._beam_detector_parse import (
    _parse_filter_rows,
    _parse_physical_detector,
    _parse_profile_beam,
    _parse_profile_detector,
)
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
from ._identity import MaterialIdentity
from ._schema import (
    _EMISSION_VALUES,
    _PROFILE_SCALAR_NUMERICS_KEYS,
    _SCAN_KEYS,
    CrystalInfo,
    CrystalSpec,
    LayerSpec,
    MaterialConfigError,
    MaterialSpec,
    MaterialValidationSpec,
    MediumSpec,
    ScanSpec,
)
from ._transport_data import TRANSPORT_ELEMENTS

logger = logging.getLogger(__name__)


def _cif_path(value: object, path: str, errors: _Errors, source: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.add(path, "must be a nonempty relative path")
        return None
    relative = Path(value)
    if relative.is_absolute() or relative.parts[:1] != ("cifs",) or ".." in relative.parts:
        errors.add(path, "must be a relative path inside cifs/")
        return None
    for root in (catalog_root(source), DATA_DIR):
        base = root.resolve()
        cif_root = (base / "cifs").resolve()
        candidate = (base / relative).resolve()
        if not cif_root.is_relative_to(base) or not candidate.is_relative_to(cif_root):
            errors.add(path, "must stay inside cifs/")
            return None
        if candidate.is_file():
            return candidate
    errors.add(path, f"file does not exist ({value})")
    return None


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
#: ASCII chemical formula: element symbols with optional integer counts, e.g.
#: ``C``, ``MoS2``, ``Al2O3``. Subscripts stay ASCII so labels survive terminals,
#: CSV exports, and filenames unchanged.
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def _optional_text(value: object, path: str, errors: _Errors) -> str | None:
    """Validate an optional metadata string: absent is fine, present must be nonempty."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.add(path, "must be a nonempty string")
        return None
    return value.strip()


def _parse_crystals(raw: object, errors: _Errors, *, source: Path) -> dict[str, CrystalSpec]:
    table = _table(raw, "crystals", errors)
    if table is None:
        return {}
    out: dict[str, CrystalSpec] = {}
    allowed = {
        "cif",
        "validation_id",
        "formula",
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
    required = {"cif", "validation_id", "formula", "B_ang2"}
    for key, value in table.items():
        path = f"crystals.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(row, path, allowed)
        for name in sorted(required - set(row)):
            errors.add(f"{path}.{name}", "missing required key")
        cif = _cif_path(row.get("cif"), f"{path}.cif", errors, source)
        validation_id = row.get("validation_id")
        if not isinstance(validation_id, str) or not validation_id.strip():
            errors.add(f"{path}.validation_id", "must be a nonempty string")
            validation_id = ""
        formula = row.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            errors.add(f"{path}.formula", "must be a nonempty string")
            formula = ""
        elif not _FORMULA_RE.match(formula):
            errors.add(f"{path}.formula", "must be an ASCII chemical formula")
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
        if (
            cif is not None
            and info is not None
            and formula
            and (beam is not None or surface is not None)
        ):
            out[key] = CrystalSpec(
                key=key,
                cif=cif,
                validation_id=str(validation_id),
                formula=str(formula).strip(),
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
    # An empty mapping opts missing catalog rows into automatic case-local
    # resolution. None is reserved for an explicit fixed line grid.
    line_grids: LineGridByEnergy | None = None if has_line else MappingProxyType({})
    if not has_line:
        energy_grid = grids.get("energy_keV")
        if energy_grid is not None:
            # dict preserves energy_keV's declared order (e.g. 30,40,...,300),
            # not set-hash order: the golden snapshot and E_grid_line_by_energy
            # consumers rely on that ordering.
            configured = list(dict.fromkeys(float(value) for value in energy_grid))
            # Partial store coverage is NOT an error (issue #101). A beam energy
            # with no stored row falls through to automatic case-local line-grid
            # resolution in campaign.sweep, so `pyrite material energy-grid
            # derive` is no longer a correctness prerequisite for a valid
            # material at a supported beam energy.
            if line_grid_store:
                covered = {
                    energy: line_grid_store[energy]
                    for energy in configured
                    if energy in line_grid_store
                }
                if covered:
                    line_grids = MappingProxyType(covered)
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
    # No line-grid key is required: neither E_grid_line nor a store row makes a
    # material invalid any more (issue #101) -- automatic case-local resolution
    # covers the gap at case-build time.
    if thickness is None or any(grids.get(key) is None for key in ordinary_required):
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
    the material's own ``[energy_grids.MATERIAL]`` rows -- happens per material in
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


#: Allowed values of ``[profiles.NAME.line_grid_policy]`` keys.
_PROFILE_LINE_GRID_POLICY_VALUES = {
    "bandwidth": BANDWIDTH_POLICIES,
    "resolution": RESOLUTION_POLICIES,
    "quadrature": LINE_QUADRATURES,
}


def _parse_profile_line_grid_policy(raw: object, path: str, errors: _Errors) -> None:
    """Validate ``[profiles.NAME.line_grid_policy]``: named line-grid policies and quadrature.

    The table becomes each case's ``Sweep.line_grid_policy``, so it joins the
    profile's case identity (#192).
    """
    table = _table(raw, path, errors)
    if table is None:
        return
    errors.keys(table, path, set(_PROFILE_LINE_GRID_POLICY_VALUES))
    for key, allowed in _PROFILE_LINE_GRID_POLICY_VALUES.items():
        if key in table and table[key] not in allowed:
            errors.add(f"{path}.{key}", f"must be one of {allowed}")


def _parse_profiles(raw: object, errors: _Errors) -> dict[str, Mapping[str, object]]:
    """Parse ``[profiles.*]`` campaign rows.

    Schema inversion (docs/adr/0005-energy-grid-schema-decisions.md decision 2):
    a profile carries scan defaults plus an optional ``materials`` list
    (absent means all catalog materials) and an optional ``overrides`` table
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
            row,
            path,
            set(_SCAN_KEYS)
            | {
                "materials",
                "overrides",
                "beam",
                "detector",
                "filters",
                "physical_detector",
                "emission",
                "line_grid_policy",
                *_PROFILE_SCALAR_NUMERICS_KEYS,
                "energy_grid_refs",
            },
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
            or not all(isinstance(item, str) and item for item in materials_list)
        ):
            errors.add(f"{path}.materials", "must be an array of nonempty material keys")
        elif isinstance(materials_list, list) and len(set(materials_list)) != len(materials_list):
            errors.add(f"{path}.materials", "must not contain duplicate material keys")
        emission = row.get("emission")
        if emission is not None and emission not in _EMISSION_VALUES:
            errors.add(f"{path}.emission", f"must be one of {_EMISSION_VALUES}")
        scalar_numerics = {name: row[name] for name in _PROFILE_SCALAR_NUMERICS_KEYS if name in row}
        try:
            validate_profile_numerics(scalar_numerics)
        except ValueError as exc:
            message = str(exc)
            field = message.split(maxsplit=1)[0]
            detail = message.removeprefix(field).strip()
            errors.add(f"{path}.{field}", detail)
        if "overrides" in row:
            _parse_profile_overrides(row["overrides"], f"{path}.overrides", errors)
        if "line_grid_policy" in row:
            _parse_profile_line_grid_policy(
                row["line_grid_policy"], f"{path}.line_grid_policy", errors
            )
        refs = row.get("energy_grid_refs")
        if refs is not None:
            refs_table = _table(refs, f"{path}.energy_grid_refs", errors)
            if refs_table is not None:
                for material_key, digest in refs_table.items():
                    if not isinstance(material_key, str) or not material_key:
                        errors.add(
                            f"{path}.energy_grid_refs", "material keys must be nonempty strings"
                        )
                    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                        errors.add(
                            f"{path}.energy_grid_refs.{material_key}",
                            "must be a 64-character lowercase SHA-256 digest",
                        )
        row_out = dict(row)
        if "beam" in row_out:
            beam_raw = row_out["beam"]
            if isinstance(beam_raw, str):
                # A bare string is a ``[beams.NAME]`` reference (decision 4);
                # TOML's own duplicate-key rule already rejects a profile
                # spelling both `beam = "NAME"` and an inline `[profiles.NAME.
                # beam]` table under the same key, so no extra check is needed
                # here. Resolved against `beams` once that table is parsed --
                # see `_load_material_catalog_cached`.
                if not beam_raw:
                    errors.add(f"{path}.beam", "must be a nonempty beam name")
                    del row_out["beam"]
            else:
                beam = _parse_profile_beam(beam_raw, f"{path}.beam", errors)
                if beam is not None:
                    row_out["beam"] = beam
                else:
                    del row_out["beam"]
        if "detector" in row_out:
            detector_raw = row_out["detector"]
            if isinstance(detector_raw, str):
                if not detector_raw:
                    errors.add(f"{path}.detector", "must be a nonempty detector name")
                    del row_out["detector"]
            else:
                detector = _parse_profile_detector(detector_raw, f"{path}.detector", errors)
                if detector is not None:
                    row_out["detector"] = detector
                else:
                    del row_out["detector"]
        if "filters" in row_out:
            filters = _parse_filter_rows(row_out["filters"], f"{path}.filters", errors)
            if filters is not None:
                row_out["filters"] = filters
            else:
                del row_out["filters"]
        if "physical_detector" in row_out:
            physical_detector = _parse_physical_detector(
                row_out["physical_detector"], f"{path}.physical_detector", errors
            )
            if physical_detector is not None:
                row_out["physical_detector"] = physical_detector
            else:
                del row_out["physical_detector"]
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


def _artifact_line_grids(identity: Mapping[str, object]) -> LineGridByEnergy:
    """Decode a verified v1 artifact's normalized line rows exactly."""
    rows = cast("list[Mapping[str, Any]]", identity["line_rows"])
    decoded: dict[float, np.ndarray] = {}
    for row in rows:
        energy = float(row["energy_keV"])
        grid = np.linspace(
            float(row["start_eV"]),
            float(row["stop_eV"]),
            int(row["num"]),
            endpoint=True,
        )
        decoded[energy] = _readonly(grid)
    return MappingProxyType(decoded)


def _load_profile_artifacts(
    source: Path,
    profiles: Mapping[str, Mapping[str, object]],
    profile_name: str,
    errors: _Errors,
) -> tuple[dict[str, tuple[str, Mapping[str, object]]], dict[str, Mapping[str, str]]]:
    """Load and verify explicit refs; leave legacy fallback entirely read-only."""
    all_refs: dict[str, Mapping[str, str]] = {}
    for name, row in profiles.items():
        raw_refs = row.get("energy_grid_refs")
        if isinstance(raw_refs, Mapping):
            refs = {
                str(material): str(digest)
                for material, digest in raw_refs.items()
                if isinstance(material, str)
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            }
            all_refs[name] = MappingProxyType(refs)

    selected: dict[str, tuple[str, Mapping[str, object]]] = {}
    store_root = catalog_root(source) / ARTIFACT_DIR
    for material, digest in all_refs.get(profile_name, {}).items():
        path = f"profiles.{profile_name}.energy_grid_refs.{material}"
        try:
            stored = load_artifact(store_root, digest)
        except ArtifactError as exc:
            errors.add(path, str(exc))
            continue
        identity = stored.identity
        if identity.get("material") != material:
            errors.add(
                path,
                f"artifact material {identity.get('material')!r} does not match ref key",
            )
            continue
        selected[material] = (digest, identity)
    return selected, all_refs


def _parse_materials(
    raw: object,
    crystals: Mapping[str, CrystalSpec],
    media: Mapping[str, MediumSpec],
    energy_grids: Mapping[str, LineGridByEnergy],
    errors: _Errors,
    profiles: Mapping[str, Mapping[str, object]],
    profile_artifacts: Mapping[str, tuple[str, Mapping[str, object]]],
    resolved_artifact_refs: dict[str, str],
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

    standard_profile = profiles.get("standard")
    if not isinstance(standard_profile, Mapping):
        standard_profile = selected_profile
    membership_raw = selected_profile.get("materials")
    selected_members = (
        {item for item in membership_raw if isinstance(item, str)}
        if isinstance(membership_raw, list)
        else None
    )

    allowed = {"display_name", "crystal", "substrate", "stack", "validation"}
    for key, value in table.items():
        path = f"materials.{key}"
        row = _table(value, path, errors)
        if row is None:
            continue
        errors.keys(row, path, allowed)
        display_name = _optional_text(row.get("display_name"), f"{path}.display_name", errors)
        crystal_key = row.get("crystal", key)
        if not isinstance(crystal_key, str) or crystal_key not in crystals:
            errors.add(f"{path}.crystal", "must reference a crystal")
            crystal_key = ""

        resolves_selected = selected_members is None or key in selected_members
        resolving_profile_name = profile_name if resolves_selected else "standard"
        resolving_profile = selected_profile if resolves_selected else standard_profile
        overrides_raw = resolving_profile.get("overrides")
        overrides = overrides_raw if isinstance(overrides_raw, Mapping) else {}
        override_raw = overrides.get(key)
        override: Mapping[str, object] = (
            cast(Mapping[str, object], override_raw) if isinstance(override_raw, Mapping) else {}
        )
        values = {name: resolving_profile[name] for name in _SCAN_KEYS if name in resolving_profile}
        if "thickness_ang" in override or "thickness_layers" in override:
            values.pop("thickness_ang", None)
            values.pop("thickness_layers", None)
        values.update({name: override[name] for name in _SCAN_KEYS if name in override})

        artifact = profile_artifacts.get(key) if resolves_selected else None
        artifact_line_grids = None
        artifact_energies: list[float] = []
        if artifact is not None:
            digest, identity = artifact
            # The artifact records which beam energies the derivation covered;
            # the profile still owns the swept axis (it is what `energy-grid
            # add` stamped into the identity in the first place). Only the
            # derived grids below come back out of the artifact.
            artifact_energies = cast("list[float]", identity["beam_energies_keV"])
            brem = cast("Mapping[str, float]", identity["brem_grid"])
            values["E_grid_brem"] = {
                "arange": {
                    "start": float(brem["start_eV"]),
                    "stop": float(brem["stop_eV"]),
                    "step": float(brem["step_eV"]),
                }
            }
            values.pop("E_grid_line", None)
            artifact_line_grids = _artifact_line_grids(identity)
            resolved_artifact_refs[key] = digest

        scan = _scan(
            values,
            f"{path}.scan",
            crystals.get(str(crystal_key)),
            errors,
            line_grid_store=(
                artifact_line_grids if artifact_line_grids is not None else energy_grids.get(key)
            ),
        )
        if artifact is not None and scan is not None:
            # A profile may sweep a subset of the derived energies; it may never
            # sweep one the derivation never covered. Catches a hand-copied ref
            # at load time instead of silently resolving the wrong axis.
            uncovered = sorted({float(value) for value in scan.energy_keV} - set(artifact_energies))
            if uncovered:
                errors.add(
                    f"profiles.{resolving_profile_name}.energy_grid_refs.{key}",
                    f"artifact {digest} covers beam energies {artifact_energies}, "
                    f"which do not include {uncovered}",
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
        validation = MaterialValidationSpec()
        validation_raw = row.get("validation")
        if validation_raw is not None:
            validation_row = _table(validation_raw, f"{path}.validation", errors)
            if validation_row is not None:
                errors.keys(
                    validation_row,
                    f"{path}.validation",
                    {"crystal_database_match"},
                )
                crystal_database_match = validation_row.get("crystal_database_match")
                if crystal_database_match not in {"verified", "unverified"}:
                    errors.add(
                        f"{path}.validation.crystal_database_match",
                        "must be 'verified' or 'unverified'",
                    )
                else:
                    validation = MaterialValidationSpec(
                        crystal_database_match=cast(
                            Literal["verified", "unverified"], crystal_database_match
                        )
                    )
        if crystal_key and scan is not None:
            crystal = crystals[str(crystal_key)]
            out[key] = MaterialSpec(
                key,
                MaterialIdentity(
                    formula=crystal.formula,
                    phase=crystal.phase,
                    full_name=crystal.full_name,
                    cut=crystal.cut,
                    cut_frame=crystal.cut_frame,
                    display_name=display_name,
                    hexagonal=crystal.hexagonal,
                ),
                resolving_profile_name,
                crystal_key,
                scan,
                substrate,
                tuple(layers),
                validation,
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
        profile_refs = profile_row.get("energy_grid_refs")
        if isinstance(profile_refs, Mapping):
            for material_key in profile_refs:
                if material_key not in table:
                    errors.add(
                        f"profiles.{profile_key}.energy_grid_refs.{material_key}",
                        "unknown material",
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

    Decision 3 (docs/adr/0005-energy-grid-schema-decisions.md): line-grid bounds
    live here, keyed by material, independent of any profile -- so profile
    edits can never delete expensive Monte-Carlo-derived bounds; only an
    explicit ``pyrite energy-grid line delete`` can. Absent entirely means no
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
