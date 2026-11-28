"""Decode and validate the scan axes of a resolved material profile."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .._cxr_vocabulary import catalog_key
from ._catalog_decode import CxrGridByEnergy, _Errors, _grid, _readonly
from ._schema import DEFAULT_BREM_GRID, CrystalSpec, ScanSpec

_GRID_UNITS = {
    "energy_keV": ("keV", "energy", None),
    "tilt_deg": ("degree", "angle", "angle"),
    "tilt_azim_deg": ("degree", "angle", "angle"),
    "E_grid_brem": ("eV", "energy", None),
    "E_grid_line": ("eV", "energy", None),
    "thickness_ang": ("angstrom", "length", None),
}


def _field_grid(value: object, name: str, path: str, errors: _Errors) -> np.ndarray | None:
    """Decode a numeric grid and parse explicit units for physical fields."""
    unit_spec = _GRID_UNITS.get(name)
    if unit_spec is None:
        return _grid(value, path, errors)
    unit, expected, kind = unit_spec
    return _grid(value, path, errors, unit=unit, expected=expected, kind=kind)


def _validate_angle_grid(name: str, grid: np.ndarray, path: str, errors: _Errors) -> bool:
    if name == "tilt_deg" and (np.any(grid < 0) or np.any(grid >= 90)):
        errors.add(path, "values must satisfy 0 <= tilt_deg < 90")
        return False
    if name == "tilt_azim_deg" and (np.any(grid < 0) or np.any(grid > 360)):
        errors.add(path, "values must satisfy 0 <= tilt_azim_deg <= 360")
        return False
    return True


def _decode_scan_grids(
    values: Mapping[str, object], path: str, errors: _Errors
) -> dict[str, np.ndarray | None]:
    """Decode photon and beam axes, preserving the default continuum grid."""
    has_cxr = "E_grid_line" in values
    grids: dict[str, np.ndarray | None] = {}
    for key in ("energy_keV", "tilt_deg", "tilt_azim_deg", "E_grid_brem"):
        if key == "E_grid_brem" and key not in values:
            grids[key] = _readonly(np.arange(*DEFAULT_BREM_GRID))
        elif key not in values:
            errors.add(f"{path}.{key}", "missing required key")
            grids[key] = None
        else:
            grids[key] = _field_grid(values[key], key, f"{path}.{key}", errors)
    if has_cxr:
        grids["E_grid_line"] = _field_grid(
            values["E_grid_line"], "E_grid_line", f"{path}.E_grid_line", errors
        )
    else:
        grids["E_grid_line"] = None
    return grids


def _scan_cxr_grids(
    has_cxr: bool, energy_grid: np.ndarray | None, cxr_grid_store: CxrGridByEnergy | None
) -> CxrGridByEnergy | None:
    """Select stored line grids in scan order, with automatic fallback."""
    # None is reserved for an explicit fixed line grid; an empty mapping
    # requests automatic case-local resolution (issue #101).
    if has_cxr:
        return None
    if energy_grid is None:
        return MappingProxyType({})
    # Retain first occurrence order, even for duplicated beam energies.
    configured = list(dict.fromkeys(float(value) for value in energy_grid))
    if not cxr_grid_store:
        return MappingProxyType({})
    # Partial store coverage is valid: missing energies resolve automatically.
    return MappingProxyType(
        {energy: cxr_grid_store[energy] for energy in configured if energy in cxr_grid_store}
    )


def _scan_thickness(
    values: Mapping[str, object], path: str, crystal: CrystalSpec | None, errors: _Errors
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Decode thickness, converting positive integer layer counts when needed."""
    thickness = None
    layer_grid = None
    if "thickness_ang" in values:
        thickness = _field_grid(
            values["thickness_ang"], "thickness_ang", f"{path}.thickness_ang", errors
        )
    elif "thickness_layers" in values:
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
    return thickness, layer_grid


def _validate_scan_grids(grids: dict[str, np.ndarray | None], path: str, errors: _Errors) -> None:
    """Validate decoded axes and clear rejected grids in place."""
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


def _scan_electron_grids(
    values: Mapping[str, object], path: str, errors: _Errors
) -> dict[str, np.ndarray | None]:
    """Decode optional positive integer electron-count axes."""
    electron_grids: dict[str, np.ndarray | None] = {}
    for key in ("n_electrons", "n_electrons_brem"):
        if key not in values:
            electron_grids[key] = None
            continue
        label = f"{path}.{catalog_key(key)}"
        grid = _grid(values[key], label, errors)
        if grid is not None and (np.any(grid <= 0) or np.any(grid != np.floor(grid))):
            errors.add(label, "values must be positive integers")
            grid = None
        electron_grids[key] = grid
    return electron_grids


def _scan(
    values: Mapping[str, object],
    path: str,
    crystal: CrystalSpec | None,
    errors: _Errors,
    *,
    cxr_grid_store: CxrGridByEnergy | None = None,
) -> ScanSpec | None:
    has_ang = "thickness_ang" in values
    has_layers = "thickness_layers" in values
    if has_ang == has_layers:
        errors.add(path, "requires exactly one of thickness_ang or thickness_layers")
    grids = _decode_scan_grids(values, path, errors)
    cxr_grids = _scan_cxr_grids("E_grid_line" in values, grids["energy_keV"], cxr_grid_store)
    thickness, layer_grid = _scan_thickness(values, path, crystal, errors)
    _validate_scan_grids(grids, path, errors)
    electron_grids = _scan_electron_grids(values, path, errors)
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
        E_grid_line_by_energy=cxr_grids,
        E_grid_brem=E_grid_brem,
        n_electrons=electron_grids["n_electrons"],
        n_electrons_brem=electron_grids["n_electrons_brem"],
        E_grid_brem_declared="E_grid_brem" in values,
    )
