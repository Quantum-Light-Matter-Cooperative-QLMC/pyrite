"""Independent regeneration of tests/data/material_catalog_golden.json.

`cxr energy-grid regen-golden` rebuilds the serialized catalog snapshot the
material-catalog golden test asserts against. It re-loads materials.toml from
disk via ``load_material_catalog`` (NOT the process-global ``CATALOG`` singleton)
so the regenerated golden always reflects on-disk state after a
``cxr energy-grid apply``. Crystal physics fingerprints come from the low-level
``pyrite.materials.crystal`` module (the same functions the golden test treats as
ground truth), never from the packaged singleton.

Independence guardrail (tests/test_energy_grid_golden.py): this module must not
import the ``CATALOG`` singleton -- forcing a fresh disk load, never a stale
in-memory catalog.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

from pyrite.materials import crystal as _crystal
from pyrite.materials.catalog import load_material_catalog

GOLDEN_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "material_catalog_golden.json"
)
_MATERIALS_TOML = Path(__file__).resolve().parent.parent / "data" / "materials.toml"
_SOURCE_CHECKOUT_ERROR = (
    "error: `cxr energy-grid regen-golden` is source-checkout-only; installed wheels "
    "do not contain tests/data/material_catalog_golden.json. Run it from an editable "
    "cxr-mc source checkout."
)

_PROVENANCE = {
    "source_commit": "df1478945a1bf52b8870a3c411a35edd55ec7d65",
    "description": "Independent serialized snapshot of the migrated CIF and registry values",
}

# n_families ranked reflection families serialized into the crystal physics block;
# matches tests/materials/test_material_catalog.py::test_catalog_matches_serialized_physics_for_every_crystal.
_PHYSICS_N_FAMILIES = 2
_PHYSICS_E_REF_EV = 1000.0


def _is_source_checkout() -> bool:
    """Return whether this module resolves from this repository's ``src`` tree."""
    module_path = Path(__file__).resolve()
    try:
        root = module_path.parents[3]
    except IndexError:
        return False
    return (
        module_path.parents[1] == (root / "src" / "pyrite").resolve()
        and (root / "pyproject.toml").is_file()
        and (root / "tests" / "data").is_dir()
    )


def _fingerprint(values) -> dict:
    array = np.asarray(values, dtype="<f8")
    return {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _hkl_list(seq) -> list:
    return [[int(x) for x in hkl] for hkl in seq]


def _serialize_crystal(spec) -> dict:
    key = spec.key
    reflections = _hkl_list(
        _crystal.dominant_reflections(key, n_families=_PHYSICS_N_FAMILIES, B_ang2=spec.B_ang2)
    )
    hkl = reflections[0]
    structure, g_mag = _crystal.structure_factor(
        key, tuple(hkl), _PHYSICS_E_REF_EV, B_ang2=spec.B_ang2
    )
    return {
        "lattice": dict(spec.lattice),
        "basis": [[element, [float(x) for x in position]] for element, position in spec.basis],
        "V_cell": spec.V_cell,
        "mosaic_fwhm_deg": spec.mosaic_fwhm_deg,
        "composition": [[element, density] for element, density in spec.composition],
        "config": {
            "B_ang2": spec.B_ang2,
            "formula": spec.formula,
            "full_name": spec.full_name,
            "phase": spec.phase,
            "beam_uvw": list(spec.beam_uvw) if spec.beam_uvw is not None else None,
            "surface_hkl": list(spec.surface_hkl) if spec.surface_hkl is not None else None,
            "E_grid": _fingerprint(spec.E_grid) if spec.E_grid is not None else None,
            "hkl_list": _hkl_list(spec.hkl_list),
            "hkl_reason": spec.hkl_reason,
            "layers_per_cell": spec.layers_per_cell,
        },
        "physics": {
            "hkl": hkl,
            "g_mag": float(g_mag),
            "structure_factor": [float(structure.real), float(structure.imag)],
            "dominant_reflections": reflections,
        },
    }


def _serialize_scan(scan) -> dict:
    return {
        "thickness_ang": _fingerprint(scan.thickness_ang),
        "energy_keV": _fingerprint(scan.energy_keV),
        "tilt_deg": _fingerprint(scan.tilt_deg),
        "tilt_azim_deg": _fingerprint(scan.tilt_azim_deg),
        "E_grid_line": None if scan.E_grid_line is None else _fingerprint(scan.E_grid_line),
        "E_grid_line_by_energy": {
            str(energy): _fingerprint(grid)
            for energy, grid in (scan.E_grid_line_by_energy or {}).items()
        },
        "E_grid_brem": _fingerprint(scan.E_grid_brem),
    }


def _serialize_material(spec) -> dict:
    return {
        "label": spec.label,
        "identity": {
            "formula": spec.formula,
            "phase": spec.phase,
            "full_name": spec.full_name,
            "cut": list(spec.cut) if spec.cut is not None else None,
            "cut_frame": spec.cut_frame,
            "display_name": spec.identity.display_name,
        },
        "profile": spec.profile,
        "crystal_key": spec.crystal_key,
        "substrate": spec.substrate,
        **(
            {
                "validation": {
                    "crystal_database_match": spec.validation.crystal_database_match,
                }
            }
            if spec.validation.crystal_database_match is not None
            else {}
        ),
        "stack": [
            {
                "material": layer.material,
                "thickness_ang": layer.thickness_ang,
                "beam_uvw": list(layer.beam_uvw) if layer.beam_uvw else None,
                "azimuth_deg": layer.azimuth_deg,
            }
            for layer in spec.stack
        ],
        "scan": _serialize_scan(spec.scan),
    }


def _serialize_resolved_stack(catalog, spec) -> dict:
    film_thickness = float(spec.scan.thickness_ang[0])
    layers = catalog.resolve_stack(spec.key, film_thickness)
    return {
        "film_thickness_ang": film_thickness,
        "layers": [
            {
                "key": layer["key"],
                "z_top_ang": layer["z_top_ang"],
                "z_bottom_ang": layer["z_bottom_ang"],
                "composition": [list(item) for item in layer["composition"]],
                "beam_uvw": list(layer["beam_uvw"]) if layer["beam_uvw"] else None,
                "azimuth_deg": layer["azimuth_deg"],
            }
            for layer in layers
        ],
    }


def _special_grids(catalog) -> dict:
    with open(_MATERIALS_TOML, "rb") as f:
        raw = tomllib.load(f)
    mote2_arange = raw["crystals"]["mote2"]["E_grid"]["arange"]
    return {
        "hbn_thickness_ang": [float(x) for x in catalog.material("hbn").scan.thickness_ang],
        "hbn_tilt_deg": [float(x) for x in catalog.material("hbn").scan.tilt_deg],
        "mote2_E_grid_descriptor": {
            "start": mote2_arange["start"],
            "stop": mote2_arange["stop"],
            "step": mote2_arange["step"],
        },
        "mote2_tilt_deg": [float(x) for x in catalog.material("mote2").scan.tilt_deg],
        "mote2_tilt_azim_deg": [float(x) for x in catalog.material("mote2").scan.tilt_azim_deg],
    }


def build_golden() -> dict:
    """Serialize the on-disk material catalog into the golden snapshot structure."""
    catalog = load_material_catalog()
    return {
        "provenance": dict(_PROVENANCE),
        "crystal_keys": list(catalog.crystals),
        "configured_crystal_keys": [
            key for key, spec in catalog.crystals.items() if spec.E_grid is not None
        ],
        "material_keys": list(catalog.material_keys),
        "crystals": {key: _serialize_crystal(spec) for key, spec in catalog.crystals.items()},
        "media": {
            key: [[element, density] for element, density in medium.composition]
            for key, medium in catalog.media.items()
        },
        "materials": {key: _serialize_material(spec) for key, spec in catalog.materials.items()},
        "resolved_stacks": {
            key: _serialize_resolved_stack(catalog, spec)
            for key, spec in catalog.materials.items()
            if spec.stack
        },
        "special_grids": _special_grids(catalog),
    }


def regen(check=False) -> int:
    """Write the golden, or (check) diff-only and return nonzero on drift."""
    if not _is_source_checkout():
        print(_SOURCE_CHECKOUT_ERROR, file=sys.stderr)
        return 1

    # NOT sort_keys: the catalog golden test asserts E_grid_line_by_energy keys in
    # numeric insertion order (30,40,...300); lexical sort would put "100.0" first.
    # build_golden() emits every dict in a deterministic, catalog-driven order.
    rebuilt = json.dumps(build_golden(), indent=2) + "\n"
    if check:
        current = GOLDEN_PATH.read_text() if GOLDEN_PATH.exists() else ""
        if rebuilt != current:
            print(
                "".join(
                    difflib.unified_diff(
                        current.splitlines(True),
                        rebuilt.splitlines(True),
                        "golden(current)",
                        "golden(rebuilt)",
                    )
                )
            )
            return 1
        return 0
    GOLDEN_PATH.write_text(rebuilt)
    print(f"[line-grid] wrote {GOLDEN_PATH}")
    return 0
