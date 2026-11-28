"""Regression coverage for material scan resolution and catalog diagnostics."""

from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import numpy as np

from pyrite.materials._catalog_decode import _Errors
from pyrite.materials._parse import _parse_materials
from pyrite.materials._schema import CrystalSpec

_SCAN = {
    "thickness_ang": 100,
    "energy_keV": 30,
    "tilt_deg": 15,
    "tilt_azim_deg": 180,
}


def _crystal() -> CrystalSpec:
    # Only the crystal fields consumed by material resolution are needed here;
    # this deliberately avoids CIF I/O and isolates the catalog parser.
    return cast(
        CrystalSpec,
        SimpleNamespace(
            formula="C",
            phase=None,
            full_name="Carbon",
            cut=(0, 0, 1),
            cut_frame="plane",
            hexagonal=False,
            composition=(("C", 0.1),),
            layers_per_cell=2,
            lattice={"c": 10.0},
        ),
    )


def test_selected_artifact_and_standard_fallback_are_isolated() -> None:
    profiles = {
        "standard": {**_SCAN, "energy_keV": 60},
        "study": {
            **_SCAN,
            "materials": ["film"],
            "E_grid_line": 100,
            "overrides": {"film": {"thickness_layers": 4}},
        },
    }
    artifact = (
        "a" * 64,
        {
            "beam_energies_keV": [30.0, 40.0],
            "brem_grid": {"start_eV": 0, "stop_eV": 100, "step_eV": 25},
            "line_rows": [
                {"energy_keV": 30, "start_eV": 200, "stop_eV": 210, "num": 3},
            ],
        },
    )
    legacy = {"film": {30.0: np.array([1.0])}, "other": {60.0: np.array([300.0])}}
    original = deepcopy(profiles)
    errors = _Errors()
    refs: dict[str, str] = {}

    parsed = _parse_materials(
        {"film": {"validation": {"crystal_database_match": "verified"}}, "other": {}},
        {"film": _crystal(), "other": _crystal()},
        {},
        legacy,
        errors,
        profiles,
        {"film": artifact, "other": artifact},
        refs,
        "study",
    )

    assert errors.items == []
    assert profiles == original
    assert refs == {"film": "a" * 64}
    film, other = parsed["film"], parsed["other"]
    assert film.profile == "study"
    assert film.validation.crystal_database_match == "verified"
    np.testing.assert_array_equal(film.scan.thickness_ang, [20.0])
    np.testing.assert_array_equal(film.scan.energy_keV, [30.0])
    np.testing.assert_array_equal(film.scan.E_grid_brem, [0, 25, 50, 75])
    assert film.scan.E_grid_line is None
    assert film.scan.E_grid_line_by_energy is not None
    np.testing.assert_array_equal(film.scan.E_grid_line_by_energy[30.0], [200, 205, 210])
    assert other.profile == "standard"
    assert other.scan.E_grid_line_by_energy is not None
    np.testing.assert_array_equal(other.scan.E_grid_line_by_energy[60.0], [300])


def test_material_errors_remain_in_catalog_order() -> None:
    errors = _Errors()
    refs: dict[str, str] = {}
    profiles = {
        "standard": {
            **_SCAN,
            "materials": ["film", "missing"],
            "overrides": {"missing": {}},
            "energy_grid_refs": {"missing": "b" * 64},
        },
    }
    parsed = _parse_materials(
        {
            "film": {
                "substrate": "film",
                "stack": [],
                "validation": {"extra": True, "crystal_database_match": "bad"},
            },
            "broken": False,
        },
        {"film": _crystal()},
        {},
        {"ghost": {}},
        errors,
        profiles,
        {},
        refs,
    )

    assert list(parsed) == ["film"]
    assert parsed["film"].substrate == "film"
    assert parsed["film"].stack == ()
    assert parsed["film"].validation.crystal_database_match is None
    assert errors.items == [
        "materials.film.stack: must be a nonempty array of layers",
        "materials.film: cannot define both substrate and stack",
        "materials.film.validation.extra: unknown key",
        "materials.film.validation.crystal_database_match: must be 'verified' or 'unverified'",
        "materials.broken: must be a table",
        "profiles.standard.materials: unknown material 'missing'",
        "profiles.standard.overrides.missing: unknown material",
        "profiles.standard.energy_grid_refs.missing: unknown material",
        "energy_grids.ghost: unknown material",
    ]


def test_uncovered_artifact_reports_error_without_replacing_profile_energy() -> None:
    errors = _Errors()
    refs: dict[str, str] = {}
    digest = "c" * 64
    parsed = _parse_materials(
        {"film": {}},
        {"film": _crystal()},
        {},
        {},
        errors,
        {"standard": {**_SCAN, "energy_keV": 50}},
        {
            "film": (
                digest,
                {
                    "beam_energies_keV": [30.0],
                    "brem_grid": {"start_eV": 0, "stop_eV": 100, "step_eV": 25},
                    "line_rows": [],
                },
            ),
        },
        refs,
    )

    assert refs == {"film": digest}
    np.testing.assert_array_equal(parsed["film"].scan.energy_keV, [50.0])
    assert errors.items == [
        f"profiles.standard.energy_grid_refs.film: artifact {digest} covers beam energies "
        "[30.0], which do not include [50.0]",
    ]
