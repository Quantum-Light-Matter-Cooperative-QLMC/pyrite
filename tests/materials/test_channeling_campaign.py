"""Profile selection reaches configured cases and isolates checkpoint identity."""

import shutil
from dataclasses import replace

import pytest
import tomlkit

from pyrite import DATA_DIR, api
from pyrite._numerics import validate_profile_numerics
from pyrite.campaign.geometry import Slab
from pyrite.campaign.profiles import (
    dataset_identity,
    profile_run_settings,
    resolve_numerics,
    variant_stem,
)
from pyrite.campaign.sweep import BeamSpec, Sweep
from pyrite.materials.catalog import load_material_catalog
from pyrite.results import Settings
from tests.helpers.user_catalog import USER_CATALOG_FIXTURE

PLANAR = {
    "channeling_model": "planar",
    "channeling_plane": [1, -1, 0],
    "radiative_model": "uncoupled",
    "channeling_basis_size": 12,
    "channeling_bloch_cells": 4,
    "channeling_dz_ang": 500.0,
    "channeling_zone_max_ang": 10000.0,
}


def _sweep():
    return Sweep(
        material="silicon",
        beam=BeamSpec(energy_keV=[5000.0]),
        target=Slab(
            "silicon", footprint=None, thickness_ang=20000.0, tilt_deg=3.0, tilt_azim_deg=45.0
        ),
    )


def test_profile_channeling_controls_resolve_and_lower():
    validate_profile_numerics(PLANAR)
    resolution = resolve_numerics(PLANAR)
    assert resolution.sources["channeling_basis_size"] == "profile"
    settings = Settings(**PLANAR)
    cases = api.build_configured_cases(_sweep(), settings)
    assert len(cases) == 1
    assert cases[0]["channeling"]["plane_hkl"] == (1, -1, 0)
    assert cases[0]["channeling"]["basis_size"] == 12
    assert cases[0]["channeling"]["zone_max_ang"] == 10000.0


def test_channeling_identity_changes_only_for_enabled_zone(monkeypatch):
    monkeypatch.setattr(
        "pyrite.campaign.profiles.resolve_bremsstrahlung_model", lambda model, elements: "eedl"
    )
    sweep = _sweep()
    off = Settings(radiative_model="uncoupled")
    identity = dataset_identity("silicon", "full", off, sweep)
    assert identity == dataset_identity(
        "silicon", "full", replace(off, channeling_basis_size=32), sweep
    )
    assert "channeling" not in identity["resolved_parameters"]
    on = Settings(**PLANAR)
    enabled = dataset_identity("silicon", "full", on, sweep)
    changed = dataset_identity("silicon", "full", replace(on, channeling_dz_ang=250.0), sweep)
    assert len({variant_stem(item) for item in (identity, enabled, changed)}) == 3


@pytest.mark.parametrize(
    "values, match",
    [
        ({"channeling_model": "planar"}, "requires channeling_plane"),
        ({**PLANAR, "channeling_basis_size": 0}, "positive integer"),
        ({**PLANAR, "channeling_plane": [0, 0, 0]}, "nonzero"),
        ({**PLANAR, "radiative_model": "auto"}, "requires radiative_model"),
    ],
)
def test_profile_channeling_rejects_invalid_controls(values, match):
    with pytest.raises(ValueError, match=match):
        validate_profile_numerics(values)


def test_catalog_retains_channeling_profile_fields(tmp_path):
    catalog_root = tmp_path / "catalog"
    shutil.copytree(DATA_DIR / "catalog", catalog_root)
    shutil.copytree(USER_CATALOG_FIXTURE, catalog_root, dirs_exist_ok=True)
    path = catalog_root / "profiles" / "hopg_short.toml"
    document = tomlkit.parse(path.read_text())
    document.update(PLANAR)
    document["emission"] = "incoherent"
    path.write_text(tomlkit.dumps(document))
    catalog = load_material_catalog(catalog_root, profile="hopg_short")
    values = catalog.profile_numerics("hopg_short")
    assert tuple(values["channeling_plane"]) == (1, -1, 0)
    for key, value in PLANAR.items():
        if key != "channeling_plane":
            assert values[key] == value
    settings = profile_run_settings(Settings(), "hopg_short", "full", catalog)
    assert settings.channeling_plane == (1, -1, 0)
    assert settings.channeling_basis_size == 12
    assert settings.precision is None
