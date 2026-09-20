"""Per-material catalog rows and immutable artifacts cannot couple materials."""

import json
import shutil

import numpy as np
import pytest
import tomlkit

from pyrite import DATA_DIR
from pyrite.campaign import config
from pyrite.campaign.profiles import case_content_key
from pyrite.campaign.sweep import build_cases
from pyrite.energy_grid import apply
from pyrite.materials import load_material_catalog


@pytest.fixture
def catalog_path(tmp_path):
    (tmp_path / "energy-grid-artifacts").mkdir()
    path = tmp_path / "materials.toml"
    shutil.copyfile(DATA_DIR / "materials.toml", path)
    return path


def _install_legacy_hopg_row(catalog_path):
    document = tomlkit.parse(catalog_path.read_text())
    document["energy_grids"] = {
        "hopg": {
            "line_by_energy": [
                {
                    "energy_keV": 30.0,
                    "grid": {"linspace": {"start": 10.0, "stop": 2500.0, "num": 831}},
                    "source": "manual",
                }
            ]
        }
    }
    catalog_path.write_text(tomlkit.dumps(document))


def _install_hopg_artifact(catalog_path):
    from pyrite import _energy_grid_artifacts as artifacts

    identity = artifacts.artifact_identity(
        "hopg",
        [{"energy_keV": 30, "start_eV": 10, "stop_eV": 2500, "num": 831}],
        {"start_eV": 0, "stop_eV": 30000, "step_eV": 10},
        [30, 35, 40, 50, 60, 100, 150, 200, 250, 300],
    )
    stored = artifacts.write_artifact(catalog_path.parent / "energy-grid-artifacts", identity)
    document = tomlkit.parse(catalog_path.read_text())
    for profile in ("standard", "hopg_hbn", "hopg_hbn_straggling", "hopg_short"):
        document["profiles"][profile]["energy_grid_refs"] = {"hopg": stored.digest}
    catalog_path.write_text(tomlkit.dumps(document))
    return stored


def _coordinates(catalog):
    return {
        key: {
            energy: np.asarray(grid, dtype="<f8").tobytes()
            for energy, grid in (catalog.material(key).scan.E_grid_line_by_energy or {}).items()
        }
        for key in catalog.material_keys
    }


def test_changing_hopg_legacy_rows_only_changes_hopg(catalog_path, monkeypatch):
    _install_legacy_hopg_row(catalog_path)
    document = tomlkit.parse(catalog_path.read_text())
    before = load_material_catalog(catalog_path)
    document["energy_grids"]["hopg"]["line_by_energy"][0]["grid"]["linspace"]["stop"] += 100
    catalog_path.write_text(tomlkit.dumps(document))
    after = load_material_catalog(catalog_path)
    old, new = _coordinates(before), _coordinates(after)
    assert old["hopg"] != new["hopg"]
    for key in before.material_keys:
        if key != "hopg":
            assert old[key] == new[key], key
    assert old["silicon"] == new["silicon"] == {}

    def identity(catalog, material):
        monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": catalog)
        case = build_cases(
            config.material_sweep(
                material,
                energy_keV=30.0,
                thickness_ang=1000.0,
                tilt_deg=15.0,
                tilt_azim_deg=180.0,
            )
        )[0]
        return case_content_key(case)

    assert identity(before, "hopg") != identity(after, "hopg")
    assert identity(before, "silicon") == identity(after, "silicon")


def test_regenerating_hopg_artifact_isolated_across_materials_and_profiles(catalog_path, tmp_path):
    stored = _install_hopg_artifact(catalog_path)
    document = tomlkit.parse(catalog_path.read_text())
    profiles = ("standard", "hopg_hbn", "hopg_hbn_straggling", "hopg_short")
    before = {profile: load_material_catalog(catalog_path, profile=profile) for profile in profiles}
    original_refs = {
        profile: dict(document["profiles"][profile]["energy_grid_refs"]) for profile in profiles
    }
    from pyrite import _energy_grid_artifacts as artifacts

    old_digest = original_refs["standard"]["hopg"]
    assert artifacts.load_artifact(tmp_path / "energy-grid-artifacts", old_digest) == stored
    rows = json.loads(json.dumps(stored.identity["line_rows"]))
    rows[0]["stop_eV"] += 100
    brem = stored.identity["brem_grid"]
    combined_path = tmp_path / "combined.json"
    combined_path.write_text(
        json.dumps(
            {
                "hopg": {
                    "line_rows": rows,
                    "brem": {
                        "stop_eV": brem["stop_eV"],
                        "step_eV": brem["step_eV"],
                        "raw_eV": brem["stop_eV"],
                    },
                }
            }
        )
    )
    refs = apply.add_file(combined_path, catalog_path=catalog_path, force=True)
    assert refs["hopg"] != old_digest
    updated = tomlkit.parse(catalog_path.read_text())
    for profile in profiles:
        expected = dict(original_refs[profile])
        if profile == "standard":
            expected["hopg"] = refs["hopg"]
        assert dict(updated["profiles"][profile]["energy_grid_refs"]) == expected
        after = load_material_catalog(catalog_path, profile=profile)
        old_coords, new_coords = _coordinates(before[profile]), _coordinates(after)
        for key in before[profile].material_keys:
            if profile == "standard" and key == "hopg":
                assert old_coords[key] != new_coords[key]
            else:
                assert old_coords[key] == new_coords[key], (profile, key)
    # The old immutable object is still readable for profiles retaining its ref.
    assert artifacts.load_artifact(tmp_path / "energy-grid-artifacts", old_digest) == stored


@pytest.mark.parametrize(
    "material",
    [
        "4h_sic",
        "6h_sic",
        "fes2",
        "gep",
        "ges",
        "mos2-on-sapphire",
        "mos2-on-sio2-si",
        "mote2_product",
        "rese2",
        "sapphire",
        "silicon",
        "tis2",
        "tise2",
        "tite2",
        "v2o5",
        "vte2",
        "ws2",
        "zrte5",
    ],
)
def test_former_shared_grid_changes_case_identity(material):
    sweep = config.material_sweep(
        material,
        energy_keV=30.0,
        thickness_ang=1000.0,
        tilt_deg=15.0,
        tilt_azim_deg=180.0,
    )
    case = build_cases(sweep)[0]
    assert case["line_grid_policy"]["resolution"]["policy"] == "sinc-nyquist"
    # Historical energy_grids.standard @ 30 keV; never derived for this material.
    legacy = config.material_sweep(
        material,
        energy_keV=30.0,
        thickness_ang=1000.0,
        tilt_deg=15.0,
        tilt_azim_deg=180.0,
        E_grid_line_by_energy={30.0: np.linspace(10.0, 2500.0, 831)},
    )
    assert case_content_key(case) != case_content_key(build_cases(legacy)[0])
