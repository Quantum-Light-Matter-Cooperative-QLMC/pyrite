"""Per-material catalog rows and immutable artifacts cannot couple materials."""

import json

import numpy as np
import pytest
import tomlkit

from pyrite import DATA_DIR
from pyrite._catalog_layout import read_text
from pyrite.campaign import config
from pyrite.campaign.profiles import case_content_key
from pyrite.campaign.sweep import build_cases
from pyrite.energy_grid import apply
from pyrite.materials import load_material_catalog


@pytest.fixture
def catalog_path(tmp_path):
    (tmp_path / "energy-grid-artifacts").mkdir()
    path = tmp_path / "materials.toml"
    path.write_text(read_text(DATA_DIR / "catalog"))
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


@pytest.mark.parametrize("enabled", [False, True])
def test_profile_coherent_window_opt_in_and_budget_reach_the_case(
    catalog_path, monkeypatch, enabled
):
    document = tomlkit.parse(catalog_path.read_text())
    policy = {"windows": enabled, "max_points": 5000000}
    document["profiles"]["hopg_short"]["line_grid_policy"] = policy
    document["profiles"]["hopg_short"]["n_electrons"] = {"values": [200]}
    catalog_path.write_text(tomlkit.dumps(document))
    catalog = load_material_catalog(catalog_path)
    monkeypatch.setattr(
        config,
        "_catalog",
        lambda catalog_profile="standard": load_material_catalog(
            catalog_path, profile=catalog_profile
        ),
    )
    sweep = config.material_sweep(
        "hopg",
        catalog_profile="hopg_short",
        energy_keV=60,
        thickness_ang=100000,
        tilt_deg=45,
        tilt_azim_deg=135,
    )
    case = build_cases(sweep, n_electrons=200, n_electrons_brem=1)[0]
    assert catalog.profile_emission("hopg_short") == "both"
    assert case["line_grid_policy"]["resolution"]["max_points"] == 5000000
    assert ("windows" in case["line_grid_policy"]) == enabled
    assert dict(catalog.profile_line_grid_policy("hopg_short")) == policy
    plain = config.material_sweep(
        "hopg", catalog_profile="hopg_short", line_grid_policy={"windows": False}
    )
    plain_case = build_cases(plain, n_electrons=200, n_electrons_brem=1)[0]
    assert "windows" not in plain_case["line_grid_policy"]
    if enabled:
        assert case_content_key(case) != case_content_key(
            {**case, "line_grid_policy": plain_case["line_grid_policy"]}
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("windows", "on"),
        ("windows", 1),
        ("windows", {}),
        ("max_points", True),
        ("max_points", 1),
        ("max_points", 2.5),
    ],
)
def test_profile_window_options_reject_malformed_catalog_values(catalog_path, field, value):
    from pyrite.materials import MaterialConfigError

    document = tomlkit.parse(catalog_path.read_text())
    document["profiles"]["hopg_short"]["line_grid_policy"] = {field: value}
    catalog_path.write_text(tomlkit.dumps(document))
    with pytest.raises(MaterialConfigError, match=rf"line_grid_policy\.{field}"):
        load_material_catalog(catalog_path)


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
    for profile in ("standard", "hopg_hbn", "hopg_short"):
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
    profiles = ("standard", "hopg_hbn", "hopg_short")
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
