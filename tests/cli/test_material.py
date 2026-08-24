from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyrite import cli
from pyrite.cli import _catalog_io
from pyrite.cli.commands import material, sweep
from tests.helpers.cli import assert_clean_result, invoke

_CATALOG = """[profiles.standard]
thickness_ang = { values = [1000.0] }
energy_keV = { values = [30.0] }
tilt_deg = { values = [5.0] }
tilt_azim_deg = { values = [95.0] }

[profiles.standard.overrides.hopg]
thickness_ang = { values = [2000.0] }
E_grid_brem = { arange = { start = 0.0, stop = 1000.0, step = 10.0 } }

[profiles.survey]
materials = ["hopg"]
thickness_ang = { values = [500.0] }
energy_keV = { values = [40.0] }
tilt_deg = { values = [10.0] }
tilt_azim_deg = { values = [45.0] }

[profiles.survey.overrides.hopg]
energy_keV = { values = [60.0] }

[materials.hopg]

[materials.mose2]
"""


def _catalog(tmp_path, monkeypatch):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(_CATALOG)
    monkeypatch.setattr(_catalog_io, "_MATERIALS_TOML", catalog)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_args: None)
    return catalog


def test_show_default_profile_reports_effective_sources(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["show", "hopg"])

    assert_clean_result(result)
    assert "hopg: profile standard" in result.stdout
    assert "thickness: [2000] (overridden)" in result.stdout
    assert "energy: [30] (inherited)" in result.stdout
    assert "profiles this material belongs to" not in result.stdout


def test_show_nonstandard_profile_json(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["show", "hopg", "--profile", "survey", "-o", "json"])

    assert_clean_result(result)
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.material.show"
    assert document["payload"]["profile"] == "survey"
    rows = {row["name"]: row for row in document["payload"]["ranges"]}
    assert rows["energy"]["values"] == [60.0]
    assert rows["energy"]["source"] == "overridden"
    assert rows["thickness"]["values"] == [500.0]
    assert rows["thickness"]["source"] == "inherited"


def test_set_nonstandard_profile_override(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        material.command,
        ["set", "mose2", "--profile", "survey", "--energy", "70"],
    )

    assert_clean_result(result, stdout="updated profile survey, material mose2\n")
    text = catalog.read_text()
    assert "[profiles.survey.overrides.mose2]" in text
    assert "energy_keV = {values = [70.0]}" in text


def test_material_set_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        material.command,
        [
            "set",
            "mose2",
            "--profile",
            "survey",
            "--energy",
            "40",
            "--energy",
            "50:70:20",
        ],
    )

    assert_clean_result(result, stdout="updated profile survey, material mose2\n")
    assert "energy_keV = {values = [40.0, 50.0, 70.0]}" in catalog.read_text()


def test_material_range_help_documents_repeatability() -> None:
    result = invoke(material.command, ["set", "--help"])

    assert_clean_result(result)
    assert " ".join(result.stdout.split()).count("repeat to combine") == 4


def test_reset_preserves_non_range_override_siblings(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(material.command, ["set", "hopg", "--reset", "thickness"])

    assert_clean_result(result, stdout="updated profile standard, material hopg\n")
    section = (
        catalog.read_text().split("[profiles.standard.overrides.hopg]", 1)[1].split("\n[", 1)[0]
    )
    assert "thickness_ang" not in section
    assert "E_grid_brem" in section


def test_overwrite_confirmation_and_dry_run(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    original = catalog.read_text()

    declined = invoke(material.command, ["set", "hopg", "--thickness", "3000"], input="n\n")
    assert declined.exit_code == 1
    assert "profile standard, material hopg" in declined.stderr
    assert catalog.read_text() == original

    dry_run = invoke(
        material.command,
        ["set", "hopg", "--thickness", "3000", "--dry-run"],
    )
    assert_clean_result(dry_run)
    assert "+thickness_ang={values=[3000.0]}" in dry_run.stdout.replace(" ", "")
    assert catalog.read_text() == original


def test_unknown_names_report_actionable_errors(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    unknown_material = invoke(material.command, ["show", "hpg"])
    unknown_profile = invoke(material.command, ["show", "hopg", "--profile", "missing"])

    assert unknown_material.exit_code == 1
    assert "unknown material: hpg. Did you mean: hopg?" in unknown_material.stderr
    assert unknown_profile.exit_code == 1
    assert "unknown profile: missing" in unknown_profile.stderr


def test_hidden_sweep_paths_warn_and_delegate(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    shown = invoke(sweep.command, ["show", "hopg", "-o", "json"])
    changed = invoke(
        sweep.command,
        ["set", "hopg", "--profile", "survey", "--azimuth", "100"],
    )

    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["schema"] == "cxr.sweep.show"
    assert "use 'pyrite material show hopg'" in shown.stderr
    assert changed.exit_code == 0
    assert "use 'pyrite material set hopg --profile survey'" in changed.stderr
    assert "tilt_azim_deg = {values = [100.0]}" in catalog.read_text()


def test_hidden_sweep_set_concatenates_repeated_range_options(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)

    result = invoke(
        sweep.command,
        [
            "set",
            "mose2",
            "--profile",
            "survey",
            "--energy",
            "40",
            "--energy",
            "50:70:20",
        ],
    )

    assert result.exit_code == 0
    assert "use 'pyrite material set mose2 --profile survey'" in result.stderr
    assert "energy_keV = {values = [40.0, 50.0, 70.0]}" in catalog.read_text()


def test_hidden_sweep_show_without_material_preserves_overview(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)

    result = invoke(sweep.command, ["show", "-o", "json"])

    assert result.exit_code == 0
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.sweep.show"
    assert [row["name"] for row in document["payload"]["profiles"]] == [
        "standard",
        "survey",
    ]
    assert "use 'pyrite profile list'" in result.stderr


def test_root_and_group_help_expose_new_ownership_only():
    root = invoke(cli.command, ["--help"])
    profile_help = invoke(cli.command, ["profile", "--help"])
    material_help = invoke(cli.command, ["material", "--help"])

    assert_clean_result(root)
    assert "material" in root.stdout
    assert "\n  blaze " not in root.stdout
    assert "\n  catalog " not in root.stdout
    assert "\n  sweep " not in root.stdout
    assert_clean_result(profile_help)
    assert "members" in profile_help.stdout
    assert "add-material" not in profile_help.stdout
    assert "remove-material" not in profile_help.stdout
    assert_clean_result(material_help)
    assert "show" in material_help.stdout
    assert "set" in material_help.stdout
    assert "validate" in material_help.stdout
    assert "blaze" in material_help.stdout
    assert "list" not in material_help.stdout


def test_material_group_lazily_routes_validate_and_blaze():
    validate_help = invoke(cli.command, ["material", "validate", "--help"])
    blaze_help = invoke(cli.command, ["material", "blaze", "--help"])

    assert_clean_result(validate_help)
    assert "Validate bundled material catalog" in validate_help.stdout
    assert_clean_result(blaze_help)
    assert "blazed-crystal MC sweep" in blaze_help.stdout


def test_simulate_formats_result_and_uses_single_scene_api(monkeypatch):
    calls = []
    spatial = SimpleNamespace(
        ray_map=SimpleNamespace(
            tile_index=np.zeros((1, 1), dtype=int),
            solid_angle_sr=np.ones((1, 1)),
            path_length_mm=np.zeros((1, 1, 1)),
        ),
        line=SimpleNamespace(
            intrinsic_by_tile=np.array([[1.0, 2.0]]),
            mu_by_filter_inv_mm=np.array([[0.0, 0.0]]),
        ),
    )
    result = SimpleNamespace(
        energy_eV=np.array([100.0, 200.0]),
        spectrum=np.array([1.0, 2.0]),
        background_energy_eV=np.array([50.0, 100.0]),
        background=np.array([0.1, 0.2]),
        spatial=spatial,
        provenance={"observation_identity_digest": "abc"},
    )
    monkeypatch.setattr(_catalog_io, "catalog_text", lambda: ("", object()))
    monkeypatch.setattr(
        material,
        "_simulation_scene",
        lambda *_args: (
            "beam",
            "target",
            "detector",
            ("filter",),
            "scorer",
            "numerics",
            "incoherent",
        ),
    )

    import pyrite.api

    def fake_simulate(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(pyrite.api, "simulate", fake_simulate)

    machine = invoke(material.command, ["simulate", "hopg", "-o", "json"])
    assert_clean_result(machine)
    payload = json.loads(machine.stdout)["payload"]
    assert payload["line"]["energy_eV"] == [100.0, 200.0]
    assert payload["pixel_grid"]["filter_count"] == 1
    assert calls[0][0][:3] == ("beam", "target", "detector")
    assert calls[0][1]["filters"] == ("filter",)
    assert calls[0][1]["pixel_scorer"] == "scorer"

    wide = invoke(material.command, ["simulate", "hopg", "-o", "wide"])
    assert_clean_result(wide)
    assert "material=hopg" in wide.stdout


def test_simulate_json_reports_resolution_errors(monkeypatch):
    monkeypatch.setattr(_catalog_io, "catalog_text", lambda: ("", object()))
    monkeypatch.setattr(
        material,
        "_simulation_scene",
        lambda *_args: (_ for _ in ()).throw(ValueError("physical_detector is required")),
    )

    result = invoke(material.command, ["simulate", "hopg", "-o", "json"])
    assert result.exit_code == 1
    document = json.loads(result.stdout)
    assert document["schema"] == "cxr.material.simulate"
    assert "physical_detector is required" in document["errors"][0]["message"]


def _single_scene_catalog(tmp_path, monkeypatch, *, energies="[30.0]"):
    from pyrite import DATA_DIR

    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    catalog = data / "materials.toml"
    with catalog.open("a") as stream:
        stream.write(
            f"""
[profiles.single]
materials = ["hopg"]
thickness_ang = {{ values = [1000.0] }}
energy_keV = {{ values = {energies} }}
tilt_deg = {{ values = [5.0] }}
tilt_azim_deg = {{ values = [95.0] }}
E_grid_line = {{ values = [100.0, 200.0] }}
E_grid_brem = {{ values = [10.0, 20.0] }}
n_electrons = {{ values = [7] }}
n_electrons_brem = {{ values = [3] }}
straggling = true
energy_model = "midpoint"
max_dE_frac = 0.1

[profiles.single.physical_detector]
distance_mm = 400.0
polar_deg = 90.0
shape = [2, 3]
pitch_mm = [0.1, 0.2]

[[profiles.single.filters]]
name = "half"
material = "silicon"
thickness_mm = 0.1
size_mm = [2.0, 3.0]
distance_mm = 200.0
polar_deg = 90.0
"""
        )
    monkeypatch.setattr(_catalog_io, "_MATERIALS_TOML", catalog)
    return _catalog_io.catalog_text()[1]


def test_simulation_scene_resolves_real_profile_objects(tmp_path, monkeypatch):
    from pyrite.campaign.model import Beam, Numerics
    from pyrite.instrument import FilterPlate, PlanarDetector

    document = _single_scene_catalog(tmp_path, monkeypatch)
    beam, target, detector, filters, scorer, numerics, emission = material._simulation_scene(
        document, "hopg", "single"
    )

    assert isinstance(beam, Beam)
    assert beam.energy_keV == 30.0
    assert target.material == "hopg"
    assert isinstance(detector, PlanarDetector)
    assert detector.pixels.shape == (2, 3)
    assert isinstance(filters[0], FilterPlate)
    assert filters[0].name == "half"
    assert isinstance(numerics, Numerics)
    assert numerics.n_electrons == 7
    assert numerics.n_electrons_brem == 3
    assert numerics.straggling is True
    assert numerics.energy_model == "midpoint"
    assert emission == "incoherent"
    assert scorer.angular_shape == (1, 1)


def test_simulation_scene_rejects_non_singleton_profile_grid(tmp_path, monkeypatch):
    document = _single_scene_catalog(tmp_path, monkeypatch, energies="[30.0, 40.0]")

    with np.testing.assert_raises_regex(
        ValueError, "requires profile 'single' to resolve one value"
    ):
        material._simulation_scene(document, "hopg", "single")


def test_simulation_artifact_uses_exact_suffixless_path_and_never_overwrites(tmp_path):
    spatial = SimpleNamespace(
        ray_map=SimpleNamespace(
            tile_index=np.zeros((1, 1), dtype=int),
            solid_angle_sr=np.ones((1, 1)),
            path_length_mm=np.zeros((1, 1, 1)),
        ),
        line=SimpleNamespace(
            intrinsic_by_tile=np.array([[1.0, 2.0]]),
            mu_by_filter_inv_mm=np.array([[0.0, 0.0]]),
        ),
    )
    result = SimpleNamespace(
        energy_eV=np.array([100.0, 200.0]),
        spectrum=np.array([1.0, 2.0]),
        background_energy_eV=np.array([50.0, 100.0]),
        background=np.array([0.1, 0.2]),
        spatial=spatial,
    )
    output = Path(tmp_path) / "spatial-output"

    material._write_simulation_artifact(output, result)

    assert output.is_file()
    assert not output.with_suffix(".npz").exists()
    with np.load(output) as archive:
        np.testing.assert_array_equal(archive["line_energy_eV"], [100.0, 200.0])
    with np.testing.assert_raises_regex(ValueError, "output file already exists"):
        material._write_simulation_artifact(output, result)
