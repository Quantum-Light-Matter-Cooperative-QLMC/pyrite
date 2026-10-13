"""End-user authoring shares domain validation and preserves machine output."""

import json

import pytest

from pyrite import _catalog_layout as layout
from pyrite import cli
from pyrite.materials import build_medium, load_material_catalog
from tests.helpers.cli import assert_clean_result, invoke
from tests.helpers.material_authoring import preserve_catalog_registry  # noqa: F401


def test_add_medium_matches_python_and_reports_source():
    result = invoke(
        cli.command,
        [
            "material",
            "add",
            "my-silica",
            "--formula",
            "SiO2",
            "--density-g-cm3",
            "2.2",
            "-o",
            "json",
        ],
    )
    assert_clean_result(result)
    payload = json.loads(result.stdout)["payload"]
    built = build_medium("my-silica", formula="SiO2", density_g_cm3=2.2)
    assert payload["composition"] == dict(built.composition)
    assert payload["identity"] == built.identity
    assert payload["source"] == "user"
    assert load_material_catalog().media[built.name] == built.medium
    shown = invoke(cli.command, ["material", "show", built.name, "-o", "json"])
    assert_clean_result(shown)
    assert json.loads(shown.stdout)["payload"]["definition"]["source"] == "user"
    listed = invoke(cli.command, ["material", "list", "-o", "json"])
    assert_clean_result(listed)
    rows = json.loads(listed.stdout)["payload"]["materials"]
    assert any(row["name"] == built.name and row["source"] == "user" for row in rows)


@pytest.mark.parametrize(
    "args, message",
    [
        (["--formula", "Si"], "density-g-cm3 is required"),
        (["--formula", "H2O", "--density-g-cm3", "1"], "unsupported"),
        (["--mass-fraction", "Si=1", "--mass-fraction", "Si=.1", "--density-g-cm3", "2"], "unique"),
        (["--formula", "Si", "--density-g-cm3", "2", "--phase", "diamond"], "require --cif"),
        (["--crystal", "silicon", "--formula", "Si"], "cannot be overridden"),
    ],
)
def test_invalid_inputs_use_error_exit_and_one_json_value(args, message):
    result = invoke(cli.command, ["material", "add", "bad", *args, "-o", "json"])
    assert result.exit_code == 1
    document = json.loads(result.stdout)
    assert message in str(document)
    assert "bad" not in load_material_catalog().media
    assert "bad" not in load_material_catalog().materials


def test_existing_crystal_and_cif_cli():
    result = invoke(cli.command, ["material", "add", "my-si", "--crystal", "silicon"])
    assert_clean_result(result)
    assert load_material_catalog().materials["my-si"].crystal_key == "silicon"
    source = layout.bundled_catalog().parent / "cifs" / "silicon.cif"
    result = invoke(
        cli.command,
        [
            "material",
            "add",
            "my-cif",
            "--cif",
            str(source),
            "--formula",
            "Si",
            "--phase",
            "diamond",
            "--surface-hkl",
            "1",
            "1",
            "1",
            "--b-ang2",
            ".46",
            "--photon-grid",
            "100",
            "400",
            "100",
        ],
    )
    assert_clean_result(result)
    assert load_material_catalog().crystals["my-cif"].E_grid.tolist() == [100, 200, 300]


def test_help_explains_units_ownership_and_limits():
    result = invoke(cli.command, ["material", "add", "--help"])
    assert_clean_result(result)
    assert "g/cm³" in result.stdout
    assert "user catalog" in result.stdout
    assert "Partial" in result.stdout
    assert "--photon-grid" in result.stdout


def test_maintainer_command_targets_checkout_bundle(
    tmp_path, monkeypatch, capsys, _isolate_user_catalog
):
    import shutil

    from pyrite.devtools import dev_cli

    checkout = tmp_path / "checkout"
    bundled = checkout / "src" / "pyrite" / "data" / "catalog"
    shutil.copytree(layout.bundled_catalog(), bundled)
    (checkout / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    monkeypatch.setattr(layout, "bundled_catalog", lambda: bundled)
    monkeypatch.setattr(dev_cli, "ROOT", checkout)
    monkeypatch.setenv("PYRITE_CATALOG", str(tmp_path / "unused-external"))
    dev_cli.main(["material", "add", "maintainer-carbon", "--formula", "C", "--density-g-cm3", "2"])
    assert "bundled" in capsys.readouterr().out
    assert (bundled / "media" / "maintainer-carbon.toml").is_file()
    assert not (_isolate_user_catalog / "media" / "maintainer-carbon.toml").exists()
    assert layout.selected_catalog() == (tmp_path / "unused-external").resolve()


def test_maintainer_command_rejects_installed_package(tmp_path, monkeypatch):
    from pyrite.devtools import dev_cli

    monkeypatch.setattr(dev_cli, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="source checkout"):
        dev_cli.main(["material", "add", "bad", "--formula", "C", "--density-g-cm3", "2"])
