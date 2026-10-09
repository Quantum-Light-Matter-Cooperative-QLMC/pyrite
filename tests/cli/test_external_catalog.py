"""A selected catalog is used for CLI edits and custom crystal files."""

import os
import shutil

import pytest
from click.testing import CliRunner

from pyrite._catalog_layout import bundled_catalog
from pyrite.cli import command
from pyrite.console import config
from pyrite.materials._schema import MaterialConfigError
from pyrite.materials.catalog import load_material_catalog


def test_catalog_option_routes_detector_edit_outside_package(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_CATALOG", raising=False)
    catalog = tmp_path / "catalog"
    shutil.copytree(bundled_catalog(), catalog)
    shutil.copyfile(
        catalog / "profiles" / "coh_test.toml", catalog / "profiles" / "local_only.toml"
    )

    listed = CliRunner().invoke(
        command,
        ["--catalog", str(catalog), "profile", "list"],
        env={"PYRITE_MC_BACKEND": "cpu"},
    )
    result = CliRunner().invoke(
        command,
        ["--catalog", str(catalog), "detector", "create", "lab", "--observation-angle", "80"],
        env={"PYRITE_MC_BACKEND": "cpu"},
    )

    assert listed.exit_code == 0, listed.output
    assert "local_only:" in listed.output
    assert result.exit_code == 0, result.output
    assert (catalog / "detectors" / "lab.toml").is_file()
    assert not (bundled_catalog() / "detectors" / "lab.toml").exists()
    assert "PYRITE_CATALOG" not in os.environ


def test_catalog_owned_cif_is_preferred_over_packaged_file(tmp_path):
    catalog = tmp_path / "catalog"
    shutil.copytree(bundled_catalog(), catalog)
    local_cif = catalog / "cifs" / "hopg.cif"
    local_cif.parent.mkdir()
    shutil.copyfile(bundled_catalog().parent / "cifs" / "hopg.cif", local_cif)

    loaded = load_material_catalog(catalog)

    assert loaded.crystals["hopg"].cif == local_cif.resolve()


def test_catalog_cifs_symlink_cannot_escape_catalog(tmp_path):
    catalog = tmp_path / "catalog"
    shutil.copytree(bundled_catalog(), catalog)
    (catalog / "cifs").symlink_to(bundled_catalog().parent / "cifs", target_is_directory=True)

    with pytest.raises(MaterialConfigError, match="must stay inside cifs/"):
        load_material_catalog(catalog)


def test_missing_selected_catalog_fails_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    missing = tmp_path / "missing"

    by_option = CliRunner().invoke(command, ["--catalog", str(missing), "profile", "list"])
    by_env = CliRunner().invoke(command, ["profile", "list"], env={"PYRITE_CATALOG": str(missing)})

    assert by_option.exit_code != 0
    assert by_env.exit_code != 0
    assert "standard:" not in by_env.output


@pytest.mark.parametrize(
    ("profile", "missing"),
    [("coh_test", ("beam",)), ("hopg_short", ("detector",)), ("standard", ("beam",))],
)
def test_selected_catalog_run_rejects_implicit_example_instrument(
    tmp_path, monkeypatch, profile, missing
):
    from pyrite.runs import scan as runs_scan

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_CATALOG", raising=False)
    monkeypatch.setattr(runs_scan, "run", lambda args: None)
    catalog = tmp_path / "catalog"
    shutil.copytree(bundled_catalog(), catalog)

    result = CliRunner().invoke(
        command,
        ["--catalog", str(catalog), "run", profile, "-m", "hopg"],
        env={"PYRITE_MC_BACKEND": "cpu"},
    )

    assert result.exit_code == 2, result.output
    assert f"Error: profile '{profile}' names no {' or '.join(missing)};" in result.stderr
    assert "removed in 0.6.0" in result.stderr


def test_bundled_catalog_run_does_not_warn_for_example_instrument(tmp_path, monkeypatch):
    from pyrite.runs import scan as runs_scan

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_CATALOG", raising=False)
    monkeypatch.setattr(runs_scan, "run", lambda args: None)

    result = CliRunner().invoke(
        command, ["run", "coh_test", "-m", "hopg"], env={"PYRITE_MC_BACKEND": "cpu"}
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
