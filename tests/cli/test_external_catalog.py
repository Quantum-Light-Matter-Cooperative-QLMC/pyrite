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
