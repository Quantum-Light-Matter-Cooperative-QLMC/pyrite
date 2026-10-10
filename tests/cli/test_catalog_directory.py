"""Catalog-editing commands write back to the owning file of a directory catalog."""

import pytest

from pyrite.cli import _catalog_io
from pyrite.cli.commands import beam, profile
from tests.helpers.cli import assert_clean_result, invoke
from tests.helpers.user_catalog import copy_full_catalog


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / "catalog"
    copy_full_catalog(root)
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", root)
    return root


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*.toml")
        if not path.name.startswith(".")
    }


def test_beam_create_rename_delete_touch_only_beam_files(catalog):
    before = _snapshot(catalog)

    created = invoke(beam.command, ["create", "lab_gun", "--rep-rate-hz", "5000"])
    assert_clean_result(created, stdout="created beam lab_gun\n")
    assert (catalog / "beams/lab_gun.toml").read_text() == "rep_rate_hz = 5000.0\n"

    renamed = invoke(beam.command, ["rename", "lab_gun", "bench_gun"])
    assert_clean_result(renamed)
    assert not (catalog / "beams/lab_gun.toml").exists()
    assert (catalog / "beams/bench_gun.toml").exists()

    deleted = invoke(beam.command, ["delete", "bench_gun", "-y"])
    assert_clean_result(deleted)
    assert _snapshot(catalog) == before
    assert not list(catalog.rglob(".*tmp"))


def test_profile_set_rewrites_only_the_profile_file(catalog):
    before = _snapshot(catalog)

    result = invoke(profile.command, ["set", "standard", "--thickness", "3000", "--yes"])

    assert_clean_result(result)
    after = _snapshot(catalog)
    changed = {name for name in after if after[name] != before.get(name)}
    assert changed == {"profiles/standard.toml"}
    assert "thickness_ang = {values = [3000.0]}" in after["profiles/standard.toml"].decode()


def test_dry_run_leaves_directory_untouched(catalog):
    before = _snapshot(catalog)

    result = invoke(profile.command, ["set", "standard", "--thickness", "3000", "--dry-run"])

    assert_clean_result(result)
    assert "+thickness_ang" in result.stdout.replace(" ", "")
    assert _snapshot(catalog) == before
