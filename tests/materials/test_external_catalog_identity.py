"""Selecting a catalog changes where definitions load from, not run identity."""

import shutil

import pytest

from pyrite._catalog_layout import bundled_catalog, selected_catalog
from pyrite.campaign.profiles import named_profile_identity
from pyrite.console import config


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_CATALOG", raising=False)
    return tmp_path


def test_catalog_selection_precedence(isolated_config, monkeypatch):
    saved, env, explicit = (isolated_config / n for n in ("saved", "env", "explicit"))
    for path in (saved, env, explicit):
        path.mkdir()

    assert selected_catalog() == bundled_catalog()
    config.set_stored("catalog.path", str(saved))
    assert selected_catalog() == saved.resolve()
    monkeypatch.setenv("PYRITE_CATALOG", str(env))
    assert selected_catalog() == env.resolve()
    assert config.catalog_path(explicit) == explicit.resolve()


def test_identity_unchanged_by_catalog_location(isolated_config, monkeypatch):
    copy = isolated_config / "moved" / "catalog"
    shutil.copytree(bundled_catalog(), copy)
    shutil.copytree(bundled_catalog().parent / "cifs", copy / "cifs")

    bundled = named_profile_identity("hopg", "survey", catalog_profile="coh_test")
    monkeypatch.setenv("PYRITE_CATALOG", str(copy))
    moved = named_profile_identity("hopg", "survey", catalog_profile="coh_test")

    assert moved == bundled
    assert str(copy) not in repr(moved)
