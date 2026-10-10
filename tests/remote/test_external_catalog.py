"""Selected catalogs join remote payload identity and batch environment."""

import pytest

from pyrite.remote import config, scripts, transport


def test_external_catalog_is_staged_and_exported(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "catalog.toml").write_text("schema_version = 1\n")
    (catalog / "cifs").mkdir()
    (catalog / "cifs" / "sample.cif").write_text("data_sample\n")
    (catalog / ".git").mkdir()
    (catalog / ".git" / "private").write_text("never stage this\n")
    monkeypatch.setenv("PYRITE_CATALOG", str(catalog))
    monkeypatch.setattr(config, "SYNC_PATHS", [])

    entries = transport._sync_entries()
    digest = transport._payload_digest(entries)
    script = scripts._slurm_batch_script("job1", "echo ok", job_name="catalog-test")

    assert {arc for arc, _ in entries} == {
        "external-catalog/catalog.toml",
        "external-catalog/cifs/sample.cif",
    }
    assert (
        f"export PYRITE_CATALOG={config.shell_word(config.remote_path('external-catalog'))}"
        in script
    )
    assert f"export PYRITE_HOME={config.shell_word(config.remote_dir())}" in script
    (catalog / "cifs" / "sample.cif").write_text("data_changed\n")
    assert transport._payload_digest(transport._sync_entries()) != digest

    outside = tmp_path / "outside.cif"
    outside.write_text("data_outside\n")
    (catalog / "cifs" / "leak.cif").symlink_to(outside)
    with pytest.raises(SystemExit, match="unsafe symlink"):
        transport._sync_entries()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.usefixtures("empty_user_catalog")
def test_bundled_catalog_is_exported_explicitly(explicit, monkeypatch):
    """Regression (#290): selecting the bundled catalog must override any
    catalog configured or staged on the box, not fall back to it."""
    from pyrite._catalog_layout import bundled_catalog

    if explicit:
        monkeypatch.setenv("PYRITE_CATALOG", str(bundled_catalog()))
    else:
        monkeypatch.delenv("PYRITE_CATALOG", raising=False)
        # built-in default, regardless of the developer's config store
        monkeypatch.setattr("pyrite._catalog_layout.selected_catalog", bundled_catalog)
    monkeypatch.setattr(config, "SYNC_PATHS", [])
    bundled = config.shell_word(config.remote_path("src/pyrite/data/catalog"))

    assert not config.external_catalog_selected()
    assert transport._sync_entries() == []
    script = scripts._slurm_batch_script("job1", "echo ok", job_name="catalog-test")
    assert f"export PYRITE_CATALOG={bundled}" in script
    assert f"PYRITE_CATALOG={bundled}" in config.remote_runtime_env()
    user = config.shell_word(config.remote_path("user-catalog"))
    assert f"export PYRITE_USER_CATALOG={user}" in script
    assert f"PYRITE_USER_CATALOG={user}" in config.remote_runtime_env()


def test_user_profiles_ship_beside_the_bundled_catalog(monkeypatch, empty_user_catalog):
    """#403: user-layer profiles and artifacts ride the sync as user-catalog/."""
    from pyrite._catalog_layout import bundled_catalog

    layer = empty_user_catalog
    (layer / "profiles").mkdir()
    (layer / "profiles" / "mine.toml").write_text('materials = ["hopg"]\n')
    (layer / "energy-grid-artifacts" / "ab").mkdir(parents=True)
    (layer / "energy-grid-artifacts" / "ab" / "abcd.json").write_text("{}\n")
    (layer / "config.toml").write_text("[remote]\n")  # never shipped
    monkeypatch.setattr("pyrite._catalog_layout.selected_catalog", bundled_catalog)
    monkeypatch.setattr(config, "SYNC_PATHS", [])

    assert [arc for arc, _ in transport._sync_entries()] == [
        "user-catalog/energy-grid-artifacts/ab/abcd.json",
        "user-catalog/profiles/mine.toml",
    ]
