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
