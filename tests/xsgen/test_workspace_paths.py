"""Generated tables and fetched SBETHE data are shared across output workspaces."""

import pytest

from pyrite.console import config
from pyrite.montecarlo.shell_configuration import _default_path
from pyrite.xsgen.sources import fetched_data_dir
from pyrite.xsgen.store import packaged_table_dir, search_dirs, user_table_dir


def test_workspace_does_not_own_generated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setenv("PYRITE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "workspace"))
    root = tmp_path / "data" / "xsgen"
    shared = root / "reference-data/sbethe/sdbase/pdatconf.p14"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"test fixture")

    assert user_table_dir() == root / "tables"
    assert fetched_data_dir("sbethe", "sdbase") == root / "reference-data/sbethe/sdbase"
    assert search_dirs() == (root / "tables", packaged_table_dir())
    assert _default_path() == shared


def test_legacy_workspace_reference_data_is_read_through(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setenv("PYRITE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "workspace"))
    legacy = tmp_path / "workspace" / "xsgen/reference-data/sbethe/sdbase/pdatconf.p14"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"test fixture")

    with pytest.warns(RuntimeWarning, match="legacy workspace data"):
        assert _default_path() == legacy
