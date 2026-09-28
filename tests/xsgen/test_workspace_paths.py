"""Generated tables and fetched SBETHE data follow an explicit workspace."""

from pyrite.console import config
from pyrite.montecarlo.shell_configuration import _default_path
from pyrite.xsgen.sources import fetched_data_dir
from pyrite.xsgen.store import search_dirs, user_table_dir


def test_explicit_workspace_owns_generated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "workspace"))
    root = tmp_path / "workspace" / "xsgen"
    selected_shell_data = root / "reference-data/sbethe/sdbase/pdatconf.p14"
    selected_shell_data.parent.mkdir(parents=True)
    selected_shell_data.write_bytes(b"test fixture")

    assert user_table_dir() == root / "tables"
    assert fetched_data_dir("sbethe", "sdbase") == root / "reference-data/sbethe/sdbase"
    assert search_dirs()[0] == root / "tables"
    assert _default_path() == selected_shell_data
