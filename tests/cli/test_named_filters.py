"""Named-filter CLI transactions preserve placements and failed-write state."""

import tomlkit

from pyrite.cli import _catalog_io
from pyrite.cli.commands.filter import command
from tests.helpers.cli import invoke


def test_named_filter_lifecycle(tmp_path, monkeypatch):
    path = tmp_path / "catalog.toml"
    path.write_text("""[filters.window]
material = "silicon"
thickness_mm = 0.1
size_mm = [20.0, 10.0]
distance_mm = 50.0
polar_deg = 60.0

[[profiles.one.filters]]
ref = "window"
name = "first"

[[profiles.two.filters]]
ref = "window"
distance_mm = 80.0
""")
    monkeypatch.setattr(_catalog_io, "active_catalog_path", lambda: path)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_: None)
    cloned = invoke(command, ["create", "copy", "--from", "window"])
    assert cloned.exit_code == 0, cloned.output
    doc = tomlkit.parse(path.read_text())
    assert doc["filters"]["copy"] == doc["filters"]["window"]
    before = path.read_bytes()
    preview = invoke(command, ["rename", "window", "renamed", "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert path.read_bytes() == before
    renamed = invoke(command, ["rename", "window", "renamed"])
    assert renamed.exit_code == 0, renamed.output
    doc = tomlkit.parse(path.read_text())
    assert doc["profiles"]["one"]["filters"] == [{"ref": "renamed", "name": "first"}]
    assert doc["profiles"]["two"]["filters"] == [{"ref": "renamed", "distance_mm": 80.0}]
    before = path.read_bytes()
    refused = invoke(command, ["delete", "renamed"])
    assert refused.exit_code != 0
    assert "cannot delete referenced" in refused.output
    assert path.read_bytes() == before
    deleted = invoke(command, ["delete", "copy"])
    assert deleted.exit_code == 0, deleted.output
    assert "copy" not in tomlkit.parse(path.read_text())["filters"]


def test_invalid_draft_does_not_write(tmp_path, monkeypatch):
    path = tmp_path / "catalog.toml"
    path.write_text('[filters.window]\nmaterial = "silicon"\n')
    monkeypatch.setattr(_catalog_io, "active_catalog_path", lambda: path)

    def reject(*_):
        raise ValueError("invalid plate thickness")

    monkeypatch.setattr(_catalog_io, "validate", reject)
    before = path.read_bytes()
    result = invoke(command, ["set", "window", "--thickness-mm", "-1"])
    assert result.exit_code != 0
    assert "invalid plate thickness" in result.output
    assert path.read_bytes() == before
