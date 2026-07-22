import cxr_mc.line_grid.provenance as p


def test_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "PROVENANCE_PATH", tmp_path / "absent.toml")
    assert p.load() == {}
    assert p.is_manual_line("hopg", 60.0) is False


def test_set_and_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "PROVENANCE_PATH", tmp_path / "prov.toml")
    p.set_line("hopg", 60.0, "manual", note="widened for detector X")
    p.set_brem("hopg", "derived job 458 (2026-07-22)")
    assert p.is_manual_line("hopg", 60.0) is True
    assert p.get_line("hopg", 60.0)["note"] == "widened for detector X"
    assert p.is_manual_brem("hopg") is False
    assert p.get_brem("hopg")["source"].startswith("derived job 458")
