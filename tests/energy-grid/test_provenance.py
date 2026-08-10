import pytest

import pyrite.energy_grid.provenance as p


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


def test_profile_provenance_is_scoped_with_standard_legacy_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "PROVENANCE_PATH", tmp_path / "prov.toml")
    p.set_line("hopg", 60.0, "manual", note="legacy")
    p.set_brem("hopg", "manual", note="legacy")
    p.set_line("hopg", 30.0, "manual", note="campaign", profile="campaign")
    p.set_brem("hopg", "derived", profile="campaign")

    assert p.is_manual_line("hopg", 60.0, profile="standard") is True
    assert p.is_manual_brem("hopg", profile="standard") is True
    assert p.is_manual_line("hopg", 60.0, profile="campaign") is False
    assert p.is_manual_line("hopg", 30.0, profile="campaign") is True
    assert p.is_manual_brem("hopg", profile="campaign") is False
    assert p.get_line("hopg", 30.0, profile="standard") is None
    assert p.profile_records("campaign")["hopg"]["line"]["30"]["note"] == "campaign"
    assert p.profile_records("standard")["hopg"]["line"]["60"]["note"] == "legacy"


def test_quotes_roundtrip_without_corrupting_toml(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "PROVENANCE_PATH", tmp_path / "prov.toml")

    p.set_line("hopg", 60.0, "manual", note='detector "X" path')

    assert p.get_line("hopg", 60.0)["note"] == 'detector "X" path'


@pytest.mark.parametrize(("field", "value"), [("source", "manual\nforged"), ("note", "bad\x1b")])
def test_persisted_provenance_rejects_controls(tmp_path, monkeypatch, field, value):
    path = tmp_path / "prov.toml"
    monkeypatch.setattr(p, "PROVENANCE_PATH", path)
    kwargs = {"source": "manual", "note": None}
    kwargs[field] = value

    with pytest.raises(ValueError, match=f"{field}.*control"):
        p.set_line("hopg", 60.0, **kwargs)

    assert not path.exists()
