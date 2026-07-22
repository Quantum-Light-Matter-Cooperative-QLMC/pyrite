import cxr_mc.line_grid.defaults as d


def test_load_missing_file_returns_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DEFAULTS_PATH", tmp_path / "absent.toml")
    got = d.load_defaults()
    assert got["thickness_ang"] == [1.0e7]
    assert got["brem_step_ev"] == 25.0
    assert got["tilts"] == []  # [] = fall back to profile angles
    assert got["energies"] == [30, 40, 50, 60, 100, 150, 200, 250, 300]


def test_update_is_partial_merge_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DEFAULTS_PATH", tmp_path / "def.toml")
    d.update_defaults(thickness_ang=[5.0e6, 1.0e7], brem_step_ev=None)
    got = d.load_defaults()
    assert got["thickness_ang"] == [5.0e6, 1.0e7]
    assert got["brem_step_ev"] == 25.0  # None ignored -> fallback kept
    assert got["materials"] == d.FALLBACK["materials"]  # untouched key preserved
