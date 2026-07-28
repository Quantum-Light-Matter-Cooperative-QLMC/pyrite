import pytest

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


def test_reset_selected_defaults_restores_inherited_and_builtin_values(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DEFAULTS_PATH", tmp_path / "def.toml")
    d.update_defaults(tilts=[5.0], azimuths=[180.0], brem_step_ev=10.0)

    got = d.reset_defaults("tilts", "brem_step_ev")

    assert got["tilts"] == []
    assert got["azimuths"] == [180.0]
    assert got["brem_step_ev"] == 25.0


def test_reset_without_keys_restores_all_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "DEFAULTS_PATH", tmp_path / "def.toml")
    d.update_defaults(materials=["wse2"], energies=[60.0])

    assert d.reset_defaults() == d.FALLBACK


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"tilts": [-0.1]}, "tilts"),
        ({"tilts": [90.0]}, "tilts"),
        ({"azimuths": [360.1]}, "azimuths"),
        ({"thickness_ang": [0.0]}, "thickness_ang"),
        ({"energies": [float("inf")]}, "energies"),
        ({"brem_step_ev": 0.0}, "brem_step_ev"),
    ],
)
def test_invalid_defaults_leave_file_unchanged(tmp_path, monkeypatch, changes, match):
    path = tmp_path / "defaults.toml"
    path.write_text("# existing defaults\n")
    monkeypatch.setattr(d, "DEFAULTS_PATH", path)

    with pytest.raises(ValueError, match=match):
        d.update_defaults(**changes)

    assert path.read_text() == "# existing defaults\n"
