import json as _json
import tomllib

from cxr_mc.line_grid import apply

BASE_TOML = """schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0, 100.0] }

[materials.hopg]
label = "HOPG"
profile = "standard"
E_grid_line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2600.0, num = 864, endpoint = true } } },
  { energy_keV = 100.0, grid = { linspace = { start = 50.0, stop = 4600.0, num = 1518, endpoint = true } } },
]
E_grid_brem = { arange = { start = 0.0, stop = 136500.0, step = 25.0 } }
"""

COMBINED = {
    "hopg": {
        "line_rows": [
            {"energy_keV": 30.0, "start_eV": 10.0, "stop_eV": 2700.0, "num": 897},
            {"energy_keV": 100.0, "start_eV": 50.0, "stop_eV": 4600.0, "num": 1518},
        ],
        "brem": {"stop_eV": 140000.0, "step_eV": 25.0, "raw_eV": 133000.0},
    }
}


class _NoManual:
    def is_manual_line(self, *a):
        return False

    def is_manual_brem(self, *a):
        return False


def test_apply_rewrites_only_owned_blocks_and_reparses():
    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED, provenance_mod=_NoManual())
    assert skipped == []
    assert "stop = 2700.0, num = 897" in new_text
    assert 'label = "HOPG"' in new_text  # untouched line preserved
    tomllib.loads(new_text)  # still valid TOML


def test_apply_skips_manual_line_unless_forced():
    class ManualHopg30:
        def is_manual_line(self, material, energy):
            return material == "hopg" and float(energy) == 30.0

        def is_manual_brem(self, *a):
            return False

    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED, provenance_mod=ManualHopg30())
    assert "hopg:30" in skipped
    assert "stop = 2600.0, num = 864" in new_text  # original 30 keV row kept
    forced, skipped2 = apply.apply_bounds(
        BASE_TOML, COMBINED, force=True, provenance_mod=ManualHopg30()
    )
    assert skipped2 == []
    assert "stop = 2700.0, num = 897" in forced


def test_apply_file_writes_stamps_provenance_and_validates(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    json_path = tmp_path / "combined.json"
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    stamped = []
    monkeypatch.setattr(
        apply._provenance, "set_line", lambda m, e, s, note=None: stamped.append((m, float(e), s))
    )
    monkeypatch.setattr(apply._provenance, "set_brem", lambda m, s, note=None: None)
    monkeypatch.setattr(apply._provenance, "is_manual_line", lambda *a: False)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *a: False)
    # The BASE_TOML fixture is a deliberately minimal single-material stub, so
    # stub the full-catalog re-parse (production validates the real catalog).
    monkeypatch.setattr(apply, "load_material_catalog", lambda p: None)
    apply.apply_file(json_path, slurm_id="458", date="2026-07-22")
    assert "stop = 2700.0, num = 897" in toml_path.read_text()
    assert ("hopg", 30.0, "derived job 458 (2026-07-22)") in stamped


def test_set_line_grid_stamps_manual_and_autocomputes_num(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    calls = []
    monkeypatch.setattr(
        apply._provenance,
        "set_line",
        lambda m, e, s, note=None: calls.append((m, float(e), s, note)),
    )
    monkeypatch.setattr(apply._provenance, "is_manual_line", lambda *a: False)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *a: False)
    apply.set_line_grid("hopg", 30.0, 3000.0, note="widen tail")
    text = toml_path.read_text()
    assert "stop = 3000.0" in text
    assert calls == [("hopg", 30.0, "manual", "widen tail")]
