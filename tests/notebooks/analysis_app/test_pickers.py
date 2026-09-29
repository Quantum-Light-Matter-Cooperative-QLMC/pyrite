"""Single Checkpoint picker and shared slice selection for the analysis apps (#236)."""

import json
from types import SimpleNamespace

import pytest

from pyrite.apps.analysis_ui import data
from pyrite.apps.analysis_ui.data import selected_checkpoint_stem, slice_results
from pyrite.apps.analysis_ui.pickers import (
    PLACEHOLDER,
    checkpoint_face,
    checkpoint_menu,
    checkpoint_notice,
    checkpoint_picker_options,
    default_checkpoint,
)
from pyrite.apps.analysis_ui.views.spectra import SPECTRA_SPECS, spectra_selection
from pyrite.results import records


def _variant(root, stem, *, material, profile, digest):
    (root / stem).mkdir()
    (root / stem / "line.pkl").touch()
    (root / stem / "meta.json").write_text(
        json.dumps(
            {
                "dataset_identity": {
                    "material": material,
                    "fidelity": "full",
                    "catalog_profile": profile,
                    "parameter_sha256": digest,
                }
            }
        )
    )


def test_menu_lists_every_checkpoint_with_its_face(tmp_path):
    (tmp_path / "hopg.pkl").touch()
    (tmp_path / "hopg_blazed.pkl").touch()
    _variant(
        tmp_path,
        "hopg@high_energy-a7b2ce",
        material="hopg",
        profile="high_energy",
        digest="a7b2ce99",
    )

    menu = checkpoint_menu("hopg", tmp_path)

    assert [(row["value"], row["label"]) for row in menu] == [
        ("hopg", "standard · flat"),
        ("hopg_blazed", "standard · blazed"),
        ("hopg@high_energy-a7b2ce", "high_energy (a7b2ce) · flat"),
    ]
    assert checkpoint_picker_options("hopg", tmp_path) == (list(menu), "hopg")
    assert checkpoint_notice("hopg", None, tmp_path) is None


def test_missing_flat_checkpoint_never_falls_back_to_another_profile(tmp_path):
    # Today's `hopg` case: only a 1-record profile run exists beside other variants.
    _variant(
        tmp_path, "hopg@tpx-test-e6cacc", material="hopg", profile="tpx-test", digest="e6cacc11"
    )
    _variant(
        tmp_path,
        "hopg@high_energy-d605b4",
        material="hopg",
        profile="high_energy",
        digest="d605b422",
    )

    menu = checkpoint_menu("hopg", tmp_path)
    rows, initial = checkpoint_picker_options("hopg", tmp_path)

    assert default_checkpoint("hopg", menu) is None
    assert initial == PLACEHOLDER
    assert rows[0] == {"value": PLACEHOLDER, "label": "— choose a checkpoint —", "disabled": True}
    assert {row["value"] for row in rows[1:]} == {"hopg@tpx-test-e6cacc", "hopg@high_energy-d605b4"}
    assert "no standard flat checkpoint" in checkpoint_notice("hopg", None, tmp_path)
    # The placeholder resolves to no stem, so nothing is loaded.
    assert selected_checkpoint_stem(SimpleNamespace(value={"value": PLACEHOLDER})) is None


def test_empty_material_has_no_menu_or_notice(tmp_path):
    assert checkpoint_menu(None, tmp_path) == ()
    assert checkpoint_picker_options("hopg", tmp_path) == ([], None)
    assert checkpoint_notice("hopg", None, tmp_path) is None


@pytest.mark.parametrize(
    ("stem", "face"),
    [("hopg", "flat"), ("hopg_blazed", "blazed"), ("hopg@x-1", "flat"), (None, None)],
)
def test_checkpoint_face_follows_the_selected_stem(stem, face):
    assert checkpoint_face("hopg", stem) == face


def test_load_context_loads_exactly_the_picked_stem(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(
        data, "load_analysis_checkpoint", lambda stem, root: loaded.append((stem, root)) or {}
    )
    material = SimpleNamespace(value={"value": "hopg"})

    context = data.load_context(material, SimpleNamespace(value={"value": "hopg_blazed"}), tmp_path)
    assert loaded == [("hopg_blazed", tmp_path)]
    assert (context.checkpoint_stem, context.selected_face) == ("hopg_blazed", "blazed")

    context = data.load_context(material, SimpleNamespace(value={"value": PLACEHOLDER}), tmp_path)
    assert len(loaded) == 1
    assert context.checkpoint_stem is None and not context.has_data


def _results():
    out = {}
    for energy, thicknesses in ((30.0, (1e3, 5e3)), (40.0, (1e3, 5e3, 1e4))):
        for tilt in (10.0, 20.0):
            for azimuth in (0.0, 90.0):
                for thickness in thicknesses:
                    case = {
                        "E0_keV": energy,
                        "tilt_deg": tilt,
                        "tilt_azim_deg": azimuth,
                        "thickness_ang": thickness,
                    }
                    name = f"t{tilt}_a{azimuth}_d{thickness}"
                    out.setdefault(name, {})[energy] = {"case": case}
    return out


def test_slice_results_pins_named_dimensions_with_thickness_fallback():
    selected = slice_results(_results(), azimuth=90.0, thickness=1e4, energy=None)
    cases = [record["case"] for record in records(selected)]

    assert {case["tilt_azim_deg"] for case in cases} == {90.0}
    # 30 keV never reached 1e4 Å, so it keeps its thickest slab, stamped.
    assert {(case["E0_keV"], case["thickness_ang"]) for case in cases} == {(30.0, 5e3), (40.0, 1e4)}
    assert all("thickness_fallback" in case for case in cases if case["E0_keV"] == 30.0)
    with pytest.raises(ValueError, match="unknown slice keys"):
        slice_results(_results(), polar=10.0)


@pytest.mark.parametrize("vary", sorted(SPECTRA_SPECS))
def test_spectra_selection_frees_only_the_varied_dimension(vary):
    spec = SPECTRA_SPECS[vary]
    slice_values = {"energy": 40.0, "tilt": 10.0, "azimuth": 0.0, "thickness": 5e3}
    all_values = {"E0_keV": [30.0, 40.0], "tilt_deg": [10.0, 20.0], "tilt_azim_deg": [0.0, 90.0]}

    selected = spectra_selection(_results(), slice_values, spec, all_values[vary])
    cases = [record["case"] for record in records(selected)]

    assert sorted({case[vary] for case in cases}) == all_values[vary]
    pinned = {"E0_keV": 40.0, "tilt_deg": 10.0, "tilt_azim_deg": 0.0}
    for key, value in pinned.items():
        if key != vary:
            assert {case[key] for case in cases} == {value}
    assert {case["thickness_ang"] for case in cases} == {5e3}
