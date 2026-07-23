"""`cxr blaze` blazed-crystal sweep driver: arg parsing, energy/spacing
pairing, forced groove geometry, checkpoint naming, and the `build_cases`
groove-tag guard (see
docs/superpowers/specs/2026-07-23-cxr-blaze-grooved-sweep-design.md)."""

import argparse

import numpy as np
import pytest

from cxr_mc import blaze
from cxr_mc.config import material_sweep
from cxr_mc.materials import CATALOG
from cxr_mc.sweep import Sweep, build_cases, fmt_thickness

MATERIAL = "hopg"  # std tilt_deg grid [5,15,30,45,60,75,85] is groove-legal


def _parse(argv):
    ap = argparse.ArgumentParser()
    blaze._build_parser(ap)
    return ap.parse_args(argv)


def _run_and_capture(monkeypatch, argv):
    """Run the blaze driver with penetration gating and run_sweep faked out,
    capturing the concatenated case list and the checkpoint_path passed to
    run_sweep."""
    captured = {}

    def fake_gate(cases, **kwargs):
        return cases, []

    def fake_run_sweep(cases, results, checkpoint_dir=None, checkpoint_path=None, **kw):
        captured["cases"] = cases
        captured["checkpoint_path"] = checkpoint_path

    monkeypatch.setattr(blaze, "gate_cases_by_penetration", fake_gate)
    monkeypatch.setattr(blaze, "run_sweep", fake_run_sweep)
    blaze.run(_parse(argv))
    return captured


# 1. Arg parsing -------------------------------------------------------------


def test_energy_required():
    with pytest.raises(SystemExit):
        _parse([MATERIAL, "--spacing", "2e-6"])


def test_spacing_required():
    with pytest.raises(SystemExit):
        _parse([MATERIAL, "--energy", "30"])


def test_spacing_meters_converted_to_angstrom(monkeypatch):
    captured = _run_and_capture(
        monkeypatch, [MATERIAL, "--energy", "30", "--spacing", "2e-6", "--angles", "25"]
    )
    assert captured["cases"]
    assert all(c["groove_spacing_ang"] == 2.0e4 for c in captured["cases"])


# 2. Energy <-> spacing pairing ----------------------------------------------


def test_pairing_equal_length_zips(monkeypatch):
    captured = _run_and_capture(
        monkeypatch,
        [MATERIAL, "--energy", "30", "50", "--spacing", "2e-6", "3e-6", "--angles", "25"],
    )
    pairs = {(c["E0_keV"], c["groove_spacing_ang"]) for c in captured["cases"]}
    assert pairs == {(30.0, 2.0e4), (50.0, 3.0e4)}


def test_pairing_single_spacing_broadcasts(monkeypatch):
    captured = _run_and_capture(
        monkeypatch,
        [MATERIAL, "--energy", "30", "50", "--spacing", "2e-6", "--angles", "25"],
    )
    pairs = {(c["E0_keV"], c["groove_spacing_ang"]) for c in captured["cases"]}
    assert pairs == {(30.0, 2.0e4), (50.0, 2.0e4)}


def test_pairing_length_mismatch_raises(monkeypatch):
    monkeypatch.setattr(blaze, "gate_cases_by_penetration", lambda cases, **kw: (cases, []))
    monkeypatch.setattr(blaze, "run_sweep", lambda *a, **kw: None)
    args = _parse(
        [MATERIAL, "--energy", "30", "50", "60", "--spacing", "2e-6", "3e-6", "--angles", "25"]
    )
    with pytest.raises(SystemExit):
        blaze.run(args)


# 3. Forced groove geometry ---------------------------------------------------


def test_forced_geometry_on_every_case(monkeypatch):
    captured = _run_and_capture(
        monkeypatch, [MATERIAL, "--energy", "30", "--spacing", "2e-6", "--angles", "25", "45"]
    )
    cases = captured["cases"]
    assert cases
    for c in cases:
        assert c["tilt_azim_deg"] == 180.0
        assert c["crystal_width_mm"] is None
        assert c["crystal_height_mm"] is None
        assert c["theta_obs_rad"] == np.deg2rad(90.0)
        assert "groove_spacing_ang" in c
        assert c["abs_layers"] is None


# 4. Angles -------------------------------------------------------------------


def test_angles_override_sets_tilt_grid(monkeypatch):
    captured = _run_and_capture(
        monkeypatch, [MATERIAL, "--energy", "30", "--spacing", "2e-6", "--angles", "25", "45"]
    )
    assert {c["tilt_deg"] for c in captured["cases"]} == {25.0, 45.0}


def test_angles_omitted_uses_std_catalog_grid(monkeypatch):
    captured = _run_and_capture(monkeypatch, [MATERIAL, "--energy", "30", "--spacing", "2e-6"])
    expected = set(np.atleast_1d(material_sweep(MATERIAL).tilt_deg).tolist())
    assert {c["tilt_deg"] for c in captured["cases"]} == expected


# 5. Checkpoint stem ------------------------------------------------------------


def test_checkpoint_targets_blazed_stem_not_flat_face(monkeypatch):
    captured = _run_and_capture(
        monkeypatch, [MATERIAL, "--energy", "30", "--spacing", "2e-6", "--angles", "25"]
    )
    path = captured["checkpoint_path"]
    assert path is not None
    assert path.endswith(f"{MATERIAL}_blazed.pkl")
    assert not path.endswith(f"/{MATERIAL}.pkl")
    assert path != f"checkpoints/{MATERIAL}.pkl"


# 6. Name encodes spacing -------------------------------------------------------


def test_case_names_encode_groove_spacing_and_stay_disjoint(monkeypatch):
    captured = _run_and_capture(
        monkeypatch,
        [MATERIAL, "--energy", "30", "50", "--spacing", "2e-6", "3e-6", "--angles", "25"],
    )
    names_2um = {c["name"] for c in captured["cases"] if c["E0_keV"] == 30.0}
    names_3um = {c["name"] for c in captured["cases"] if c["E0_keV"] == 50.0}
    assert all("groove=2um" in name for name in names_2um)
    assert all("groove=3um" in name for name in names_3um)
    assert names_2um.isdisjoint(names_3um)


# 7. Regression: flat-face build_cases name unchanged ---------------------------


def test_flat_build_cases_name_unchanged():
    sweep = Sweep(
        material=MATERIAL,
        thickness_ang=1.0e4,
        energy_keV=30.0,
        tilt_deg=45.0,
        tilt_azim_deg=180.0,
        crystal_width_mm=None,
        crystal_height_mm=None,
    )
    cases = build_cases(sweep)
    label = CATALOG.materials[MATERIAL].label
    expected_name = f"{label} {fmt_thickness(1.0e4)} pol=45 az=180"
    assert cases[0]["name"] == expected_name
    assert "groove=" not in cases[0]["name"]
