"""Sweep / build_cases: the Cartesian expansion and the required-material guard."""

import numpy as np
import pytest

from cxr_mc.config import (
    MATERIALS,
    PENETRATION_TILT_DEG,
    material_grid,
    material_sweep,
    trajectory_sweep,
)
from cxr_mc.sweep import MATERIAL_LABELS, Sweep, build_cases, crystal_params

ALL = [
    "mose2",
    "wse2",
    "mote2",
    "mos2",
    "ws2",
    "ptse2",
    "hfse2",
    "zrse2",
    "diamond",
    "silicon",
    "hopg",
    "hbn",
]


def test_sweep_requires_material():
    with pytest.raises(TypeError):
        Sweep(thickness_ang=1e4)  # type: ignore[call-arg]  # material has no default


@pytest.mark.parametrize("material", ALL)
def test_crystal_params_complete(material):
    cp = crystal_params(material)

    for key in ("crystal", "composition", "hkl_list", "beam_uvw", "B_ang2"):
        assert key in cp

    assert cp["composition"] and all(n > 0 for _, n in cp["composition"])


def test_crystal_params_unknown_raises():
    with pytest.raises(ValueError):
        crystal_params("unobtanium")


def test_build_cases_is_cartesian_product():
    sw = Sweep(
        material="mose2",
        thickness_ang=1e4,
        energy_keV=[30, 45],
        tilt_deg=[-30, -10],
        tilt_azim_deg=[-45],
        E_grid_line=np.arange(50.0, 100.0, 5.0),
        E_grid_brem=np.arange(0.0, 1000.0, 100.0),
    )

    cases = build_cases(sw)

    assert len(cases) == 2 * 2 * 1  # energies x polar tilts x azimuths

    required = {
        "crystal",
        "composition",
        "hkl_list",
        "B_ang2",
        "E0_keV",
        "thickness_ang",
        "theta_obs_rad",
        "tilt_deg",
    }

    assert required <= set(cases[0])


def test_brem_grid_upper_limit_tracks_case_beam_energy():
    sw = Sweep(
        material="mose2",
        thickness_ang=1e4,
        energy_keV=[25, 35],
        tilt_deg=[-10],
        tilt_azim_deg=[0],
        E_grid_line=np.arange(50.0, 100.0, 5.0),
        E_grid_brem=np.arange(0.0, 60000.0, 5000.0),
    )

    cases = build_cases(sw)

    for case in cases:
        start, stop, step = case["E_grid_brem"]
        grid = np.arange(start, stop, step)
        assert grid[-1] == pytest.approx(case["E0_keV"] * 1e3)


def test_mote2_registered():
    assert MATERIAL_LABELS["mote2"] == "MoTe2"

    assert "mote2" in MATERIALS


def test_mote2_material_grid_is_bulk():
    # Bulk 2H-MoTe2 without substrate (default ~10 um thickness).
    grid = material_grid("mote2")

    assert "substrate" not in grid  # bulk material, no substrate in default grid
    assert grid["thickness_ang"] == 10e4

    sweep = material_sweep("mote2")
    assert sweep.substrate is None
    assert sweep.thickness_ang == 10e4


def test_mote2_product_material_grid_matches_few_layer_sapphire():
    # Product target: 3-6 layers of 2H-MoTe2 on c-cut crystalline sapphire.
    layer_pitch_ang = 13.41 / 2.0
    grid = material_grid("mote2_product")

    assert grid["substrate"] == "sapphire"
    np.testing.assert_allclose(grid["thickness_ang"], layer_pitch_ang * np.arange(3, 7))

    sweep = material_sweep("mote2_product")
    assert sweep.substrate == "sapphire"
    np.testing.assert_allclose(sweep.thickness_ang, grid["thickness_ang"])

    override = material_sweep("mote2", substrate="sio2")
    assert override.substrate == "sio2"


def test_named_stack_registered():
    # a registry key can name a full STACK: film crystal + substrate-side layers,
    # runnable via `cxr scan <key>` like any single material
    assert "mos2-on-sio2-si" in MATERIALS
    sweep = material_sweep("mos2-on-sio2-si")
    assert sweep.material == "mos2"
    assert sweep.stack is not None
    assert [lay.material for lay in sweep.stack] == ["sio2", "silicon"]

    case = build_cases(sweep, 10, 5)[0]
    assert case["crystal"] == "mos2"
    assert len(case["abs_layers"]) == 3

    # the penetration-figure sweep resolves the FILM crystal too
    assert trajectory_sweep("mos2-on-sio2-si").material == "mos2"


def test_trajectory_sweep_uses_penetration_angle_set():
    sweep = trajectory_sweep("hopg")
    cases = build_cases(sweep, 10, 5)

    assert tuple(sweep.tilt_deg) == PENETRATION_TILT_DEG
    assert sorted({c["tilt_deg"] for c in cases}) == sorted(PENETRATION_TILT_DEG)
    assert len(cases) == len(PENETRATION_TILT_DEG) * 2


def test_scan_checkpoints_under_registry_name(monkeypatch, tmp_path):
    # the checkpoint must be named for the REGISTRY key, not the film crystal --
    # otherwise `cxr scan mos2-on-sio2-si` would clobber/resume plain mos2.pkl
    # (run_sweep's default derives the name from cases[0]["crystal"]).
    import argparse

    from cxr_mc import scan

    seen = {}

    def fake_run_sweep(cases, results, checkpoint_dir=None, checkpoint_path=None, **kw):
        seen["path"] = checkpoint_path

    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)
    args = argparse.Namespace(
        material="mos2-on-sio2-si", workers=0, quick=False, checkpoint_dir=str(tmp_path)
    )
    scan.run(args)
    assert seen["path"] is not None
    assert seen["path"].endswith("mos2-on-sio2-si.pkl")
