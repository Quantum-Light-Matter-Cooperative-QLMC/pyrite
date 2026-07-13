"""Sweep / build_cases: the Cartesian expansion and the required-material guard."""

import numpy as np
import pytest

from cxr_mc import materials as material_registry
from cxr_mc.config import (
    MATERIALS,
    PENETRATION_TILT_DEG,
    material_grid,
    material_sweep,
    trajectory_sweep,
)
from cxr_mc.materials import (
    CRYSTAL_PARAMS,
    MATERIAL_CONFIGS,
    MATERIAL_GRIDS,
    material_crystal_key,
)
from cxr_mc.sweep import MATERIAL_LABELS, Layer, Sweep, build_cases, crystal_params

ALL = [
    "mose2",
    "wse2",
    "mote2",
    "mos2",
    "ws2",
    "ptse2",
    "hfs2",
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
        tilt_deg=[30, 10],
        tilt_azim_deg=[45],
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
        tilt_deg=[10],
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


def test_material_registry_projects_scan_and_crystal_views():
    assert set(MATERIAL_GRIDS) <= set(MATERIAL_CONFIGS)
    assert set(CRYSTAL_PARAMS) <= set(MATERIAL_CONFIGS)
    assert Layer is material_registry.Layer

    assert material_crystal_key("mos2-on-sio2-si") == "mos2"
    assert MATERIAL_GRIDS["mos2-on-sio2-si"]["stack"][0].material == "sio2"
    assert CRYSTAL_PARAMS["mos2"]["beam_uvw"] == (0, 0, 2)


@pytest.mark.parametrize("material", ["hopg", "hbn"])
def test_pinned_hkl_materials_carry_reason(material):
    row = MATERIAL_CONFIGS[material]

    assert row["hkl_list_reason"]
    assert row["hkl_list"] == CRYSTAL_PARAMS[material]["hkl_list"]
    assert crystal_params(material, n_families=999)["hkl_list"] == row["hkl_list"]


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


def test_trajectory_sweep_accepts_explicit_penetration_thickness():
    sweep = trajectory_sweep("hbn", thickness_ang=100000.0)
    cases = build_cases(sweep, 10, 5)

    assert sweep.thickness_ang == 100000.0
    assert {c["thickness_ang"] for c in cases} == {100000.0}


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
        material="mos2-on-sio2-si",
        workers=0,
        quick=False,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
    )
    scan.run(args)
    assert seen["path"] is not None
    assert seen["path"].endswith("mos2-on-sio2-si.pkl")


def test_scan_forwards_n_families_and_beam_uvw_overrides(monkeypatch, tmp_path):
    # mose2 auto-selects hkl_list via dominant_reflections (unlike HOPG/h-BN,
    # which hand-pin it), so n_families=6 must change the resolved reflection
    # count and beam_uvw=(1, 0, 0) must override the material's (0, 0, 2) default.
    import argparse

    from cxr_mc import scan

    default_hkl = crystal_params("mose2", n_families=4)["hkl_list"]
    override_hkl = crystal_params("mose2", n_families=6)["hkl_list"]
    assert len(override_hkl) != len(default_hkl)

    seen = {}

    def fake_run_sweep(cases, results, checkpoint_dir=None, checkpoint_path=None, **kw):
        seen["cases"] = cases

    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)
    args = argparse.Namespace(
        material="mose2",
        workers=0,
        quick=True,
        n_families=6,
        beam_uvw=[1, 0, 0],
        checkpoint_dir=str(tmp_path),
    )
    scan.run(args)

    case = seen["cases"][0]
    assert case["beam_uvw"] == (1, 0, 0)
    assert len(case["hkl_list"]) == len(override_hkl)
