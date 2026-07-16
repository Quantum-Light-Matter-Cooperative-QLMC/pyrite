"""Sweep / build_cases: the Cartesian expansion and the required-material guard."""

import json

import numpy as np
import pytest

from cxr_mc import materials as material_registry
from cxr_mc._energy_grid import decode_energy_grid
from cxr_mc.config import (
    MATERIALS,
    PENETRATION_TILT_DEG,
    material_grid,
    material_sweep,
    trajectory_sweep,
)
from cxr_mc.materials import (
    CATALOG,
    LayerSpec,
)
from cxr_mc.montecarlo import runner
from cxr_mc.sweep import (
    MATERIAL_LABELS,
    Sweep,
    build_cases,
    crystal_params,
    geometry_table,
)

ALL = [
    "mose2",
    "wse2",
    "nbs2",
    "nbse2",
    "mote2",
    "mos2",
    "ws2",
    "ptse2",
    "pts2",
    "pdse2",
    "hfs2",
    "hfte2",
    "hfse2",
    "zrse2",
    "diamond",
    "silicon",
    "hopg",
    "hbn",
    "v2o5",
    "tis2",
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


def test_build_cases_selects_and_encodes_line_grid_for_each_beam_energy():
    grids = {
        30.0: np.linspace(10.0, 2500.0, 831),
        50.0: np.linspace(10.0, 3000.0, 998),
    }
    cases = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            energy_keV=[30.0, 50.0],
            tilt_deg=0.0,
            E_grid_line_by_energy=grids,
            E_grid_brem=75.0,
        )
    )

    assert [case["E0_keV"] for case in cases] == [30.0, 50.0]
    for case in cases:
        expected = grids[case["E0_keV"]]
        np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), expected)
        assert case["E_grid"] is case["E_grid_line"]


def test_catalog_line_grid_is_selected_for_each_standard_beam_energy():
    sweep = material_sweep("mose2")
    cases = build_cases(sweep)

    assert {case["E0_keV"] for case in cases} == set(sweep.energy_keV)
    for case in cases:
        expected = CATALOG.material("mose2").scan.E_grid_line_by_energy[case["E0_keV"]]
        np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), expected)


def test_fixed_line_grid_takes_precedence_over_per_beam_mapping():
    fixed = np.array([75.0, 78.0])
    cases = build_cases(
        Sweep(
            material="mose2",
            energy_keV=[30.0, 50.0],
            E_grid_line=fixed,
            E_grid_line_by_energy={30.0: np.array([10.0]), 50.0: np.array([20.0])},
        )
    )
    for case in cases:
        np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), fixed)


def test_deprecated_line_grid_alias_takes_precedence_over_per_beam_mapping():
    fixed = np.array([75.0, 78.0])
    cases = build_cases(
        Sweep(
            material="mose2",
            energy_keV=[30.0, 50.0],
            e_grid_eV=fixed,
            E_grid_line_by_energy={30.0: np.array([10.0]), 50.0: np.array([20.0])},
        )
    )
    for case in cases:
        np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), fixed)


def test_missing_per_beam_line_grid_fails_before_cases_are_built():
    with pytest.raises(ValueError, match=r"no E_grid_line configured for beam energy 50"):
        build_cases(
            Sweep(
                material="mose2",
                energy_keV=[30.0, 50.0],
                E_grid_line_by_energy={30.0: np.array([10.0])},
            )
        )


def test_empty_per_beam_line_grid_fails_with_selected_energy():
    with pytest.raises(ValueError, match=r"no E_grid_line configured for beam energy 30 keV"):
        build_cases(
            Sweep(
                material="mose2",
                energy_keV=30.0,
                E_grid_line_by_energy={},
            )
        )


def test_implicit_brem_grid_starts_at_lowest_per_beam_line_grid_start():
    cases = build_cases(
        Sweep(
            material="mose2",
            energy_keV=[30.0, 50.0],
            E_grid_line_by_energy={
                30.0: np.array([25.0, 50.0]),
                50.0: np.array([10.0, 50.0]),
            },
        )
    )

    assert {case["E_grid_brem"][0] for case in cases} == {10.0}


def test_build_cases_quantizes_angles_symmetrically_and_removes_duplicates():
    cases = build_cases(
        Sweep(
            material="mose2",
            energy_keV=30.0,
            thickness_ang=100.0,
            tilt_deg=[1.24, 1.26, 1.25, 1.24],
            tilt_azim_deg=[-1.24, -1.26, -1.25, -1.24],
            E_grid_line=np.array([75.0]),
            E_grid_brem=np.array([75.0]),
        )
    )

    assert [(case["tilt_deg"], case["tilt_azim_deg"]) for case in cases] == [
        (1.0, -1.0),
        (1.0, -1.5),
        (1.5, -1.0),
        (1.5, -1.5),
    ]
    assert [case["seed"] for case in cases] == [1, 1001, 2001, 3001]
    assert [case["name"] for case in cases] == [
        "MoSe2 10nm pol=1 az=-1",
        "MoSe2 10nm pol=1 az=-1.5",
        "MoSe2 10nm pol=1.5 az=-1",
        "MoSe2 10nm pol=1.5 az=-1.5",
    ]
    for case in cases:
        assert np.deg2rad(case["tilt_deg"]) == pytest.approx(
            np.deg2rad(float(case["name"].split("pol=")[1].split()[0]))
        )
        assert np.deg2rad(case["tilt_azim_deg"]) == pytest.approx(
            np.deg2rad(float(case["name"].split("az=")[1]))
        )


@pytest.mark.parametrize(
    ("line_grid", "brem_grid"),
    [
        (np.array([75.0]), np.array([10.0, 100.0, 1000.0])),
        (np.array([10.0, 100.0, 1000.0]), np.array([75.0])),
    ],
)
def test_build_cases_and_runner_preserve_exact_nonuniform_and_scalar_energy_grids(
    monkeypatch, line_grid, brem_grid
):
    case = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            energy_keV=30.0,
            tilt_deg=0.0,
            E_grid_line=line_grid,
            E_grid_brem=brem_grid,
        ),
        n_electrons=1,
        n_electrons_brem=1,
    )[0]

    assert isinstance(case["E_grid_line"], np.ndarray)
    assert isinstance(case["E_grid_brem"], np.ndarray)
    monkeypatch.setattr(runner, "simulate_trajectories", lambda *args, **kwargs: {})

    transport = runner._transport_case(case)

    np.testing.assert_array_equal(transport["E_grid"], line_grid)
    np.testing.assert_array_equal(transport["E_brem"], brem_grid)


def test_build_cases_keeps_legacy_triples_for_uniform_energy_grids():
    case = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            energy_keV=30.0,
            tilt_deg=0.0,
            E_grid_line=np.arange(50.0, 100.0, 5.0),
            E_grid_brem=np.arange(0.0, 1000.0, 100.0),
        )
    )[0]

    assert case["E_grid_line"] == (50.0, 100.0, 5.0)
    assert case["E_grid"] == (50.0, 100.0, 5.0)
    assert case["E_grid_brem"] == (0.0, 30100.0, 100.0)


def test_uniform_linspace_endpoint_grid_roundtrips_through_legacy_triple(monkeypatch):
    line_grid = np.linspace(0.0, 1.0, 10, endpoint=True)
    case = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            energy_keV=30.0,
            tilt_deg=0.0,
            E_grid_line=line_grid,
            E_grid_brem=75.0,
        ),
        n_electrons=1,
        n_electrons_brem=1,
    )[0]

    assert isinstance(case["E_grid_line"], tuple)
    monkeypatch.setattr(runner, "simulate_trajectories", lambda *args, **kwargs: {})

    transport = runner._transport_case(case)

    np.testing.assert_array_equal(transport["E_grid"], line_grid)


@pytest.mark.parametrize(
    "line_grid",
    [
        75.0,
        np.array([75.0, 75.0]),
        np.array([50.0, 51.0, 52.0 + 1e-13]),
    ],
)
def test_scalar_constant_and_near_uniform_energy_grids_stay_exact(monkeypatch, line_grid):
    expected = np.atleast_1d(np.asarray(line_grid, dtype=float))
    case = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            energy_keV=[30.0, 40.0],
            tilt_deg=0.0,
            E_grid_line=line_grid,
            E_grid_brem=75.0,
        ),
        n_electrons=1,
        n_electrons_brem=1,
    )[0]

    assert isinstance(case["E_grid_line"], np.ndarray)
    assert case["E_grid_line"].shape == expected.shape
    assert not case["E_grid_line"].flags.writeable
    assert not case["E_grid_brem"].flags.writeable
    with pytest.raises(ValueError):
        case["E_grid_line"][0] = -1.0

    monkeypatch.setattr(runner, "simulate_trajectories", lambda *args, **kwargs: {})
    transport = runner._transport_case(case)
    np.testing.assert_array_equal(transport["E_grid"], expected)
    np.testing.assert_array_equal(transport["E_brem"], [75.0])


def test_build_cases_sweeps_rectangular_footprints_and_labels_them():
    cases = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            energy_keV=30.0,
            tilt_deg=0.0,
            crystal_width_mm=[0.1, 0.2],
            crystal_height_mm=[0.3, 0.4],
        )
    )

    assert {(c["crystal_width_mm"], c["crystal_height_mm"]) for c in cases} == {
        (0.1, 0.3),
        (0.1, 0.4),
        (0.2, 0.3),
        (0.2, 0.4),
    }
    assert all(
        c["name"].endswith(f"footprint={c['crystal_width_mm']:g}x{c['crystal_height_mm']:g}mm")
        for c in cases
    )

    table = geometry_table(cases)
    assert {"width [mm]", "height [mm]"} <= set(table.columns)
    assert set(table["width [mm]"]) == {0.1, 0.2}
    assert set(table["height [mm]"]) == {0.3, 0.4}


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0.1, None),
        (None, 0.1),
        (0.0, 0.1),
        (0.1, -0.1),
        (float("inf"), 0.1),
        (0.1, float("inf")),
    ],
)
def test_build_cases_rejects_invalid_footprint(width, height):
    with pytest.raises(ValueError):
        build_cases(Sweep(material="mose2", crystal_width_mm=width, crystal_height_mm=height))


def test_build_cases_preserves_legacy_name_for_omitted_footprint():
    case = build_cases(Sweep(material="mose2", thickness_ang=100.0, energy_keV=30.0, tilt_deg=0.0))[
        0
    ]

    assert case["name"] == "MoSe2 10nm pol=0 az=0"
    assert case["crystal_width_mm"] is None
    assert case["crystal_height_mm"] is None


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


@pytest.mark.parametrize(
    ("material", "label", "chalcogen"),
    [("nbs2", "NbS2", "S"), ("nbse2", "NbSe2", "Se")],
)
def test_niobium_dichalcogenide_registered_and_runnable(material, label, chalcogen):
    assert MATERIAL_LABELS[material] == label
    assert material in MATERIALS

    grid = material_grid(material)
    assert grid["thickness_ang"] == 10e4
    assert "substrate" not in grid

    params = crystal_params(material)
    composition = dict(params["composition"])
    assert params["beam_uvw"] == (0, 0, 2)
    assert params["hkl_list"]
    assert composition[chalcogen] == pytest.approx(2.0 * composition["Nb"])

    sweep = Sweep(
        material=material,
        thickness_ang=100.0,
        energy_keV=30.0,
        tilt_deg=30.0,
        tilt_azim_deg=0.0,
        E_grid_line=np.arange(500.0, 520.0, 5.0),
        E_grid_brem=np.arange(0.0, 1000.0, 100.0),
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=1)[0]
    assert case["crystal"] == material
    assert case["composition"] == params["composition"]


@pytest.mark.parametrize(
    ("material", "label", "beam_uvw", "hkl_list", "ratio"),
    [
        ("v2o5", "V2O5 (010)", (0, 0, 1), [(0, 0, 1), (0, 0, -1)], {"V": 1, "O": 2.5}),
        ("tis2", "1T-TiS2 (003)", (0, 0, 1), [(0, 0, 3), (0, 0, -3)], {"Ti": 1, "S": 2}),
        ("hfte2", "1T-HfTe2 (001)", (0, 0, 1), [(0, 0, 1), (0, 0, -1)], {"Hf": 1, "Te": 2}),
        ("tise2", "1T-TiSe2 (001)", (0, 0, 1), [(0, 0, 1), (0, 0, -1)], {"Ti": 1, "Se": 2}),
        (
            "black_phosphorus",
            "black phosphorus (020)",
            (0, 1, 0),
            [(0, 2, 0), (0, -2, 0)],
            {"P": 1},
        ),
        ("4h_sic", "4H-SiC (0004)", (0, 0, 1), [(0, 0, 4), (0, 0, -4)], {"Si": 1, "C": 1}),
        ("6h_sic", "6H-SiC (0006)", (0, 0, 1), [(0, 0, 6), (0, 0, -6)], {"Si": 1, "C": 1}),
    ],
)
def test_oriented_materials_are_registered_as_symmetric_cuts(
    material, label, beam_uvw, hkl_list, ratio
):
    assert MATERIAL_LABELS[material] == label
    assert material in MATERIALS

    grid = material_grid(material)
    assert grid["thickness_ang"] == 10e4
    assert "substrate" not in grid

    params = crystal_params(material, n_families=999)
    composition = dict(params["composition"])
    assert params["beam_uvw"] == beam_uvw
    assert params["hkl_list"] == hkl_list
    assert CATALOG.crystal(material).hkl_reason
    for element, count in ratio.items():
        assert composition[element] / min(composition.values()) == pytest.approx(count)

    sweep = Sweep(
        material=material,
        thickness_ang=100.0,
        energy_keV=30.0,
        tilt_deg=30.0,
        tilt_azim_deg=0.0,
        E_grid_line=np.arange(500.0, 520.0, 5.0),
        E_grid_brem=np.arange(0.0, 1000.0, 100.0),
    )
    case = build_cases(sweep, n_electrons=2, n_electrons_brem=1)[0]
    assert case["crystal"] == material
    assert case["beam_uvw"] == beam_uvw
    assert case["hkl_list"] == hkl_list


def test_material_registry_projects_scan_and_crystal_views():
    assert MATERIALS == CATALOG.material_keys
    assert MATERIAL_LABELS == {key: material.label for key, material in CATALOG.materials.items()}
    assert CATALOG.material("mos2-on-sio2-si").crystal_key == "mos2"
    assert CATALOG.material("mos2-on-sio2-si").stack[0].material == "sio2"
    assert CATALOG.crystal("mos2").beam_uvw == (0, 0, 2)
    assert material_registry.LayerSpec is LayerSpec


@pytest.mark.parametrize("material", ["hopg", "hbn"])
def test_pinned_hkl_materials_carry_reason(material):
    spec = CATALOG.crystal(material)

    assert spec.hkl_reason
    assert crystal_params(material, n_families=999)["hkl_list"] == list(spec.hkl_list)


def test_layer_spec_is_the_frozen_stack_type():
    from dataclasses import FrozenInstanceError

    layer = LayerSpec("silicon", 5e6)
    with pytest.raises(FrozenInstanceError):
        layer.thickness_ang = 1.0  # type: ignore[misc]


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
    assert sweep.energy_keV == [30, 50]
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


def test_scan_progress_record_is_atomically_replaced(tmp_path):
    from cxr_mc.scan import _write_progress_record

    path = tmp_path / "progress" / "hopg.json"
    _write_progress_record(
        path,
        material="hopg",
        total_cases=4,
        cached_cases=1,
        completed_new_cases=0,
        state="running",
    )
    _write_progress_record(
        path,
        material="hopg",
        total_cases=4,
        cached_cases=1,
        completed_new_cases=3,
        state="done",
    )

    assert json.loads(path.read_text()) == {
        "material": "hopg",
        "total_cases": 4,
        "cached_cases": 1,
        "completed_new_cases": 3,
        "state": "done",
    }
    assert list(path.parent.glob("*.tmp")) == []


def test_scan_progress_record_tracks_running_and_done(monkeypatch, tmp_path):
    import argparse

    from cxr_mc import scan

    path = tmp_path / "progress" / "hopg.json"
    observed = []

    def fake_run_sweep(_cases, _results, **kwargs):
        callback = kwargs["on_progress"]
        callback(0, 2, 1)
        observed.append(json.loads(path.read_text()))
        callback(1, 2, 1)

    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)
    args = argparse.Namespace(
        workers=0,
        quick=False,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
        progress_file=str(path),
        no_progress=True,
    )

    scan._run_material(args, "hopg")

    assert observed[0]["state"] == "running"
    assert observed[0]["cached_cases"] == 1
    assert json.loads(path.read_text())["state"] == "done"


def test_scan_progress_record_tracks_failure(monkeypatch, tmp_path):
    import argparse

    from cxr_mc import scan

    path = tmp_path / "progress" / "hopg.json"

    def fake_run_sweep(_cases, _results, **kwargs):
        kwargs["on_progress"](0, 2, 0)
        raise RuntimeError("scan failed")

    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)
    args = argparse.Namespace(
        workers=0,
        quick=False,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
        progress_file=str(path),
        no_progress=True,
    )

    with pytest.raises(RuntimeError, match="scan failed"):
        scan._run_material(args, "hopg")

    assert json.loads(path.read_text())["state"] == "failed"


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

    assert {case["E0_keV"] for case in seen["cases"]} == {30.0, 50.0}
    case = seen["cases"][0]
    assert case["beam_uvw"] == (1, 0, 0)
    assert len(case["hkl_list"]) == len(override_hkl)


def test_scan_all_runs_every_material_in_toml_manifest(monkeypatch, tmp_path):
    """``cxr scan --all`` consumes the configured materials in file order."""
    import argparse

    from cxr_mc import scan

    manifest = tmp_path / "mats_to_sim.toml"
    manifest.write_text('materials = ["hopg", "hbn"]\n')
    monkeypatch.setattr(scan, "MATS_FILE", manifest)
    seen = []
    monkeypatch.setattr(
        scan, "run_sweep", lambda cases, results, **kw: seen.append(kw["checkpoint_path"])
    )

    scan.run(
        argparse.Namespace(
            material=None,
            all=True,
            workers=0,
            quick=True,
            n_families=None,
            beam_uvw=None,
            checkpoint_dir=str(tmp_path),
        )
    )

    assert seen == [str(tmp_path / "hopg_quick.pkl"), str(tmp_path / "hbn_quick.pkl")]


def test_scan_all_rejects_unknown_manifest_material_before_running(monkeypatch, tmp_path):
    import argparse

    from cxr_mc import scan

    manifest = tmp_path / "mats_to_sim.toml"
    manifest.write_text('materials = ["hopg", "missing"]\n')
    monkeypatch.setattr(scan, "MATS_FILE", manifest)
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: pytest.fail("must not run"))

    with pytest.raises(SystemExit, match="unknown material"):
        scan.run(
            argparse.Namespace(
                material=None,
                all=True,
                workers=0,
                quick=False,
                n_families=None,
                beam_uvw=None,
                checkpoint_dir=str(tmp_path),
            )
        )


@pytest.mark.parametrize(
    ("manifest_text", "message"),
    [
        ('materials = ["hopg", ""]\n', "non-empty strings"),
        ('materials = ["hopg", "hopg"]\n', "duplicate material"),
    ],
)
def test_scan_all_rejects_invalid_manifest_entries_before_running(
    monkeypatch, tmp_path, manifest_text, message
):
    import argparse

    from cxr_mc import scan

    manifest = tmp_path / "mats_to_sim.toml"
    manifest.write_text(manifest_text)
    monkeypatch.setattr(scan, "MATS_FILE", manifest)
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: pytest.fail("must not run"))

    with pytest.raises(SystemExit, match=message):
        scan.run(
            argparse.Namespace(
                material=None,
                all=True,
                workers=0,
                quick=False,
                n_families=None,
                beam_uvw=None,
                checkpoint_dir=str(tmp_path),
            )
        )


def test_scan_rejects_explicit_unknown_material_before_building(monkeypatch, tmp_path):
    import argparse

    from cxr_mc import scan

    monkeypatch.setattr(
        scan, "material_sweep", lambda *a, **kw: pytest.fail("must validate before building")
    )

    with pytest.raises(SystemExit, match="unknown material"):
        scan.run(
            argparse.Namespace(
                material="not-in-catalog",
                all=False,
                workers=0,
                quick=False,
                n_families=None,
                beam_uvw=None,
                checkpoint_dir=str(tmp_path),
            )
        )
