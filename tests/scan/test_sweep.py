"""Sweep / build_cases: the Cartesian expansion and the required-material guard."""

import json
from dataclasses import replace

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
from cxr_mc.detectors import DetectorSpec
from cxr_mc.energy_grid.encoding import decode_energy_grid
from cxr_mc.materials import (
    CATALOG,
    LayerSpec,
)
from cxr_mc.montecarlo import runner
from cxr_mc.sweep import (
    MATERIAL_LABELS,
    BeamSpec,
    Sweep,
    build_cases,
    case_cost,
    crystal_params,
    fmt_thickness,
    geometry_table,
    scan_grid_rows,
    sweep_cost_weights,
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


def test_fmt_thickness_uses_millimetres_at_one_mm():
    assert fmt_thickness(10_000_000.0) == "1mm"


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


def test_real_manifest_materials_are_unique_and_buildable():
    """The shipped ``mats_to_sim.toml`` resolves to unique catalog keys, each of
    which builds at least one case through the standard sweep path."""
    from cxr_mc.scan import MATS_FILE, load_all_materials

    materials = load_all_materials(MATS_FILE)
    assert materials, "manifest is empty"
    assert len(materials) == len(set(materials)), "manifest has duplicate keys"
    assert all(key in CATALOG.materials for key in materials)

    for key in materials:
        sweep = Sweep(
            material=key, thickness_ang=100.0, beam=BeamSpec(energy_keV=30.0), tilt_deg=5.0
        )
        cases = build_cases(sweep)
        assert cases, f"{key} produced no cases"


def test_build_cases_is_cartesian_product():
    sw = Sweep(
        material="mose2",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=[30, 45]),
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


def test_standard_sweep_owns_ninety_degree_detector_and_builds_pi_over_two_case():
    sweep = material_sweep("mose2")
    case = build_cases(sweep)[0]

    assert sweep.detector == DetectorSpec(observation_angle_deg=90.0)
    assert case["theta_obs_rad"] == pytest.approx(np.pi / 2.0)


def test_legacy_flat_detector_inputs_normalize_to_detector_and_cases():
    sweep = Sweep(
        material="mose2",
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=5.0,
        theta_obs_deg=119.0,
        dtheta_obs_deg=16.6,
        domega_sr=0.066,
    )
    case = build_cases(sweep)[0]

    assert sweep.detector == DetectorSpec(119.0, 16.6, 0.066)
    assert case["theta_obs_rad"] == pytest.approx(np.deg2rad(119.0))
    assert case["dtheta_obs_rad"] == pytest.approx(np.deg2rad(16.6))
    assert case["domega_sr"] == pytest.approx(0.066)


def test_conflicting_nested_and_flat_detector_inputs_fail_actionably():
    with pytest.raises(ValueError, match="conflicting nested detector.*theta_obs_deg"):
        Sweep(
            material="mose2",
            detector=DetectorSpec(observation_angle_deg=119.0),
            theta_obs_deg=90.0,
        )


def test_reserved_detector_fields_remain_inert_in_case_construction():
    active = DetectorSpec(119.0, 16.6, 0.066)
    described = replace(
        active,
        response_model="registry/test",
        qe_curve="package-data/qe/test.csv",
        pixel_pitch_um=55.0,
        sensor_thickness_um=500.0,
        distance_mm=400.0,
        threshold_eV=100.0,
    )
    base = Sweep(
        material="mose2",
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=5.0,
        detector=active,
    )
    plain_case = build_cases(base)[0]
    described_case = build_cases(replace(base, detector=described))[0]

    for key in ("theta_obs_rad", "dtheta_obs_rad", "domega_sr"):
        assert described_case[key] == plain_case[key]
    for key in (
        "response_model",
        "qe_curve",
        "pixel_pitch_um",
        "sensor_thickness_um",
        "distance_mm",
        "threshold_eV",
    ):
        assert key not in described_case


def test_build_cases_selects_and_encodes_line_grid_for_each_beam_energy():
    grids = {
        30.0: np.linspace(10.0, 2500.0, 831),
        50.0: np.linspace(10.0, 3000.0, 998),
    }
    cases = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            beam=BeamSpec(energy_keV=[30.0, 50.0]),
            tilt_deg=5.0,
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

    assert {case["E0_keV"] for case in cases} == set(sweep.beam.energy_keV)
    for case in cases:
        expected = CATALOG.material("mose2").scan.E_grid_line_by_energy[case["E0_keV"]]
        np.testing.assert_array_equal(decode_energy_grid(case["E_grid_line"]), expected)


def test_fixed_line_grid_takes_precedence_over_per_beam_mapping():
    fixed = np.array([75.0, 78.0])
    cases = build_cases(
        Sweep(
            material="mose2",
            beam=BeamSpec(energy_keV=[30.0, 50.0]),
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
            beam=BeamSpec(energy_keV=[30.0, 50.0]),
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
                beam=BeamSpec(energy_keV=[30.0, 50.0]),
                E_grid_line_by_energy={30.0: np.array([10.0])},
            )
        )


def test_empty_per_beam_line_grid_fails_with_selected_energy():
    with pytest.raises(ValueError, match=r"no E_grid_line configured for beam energy 30 keV"):
        build_cases(
            Sweep(
                material="mose2",
                beam=BeamSpec(energy_keV=30.0),
                E_grid_line_by_energy={},
            )
        )


def test_implicit_brem_grid_starts_at_lowest_per_beam_line_grid_start():
    cases = build_cases(
        Sweep(
            material="mose2",
            beam=BeamSpec(energy_keV=[30.0, 50.0]),
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
            beam=BeamSpec(energy_keV=30.0),
            thickness_ang=100.0,
            tilt_deg=[1.24, 1.26, 1.25, 1.24],
            tilt_azim_deg=[-1.24, -1.26, -1.25, -1.24],
            crystal_width_mm=None,
            crystal_height_mm=None,
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
            beam=BeamSpec(energy_keV=30.0),
            tilt_deg=5.0,
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
            beam=BeamSpec(energy_keV=30.0),
            tilt_deg=5.0,
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
            beam=BeamSpec(energy_keV=30.0),
            tilt_deg=5.0,
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
            beam=BeamSpec(energy_keV=[30.0, 40.0]),
            tilt_deg=5.0,
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
            beam=BeamSpec(energy_keV=30.0),
            tilt_deg=5.0,
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


def test_build_cases_preserves_legacy_name_for_explicit_none_footprint():
    case = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            beam=BeamSpec(energy_keV=30.0),
            tilt_deg=5.0,
            crystal_width_mm=None,
            crystal_height_mm=None,
        )
    )[0]

    assert case["name"] == "MoSe2 10nm pol=5 az=0"
    assert case["crystal_width_mm"] is None
    assert case["crystal_height_mm"] is None


def test_build_cases_defaults_to_finite_footprint_and_beam_spot():
    """A bare Sweep() -- no explicit footprint or beam size override -- gets a
    finite 5x5 mm crystal footprint and a 1 mm FWHM Gaussian beam spot, so
    default sweeps transport a physically finite beam into a physically finite
    crystal rather than the legacy point-beam / laterally-infinite slab."""
    case = build_cases(
        Sweep(material="mose2", thickness_ang=100.0, beam=BeamSpec(energy_keV=30.0), tilt_deg=5.0)
    )[0]

    assert case["crystal_width_mm"] == 5.0
    assert case["crystal_height_mm"] == 5.0
    assert case["beam_fwhm_mm"] == 1.0
    assert case["name"] == "MoSe2 10nm pol=5 az=0 footprint=5x5mm"


def test_build_cases_carries_pulse_source_fields_from_beam():
    sweep = Sweep(
        material="mose2",
        thickness_ang=100.0,
        beam=BeamSpec(energy_keV=30.0, bunch_charge_pc=2.0, rep_rate_hz=10_000.0),
        tilt_deg=5.0,
    )
    case = build_cases(sweep)[0]
    assert case["bunch_charge_pc"] == 2.0
    assert case["rep_rate_hz"] == 10_000.0


def test_brem_grid_upper_limit_tracks_case_beam_energy():
    sw = Sweep(
        material="mose2",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=[25, 35]),
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
    np.testing.assert_array_equal(
        grid["thickness_ang"],
        [
            1000.0,
            5000.0,
            10000.0,
            40000.0,
            100000.0,
            200000.0,
            500000.0,
            1000000.0,
            10000000.0,
        ],
    )
    assert "substrate" not in grid

    params = crystal_params(material)
    composition = dict(params["composition"])
    assert params["beam_uvw"] == (0, 0, 2)
    assert params["hkl_list"]
    assert composition[chalcogen] == pytest.approx(2.0 * composition["Nb"])

    sweep = Sweep(
        material=material,
        thickness_ang=100.0,
        beam=BeamSpec(energy_keV=30.0),
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
            "BP (020)",
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
    np.testing.assert_array_equal(
        grid["thickness_ang"],
        [
            1000.0,
            5000.0,
            10000.0,
            40000.0,
            100000.0,
            200000.0,
            500000.0,
            1000000.0,
            10000000.0,
        ],
    )
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
        beam=BeamSpec(energy_keV=30.0),
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
    # Bulk 2H-MoTe2 without substrate (default thickness sweep).
    grid = material_grid("mote2")

    assert "substrate" not in grid  # bulk material, no substrate in default grid
    np.testing.assert_array_equal(
        grid["thickness_ang"],
        [
            1000.0,
            5000.0,
            10000.0,
            40000.0,
            100000.0,
            200000.0,
            500000.0,
            1000000.0,
            10000000.0,
        ],
    )

    sweep = material_sweep("mote2")
    assert sweep.substrate is None
    np.testing.assert_array_equal(
        sweep.thickness_ang,
        [
            1000.0,
            5000.0,
            10000.0,
            40000.0,
            100000.0,
            200000.0,
            500000.0,
            1000000.0,
            10000000.0,
        ],
    )


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
    # runnable via `cxr run standard -m <key>` like any single material
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
    assert sweep.beam.energy_keV == [30, 50]
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
    # otherwise `cxr run standard -m mos2-on-sio2-si` would clobber/resume plain mos2.pkl
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
        quick=True,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
    )
    scan.run(args)
    assert seen["path"] is not None
    assert seen["path"].endswith("mos2-on-sio2-si_quick")


def test_run_material_applies_penetration_watchdog(monkeypatch, tmp_path):
    import argparse

    from cxr_mc import scan

    captured = {}

    def fake_gate(cases, **kwargs):
        captured["input_len"] = len(cases)
        half = len(cases) // 2
        return cases[:half], cases[half:]

    def fake_run_sweep(cases, results, checkpoint_dir=None, checkpoint_path=None, **kw):
        captured["run_sweep_cases"] = len(cases)

    monkeypatch.setattr(scan, "gate_cases_by_penetration", fake_gate)
    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)

    args = argparse.Namespace(
        material="mose2",
        workers=0,
        quick=True,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
    )
    scan.run(args)

    assert captured["input_len"] > 0
    assert captured["run_sweep_cases"] == captured["input_len"] // 2


def test_build_cases_carries_groove_spacing():
    sweep = Sweep(
        material="hopg",
        tilt_deg=45.0,
        tilt_azim_deg=180.0,
        groove_spacing_ang=2.0e4,
        thickness_ang=2.0e5,
        beam=BeamSpec(energy_keV=100.0),
        crystal_width_mm=None,
        crystal_height_mm=None,
    )
    cases = build_cases(sweep)
    assert all(c["groove_spacing_ang"] == 2.0e4 for c in cases)


def test_build_cases_omits_groove_spacing_when_unset():
    cases = build_cases(
        Sweep(material="hopg", thickness_ang=100.0, beam=BeamSpec(energy_keV=30.0), tilt_deg=5.0)
    )
    assert "groove_spacing_ang" not in cases[0]


def test_build_cases_groove_requires_azim_180():
    with pytest.raises(ValueError):
        build_cases(
            Sweep(
                material="hopg",
                tilt_deg=45.0,
                tilt_azim_deg=0.0,
                groove_spacing_ang=2.0e4,
                crystal_width_mm=None,
                crystal_height_mm=None,
            )
        )


def test_build_cases_groove_requires_tilt_between_0_and_90():
    with pytest.raises(ValueError):
        build_cases(
            Sweep(
                material="hopg",
                tilt_deg=0.0,
                tilt_azim_deg=180.0,
                groove_spacing_ang=2.0e4,
                crystal_width_mm=None,
                crystal_height_mm=None,
                allow_normal_incidence=True,
            )
        )


def test_build_cases_groove_rejects_substrate():
    with pytest.raises(ValueError):
        build_cases(
            Sweep(
                material="hopg",
                tilt_deg=45.0,
                tilt_azim_deg=180.0,
                groove_spacing_ang=2.0e4,
                substrate="silicon",
            )
        )


def test_build_cases_groove_allows_default_finite_footprint():
    # A grooved sweep keeps the default finite 5x5 mm footprint (same as flat
    # sweeps), so it records a real electron hit/miss fraction; the footprint and
    # the sub-micron groove phase are independent in transport.
    cases = build_cases(
        Sweep(
            material="hopg",
            tilt_deg=45.0,
            tilt_azim_deg=180.0,
            groove_spacing_ang=2.0e4,
        )
    )
    assert cases
    for c in cases:
        assert c["crystal_width_mm"] == 5.0
        assert c["crystal_height_mm"] == 5.0
        assert c["groove_spacing_ang"] == 2.0e4


def test_build_cases_groove_rejects_nonpositive_spacing():
    with pytest.raises(ValueError):
        build_cases(
            Sweep(
                material="hopg",
                tilt_deg=45.0,
                tilt_azim_deg=180.0,
                groove_spacing_ang=0.0,
                crystal_width_mm=None,
                crystal_height_mm=None,
            )
        )


def test_build_cases_groove_requires_theta_obs_90():
    with pytest.raises(ValueError):
        build_cases(
            Sweep(
                material="hopg",
                tilt_deg=45.0,
                tilt_azim_deg=180.0,
                groove_spacing_ang=2.0e4,
                crystal_width_mm=None,
                crystal_height_mm=None,
                theta_obs_deg=45.0,
            )
        )


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
        quick=True,
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
        quick=True,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
        progress_file=str(path),
        no_progress=True,
    )

    with pytest.raises(RuntimeError, match="scan failed"):
        scan._run_material(args, "hopg")

    assert json.loads(path.read_text())["state"] == "failed"


def test_scan_performance_profile_records_resolved_beam(monkeypatch, tmp_path):
    import argparse

    from cxr_mc import performance_profile, scan

    observed = {}

    class FakePerformanceLogger:
        def __init__(self, _path, *, static, context, **_kwargs):
            observed.update(static)
            observed["context"] = context

        def start(self):
            pass

        def close(self, _state):
            pass

    def fake_run_sweep(cases, _results, **kwargs):
        kwargs["on_runtime"]({"engine": "serial", "effective_workers": 1})
        kwargs["on_activity"](
            {
                "phase": "spectrum",
                "case_index": 0,
                "case": cases[0],
                "in_flight_case_count": 1,
            }
        )
        kwargs["on_timing"](
            {
                "case_index": 0,
                "timed_case_count": 1,
                "transport_seconds": 2.0,
                "spectrum_seconds": 3.0,
                "driver_wait_seconds": 0.5,
                "transport_seconds_total": 2.0,
                "spectrum_seconds_total": 3.0,
                "driver_wait_seconds_total": 0.5,
                "gpu_oom_retry_count": 1,
            }
        )
        kwargs["on_timing"]({"checkpoint_seconds": 0.25})
        kwargs["on_case"](cases[0])
        return True

    monkeypatch.setattr(performance_profile, "PerformanceLogger", FakePerformanceLogger)
    monkeypatch.setattr(scan, "run_sweep", fake_run_sweep)
    args = argparse.Namespace(
        workers=0,
        quick=True,
        n_families=None,
        beam_uvw=None,
        checkpoint_dir=str(tmp_path),
        progress_file=None,
        no_progress=True,
        performance_profile="standard",
        performance_dir=str(tmp_path / "performance"),
    )

    scan._run_material(args, "hopg")

    # The beam the performance log records is the one the profile resolved --
    # there is no per-run override path (tests/test_scan_beam_options.py), so
    # these are the profile's own values, not anything the caller passed in.
    beam = observed["beam_parameters"]
    assert beam["energy_keV"] == [30, 50]
    assert beam["transverse_fwhm_x_mm"] == 1.0
    assert beam["transverse_fwhm_y_mm"] == 1.0
    assert beam["bunch_length_fs"] is None
    assert beam["long_shape"] == "gaussian"
    assert beam["rep_rate_hz"] == 5000.0
    assert beam["bunch_charge_pc"] == 1.0
    context = observed["context"]()
    assert context["current"]["configuration"]
    assert context["active_case"]["configuration"]
    assert context["phase"] == "spectrum"
    assert context["transport_seconds_total"] == 2.0
    assert context["spectrum_seconds_total"] == 3.0
    assert context["gpu_feed_wait_fraction"] == 0.5 / 3.5
    assert context["gpu_oom_retry_count_total"] == 1
    assert context["checkpoint_seconds_total"] == 0.25


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


# ---- compute-cost proxy (progress weighting; instrumentation) ----------------
def _hopg_cases(**kw):
    """Flat-face HOPG cases at a legal groove-free tilt for the cost tests."""
    return build_cases(Sweep(material="hopg", tilt_deg=5.0, tilt_azim_deg=180.0, **kw))


def test_case_cost_is_positive_and_deterministic():
    (case,) = _hopg_cases(thickness_ang=2e4, beam=BeamSpec(energy_keV=60.0))
    assert case_cost(case) > 0.0
    assert case_cost(case) == case_cost(case)  # pure fn of the case dict


def test_case_cost_rises_with_beam_energy_for_a_thick_slab():
    # A thick slab stops the beam inside it, so the CSDA path length (segment
    # count) grows with E0 -- the very term the flat "N of M" bar ignores.
    cases = _hopg_cases(thickness_ang=1e7, beam=BeamSpec(energy_keV=[30.0, 60.0, 100.0, 200.0]))
    costs = [case_cost(c) for c in cases]
    assert all(lo < hi for lo, hi in zip(costs[:-1], costs[1:], strict=True))


def test_case_cost_walks_a_multilayer_stack():
    # Adding a substrate lets electrons keep depositing past the film, so the
    # stacked case must cost strictly more than the free-standing film.
    (film,) = _hopg_cases(thickness_ang=2e4, beam=BeamSpec(energy_keV=60.0))
    (stacked,) = build_cases(
        Sweep(
            material="hopg",
            tilt_deg=5.0,
            tilt_azim_deg=180.0,
            thickness_ang=2e4,
            beam=BeamSpec(energy_keV=60.0),
            substrate="silicon",
        )
    )
    assert stacked.get("abs_layers") is not None
    assert case_cost(stacked) > case_cost(film)


def test_sweep_cost_weights_match_case_cost_and_sum_to_total():
    cases = _hopg_cases(thickness_ang=2e4, beam=BeamSpec(energy_keV=[30.0, 60.0]))
    weights, total = sweep_cost_weights(cases)
    assert set(weights) == {(c["name"], c["E0_keV"]) for c in cases}
    for c in cases:
        assert weights[(c["name"], c["E0_keV"])] == case_cost(c)
    assert total == pytest.approx(sum(weights.values()))


def test_scan_grid_rows_aggregate_state_fractions_by_tilt():
    cases = build_cases(
        Sweep(
            material="hopg",
            tilt_deg=[5.0, 15.0],
            tilt_azim_deg=180.0,
            thickness_ang=2e4,
            beam=BeamSpec(energy_keV=[30.0, 60.0]),
        )
    )
    cached = {(cases[0]["name"], cases[0]["E0_keV"])}
    energies, rows = scan_grid_rows(cases, cached=cached)
    assert energies == [30.0, 60.0]
    assert [r["label"] for r in rows] == ["5°", "15°"]
    # every requested (tilt, E0) slot is populated (no None) for this dense grid
    assert all(cell is not None for r in rows for cell in r["cells"])
    # the one cached pair shows up as a nonzero cached fraction; nothing excluded
    total_cached = sum(cell["cached"] for r in rows for cell in r["cells"])
    assert total_cached > 0.0
    assert all(cell["excluded"] == 0.0 for r in rows for cell in r["cells"])


def test_scan_grid_rows_mark_excluded_and_empty_slots():
    cases = _hopg_cases(thickness_ang=2e4, beam=BeamSpec(energy_keV=[30.0, 60.0]))
    dropped = [cases[0]]  # pretend the 30 keV case was penetration-excluded
    energies, rows = scan_grid_rows(cases, excluded=dropped)
    (row,) = rows  # single tilt
    excluded_at_30 = row["cells"][energies.index(cases[0]["E0_keV"])]["excluded"]
    assert excluded_at_30 == pytest.approx(1.0)
