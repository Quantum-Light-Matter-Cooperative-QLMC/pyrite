"""Reciprocal cleavage-plane orientation and pipeline regression tests."""

import numpy as np
import pytest

from cxr_mc.materials import LayerSpec, load_material_catalog
from cxr_mc.materials.crystal import HBARC_EV_ANG
from cxr_mc.montecarlo import beta_from_keV, mc_spectrum
from cxr_mc.montecarlo.geometry import _orientation_R
from cxr_mc.sweep import BeamSpec, Sweep, build_cases, layer_radiator

GENERAL_LATTICE = {
    "system": "general",
    "a": 3.2,
    "b": 4.1,
    "c": 5.3,
    "alpha": 71.0,
    "beta": 82.0,
    "gamma": 76.0,
}


def test_surface_hkl_aligns_nonorthogonal_plane_normal_with_sample_z():
    surface_hkl = (2, 0, -1)
    a, b, c = (GENERAL_LATTICE[name] for name in ("a", "b", "c"))
    alpha, beta, gamma = np.deg2rad([GENERAL_LATTICE[name] for name in ("alpha", "beta", "gamma")])
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    a3 = np.array(
        [
            c * np.cos(beta),
            c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
            0.0,
        ]
    )
    a3[2] = np.sqrt(c**2 - a3[0] ** 2 - a3[1] ** 2)
    # (2, 0, -1) plane translations satisfy 2*u - w = 0. Their Cartesian
    # cross product is an independently constructed surface normal.
    plane_t1 = a2
    plane_t2 = a1 + 2.0 * a3
    g_surface = np.cross(plane_t1, plane_t2)
    if g_surface @ a1 < 0.0:  # choose the +h reciprocal-normal direction
        g_surface = -g_surface

    for translation in (plane_t1, plane_t2):
        cosine = (g_surface @ translation) / (
            np.linalg.norm(g_surface) * np.linalg.norm(translation)
        )
        assert cosine == pytest.approx(0.0, abs=1e-15)

    R = _orientation_R(
        GENERAL_LATTICE,
        None,
        0.0,
        surface_hkl=surface_hkl,
    )

    np.testing.assert_allclose(
        R @ (g_surface / np.linalg.norm(g_surface)),
        [0.0, 0.0, 1.0],
        rtol=0.0,
        atol=1e-15,
    )


def test_surface_hkl_matches_direct_axis_for_orthogonal_lattice_numerically():
    lattice = {"system": "orthorhombic", "a": 3.0, "b": 4.0, "c": 5.0}

    direct = _orientation_R(lattice, (1, 0, 0), 0.23)
    reciprocal = _orientation_R(lattice, None, 0.23, surface_hkl=(1, 0, 0))

    np.testing.assert_allclose(reciprocal, direct, rtol=0.0, atol=2e-15)


def test_surface_orientation_preserves_handedness_and_applies_azimuth_about_z():
    surface_hkl = (2, 0, -1)
    azimuth = 0.41
    base = _orientation_R(GENERAL_LATTICE, None, 0.0, surface_hkl=surface_hkl)
    rolled = _orientation_R(GENERAL_LATTICE, None, azimuth, surface_hkl=surface_hkl)
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    expected_roll = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])

    np.testing.assert_allclose(rolled, expected_roll @ base, rtol=0.0, atol=1e-15)
    assert np.linalg.det(rolled) > 0.0


def test_legacy_beam_orientation_is_numerically_frozen():
    expected = np.array(
        [
            [0.9099727566197751, -0.40011831354967475, -0.10888028918023611],
            [0.21950802825760019, 0.6875657648577284, -0.6921484988975038],
            [0.3518036494129478, 0.6059361799379299, 0.7134952964820164],
        ]
    )

    actual = _orientation_R(GENERAL_LATTICE, (1, 2, 3), 0.37)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)


def test_surface_orientation_reaches_case_and_explicit_sweep_beam_clears_it(monkeypatch):
    import cxr_mc.sweep as sweep_module

    real = sweep_module.crystal_params("mose2")
    surface_params = {**real, "beam_uvw": None, "surface_hkl": (2, 0, -1)}
    monkeypatch.setattr(sweep_module, "crystal_params", lambda *_args, **_kwargs: surface_params)

    base = dict(
        material="mose2",
        thickness_ang=100.0,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=5.0,
        E_grid_line=np.array([100.0]),
        E_grid_brem=np.array([100.0]),
    )
    catalog_case = build_cases(Sweep(**base))[0]
    override_case = build_cases(Sweep(**base, beam_uvw=(1, 0, 0)))[0]

    assert catalog_case["surface_hkl"] == (2, 0, -1)
    assert catalog_case["beam_uvw"] is None
    assert override_case["surface_hkl"] is None
    assert override_case["beam_uvw"] == (1, 0, 0)


def test_explicit_layer_beam_override_clears_catalog_surface(monkeypatch):
    import cxr_mc.sweep as sweep_module

    monkeypatch.setattr(
        sweep_module,
        "substrate_radiator",
        lambda *_args, **_kwargs: {
            "crystal": "synthetic",
            "hkl_list": [(1, 0, 0)],
            "B_ang2": 0.6,
            "beam_uvw": None,
            "surface_hkl": (2, 0, -1),
        },
    )

    rad = layer_radiator(LayerSpec("synthetic", 100.0, beam_uvw=(0, 1, 0)))

    assert rad["beam_uvw"] == (0, 1, 0)
    assert rad["surface_hkl"] is None


def test_detector_mosaic_geometry_uses_case_surface_hkl(monkeypatch):
    from cxr_mc.montecarlo import detector

    captured = {}

    def orientation(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(detector, "_orientation_R", orientation)
    case = build_cases(
        Sweep(
            material="mose2",
            thickness_ang=100.0,
            beam=BeamSpec(energy_keV=30.0),
            tilt_deg=20.0,
            E_grid_line=np.array([100.0]),
            E_grid_brem=np.array([100.0]),
        )
    )[0]
    case.update(beam_uvw=None, surface_hkl=(2, 0, -1))

    detector.mosaic_psi_rad(case, 100.0)

    assert captured["surface_hkl"] == (2, 0, -1)


def test_runner_forwards_surface_hkl_to_spectrum(monkeypatch):
    from cxr_mc.montecarlo import runner

    captured = {}
    monkeypatch.setattr(
        runner,
        "mc_spectrum",
        lambda *_args, **kwargs: captured.update(kwargs) or np.zeros(1),
    )
    monkeypatch.setattr(
        runner,
        "_brem_wide_from_segments",
        lambda *_args, **_kwargs: np.zeros(1),
    )
    case = {
        "crystal": "mose2",
        "hkl_list": [(0, 0, 2)],
        "B_ang2": 0.6,
        "composition": [("Mo", 0.01), ("Se", 0.02)],
        "beam_uvw": None,
        "surface_hkl": (2, 0, -1),
        "E0_keV": 30.0,
        "spec_chunk": None,
        "layer_radiators": None,
    }
    segments = {
        "n_backscattered": 0,
        "n_missed": 0,
        "Ne": 1,
        "L_ang": np.array([1.0]),
    }
    transport = {
        "E_grid": np.array([100.0]),
        "E_brem": np.array([100.0]),
        "n_hat": np.array([1.0, 0.0, 0.0]),
        "segs": segments,
        "segs_b": segments,
    }

    runner._spectrum_case(case, transport)

    assert captured["surface_hkl"] == (2, 0, -1)


def test_orientation_rejects_conflicting_direct_and_reciprocal_contracts():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _orientation_R(
            GENERAL_LATTICE,
            (0, 0, 1),
            0.0,
            surface_hkl=(0, 0, 1),
        )


def test_real_surface_catalog_case_changes_real_cpu_spectrum(tmp_path, monkeypatch):
    import cxr_mc.sweep as sweep_module

    catalog_text = """
schema_version = 1
[profiles.standard]
thickness_ang = 100.0
energy_keV = 30.0
tilt_deg = 0.0
tilt_azim_deg = 0.0
E_grid_line = { arange = { start = 800.0, stop = 3500.0, step = 5.0 } }
E_grid_brem = 100.0
[crystals.mos2]
cif = "cifs/mos2.cif"
validation_id = "mos2-cif-migration"
B_ang2 = 0.6
surface_hkl = [1, 0, 0]
E_grid = { arange = { start = 800.0, stop = 3500.0, step = 5.0 } }
hkl_families = [[1, 0, 0]]
hkl_reason = "test surface-parallel reflection"
[media.sio2]
composition = { Si = 0.02205, O = 0.04410 }
[materials.sample]
label = "surface sample"
crystal = "mos2"
"""
    path = tmp_path / "materials.toml"
    path.write_text(catalog_text)
    catalog = load_material_catalog(path)
    monkeypatch.setattr(sweep_module, "CATALOG", catalog)

    grid = np.arange(800.0, 3500.0, 5.0)
    base = dict(
        material="sample",
        thickness_ang=100.0,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=5.0,
        E_grid_line=grid,
        E_grid_brem=np.array([100.0]),
    )
    surface_case = build_cases(Sweep(**base), n_electrons=1, n_electrons_brem=1)[0]
    direct_case = build_cases(Sweep(**base, beam_uvw=(1, 0, 0)), n_electrons=1, n_electrons_brem=1)[
        0
    ]
    segments = {
        "r_mid": np.array([[0.0, 0.0, 50.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([1000.0]),
        "E_keV": np.array([30.0]),
        "Ne": 1,
        "thickness_ang": 100.0,
    }
    n_hat = np.array([1.0, 0.0, 0.2])

    def spectrum(case):
        return mc_spectrum(
            segments,
            grid,
            crystal=case["crystal"],
            hkl_list=case["hkl_list"],
            B_ang2=case["B_ang2"],
            composition=case["composition"],
            beam_uvw=case["beam_uvw"],
            surface_hkl=case["surface_hkl"],
            n_hat=n_hat,
            chunk=1,
        )

    surface_spec = spectrum(surface_case)
    direct_spec = spectrum(direct_case)

    assert surface_case["surface_hkl"] == (1, 0, 0)
    assert surface_case["beam_uvw"] is None
    assert float(np.max(surface_spec)) > 0.0
    a_ang = float(catalog.crystal("mos2").lattice["a"])
    g_100 = 4.0 * np.pi / (np.sqrt(3.0) * a_ang)
    beta_30 = beta_from_keV(30.0)
    n_unit = n_hat / np.linalg.norm(n_hat)
    expected_resonance_eV = HBARC_EV_ANG * beta_30 * g_100 / (1.0 - beta_30 * n_unit[2])
    assert grid[int(np.argmax(surface_spec))] == pytest.approx(expected_resonance_eV, abs=5.0)
    assert not np.allclose(surface_spec, direct_spec, rtol=1e-6, atol=0.0)
