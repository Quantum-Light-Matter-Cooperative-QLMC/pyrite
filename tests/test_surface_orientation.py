"""Reciprocal cleavage-plane orientation and pipeline regression tests."""

import numpy as np
import pytest

from cxr_mc.materials import LayerSpec
from cxr_mc.materials.crystal import reciprocal_g_vector
from cxr_mc.montecarlo.geometry import _orientation_R
from cxr_mc.sweep import Sweep, build_cases, layer_radiator

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
    g_surface, _ = reciprocal_g_vector(surface_hkl, GENERAL_LATTICE)

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


def test_surface_hkl_matches_direct_axis_for_orthogonal_lattice():
    lattice = {"system": "orthorhombic", "a": 3.0, "b": 4.0, "c": 5.0}

    direct = _orientation_R(lattice, (1, 0, 0), 0.23)
    reciprocal = _orientation_R(lattice, None, 0.23, surface_hkl=(1, 0, 0))

    np.testing.assert_array_equal(reciprocal, direct)


def test_surface_orientation_preserves_handedness_and_applies_azimuth_about_z():
    surface_hkl = (2, 0, -1)
    azimuth = 0.41
    base = _orientation_R(GENERAL_LATTICE, None, 0.0, surface_hkl=surface_hkl)
    rolled = _orientation_R(GENERAL_LATTICE, None, azimuth, surface_hkl=surface_hkl)
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    expected_roll = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])

    np.testing.assert_allclose(rolled, expected_roll @ base, rtol=0.0, atol=1e-15)
    assert np.linalg.det(rolled) > 0.0


def test_legacy_beam_orientation_is_bitwise_frozen():
    expected = np.array(
        [
            [0.9099727566197751, -0.40011831354967475, -0.10888028918023611],
            [0.21950802825760019, 0.6875657648577284, -0.6921484988975038],
            [0.3518036494129478, 0.6059361799379299, 0.7134952964820164],
        ]
    )

    actual = _orientation_R(GENERAL_LATTICE, (1, 2, 3), 0.37)

    np.testing.assert_array_equal(actual, expected)


def test_surface_orientation_reaches_case_and_explicit_sweep_beam_clears_it(monkeypatch):
    import cxr_mc.sweep as sweep_module

    real = sweep_module.crystal_params("mose2")
    surface_params = {**real, "beam_uvw": None, "surface_hkl": (2, 0, -1)}
    monkeypatch.setattr(sweep_module, "crystal_params", lambda *_args, **_kwargs: surface_params)

    base = dict(
        material="mose2",
        thickness_ang=100.0,
        energy_keV=30.0,
        tilt_deg=0.0,
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
            energy_keV=30.0,
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
