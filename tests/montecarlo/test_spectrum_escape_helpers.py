"""Unit coverage for the shared observation-direction and escape-path helpers."""

import numpy as np
import pytest

from cxr_mc.montecarlo.groove import blazed_groove_spec
from cxr_mc.montecarlo.spectrum import (
    _escape_length,
    _observation_direction,
    _segment_escape_distance,
    mc_brem_spectrum,
    mc_spectrum,
)

_TP = np.deg2rad(45.0)
_GROOVE = blazed_groove_spec(20_000.0, np.pi / 2, _TP, np.pi)
_GROOVE_N_HAT = np.array([np.cos(_TP), 0.0, -np.sin(_TP)])


def test_observation_direction_defaults_to_the_polar_observer_direction():
    theta = np.deg2rad(119.0)

    actual = _observation_direction(theta, n_hat=None)

    np.testing.assert_allclose(actual, [np.sin(theta), 0.0, np.cos(theta)])


def test_observation_direction_normalizes_explicit_direction():
    np.testing.assert_allclose(_observation_direction(0.0, [0.0, 0.0, -4.0]), [0.0, 0.0, -1.0])


def test_escape_length_uses_the_nearest_exit_face_for_each_direction():
    z_mid = np.array([2.0, 7.0])

    np.testing.assert_allclose(_escape_length(z_mid, thickness=10.0, n_z=-0.5), [4.0, 14.0])
    np.testing.assert_allclose(_escape_length(z_mid, thickness=10.0, n_z=0.5), [16.0, 6.0])


def test_segment_escape_distance_prefers_near_side_face():
    segments = {
        "r_mid": np.array([[4.0, 0.0, 5.0]]),
        "thickness_ang": 100.0,
        "crystal_width_ang": 10.0,
        "crystal_height_ang": 10.0,
    }

    np.testing.assert_allclose(
        _segment_escape_distance(segments, np.array([1.0, 0.0, 0.01]), xp=np), [1.0]
    )


def _finite_segment(width_ang):
    return {
        "r_mid": np.array([[4.0, 0.0, 5.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([10.0]),
        "E_keV": np.array([30.0]),
        "t_ang": np.array([0.0]),
        "elec_id": np.array([0]),
        "layer": np.array([0]),
        "Ne": 1,
        "thickness_ang": 10.0,
        "crystal_width_ang": width_ang,
        "crystal_height_ang": 10.0,
    }


def test_finite_side_exit_shortens_coherent_and_brem_self_absorption():
    n_hat = np.array([1.0, 0.0, 0.01])
    short = _finite_segment(10.0)
    wide = _finite_segment(1000.0)

    line_kw = dict(
        crystal="hopg",
        hkl_list=[(0, 0, 2)],
        B_ang2=0.8,
        n_hat=n_hat,
    )
    short_line = mc_spectrum(short, np.arange(700.0, 1500.0), **line_kw)
    wide_line = mc_spectrum(wide, np.arange(700.0, 1500.0), **line_kw)
    assert np.sum(short_line) > np.sum(wide_line)

    brem_kw = dict(composition=[("C", 0.176)], n_hat=n_hat)
    short_brem = mc_brem_spectrum(short, np.arange(700.0, 5000.0, 50.0), **brem_kw)
    wide_brem = mc_brem_spectrum(wide, np.arange(700.0, 5000.0, 50.0), **brem_kw)
    assert np.sum(short_brem) > np.sum(wide_brem)


@pytest.mark.filterwarnings("error")
def test_finite_brem_pure_lateral_escape_has_no_divide_by_zero_warning():
    spectrum = mc_brem_spectrum(
        _finite_segment(10.0),
        np.arange(700.0, 5000.0, 50.0),
        composition=[("C", 0.176)],
        n_hat=np.array([1.0, 0.0, 0.0]),
    )

    assert np.all(np.isfinite(spectrum))
    assert np.sum(spectrum) > 0.0


def test_finite_side_exit_layered_absorption_stays_in_emission_layer():
    segments = _finite_segment(10.0)
    n_hat = np.array([1.0, 0.0, 0.01])
    carbon = [("C", 0.176)]
    layers = [(0.0, 10.0, carbon), (10.0, 1000.0, [("W", 0.0632)])]

    line_kw = dict(
        crystal="hopg",
        hkl_list=[(0, 0, 2)],
        B_ang2=0.8,
        n_hat=n_hat,
        composition=carbon,
    )
    reference_line = mc_spectrum(segments, np.arange(700.0, 1500.0), **line_kw)
    layered_line = mc_spectrum(segments, np.arange(700.0, 1500.0), layers=layers, **line_kw)
    # GPU spectra accumulate in float32, so equivalent absorption paths can
    # differ by slightly more than NumPy's default 1e-7 relative tolerance.
    # Observed drift on RTX 3060 Ti-class hardware reaches ~8e-6.
    np.testing.assert_allclose(layered_line, reference_line, rtol=1e-5)

    brem_kw = dict(composition=carbon, n_hat=n_hat)
    reference_brem = mc_brem_spectrum(segments, np.arange(700.0, 5000.0, 50.0), **brem_kw)
    layered_brem = mc_brem_spectrum(
        segments, np.arange(700.0, 5000.0, 50.0), layers=layers, **brem_kw
    )
    np.testing.assert_allclose(layered_brem, reference_brem, rtol=2e-7)


def test_all_none_footprint_retains_z_only_spectrum_results():
    omitted = _finite_segment(None)
    omitted.pop("crystal_width_ang")
    omitted.pop("crystal_height_ang")
    explicit_none = _finite_segment(None)
    explicit_none["crystal_height_ang"] = None
    n_hat = np.array([0.5, 0.0, 1.0])

    line_kw = dict(crystal="hopg", hkl_list=[(0, 0, 2)], B_ang2=0.8, n_hat=n_hat)
    np.testing.assert_array_equal(
        mc_spectrum(omitted, np.arange(700.0, 1500.0), **line_kw),
        mc_spectrum(explicit_none, np.arange(700.0, 1500.0), **line_kw),
    )

    brem_kw = dict(composition=[("C", 0.176)], n_hat=n_hat)
    np.testing.assert_array_equal(
        mc_brem_spectrum(omitted, np.arange(700.0, 5000.0, 50.0), **brem_kw),
        mc_brem_spectrum(explicit_none, np.arange(700.0, 5000.0, 50.0), **brem_kw),
    )


def test_brem_groove_none_retains_flat_result_bit_for_bit():
    segments = _finite_segment(10.0)
    grid = np.arange(700.0, 5000.0, 50.0)
    kwargs = dict(composition=[("C", 0.176)], n_hat=_GROOVE_N_HAT)

    np.testing.assert_array_equal(
        mc_brem_spectrum(segments, grid, **kwargs),
        mc_brem_spectrum(segments, grid, groove=None, **kwargs),
    )


def test_brem_groove_escape_stays_periodic_with_finite_footprint():
    finite = _finite_segment(10.0)
    periodic = dict(finite)
    periodic.pop("crystal_width_ang")
    periodic.pop("crystal_height_ang")
    grid = np.arange(700.0, 5000.0, 50.0)
    kwargs = dict(
        composition=[("C", 0.176)],
        n_hat=_GROOVE_N_HAT,
        groove=_GROOVE,
    )

    np.testing.assert_array_equal(
        mc_brem_spectrum(finite, grid, **kwargs),
        mc_brem_spectrum(periodic, grid, **kwargs),
    )
