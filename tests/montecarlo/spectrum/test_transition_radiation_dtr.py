"""First-Born boundary diffraction diagnostic; no production phase certification.

Validation: dtr-kinematic-boundary
"""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.dtr import dtr_crossing_amplitudes, pxr_straight_amplitude
from pyrite.montecarlo.spectrum.transition_radiation import planar_transition_radiation


def _passage(thickness=80.0, velocity=(0.25, 0.0, 0.8)):
    v = np.asarray(velocity)
    z = np.array([0.0, thickness])
    return dict(
        velocity=np.tile(v, (2, 1)),
        time_ang=z / v[2],
        position_ang=z[:, None] * v / v[2],
        interface=np.array([0, 1]),
    )


def _base(**overrides):
    values = dict(
        photon_energy_eV=np.linspace(500.0, 540.0, 7),
        direction=np.array([0.6, 0.0, 0.8]),
        layers=[(0.98 + 0.0003j, 80.0)],
        substrate_eps=1.0,
        crystal_layer=0,
        reciprocal_vector_ang=np.array([0.0, 0.0, 0.04]),
        chi_transfer=0.002 + 0.00001j,
    )
    return values | overrides


def _pxr(config, passage):
    return pxr_straight_amplitude(
        **config,
        velocity=passage["velocity"][0],
        time_ang=passage["time_ang"][0],
        position_ang=passage["position_ang"][0],
    )


def test_zero_harmonic_and_zero_contrast():
    passage = _passage()
    for config in (_base(chi_transfer=0.0), _base(layers=[(1.0, 80.0)])):
        np.testing.assert_array_equal(dtr_crossing_amplitudes(**config, **passage), 0.0)
    np.testing.assert_array_equal(_pxr(_base(chi_transfer=0.0), passage), 0.0)


def test_transverse_complex_field_and_harmonic_linearity():
    config, passage = _base(), _passage()
    field = dtr_crossing_amplitudes(**config, **passage)
    assert field.shape == (7, 2, 3) and np.max(np.abs(field)) > 0
    np.testing.assert_allclose(np.sum(field * config["direction"], axis=-1), 0, atol=1e-18)
    scaled = dtr_crossing_amplitudes(
        **(config | {"chi_transfer": 2j * config["chi_transfer"]}), **passage
    )
    np.testing.assert_allclose(scaled, 2j * field, rtol=1e-12, atol=1e-18)


def test_translation_and_termination_phase():
    config = _base(reciprocal_vector_ang=np.array([0.015, 0.01, 0.04]))
    passage = _passage()
    field = dtr_crossing_amplitudes(**config, **passage)
    delta, dt = np.array([7.0, -4.0, 0.0]), 9.0
    moved = passage | {
        "time_ang": passage["time_ang"] + dt,
        "position_ang": passage["position_ang"] + delta,
    }
    shifted = dtr_crossing_amplitudes(**config, **moved)
    phase = np.exp(
        1j
        * (
            config["photon_energy_eV"] * (dt - config["direction"] @ delta) / HBARC_EV_ANG
            - config["reciprocal_vector_ang"] @ delta
        )
    )
    np.testing.assert_allclose(shifted, field * phase[:, None, None], rtol=1e-12, atol=1e-18)
    origin = np.array([1.0, 2.0, 3.0])
    terminated = dtr_crossing_amplitudes(**(config | {"lattice_origin_ang": origin}), **passage)
    np.testing.assert_allclose(
        terminated,
        field * np.exp(1j * config["reciprocal_vector_ang"] @ origin),
        rtol=1e-12,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        _pxr(config | {"lattice_origin_ang": origin}, passage),
        _pxr(config, passage) * np.exp(1j * config["reciprocal_vector_ang"] @ origin),
        rtol=1e-12,
    )


def test_straight_point_reference_does_not_change_pxr():
    config, passage = _base(), _passage()
    start = _pxr(config, passage)
    v = passage["velocity"][0]
    midpoint = pxr_straight_amplitude(**config, velocity=v, time_ang=37.0, position_ang=37.0 * v)
    np.testing.assert_allclose(midpoint, start, rtol=1e-12, atol=1e-18)


def test_pxr_and_dtr_share_translation_phase():
    config = _base(reciprocal_vector_ang=np.array([0.015, 0.01, 0.04]))
    passage = _passage()
    delta = np.array([11.0, -8.0, 0.0])
    moved = passage | {"position_ang": passage["position_ang"] + delta}
    before = (
        np.sum(dtr_crossing_amplitudes(**config, **passage), axis=-2) * _pxr(config, passage).conj()
    )
    after = np.sum(dtr_crossing_amplitudes(**config, **moved), axis=-2) * _pxr(config, moved).conj()
    np.testing.assert_allclose(after, before, rtol=2e-12, atol=1e-22)


@pytest.mark.slow
@pytest.mark.parametrize("velocity", [(0.25, 0.1, 0.8), (-0.25, 0.1, -0.8)])
@pytest.mark.parametrize("caps", [False, True])
def test_full_field_matches_periodic_dielectric_finite_difference(velocity, caps):
    """Independent oracle: differentiate the ordinary Maxwell stack solve.

    A fine staircase approximates eps(z)=eps0+h exp(-i g_z z). Its complex
    vacuum-amplitude derivative must be (PXR+DTR)/chi, including absolute
    phase, charge normalization, interface response and polarization. This
    does not use the diagnostic's integral or sheet propagator as the oracle.
    """
    thickness, gz, step = 80.0, 0.04, 2e-6
    config = _base(
        photon_energy_eV=520.0, chi_transfer=1.0, direction=np.array([0.6, 0.2, np.sqrt(0.6)])
    )
    lo = [(0.95 + 0.001j, 17.0)] if caps else []
    hi = [(0.99 + 0.0005j, 19.0)] if caps else []
    config["layers"] = lo + config["layers"] + hi
    config["crystal_layer"] = int(caps)
    z = np.r_[0.0, np.cumsum([d for _, d in config["layers"]])]
    v = np.asarray(velocity)
    passage = dict(
        velocity=np.tile(v, (len(z), 1)),
        time_ang=z / v[2],
        position_ang=z[:, None] * v / v[2],
        interface=np.arange(len(z)),
    )
    predicted = _pxr(config, passage) + dtr_crossing_amplitudes(**config, **passage).sum(axis=-2)

    def exact(h, count):
        dz = thickness / count
        centers = (np.arange(count) + 0.5) * dz + (17.0 if caps else 0.0)
        layers = lo + [(0.98 + 0.0003j + h * np.exp(-1j * gz * z), dz) for z in centers] + hi
        return planar_transition_radiation(
            520.0, config["direction"], passage["velocity"][0], layers, 1.0, return_amplitude=True
        )[1]

    coarse = (exact(step, 96) - exact(-step, 96)) / (2 * step)
    fine = (exact(step, 192) - exact(-step, 192)) / (2 * step)
    # The midpoint staircase is second order. Opposite-going wave phases
    # vary rapidly for v_z < 0, so extrapolate rather than confusing its mesh
    # error with the complex boundary field. Require mesh convergence too.
    assert np.linalg.norm(fine - predicted) < np.linalg.norm(coarse - predicted) / 3
    numerical = (4 * fine - coarse) / 3
    np.testing.assert_allclose(predicted, numerical, rtol=1e-5, atol=1e-9)


def test_vacuum_pxr_matches_closed_volume_dipole_integral():
    """Analytic outgoing dyadic Green function, no dielectric stack interfaces."""
    config = _base(photon_energy_eV=520.0, layers=[(1.0, 80.0)])
    passage = _passage()
    energy, n, g = 520.0, config["direction"], config["reciprocal_vector_ang"] * HBARC_EV_ANG
    v = passage["velocity"][0]
    kpar = energy * n[:2] + g[:2]
    q = (energy - kpar @ v[:2]) / v[2]
    k = np.r_[kpar, q]
    from pyrite.materials.crystal import ALPHA_FS

    own = 4j * np.pi * np.sqrt(ALPHA_FS) * (energy * v - k) / (abs(v[2]) * (k @ k - energy**2))
    d = 80.0 / HBARC_EV_ANG
    mismatch = q - g[2] - energy * n[2]
    integral = (
        d * np.exp(1j * (energy * n[2] + mismatch / 2) * d) * np.sinc(mismatch * d / (2 * np.pi))
    )
    expected = 0.5j * energy / n[2] * config["chi_transfer"] * integral * (own - n * (n @ own))
    np.testing.assert_allclose(_pxr(config, passage), expected, rtol=1e-12, atol=1e-17)


def test_zero_thickness_limit_and_absorbing_stack_stay_finite():
    config = _base(photon_energy_eV=520.0)
    for d in (1e-7, 1e6):
        config["layers"] = [(0.98 + 0.03j, d)]
        field = dtr_crossing_amplitudes(**config, **_passage(d))
        assert np.all(np.isfinite(field))
        if d < 1:
            assert np.linalg.norm(field.sum(axis=-2)) < 1e-12


def test_exact_diffraction_matching_is_a_finite_continuous_film_integral():
    energy, eps = 520.0, 0.98
    config = _base(photon_energy_eV=energy, layers=[(eps, 80.0)])
    k_out = energy * config["direction"][0]
    gx = 0.02
    lam_in = np.sqrt(eps * energy**2 - (k_out + gx * HBARC_EV_ANG) ** 2)
    lam_out = np.sqrt(eps * energy**2 - k_out**2)
    gz = (lam_in - lam_out) / HBARC_EV_ANG
    config["reciprocal_vector_ang"] = [gx, 0.0, gz]
    matched = dtr_crossing_amplitudes(**config, **_passage())
    assert np.all(np.isfinite(matched)) and np.linalg.norm(matched) > 0
    nearby = dtr_crossing_amplitudes(
        **(config | {"reciprocal_vector_ang": [gx, 0.0, gz + 1e-10]}), **_passage()
    )
    np.testing.assert_allclose(matched, nearby, rtol=1e-7, atol=1e-15)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"direction": [0.0, 0.0, -1.0]}, "n_z"),
        ({"direction": [0.0, 0.0, 2.0]}, "unit"),
        ({"crystal_layer": -1}, "crystal_layer"),
        ({"layers": [(0.98, -1.0)]}, "thickness"),
        ({"reciprocal_vector_ang": [0, 0, 0]}, "nonzero"),
    ],
)
def test_invalid_geometry_refuses(changes, message):
    with pytest.raises(ValueError, match=message):
        dtr_crossing_amplitudes(**_base(**changes), **_passage())
