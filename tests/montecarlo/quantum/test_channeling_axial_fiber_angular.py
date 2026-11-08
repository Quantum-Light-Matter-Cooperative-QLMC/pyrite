"""Off-axis polarization source vs on-axis estimator, exact regression and Liénard–Wiechert."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_fiber_angular import (
    angular_coefficients,
    fiber_angular_source,
    integrate_solid_angle,
    polarization_spectrum,
    stokes,
)
from pyrite.montecarlo.channeling.dipole import dipole_far_field

from .test_channeling_axial_fiber_regression import (
    BETA,
    DEPTHS,
    GAMMA,
    HBAR_V,
    KINETIC,
    ORDERS,
    _channels,
    _source,
)

ENERGIES = np.linspace(0.05, 1.2, 18)
THETA = np.array([0.0, 0.5 / GAMMA, 1.0 / GAMMA, 2.0 / GAMMA])
PHI = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)


def _entry():
    entry = np.zeros(len(ORDERS), dtype=complex)
    entry[4] = 1.0  # order (0, 0)
    return entry


def _angular(channels, n, **extra):
    return fiber_angular_source(
        channels,
        _entry(),
        kinetic_energy_eV=KINETIC,
        hbar_v_eV_ang=HBAR_V,
        depths_ang=DEPTHS,
        photon_energy_eV=extra.pop("photon_energy_eV", ENERGIES),
        theta_rad=extra.pop("theta", THETA),
        phi_rad=extra.pop("phi", PHI),
        incident_ids=range(n),
        seed=11,
        search_step_ang=500.0,
        **extra,
    )


def test_on_axis_trace_and_draws_match_the_on_axis_source_with_jumps():
    channels = _channels(0.004, 0.006)
    on_axis = _source(channels, _entry(), 40)
    angular = _angular(channels, 40, batches=4)
    assert [e.kind for e in angular.events] == [e.kind for e in on_axis.events]
    assert [e.depth_ang for e in angular.events] == [e.depth_ang for e in on_axis.events]
    np.testing.assert_allclose(
        angular.correlation.values, on_axis.correlation.values, rtol=1e-10, atol=1e-16
    )
    intensity = angular.stokes[0, :, 0]
    for row in intensity:  # every azimuth on axis
        np.testing.assert_allclose(
            row, on_axis.spectrum, rtol=1e-9, atol=1e-9 * np.abs(on_axis.spectrum).max()
        )
    assert sum(e.kick_jumps + e.lattice_jumps for e in angular.events) > 0
    assert np.any(angular.stokes_sem[..., 0, :] > 0)


def test_map_is_an_independent_double_sum_and_c4_symmetric():
    """No jumps: S from an explicit loop over depth pairs; the square toy is C4 symmetric."""
    from pyrite.materials.crystal import ALPHA_FS

    channels = _channels(0.0, 0.0)
    source = _angular(channels, 3, batches=3)
    g = source.correlation.values
    spacing = np.diff(DEPTHS)
    w = np.r_[spacing[0], spacing[:-1] + spacing[1:], spacing[-1]] / 2
    for t, angle in enumerate(THETA):
        doppler = 1.0 - BETA * np.cos(angle)
        for p, phi in enumerate(PHI):
            n_hat = np.array([np.cos(phi), np.sin(phi)])
            m = np.stack([n_hat * (np.cos(angle) - BETA) / doppler, [-np.sin(phi), np.cos(phi)]])
            for k, e in enumerate(ENERGIES):
                v = w * np.exp(1j * e * doppler * DEPTHS / HBAR_V)
                amp = np.einsum("i,ijab,j->ab", v, g, v.conj())
                s_mat = m @ amp @ m.T * ALPHA_FS * e / (4 * np.pi**2 * HBAR_V**2)
                np.testing.assert_allclose(
                    source.polarization[t, p, :, :, k], s_mat, rtol=1e-8, atol=1e-14
                )
    i_map = source.stokes[:, :, 0]
    np.testing.assert_allclose(i_map, np.roll(i_map, 2, axis=1), rtol=1e-8, atol=1e-12)


def test_c4_symmetric_correlation_gives_azimuthally_uniform_intensity():
    """C4 (J_x -> J_y -> -J_x): G_xx = G_yy, G_yx = -G_xy with G_xy anti-Hermitian in depth."""
    rng = np.random.default_rng(3)
    z = np.linspace(0.0, 2000.0, 9)
    base = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    base = base @ base.conj().T
    herm = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    cross = 1j * (herm + herm.conj().T)
    g = np.zeros((9, 9, 2, 2), dtype=complex)
    g[..., 0, 0] = g[..., 1, 1] = base
    g[..., 0, 1], g[..., 1, 0] = cross, -cross
    i_map = stokes(polarization_spectrum(g, z, ENERGIES, KINETIC, THETA, PHI))[:, :, 0]
    np.testing.assert_allclose(i_map, i_map[:, :1].repeat(PHI.size, 1), rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("dipole", [(1.0, 0.0), (1.0, 1j), (0.3 - 0.2j, 0.9 + 0.1j)])
def test_resonant_amplitude_is_the_lienard_wiechert_field(dipole):
    d = np.asarray(dipole) / np.linalg.norm(dipole)
    theta = np.array([0.0, 0.3, 1.0, 1.7, 3.0]) / GAMMA
    phi = np.linspace(0.0, 2 * np.pi, 7, endpoint=False)
    for t in theta:
        doppler = 1.0 - BETA * np.cos(t)
        ours = angular_coefficients([t], phi, BETA)[0] @ d
        circ = dipole_far_field(t * np.cos(phi), t * np.sin(phi), GAMMA, d)
        fx, fy, fz = (
            (circ[0] + circ[1]) / np.sqrt(2),
            1j * (circ[0] - circ[1]) / np.sqrt(2),
            circ[2],
        )
        e1 = np.stack([np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), -np.sin(t) + 0 * phi])
        e2 = np.stack([-np.sin(phi), np.cos(phi), 0 * phi])
        field = np.stack([fx, fy, fz])
        projected = np.stack([np.sum(e1 * field, 0), np.sum(e2 * field, 0)], -1)
        scale = -np.sqrt(3 / (8 * np.pi)) / (GAMMA * doppler)
        np.testing.assert_allclose(projected, scale * ours, rtol=1e-10, atol=1e-13)


def test_single_line_resonance_is_the_lienard_wiechert_line():
    """At E (1 - beta cos theta) = gap the depth phase cancels exactly: S = alpha E L^2 A A* / (2 pi hbar v)^2."""
    from pyrite.materials.crystal import ALPHA_FS

    gap, length = 0.02, 4.0e5
    z = np.linspace(0.0, length, 201)
    d = np.array([1.0, 0.0])
    comp = d.astype(complex)
    phase = np.exp(-1j * gap * (z[:, None] - z[None, :]) / HBAR_V)
    values = phase[:, :, None, None] * comp[None, None, :, None] * comp[None, None, None, :].conj()
    theta = np.array([0.0, 0.6 / GAMMA, 1.0 / GAMMA, 2.5 / GAMMA])
    doppler = 1.0 - BETA * np.cos(theta)
    phi = np.array([0.0, np.pi / 2, 0.7])
    resonance = gap / doppler
    st = stokes(polarization_spectrum(values, z, resonance, KINETIC, theta, phi))
    diag = st[np.arange(theta.size), :, :, np.arange(theta.size)]  # (theta, phi, stokes)
    a1 = np.cos(phi)[None] * ((np.cos(theta) - BETA) / doppler)[:, None]
    a2 = -np.sin(phi)[None] + 0 * a1
    scale = ALPHA_FS * resonance[:, None] * length**2 / (2 * np.pi * HBAR_V) ** 2
    np.testing.assert_allclose(diag[..., 0], scale * (a1**2 + a2**2), rtol=1e-10)
    np.testing.assert_allclose(
        diag[..., 1], scale * (a1**2 - a2**2), rtol=1e-10, atol=1e-12 * scale.max()
    )
    np.testing.assert_allclose(
        diag[..., 2], 2 * scale * a1 * a2, rtol=1e-10, atol=1e-12 * scale.max()
    )
    np.testing.assert_allclose(diag[..., 3], 0.0, atol=1e-12 * scale.max())
    # long-slab line area ~ S(resonance) * width ~ S / (1 - beta cos theta): Lienard-Wiechert pattern
    area = diag[..., 0] / doppler[:, None]
    ff = np.sum(
        np.abs(
            dipole_far_field(theta[:, None] * np.cos(phi), theta[:, None] * np.sin(phi), GAMMA, d)
        )
        ** 2,
        axis=0,
    )
    np.testing.assert_allclose(area / area[0, 0], ff / ff[0, 0], rtol=1e-10)


def test_solid_angle_integral():
    theta = np.linspace(0.0, 0.3, 601)
    phi = np.linspace(0.0, 2 * np.pi, 6, endpoint=False) + 0.1
    values = (1.0 + np.cos(2 * phi)[None, :] + 0 * theta[:, None])[..., None]
    expected = 2 * np.pi * (1 - np.cos(0.3))
    np.testing.assert_allclose(integrate_solid_angle(values, theta, phi)[0], expected, rtol=1e-5)
    with pytest.raises(ValueError, match="uniform"):
        integrate_solid_angle(values[:, :5], theta, phi[:5] ** 1.1)


def test_refusals():
    channels = _channels(0.0, 0.0)
    with pytest.raises(ValueError, match="batches"):
        _angular(channels, 3, batches=1)
    with pytest.raises(ValueError, match="batches"):
        _angular(channels, 3, batches=4)
    with pytest.raises(ValueError, match="pi/2"):
        _angular(channels, 3, theta=[1.6])
    with pytest.raises(ValueError, match="nonnegative"):
        _angular(channels, 3, theta=[-0.1])
    z = DEPTHS
    with pytest.raises(ValueError, match="shape"):
        polarization_spectrum(np.zeros((z.size, z.size, 4, 4)), z, ENERGIES, KINETIC, [0.0], [0.0])
    with pytest.raises(ValueError, match="undersamples"):
        _angular(
            channels, 3, batches=3, theta=[3.0 / GAMMA], photon_energy_eV=np.array([0.5, 2000.0])
        )
