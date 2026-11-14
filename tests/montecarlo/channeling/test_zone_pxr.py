"""Zone-row PXR + DCR source kernel (#1166).

Synthetic Laue geometry: beam along z, plane normal x, entrance face normal z,
photon at 45 deg in the x-z plane. Reflection ``g = (-a, 0, b)`` puts the
Bragg pole about 40 eV (``a = 3``) or 1.3 keV (``a = 2``) above the free root.

Validation: cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.materials.recoil import recoil_resonance_eV
from pyrite.montecarlo.channeling import zone_pxr as zp

KINETIC_EV = 17e6
GAMMA = 1.0 + KINETIC_EV / M_E_EV
BETA = float(np.sqrt(1.0 - 1.0 / GAMMA**2))
PERIOD = 1.920113109
LENGTH = 20000.0
GEO = zp.SlabGeometry(
    beam=np.array([0.0, 0.0, 1.0]),
    x_hat=np.array([1.0, 0.0, 0.0]),
    s_hat=np.array([0.0, 0.0, 1.0]),
    path_ang=LENGTH,
)
N_HAT = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
CHI_G = 4e-6


def _g(a):
    return np.array([-a, 0.0, 1.19])


def _root(g):
    return float(recoil_resonance_eV(KINETIC_EV, GEO.beam, g, N_HAT))


def _pole_offset(g, E0):
    return -HBARC_EV_ANG * (g @ g) / (2.0 * N_HAT @ g) - E0


def test_two_beam_profile_solves_the_mode_equations():
    g = _g(3.0)
    E0 = _root(g)
    chi_g, chi_0 = 1e-4, complex(-1e-5, 2e-7)  # strong coupling, sensitive to branches
    E = E0 + np.array([-30.0, 0.0, 25.0, 60.0])
    mode = zp.photon_mode(E, N_HAT, g, GEO, BETA, chi_g, chi_0)
    z = np.linspace(0.0, LENGTH, 7)
    got = mode.profile(z, 1.0, LENGTH)
    k = E / HBARC_EV_ANG
    Q = k[:, None] * N_HAT + g
    Qs, ns = Q[:, 2], N_HAT[2]
    q = -(g @ g + 2 * k * (N_HAT @ g)) / (2 * Qs)
    es = np.cross(N_HAT, g) / np.linalg.norm(np.cross(N_HAT, g))
    for s, e in enumerate((es, np.cross(es, N_HAT))):
        for j in range(E.size):
            Qh = Q[j] / np.linalg.norm(Q[j])
            C = np.linalg.norm(e - Qh * (Qh @ e))
            a = k[j] ** 2 * chi_g * C / (2 * k[j] * ns)
            b = k[j] ** 2 * chi_g * C / (2 * Qs[j])
            N = np.array(
                [
                    [k[j] * chi_0 / (2 * ns), a],
                    [b, q[j] + k[j] ** 2 * chi_0 / (2 * Qs[j])],
                ]
            )
            for iz, zz in enumerate(z):
                Dh = (expm(1j * N * (LENGTH - zz)) @ np.array([1.0, 0.0]))[1]
                assert got[s, iz, j] == pytest.approx(Dh * -q[j] / b, rel=1e-8, abs=1e-12)


def test_kernels_integrate_the_profile_and_continue_it_upstream():
    g = _g(3.0)
    E0 = _root(g)
    E = E0 + np.array([-5.0, 3.0, 40.0])
    mode = zp.photon_mode(E, N_HAT, g, GEO, BETA, CHI_G, complex(-1e-5, 1.5e-7))
    D = 1.0 - BETA * N_HAT[2]
    delta = D * (E - (E0 + 7.0)) / (HBARC_EV_ANG * BETA)  # a transverse transition 7 eV up
    delta_pre = D * (E - E0) / (HBARC_EV_ANG * BETA)
    K, K_pre = mode.kernels(delta, delta_pre, 1.0, LENGTH)
    z = np.linspace(0.0, LENGTH, 400001)
    det = g @ g + 2 * E / HBARC_EV_ANG * (N_HAT @ g)
    P = mode.profile(z, 1.0, LENGTH)
    numeric = np.trapezoid(np.exp(1j * delta[None, None, :] * z[None, :, None]) * P, z, axis=1)
    np.testing.assert_allclose(K, numeric / det, rtol=1e-6)
    upstream = P[:, 0, :] / (1j * (delta_pre - mode.q)) / det
    np.testing.assert_allclose(K_pre, upstream, rtol=1e-12)


def test_kinematic_elastic_pair_is_the_first_order_born_line():
    g = _g(3.0)
    E0 = _root(g)
    E = E0 + np.linspace(-50.0, 50.0, 11)
    mode = zp.photon_mode(E, N_HAT, g, GEO, BETA, CHI_G, 0.0, dynamical=False)
    D = 1.0 - BETA * N_HAT[2]
    delta = D * (E - E0) / (HBARC_EV_ANG * BETA)
    K, K_pre = mode.kernels(delta, delta, 1.0, LENGTH)
    det = g @ g + 2 * E / HBARC_EV_ANG * (N_HAT @ g)
    qs = mode.q
    born = zp.phi(delta, LENGTH) * (-qs) / ((delta - qs) * det)
    np.testing.assert_allclose(K + K_pre, np.broadcast_to(born, K.shape), rtol=1e-9)


@pytest.mark.parametrize("dynamical", [False, True])
def test_free_electron_far_from_a_pole_reproduces_the_native_band_yield(dynamical):
    g = _g(2.0)
    E0 = _root(g)
    assert _pole_offset(g, E0) > 1000.0
    grid = E0 + np.arange(-400.0, 400.0, 0.05)
    states = zp.free_states(np.arange(-8, 9), PERIOD, GAMMA)
    spec = zp.reflection_spectrum(states, GEO, g, N_HAT, E0, grid, CHI_G, 0.0, dynamical=dynamical)
    native = zp.plane_wave_band_yield(GEO, g, N_HAT, E0, GAMMA, CHI_G)
    assert np.trapezoid(spec, grid) == pytest.approx(native, rel=3e-3)
    # With the potential coupling, the free line carries native's PXR + CBS interference.
    u_g = -8.0 * CHI_G  # Si: same structure-factor phase as chi_g, opposite sign, ~8x
    spec = zp.reflection_spectrum(
        states, GEO, g, N_HAT, E0, grid, CHI_G, 0.0, u_g=u_g, dynamical=dynamical
    )
    native = zp.plane_wave_band_yield(GEO, g, N_HAT, E0, GAMMA, CHI_G, u_g=u_g)
    assert native != pytest.approx(
        zp.plane_wave_band_yield(GEO, g, N_HAT, E0, GAMMA, CHI_G), rel=0.05
    )
    assert np.trapezoid(spec, grid) == pytest.approx(native, rel=3e-3)


def test_entrance_vertex_closes_over_bound_states_off_the_plane():
    """With transverse energies flattened, any basis must return the free Born result.

    Holds only if the pre-crystal term sits on the ``exp(-i Q_x x)`` vertex of
    every pair with the entrance velocity, here at ``Q_x != 0`` with a deep
    cosine potential (bound states) and a tilted entrance wave ``kappa``.
    """
    g = _g(3.0)
    E0 = _root(g)
    orders = np.arange(-10, 11)
    G = 2 * np.pi / PERIOD
    kappa = 0.3
    C = HBARC_EV_ANG**2 / (2 * GAMMA * M_E_EV)
    potential = -10.0 * (np.eye(orders.size, k=1) + np.eye(orders.size, k=-1)).astype(complex)
    energies, vectors = np.linalg.eigh(potential + np.diag(C * (kappa + orders * G) ** 2))
    states = zp.ZoneStates(
        potential[None], energies[None], vectors[None], np.ones((1, orders.size)),
        orders, np.array([kappa]), PERIOD, GAMMA,
    )  # fmt: skip
    assert np.count_nonzero(energies < 0) >= 2  # bound below the barrier top
    E = E0 + np.linspace(-20.0, 20.0, 9)
    mode = zp.photon_mode(E, N_HAT, g, GEO, BETA, CHI_G, 0.0, qx_at_eV=E0)
    assert abs(mode.Qx) > 0.1
    A, B, _ = zp.fibre_elements(states, 0, mode.Qx)
    D = 1.0 - BETA * N_HAT[2]
    delta = D * (E - E0) / (HBARC_EV_ANG * BETA) - 2 * C * kappa * mode.Qx / (HBARC_EV_ANG * BETA)
    K, K_pre = mode.kernels(delta, delta, 1.0, LENGTH)
    c = vectors[orders == 0][0].conj()  # <i|kappa>
    pre = mode.entrance_vertex(kappa, GAMMA)
    amp = (
        mode.alpha[:, None, None, :] * A[None, :, :, None]
        + mode.bx[:, None, None, :] * B[None, :, :, None]
    ) * K[:, None, None, :] + (pre[:, None, None, :] * A[None, :, :, None]) * K_pre[
        :, None, None, :
    ]
    coherent = np.einsum("sfie,i->sfe", amp, c)
    got = np.sum(np.abs(coherent) ** 2, axis=(0, 1))
    want = np.sum(np.abs(pre * (K + K_pre)) ** 2, axis=0)
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_bin_mean_output_conserves_the_band_and_node_output_samples_it():
    g = _g(3.0)
    E0 = _root(g)
    states = zp.free_states(np.arange(-8, 9), PERIOD, GAMMA)
    chi_0 = complex(-1e-5, 1.5e-7)
    fine = E0 + np.arange(-400.0, 400.0, 0.05)
    band = np.trapezoid(zp.reflection_spectrum(states, GEO, g, N_HAT, E0, fine, CHI_G, chi_0), fine)
    grid = E0 - 403.0 + 7.0 * np.arange(116)  # coarse bins straddling the line
    binned = zp.binned_reflection_spectrum(states, GEO, g, N_HAT, E0, grid, CHI_G, chi_0)
    assert binned.sum() * 7.0 == pytest.approx(band, rel=1e-3)
    nodes = zp.binned_reflection_spectrum(
        states, GEO, g, N_HAT, E0, fine[::40], CHI_G, chi_0, quadrature="node"
    )
    exact = zp.reflection_spectrum(states, GEO, g, N_HAT, E0, fine[::40], CHI_G, chi_0)
    np.testing.assert_allclose(nodes, exact, rtol=1e-12, atol=1e-5 * exact.max())


@pytest.mark.parametrize("order", [2, 4])
def test_entrance_beyond_half_g_uses_the_true_entrance_momentum(order):
    """A plane wave entering at ``kappa + nu G`` (tilted or divergent beam) near a mirror pole.

    The pre-crystal term must take the true ``p``, not ``kappa`` folded into
    ``[-G/2, G/2)``: the band must equal the free Born line at that ``p``
    (fresh-context verifier, #1166; the folded momentum was 7-17% off).
    """
    from dataclasses import replace

    G = 2 * np.pi / PERIOD
    C = HBARC_EV_ANG**2 / (2 * GAMMA * M_E_EV)
    D = 1.0 - BETA * N_HAT[2]
    scan = np.linspace(2.5, 2.95, 451)
    offsets = np.array([_pole_offset(_g(a), _root(_g(a))) for a in scan])
    g = _g(scan[np.argmin(offsets)])  # closest approach, ~12 eV below the pole
    E0 = _root(g)
    assert 5.0 < _pole_offset(g, E0) < 20.0
    p = 0.3 + order * G
    states = zp.free_states(np.arange(-8, 9), PERIOD, GAMMA, kappa=p)
    entered = replace(states, entrance=states.populations)  # the m = 0 wave at kappa = p
    centre = zp.photon_mode([E0], N_HAT, g, GEO, BETA, 0.0, 0.0, False, qx_at_eV=E0)
    Ec = E0 + 2 * C * p * centre.Qx / D
    E = Ec + np.arange(-300.0, 300.0, 0.01)
    mode = zp.photon_mode(E, N_HAT, g, GEO, BETA, 0.0, 0.0, False, qx_at_eV=E0)
    delta = D * (E - Ec) / (HBARC_EV_ANG * BETA)
    det = g @ g + 2 * E / HBARC_EV_ANG * (N_HAT @ g)
    born = zp.phi(delta, LENGTH) * (-mode.q) / ((delta - mode.q) * det)
    pref = zp.ALPHA_FS * (E / HBARC_EV_ANG) / (4 * np.pi**2 * HBARC_EV_ANG * BETA**2)
    want = np.trapezoid(
        pref * np.sum(np.abs(CHI_G * mode.entrance_vertex(p, GAMMA) * born) ** 2, axis=0), E
    )

    def band(s):
        return np.trapezoid(
            zp.reflection_spectrum(s, GEO, g, N_HAT, E0, E, CHI_G, dynamical=False), E
        )

    assert band(entered) == pytest.approx(want, rel=2e-3)
    assert abs(band(states) / want - 1.0) > 0.03  # the folded entrance is measurably wrong


def test_mixed_entry_returns_the_incoherent_free_born_sum():
    """Two entering orders: each state interferes only with the wave that fed it.

    Free plane waves at ``kappa`` and ``kappa + 2G`` entering with weights 0.7
    and 0.3 must give the weighted sum of their own Born lines (fresh-context
    verifier, #1166: entry-weight-only averaging was 3.9% off).
    """
    from dataclasses import replace

    G = 2 * np.pi / PERIOD
    C = HBARC_EV_ANG**2 / (2 * GAMMA * M_E_EV)
    D = 1.0 - BETA * N_HAT[2]
    scan = np.linspace(2.5, 2.95, 451)
    offsets = np.array([_pole_offset(_g(a), _root(_g(a))) for a in scan])
    g = _g(scan[np.argmin(offsets)])
    E0 = _root(g)
    orders = np.arange(-8, 9)
    weight = np.where(orders == 0, 0.7, 0.0) + np.where(orders == 2, 0.3, 0.0)
    states = replace(
        zp.free_states(orders, PERIOD, GAMMA, kappa=0.3),
        populations=weight[None],
        entrance=weight[None],
    )
    centre = zp.photon_mode([E0], N_HAT, g, GEO, BETA, 0.0, 0.0, False, qx_at_eV=E0)
    lo = E0 + 2 * C * 0.3 * centre.Qx / D
    hi = E0 + 2 * C * (0.3 + 2 * G) * centre.Qx / D
    E = np.arange(min(lo, hi) - 300.0, max(lo, hi) + 300.0, 0.01)
    mode = zp.photon_mode(E, N_HAT, g, GEO, BETA, 0.0, 0.0, False, qx_at_eV=E0)
    det = g @ g + 2 * E / HBARC_EV_ANG * (N_HAT @ g)
    pref = zp.ALPHA_FS * (E / HBARC_EV_ANG) / (4 * np.pi**2 * HBARC_EV_ANG * BETA**2)
    want = 0.0
    for p, w in ((0.3, 0.7), (0.3 + 2 * G, 0.3)):
        delta = D * (E - (E0 + 2 * C * p * centre.Qx / D)) / (HBARC_EV_ANG * BETA)
        born = zp.phi(delta, LENGTH) * (-mode.q) / ((delta - mode.q) * det)
        line = np.abs(CHI_G * mode.entrance_vertex(p, GAMMA) * born) ** 2
        want = want + w * pref * np.sum(line, axis=0)
    got = zp.reflection_spectrum(states, GEO, g, N_HAT, E0, E, CHI_G, dynamical=False)
    assert np.trapezoid(got, E) == pytest.approx(np.trapezoid(want, E), rel=2e-3)
