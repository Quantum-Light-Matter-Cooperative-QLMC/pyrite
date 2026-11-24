"""Lattice-coherent share of the BremsLib continuum (#1250).

Validation: brem-lattice-coherent-fraction
"""

import numpy as np
import pytest

from pyrite._bremslib_table import solid_angle_integral
from pyrite.montecarlo.spectrum.brem_lattice_share import (
    LatticeShareKernel,
    kernel_x,
    lattice_share_density,
    lattice_share_table,
    reflection_shell_weights,
    shell_kernel_x,
)

X_EDGES = np.linspace(0.0, 1.05, 106)
N_SI = 0.049942993373787194


@pytest.fixture(scope="module")
def kernel():
    """A 2x2x2-node kernel around Si (111)/(220) at 20-45 keV, 30-60 degrees."""
    g = np.array([1.5, 3.4])
    T = np.array([20.0, 45.0])
    th = np.array([30.0, 60.0])
    mass = np.array(
        [
            [
                [shell_kernel_x(gi, Ti, np.deg2rad(t), X_EDGES, n_mu=200, n_phi=12) for t in th]
                for Ti in T
            ]
            for gi in g
        ]
    )
    return LatticeShareKernel(g, T, th, X_EDGES, mass, 200, 12)


def _si_low_order():
    hkls = [(1, 1, 1), (-1, -1, -1), (2, 2, 0), (-2, -2, 0), (3, 1, 1)]
    return reflection_shell_weights("silicon", hkls, 0.46, [("Si", N_SI)])


def test_resonance_variable_is_g_dot_v_for_the_classical_root():
    # Classical root: hbar omega = hbar c g beta mu / (1 - beta cos theta).
    T, th, g, mu = 30.0, np.deg2rad(45.0), 2.0, 0.37
    from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV

    gamma = 1 + T * 1e3 / M_E_EV
    beta = np.sqrt(1 - 1 / gamma**2)
    E = HBARC_EV_ANG * g * beta * mu / (1 - beta * np.cos(th))
    assert kernel_x(E, g, T, th) == pytest.approx(mu, rel=1e-12)


def test_shell_kernel_lives_on_the_physical_x_range():
    mass = shell_kernel_x(2.2, 30.0, np.deg2rad(45.0), X_EDGES, n_mu=200, n_phi=12)
    assert mass.sum() > 0
    assert np.all(mass >= 0)
    # x = mu <= 1 classically; recoil only lowers the photon energy.
    assert mass[X_EDGES[1:] > 1.0 + 1e-9].sum() == 0.0


def test_large_debye_waller_and_empty_sets_remove_nothing(kernel):
    g, w = reflection_shell_weights("silicon", [(1, 1, 1)], 1e4, [("Si", N_SI)])
    assert w["Si"].max() < 1e-30  # B -> infinity: e^{-2W} -> 0
    S = lattice_share_density(kernel, g, w["Si"], 30.0, np.linspace(500.0, 9000.0, 30))
    assert np.abs(S).max() < 1e-30
    empty = lattice_share_density(kernel, np.array([]), np.array([]), 30.0, [1000.0])
    assert not empty.any()  # G -> 0: no reflections, no share


def test_share_is_additive_over_reflections(kernel):
    g, w = _si_low_order()
    k = np.linspace(300.0, 20000.0, 40)
    whole = lattice_share_density(kernel, g, w["Si"], 30.0, k)
    parts = sum(
        lattice_share_density(kernel, g[i : i + 1], w["Si"][i : i + 1], 30.0, k)
        for i in range(g.size)
    )
    np.testing.assert_allclose(whole, parts, rtol=1e-12, atol=0)


def test_kernel_nodes_reproduce_the_direct_shell(kernel):
    # At a node (g, T, theta), the interpolated density is the direct kernel's.
    g0, T0, th0 = 1.5, 20.0, 30.0
    mass = shell_kernel_x(g0, T0, np.deg2rad(th0), X_EDGES, n_mu=200, n_phi=12)
    from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV

    beta = np.sqrt(1 - 1 / (1 + T0 * 1e3 / M_E_EV) ** 2)
    scale = HBARC_EV_ANG * g0 * beta / (1 - beta * np.cos(np.deg2rad(th0)))
    centres = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])
    k = centres[10:90] * scale
    S = lattice_share_density(kernel, np.array([g0]), np.array([1.0]), T0, k)[0]
    expected = mass[10:90] / np.diff(X_EDGES)[10:90] / scale
    np.testing.assert_allclose(S, expected, rtol=1e-10)


def test_kernel_rows_interpolate_alone():
    # The share interpolates only the |g| nodes its shells bracket (#1281);
    # each row must equal the full interpolation's, zero-mass rows included.
    from pyrite.montecarlo.spectrum.brem_lattice_share import _kernel_at_T

    rng = np.random.default_rng(1281)
    mass = rng.random((5, 3, 2, 7))
    mass[1, 0] = 0.0  # empty at the lower T node
    mass[3, 2, 1] = 0.0  # empty at the upper T node
    kernel = LatticeShareKernel(
        np.geomspace(1.0, 9.0, 5),
        np.array([10.0, 30.0, 90.0]),
        np.array([30.0, 60.0]),
        np.linspace(0.0, 1.05, 8),
        mass,
        1,
        1,
    )
    rows = np.array([1, 3, 4])
    for T in (10.0, 17.0, 30.0, 61.0, 90.0):
        np.testing.assert_array_equal(_kernel_at_T(kernel, T, rows), _kernel_at_T(kernel, T)[rows])
    assert _kernel_at_T(kernel, 5.0, rows) is None


def test_out_of_range_energy_is_left_alone(kernel):
    g, w = _si_low_order()
    assert not lattice_share_density(kernel, g, w["Si"], 5.0, [1000.0]).any()
    assert not lattice_share_density(kernel, g, w["Si"], 100.0, [1000.0]).any()


def test_table_loses_exactly_k_S_over_n_Z2_and_stays_self_consistent(kernel):
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    table = load_bremsstrahlung_tables(["Si"])["Si"]
    g, w = _si_low_order()
    corrected, info = lattice_share_table(table, kernel, "Si", N_SI, g, w["Si"])
    assert info.clamped_nodes == 0
    assert 0.0 < info.removed_fraction < 0.2
    # Keys name the BremsLib parent (the coupled partition check compares them).
    assert (corrected.key, corrected.digest) == (table.key, table.digest)
    T = np.asarray(table.incident_energy_keV)
    i = int(np.argmin(np.abs(T - 30.0)))
    kap = np.array(table.nominal_reduced_energy, dtype=float)
    kap[-1] = float(table.top_reduced_energy[i])
    k = kap * T[i] * 1e3
    S = lattice_share_density(kernel, g, w["Si"], T[i], k)
    j = int(np.argmin(np.abs(np.rad2deg(table.theta_rad) - 45.0)))
    assert np.rad2deg(table.theta_rad[j]) == pytest.approx(45.0, abs=1e-9)
    # Kernel theta nodes are 30 and 60 degrees; the table interpolates between them.
    delta = 1e11 * k * 0.5 * (S[0] + S[1]) / (N_SI * 14.0**2)
    removed = table.scaled_ddcs_mb_sr[i, :, j] - corrected.scaled_ddcs_mb_sr[i, :, j]
    np.testing.assert_allclose(removed, delta, rtol=1e-5, atol=1e-12 * delta.max())
    # SDCS remains the linear-in-theta solid-angle integral of the DDCS.
    np.testing.assert_allclose(
        solid_angle_integral(corrected.theta_rad, corrected.scaled_ddcs_mb_sr)[i],
        corrected.scaled_sdcs_mb[i],
        rtol=1e-5,
    )
    # Energies outside the kernel keep the parent table bit for bit.
    outside = (T < 20.0) | (T > 45.0)
    np.testing.assert_array_equal(
        corrected.scaled_ddcs_mb_sr[outside], table.scaled_ddcs_mb_sr[outside]
    )


def test_no_share_is_removed_below_the_line_electron_cutoff(kernel):
    # The line route scores no CBS below E_cut_lines; those nodes keep the full DDCS.
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    table = load_bremsstrahlung_tables(["Si"])["Si"]
    g, w = _si_low_order()
    full, _ = lattice_share_table(table, kernel, "Si", N_SI, g, w["Si"])
    floored, _ = lattice_share_table(table, kernel, "Si", N_SI, g, w["Si"], T_floor_keV=30.0)
    T = np.asarray(table.incident_energy_keV)
    below, above = T < 30.0, T >= 30.0
    np.testing.assert_array_equal(floored.scaled_ddcs_mb_sr[below], table.scaled_ddcs_mb_sr[below])
    np.testing.assert_array_equal(floored.scaled_ddcs_mb_sr[above], full.scaled_ddcs_mb_sr[above])
    assert (
        full.scaled_ddcs_mb_sr[below & (T >= 20.0)] < table.scaled_ddcs_mb_sr[below & (T >= 20.0)]
    ).any()


#: Released-kernel scene anchor (#1250 P5d): Si, B = 0.46 A^2, every
#: reflection with |g| <= 30 1/A (13,688), T = 30 keV, theta = 45 deg, 0.5-10 keV
#: in 100 eV cells. ``S_P3`` is the independent P3 direct quadrature
#: (pyrite-output/validation/issue-1250/2026-11-17-p3/share-tables.npz).
#: ``DROP_OVER_BREM`` re-pinned for the BremsLib k/T1 sub-node scheme (#1349).
SCENE_S_PER_ANG_SR = 3.6008354848850275e-10
SCENE_S_P3_PER_ANG_SR = 3.590115293509795e-10
SCENE_DROP_OVER_BREM = 0.12968644980404118


def test_released_kernel_pins_the_si_30kev_g30_share():
    """One scene value against the released ``brem-lattice-share`` dataset.

    The released kernel's share is within 0.3 % of the independent P3
    quadrature; the production scorer with the corrected table removes 0.99 S
    (BremsLib k-node sampling) and 13.0 % of this segment's in-band brem.
    """
    from pyrite.datasets import require_verified
    from pyrite.materials.crystal import dominant_reflections
    from pyrite.montecarlo import mc_brem_spectrum
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    released = LatticeShareKernel.load(require_verified("brem-lattice-share"))
    hkls = dominant_reflections("silicon", n_families=10**9, B_ang2=0.46, g_max_invang=30.0)
    assert len(hkls) == 13688
    g, w = reflection_shell_weights("silicon", hkls, 0.46, [("Si", N_SI)])
    k = np.arange(550.0, 10000.0, 100.0)
    theta = int(np.argmin(np.abs(released.theta_deg - 45.0)))
    S = lattice_share_density(released, g, w["Si"], 30.0, k)[theta].sum() * 100.0
    assert S == pytest.approx(SCENE_S_PER_ANG_SR, rel=1e-6)
    assert S == pytest.approx(SCENE_S_P3_PER_ANG_SR, rel=5e-3)

    table = load_bremsstrahlung_tables(["Si"])["Si"]
    corrected, _ = lattice_share_table(table, released, "Si", N_SI, g, w["Si"], T_floor_keV=20.0)
    v = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    segment = dict(
        r_mid=np.array([[0.0, 0.0, 0.5]]),
        v_hat=v[None],
        L_ang=np.array([1.0]),
        E_keV=np.array([30.0]),
        t_ang=np.zeros(1),
        t0_ang=np.zeros(1),
        elec_id=np.zeros(1, dtype=int),
        layer=np.zeros(1, dtype=int),
        Ne=1,
        thickness_ang=1.0,
        crystal_width_ang=None,
        crystal_height_ang=None,
        n_backscattered=0,
        n_missed=0,
    )

    def brem(tab):
        return np.asarray(
            mc_brem_spectrum(
                segment,
                k,
                element="Si",
                n_atoms_per_ang3=N_SI,
                n_hat=np.array([0.0, 0.0, 1.0]),
                cross_section_model="bremslib",
                bremslib_tables={"Si": tab},
            )
        )

    full, reduced = brem(table), brem(corrected)
    drop = (full - reduced).sum()
    assert drop / full.sum() == pytest.approx(SCENE_DROP_OVER_BREM, rel=1e-6)
    assert 0.98 < drop * 100.0 / S < 1.0


@pytest.mark.slow  # ~6 s: two full-table G30 share builds
def test_isotropic_share_is_the_sdcs_the_bremslib_route_removes():
    """#1261 acceptance: the theta-integrated EEDL correction on the pinned Si G30 scene.

    The BremsLib table's theta nodes contain the kernel's, so its removed SDCS
    (``Z^2/k`` x scaled SDCS, in mb) equals the isotropic share per atom to
    rounding; T between nodes adds only the ln T interpolation error.
    """
    from pyrite.datasets import require_verified
    from pyrite.materials.crystal import dominant_reflections
    from pyrite.montecarlo.spectrum.brem_lattice_share import (
        IsotropicLatticeShare,
        isotropic_share_sigma,
    )
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    released = LatticeShareKernel.load(require_verified("brem-lattice-share"))
    hkls = dominant_reflections("silicon", n_families=10**9, B_ang2=0.46, g_max_invang=30.0)
    g, w = reflection_shell_weights("silicon", hkls, 0.46, [("Si", N_SI)])
    table = load_bremsstrahlung_tables(["Si"])["Si"]
    corrected, info = lattice_share_table(table, released, "Si", N_SI, g, w["Si"])
    assert info.clamped_nodes == 0
    T = np.asarray(table.incident_energy_keV)
    for T_target in (30.0, 100.0, 300.0):
        i = int(np.argmin(np.abs(T - T_target)))
        kap = np.array(table.nominal_reduced_energy, dtype=float)
        kap[-1] = float(table.top_reduced_energy[i])
        k = kap * T[i] * 1e3
        removed = np.asarray(table.scaled_sdcs_mb[i]) - np.asarray(corrected.scaled_sdcs_mb[i])
        removed_cm2 = removed * 14.0**2 / np.where(k > 0, k, 1.0) * 1e-27
        iso = isotropic_share_sigma(released, g, w["Si"], N_SI, T[i], k)
        band = (k > 500.0) & (k < 10000.0)
        assert iso[band].min() > 0
        np.testing.assert_allclose(removed_cm2[band], iso[band], rtol=1e-9)

    # Between T nodes: within the kernel's ln T interpolation tolerance.
    # A 200 keV floor keeps the table to the nodes this check reads.
    share = IsotropicLatticeShare(released, g, w["Si"], N_SI, T_floor_keV=200.0)
    k = np.arange(550.0, 10000.0, 100.0)
    for T_off in (250.0, 610.0):
        direct = isotropic_share_sigma(released, g, w["Si"], N_SI, T_off, k)
        tabled = share.sigma(np.array([T_off]), k)[0]
        assert abs(tabled.sum() / direct.sum() - 1.0) < 1e-3
        assert np.abs(tabled - direct).max() < 3e-3 * direct.max()


def test_isotropic_share_limits_and_floor(kernel):
    from pyrite.montecarlo.spectrum.brem_lattice_share import IsotropicLatticeShare

    k = np.linspace(500.0, 9000.0, 30)
    g, w = reflection_shell_weights("silicon", [(1, 1, 1)], 1e4, [("Si", N_SI)])
    assert np.abs(IsotropicLatticeShare(kernel, g, w["Si"], N_SI).sigma([30.0], k)).max() < 1e-40
    empty = IsotropicLatticeShare(kernel, np.array([]), np.array([]), N_SI)
    assert not empty.sigma([30.0], k).any()  # G -> 0
    g, w = _si_low_order()
    share = IsotropicLatticeShare(kernel, g, w["Si"], N_SI, T_floor_keV=30.0)
    out = share.sigma(np.array([25.0, 30.0, 40.0, 50.0]), k)
    assert not out[0].any() and not out[3].any()  # below the floor, above the kernel
    assert out[1].sum() > 0 and out[2].sum() > 0


def test_eedl_spectrum_loses_exactly_the_isotropic_share(kernel):
    from pyrite.montecarlo import mc_brem_spectrum
    from pyrite.montecarlo.spectrum.brem import _bremsstrahlung_dsigma_dk
    from pyrite.montecarlo.spectrum.brem_lattice_share import IsotropicLatticeShare

    g, w = _si_low_order()
    share = IsotropicLatticeShare(kernel, g, w["Si"], N_SI)
    k = np.arange(550.0, 10000.0, 100.0)
    v = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    segment = dict(
        r_mid=np.array([[0.0, 0.0, 0.5]]),
        v_hat=v[None],
        L_ang=np.array([1.0]),
        E_keV=np.array([30.0]),
        t_ang=np.zeros(1),
        t0_ang=np.zeros(1),
        elec_id=np.zeros(1, dtype=int),
        layer=np.zeros(1, dtype=int),
        Ne=1,
        thickness_ang=1.0,
        crystal_width_ang=None,
        crystal_height_ang=None,
        n_backscattered=0,
        n_missed=0,
    )

    def brem(**extra):
        return np.asarray(
            mc_brem_spectrum(
                segment,
                k,
                element="Si",
                n_atoms_per_ang3=N_SI,
                n_hat=np.array([0.0, 0.0, 1.0]),
                cross_section_model="eedl",
                **extra,
            )
        )

    full = brem()
    reduced = brem(isotropic_lattice_share={"Si": share})
    sigma = np.asarray(
        _bremsstrahlung_dsigma_dk("Si", np.array([30.0]), k, cross_section_model="eedl")
    )[0]
    expected = full * (1.0 - share.sigma(np.array([30.0]), k)[0] / sigma)
    np.testing.assert_allclose(reduced, expected, rtol=1e-9)
    assert 0.0 < 1.0 - reduced.sum() / full.sum() < 0.2
