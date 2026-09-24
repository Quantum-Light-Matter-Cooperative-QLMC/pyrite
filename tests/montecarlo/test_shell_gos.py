"""PENELOPE-2024 shell GOS energy-loss moments for electrons (§§3.2.2–3.2.4)."""

import numpy as np
import pytest
from scipy.constants import c, e, m_e, physical_constants
from scipy.integrate import quad

from pyrite.materials.attenuation import plasma_energy_eV
from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import shell_gos as sg
from pyrite.montecarlo.transport import shell_oscillators as so
from pyrite.xsgen.sbethe.catalog import catalog_material, resolve_catalog_table

MC2 = m_e * c * c / e
RE_CM = 100.0 * physical_constants["classical electron radius"][0]
KEYS = ("silicon", "sio2", "mos2")


def _shell(z, designator, label, orbital, occupation, energy):
    return config.AtomicShell(z, designator, label, orbital, occupation, energy, 0.1, 0.0, 0.0)


def _silicon_fixture(resonance=16.7):
    shells = {
        14: (
            _shell(14, 1, "K", "1s1/2", 2, 1844.0),
            _shell(14, 2, "L1", "2s1/2", 2, 154.0),
            _shell(14, 3, "L2", "2p1/2", 2, 104.0),
            _shell(14, 4, "L3", "2p3/2", 4, 104.0),
            _shell(14, 5, "M1", "3s1/2", 2, 13.46),
            _shell(14, 6, "M2", "3p1/2", 2, 8.151),
        )
    }
    band = so.ConductionBand(4.0, resonance, "fixture", {14: 1.0})
    return so.build_shell_oscillators({14: 1.0}, 173.0, 31.05, shells, band)


def _one_shell(u=13.6, i=19.2, omega=0.3):
    shells = {1: (_shell(1, 1, "K", "1s1/2", 1, u),)}
    return so.build_shell_oscillators({1: 1.0}, i, omega, shells, default_threshold_eV=1.0)


def _catalog(key):
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    material = catalog_material(key)
    return so.build_shell_oscillators(
        material.composition,
        material.mean_excitation_eV,
        plasma_energy_eV(key),
        config.load_atomic_shells(),
        so.load_conduction_bands()[key],
    )


def _stp(key, energy):
    arrays = resolve_catalog_table(key).arrays()
    grid, stopping = arrays["stopping_energy_eV"], arrays["stopping_cs_eV_cm2"]
    return float(np.exp(np.interp(np.log(energy), np.log(grid), np.log(stopping))))


def _pref(energy):
    gamma = 1.0 + energy / MC2
    return 2.0 * np.pi * RE_CM**2 * MC2 / (1.0 - 1.0 / gamma**2)


@pytest.mark.parametrize("energy", [40.0, 200.0, 1e3, 2e3, 1e4, 1e5, 1e6, 1e8, 1e9])
def test_moments_are_finite_non_negative_and_partition(energy):
    material = _silicon_fixture()
    moments = sg.shell_gos_moments(material, energy)
    channels = (moments.distant_longitudinal, moments.distant_transverse, moments.close)
    for channel in channels:
        assert channel.shape == (len(material.oscillators), 3)
        assert np.all(np.isfinite(channel)) and np.all(channel >= 0.0)
    assert np.allclose(moments.per_shell, sum(channels), rtol=1e-15, atol=0.0)
    assert np.allclose(moments.total, moments.per_shell.sum(axis=0), rtol=1e-15, atol=0.0)
    assert np.all(moments.total > 0.0)
    number = sg.formula_units_per_angstrom3(material)
    assert np.allclose(sg.path_moments(moments, number), moments.total * number * 1e16, atol=0.0)


@pytest.mark.parametrize("energy", [2e3, 2e4, 3e5])
def test_close_moments_match_direct_moller_quadrature(energy):
    """Eqs. 3.106–3.110 against direct quadrature of Eq. 3.87 with E' = E + U.

    The close lower limit is Q_k = U_k for bound shells (Eq. 3.96) and W_cb for
    the conduction band.
    """
    material = _silicon_fixture()
    moments = sg.shell_gos_moments(material, energy)
    a = (energy / (energy + MC2)) ** 2
    for osc, close in zip(material.oscillators, moments.close, strict=True):
        u = osc.ionization_energy_eV
        if energy <= u:
            assert not np.any(close)
            continue
        prime, upper = energy + u, 0.5 * (energy + u)
        lower = u if u > 0.0 else osc.resonance_energy_eV

        def moller(w, prime=prime):
            r = w / (prime - w)
            return 1.0 + r * r - (1.0 - a) * r + a * (w / prime) ** 2

        for n in range(3):
            expected, _ = quad(
                lambda w, n=n: w ** (n - 2) * moller(w), lower, upper, epsrel=1e-12, limit=200
            )
            expected *= _pref(energy) * osc.strength
            assert close[n] == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_bound_shell_close_moments_start_at_binding_and_vanish_at_threshold():
    """Eq. 3.96: no close loss below U_k, so moments vanish continuously at E -> U_k."""
    material = _silicon_fixture()
    k_shell = next(i for i, o in enumerate(material.oscillators) if o.label == "K")
    u = material.oscillators[k_shell].ionization_energy_eV
    previous = None
    for excess in (1e-1, 1e-2, 1e-3):
        close = sg.shell_gos_moments(material, u * (1.0 + excess)).close[k_shell]
        assert np.all(close >= 0.0)
        if previous is not None:
            assert close[0] < 0.2 * previous
        previous = close[0]
    # Mean close loss lies above U_k: sigma1/sigma0 >= U_k.
    close = sg.shell_gos_moments(material, 1.2 * u).close[k_shell]
    assert close[1] / close[0] >= u


def test_full_triangle_mean_loss_is_the_resonance():
    """Eq. 3.77: an untruncated p_dis has <W> = W_k, so sigma_dis^(2) = W_k sigma_dis^(1)."""
    material = _silicon_fixture()
    moments = sg.shell_gos_moments(material, 1e6)
    distant = moments.distant_longitudinal + moments.distant_transverse
    for osc, row in zip(material.oscillators, distant, strict=True):
        assert row[2] == pytest.approx(osc.resonance_energy_eV * row[1], rel=1e-12, abs=0.0)
        lower, top = osc.ionization_energy_eV, 3.0 * osc.resonance_energy_eV
        if lower > 0.0:
            top -= 2.0 * lower
            inverse = 2.0 / (top - lower) ** 2 * (top * np.log(top / lower) - (top - lower))
            assert row[0] == pytest.approx(inverse * row[1], rel=1e-12, abs=0.0)
        else:
            assert row[0] == pytest.approx(row[1] / osc.resonance_energy_eV, rel=1e-12, abs=0.0)


def test_one_shell_distant_stopping_is_the_closed_form():
    """Single bound shell, W = I (Eqs. 3.60–3.61), E >> U: Eqs. 3.104–3.105 with <1/W>=1."""
    material = _one_shell()
    (osc,) = material.oscillators
    assert osc.resonance_energy_eV == pytest.approx(19.2, rel=1e-12, abs=0.0)
    energy = 5e4
    moments = sg.shell_gos_moments(material, energy)
    gamma = 1.0 + energy / MC2
    beta2 = 1.0 - 1.0 / gamma**2
    p0 = np.sqrt(energy * (energy + 2.0 * MC2))
    p1 = np.sqrt((energy - 19.2) * (energy - 19.2 + 2.0 * MC2))
    q_minus = np.sqrt((p0 - p1) ** 2 + MC2**2) - MC2
    longitudinal = np.log(13.6 * (q_minus + 2.0 * MC2) / (q_minus * (13.6 + 2.0 * MC2)))
    transverse = np.log(gamma**2) - beta2 - moments.density_effect
    expected = _pref(energy) * (longitudinal + transverse)
    distant = moments.distant_longitudinal[0, 1] + moments.distant_transverse[0, 1]
    assert distant == pytest.approx(expected, rel=1e-6, abs=0.0)
    assert moments.density_effect == 0.0


def test_shell_below_threshold_is_closed_and_threshold_modification_is_continuous():
    material = _silicon_fixture()
    k_shell = next(i for i, o in enumerate(material.oscillators) if o.label == "K")
    osc = material.oscillators[k_shell]
    below = sg.shell_gos_moments(material, osc.ionization_energy_eV)
    assert not np.any(below.per_shell[k_shell])
    edge = 3.0 * osc.resonance_energy_eV - 2.0 * osc.ionization_energy_eV
    lo = sg.shell_gos_moments(material, edge * (1.0 - 1e-9)).per_shell[k_shell]
    hi = sg.shell_gos_moments(material, edge * (1.0 + 1e-9)).per_shell[k_shell]
    assert np.allclose(lo, hi, rtol=1e-6, atol=0.0)


def test_conduction_band_resonance_and_cutoff():
    """Q_cb = W_cb and a delta resonance (Fig. 3.8b): no distant loss once W_cb >= E/2."""
    material = _silicon_fixture()
    cb = material.oscillators[0]
    assert cb.label == "cb" and cb.ionization_energy_eV == 0.0
    slow = sg.shell_gos_moments(material, 1.9 * cb.resonance_energy_eV)
    assert not np.any(slow.per_shell[0])
    moments = sg.shell_gos_moments(material, 1e4)
    assert moments.per_shell[0, 0] / moments.total[0] > 0.5


def test_conduction_resonance_moves_imfp_not_stopping():
    """Manual Fig. 3.11 discussion: W_cb strongly changes lambda_in; S_in is insensitive."""
    low, high = _silicon_fixture(14.0), _silicon_fixture(20.0)
    for energy in (1e4, 1e5):
        a, b = sg.shell_gos_moments(low, energy).total, sg.shell_gos_moments(high, energy).total
        assert abs(b[1] / a[1] - 1.0) < 0.01
        assert abs(b[0] / a[0] - 1.0) > 0.05


@pytest.mark.parametrize("energy", [1e6, 1e7, 1e8, 1e9])
def test_high_energy_stopping_is_the_bethe_formula(energy):
    """Eqs. 3.115–3.121; residual terms are O(U_k/E) and O(Q_-/Q_k)."""
    material = _silicon_fixture()
    raw = sg.shell_gos_moments(material, energy).total[1]
    assert raw == pytest.approx(sg.bethe_stopping_cs(material, energy), rel=2e-4, abs=0.0)


def test_density_effect_reaches_ultrarelativistic_limit():
    """Eq. 3.73: delta_F -> ln(Omega_p^2/((1 - beta^2) I^2)) - 1."""
    material = _silicon_fixture()
    energy = 1e10
    gamma = 1.0 + energy / MC2
    expected = np.log(31.05**2 * gamma**2 / 173.0**2) - 1.0
    assert sg.density_effect_correction(material, energy) == pytest.approx(expected, abs=1e-3)


def test_formula_density_follows_plasma_energy():
    """Eq. 3.51 inverts silicon's catalog plasma energy to its atom density."""
    material = _silicon_fixture()
    assert sg.formula_units_per_angstrom3(material) == pytest.approx(0.04994, rel=1e-3, abs=0.0)


@pytest.mark.parametrize("energy", [float("nan"), 0.0, -1.0])
def test_rejects_non_physical_energy(energy):
    with pytest.raises(ValueError, match="finite and positive"):
        sg.shell_gos_moments(_silicon_fixture(), energy)


@pytest.mark.parametrize("key", KEYS)
def test_fetched_raw_stopping_reaches_bethe_and_sbethe_at_high_energy(key):
    """Above 100 keV both GOS and SBETHE approach Bethe with the same I.

    Remaining difference is SBETHE's shell correction and its δ_F from a
    continuous OOS versus Eq. 3.72's oscillators (≤ 2% up to 10 MeV).
    """
    material = _catalog(key)
    for energy in (1e5, 1e6, 1e7, 1e8, 1e9):
        raw = sg.shell_gos_moments(material, energy).total[1]
        assert raw == pytest.approx(sg.bethe_stopping_cs(material, energy), rel=1e-3, abs=0.0)
        assert raw == pytest.approx(_stp(key, energy), rel=0.02, abs=0.0)
    raw = sg.shell_gos_moments(material, 1e5).total[1]
    assert raw == pytest.approx(_stp(key, 1e5), rel=0.006, abs=0.0)


@pytest.mark.xfail(
    strict=True,
    reason="raw δ-oscillator GOS exceeds shell-corrected stp.dat at 1 keV: Si 13%, SiO2 21%, MoS2 15%",
)
@pytest.mark.parametrize("key", KEYS)
def test_fetched_raw_stopping_matches_sbethe_below_10_kev(key):
    material = _catalog(key)
    for energy in (1e3, 2e3, 5e3, 1e4):
        raw = sg.shell_gos_moments(material, energy).total[1]
        assert raw == pytest.approx(_stp(key, energy), rel=0.02, abs=0.0)


@pytest.mark.parametrize(
    "key, ratios",
    [
        ("silicon", (1.1333, 1.0309, 1.0139, 1.0003)),
        ("sio2", (1.2142, 1.0635, 1.0320, 0.9996)),
        ("mos2", (1.1525, 1.1150, 1.0675, 1.0052)),
    ],
)
def test_fetched_raw_to_sbethe_stopping_ratios_are_recorded(key, ratios):
    """Pin the recorded 1, 5, 10, 100 keV GOS/stp.dat ratios (derivation doc table)."""
    material = _catalog(key)
    for energy, ratio in zip((1e3, 5e3, 1e4, 1e5), ratios, strict=True):
        raw = sg.shell_gos_moments(material, energy).total[1]
        assert raw / _stp(key, energy) == pytest.approx(ratio, abs=2e-4)


@pytest.mark.parametrize("energy", [2e5, 1e6, 1e8])
def test_single_oscillator_density_effect_matches_closed_form(energy):
    """One oscillator: L^2 = Omega^2/(1-beta^2) - W^2 solves Eq. 3.71 exactly."""
    omega, resonance = 30.0, 20.0
    material = so.MaterialShellOscillators(
        (so.ShellOscillator(0, "cb", 4.0, 0.0, resonance),),
        4.0,
        resonance,
        omega,
        1.0,
        "fixture",
        (),
    )
    gamma = 1.0 + energy / MC2
    one_minus_beta2 = 1.0 / (gamma * gamma)
    assert one_minus_beta2 < (omega / resonance) ** 2
    ratio = omega**2 / (one_minus_beta2 * resonance**2)
    expected = np.log(ratio) - 1.0 + 1.0 / ratio
    assert sg.density_effect_correction(material, energy) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )
