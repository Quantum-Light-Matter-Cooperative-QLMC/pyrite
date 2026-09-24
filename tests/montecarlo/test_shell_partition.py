"""Soft/hard partition of the closed PENELOPE-2024 shell GOS (Eqs. 3.124, 4.44–4.47)."""

import numpy as np
import pytest
from scipy.constants import c, e, m_e, physical_constants
from scipy.integrate import quad

from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import shell_gos as sg
from pyrite.montecarlo.transport import shell_oscillators as so
from pyrite.montecarlo.transport import shell_partition as sp
from pyrite.montecarlo.transport import shell_rates as sr
from pyrite.montecarlo.transport import shell_sampling as ss

MC2 = m_e * c * c / e
RE_CM = 100.0 * physical_constants["classical electron radius"][0]
KEYS = ("silicon", "sio2", "mos2")
CHANNELS = ("distant_longitudinal", "distant_transverse", "close")


def _shell(z, designator, label, orbital, occupation, energy):
    return config.AtomicShell(z, designator, label, orbital, occupation, energy, 0.1, 0.0, 0.0)


def _silicon_fixture():
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
    band = so.ConductionBand(4.0, 16.7, "fixture", {14: 1.0})
    return so.build_shell_oscillators({14: 1.0}, 173.0, 31.05, shells, band)


def _fixture_closure(energy, stopping_scale=1.05):
    """Fixture closure with K and L1 substituted at 0.8 times their GOS rate."""
    material = _silicon_fixture()
    raw = sg.shell_gos_moments(material, energy)
    inner = {
        (14, o.label): 0.8 * raw.per_shell[k, 0]
        for k, o in enumerate(material.oscillators)
        if o.label in ("K", "L1") and raw.per_shell[k, 0] > 0.0
    }
    stopping = stopping_scale * raw.total[1]
    return material, sr.close_shell_rates(material, energy, stopping, inner)


def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _pref(energy):
    gamma = 1.0 + energy / MC2
    return 2.0 * np.pi * RE_CM**2 * MC2 / (1.0 - 1.0 / gamma**2)


@pytest.mark.parametrize("energy", [40.0, 1e3, 5e3, 1e5, 1e8])
def test_full_window_is_bitwise_the_unrestricted_moments(energy):
    material = _silicon_fixture()
    full = sg.shell_gos_moments(material, energy)
    window = sg.windowed_shell_gos_moments(material, energy, 0.0, np.inf)
    for name in CHANNELS:
        assert np.array_equal(getattr(window, name), getattr(full, name))


@pytest.mark.parametrize("energy", [1e3, 5e3, 1e5])
@pytest.mark.parametrize("cutoff", [0.0, 5.0, 16.7, 50.0, 120.0, 400.0, 2e3, 1e9])
def test_soft_and_hard_windows_sum_to_the_unrestricted_moments(energy, cutoff):
    material = _silicon_fixture()
    full = sg.shell_gos_moments(material, energy)
    below = sg.windowed_shell_gos_moments(material, energy, 0.0, cutoff)
    above = sg.windowed_shell_gos_moments(material, energy, cutoff, np.inf)
    for name in CHANNELS:
        total = getattr(below, name) + getattr(above, name)
        assert np.all(getattr(below, name) >= 0.0) and np.all(getattr(above, name) >= 0.0)
        assert np.allclose(total, getattr(full, name), rtol=1e-12, atol=0.0)


def test_conduction_band_delta_at_the_cutoff_is_soft():
    material = _silicon_fixture()
    w_cb = material.oscillators[0].resonance_energy_eV
    below = sg.windowed_shell_gos_moments(material, 1e4, 0.0, w_cb)
    above = sg.windowed_shell_gos_moments(material, 1e4, w_cb, np.inf)
    assert below.distant_longitudinal[0, 1] > 0.0 and below.distant_transverse[0, 1] > 0.0
    assert not np.any(above.distant_longitudinal[0]) and not np.any(above.distant_transverse[0])


@pytest.mark.parametrize("energy", [3e3, 2e4])
@pytest.mark.parametrize("window", [(0.0, 120.0), (120.0, 600.0), (600.0, np.inf)])
def test_windowed_moments_match_direct_quadrature(energy, window):
    """Triangle p_dis (Eq. 3.76) and Møller F^(-) (Eq. 3.87) integrated in a loss window."""
    material = _silicon_fixture()
    lower, upper = window
    moments = sg.windowed_shell_gos_moments(material, energy, lower, upper)
    full = sg.shell_gos_moments(material, energy)
    a = (energy / (energy + MC2)) ** 2
    for k, osc in enumerate(material.oscillators):
        u, w = osc.ionization_energy_eV, osc.resonance_energy_eV
        if u == 0.0 or energy <= u:
            continue
        w_max = 0.5 * (energy + u)
        w_dis = 3.0 * w - 2.0 * u
        if energy <= w_dis:
            w_dis = 3.0 * (energy + 2.0 * u) / 3.0 - 2.0 * u
        full_loss = min(w_dis, w_max)
        prime = energy + u

        def moller(x, prime=prime):
            r = x / (prime - x)
            return 1.0 + r * r - (1.0 - a) * r + a * (x / prime) ** 2

        for n in range(3):
            # Distant: the loss window scales the full-window channel by the restricted integral.
            lo, hi = max(u, lower), min(full_loss, upper)

            def density(x, n=n, w_dis=w_dis):
                return x ** (n - 1) * (w_dis - x)

            part = quad(density, lo, hi, epsrel=1e-12)[0] if hi > lo else 0.0
            whole = quad(density, u, full_loss, epsrel=1e-12)[0]
            expected = full.distant_longitudinal[k, n] * part / whole
            assert moments.distant_longitudinal[k, n] == pytest.approx(expected, rel=1e-10, abs=0.0)
            lo, hi = max(u, lower), min(w_max, upper)
            close = quad(lambda x, n=n: x ** (n - 2) * moller(x), lo, hi, epsrel=1e-12, limit=200)
            expected = _pref(energy) * osc.strength * close[0] if hi > lo else 0.0
            assert moments.close[k, n] == pytest.approx(expected, rel=1e-9, abs=0.0)


@pytest.mark.parametrize(
    "window", [(-1.0, 10.0), (10.0, 5.0), (np.inf, np.inf), (np.nan, 10.0), (0.0, np.nan)]
)
def test_window_rejects_invalid_bounds(window):
    with pytest.raises(ValueError, match="loss window"):
        sg.windowed_shell_gos_moments(_silicon_fixture(), 1e4, *window)


@pytest.mark.parametrize("energy", [2e3, 3e4])
@pytest.mark.parametrize("cutoff", [0.0, 10.0, 50.0, 200.0, 1e3, 1e9])
def test_fixture_partition_closes_stopping_and_keeps_inner_shells_hard(energy, cutoff):
    material, closure = _fixture_closure(energy)
    partition = sp.partition_shell_rates(material, closure, cutoff)
    for name in CHANNELS:
        total = getattr(partition.soft, name) + getattr(partition.hard, name)
        assert np.allclose(total, getattr(closure.moments, name), rtol=1e-12, atol=0.0)
    stopping = partition.soft_stopping_eV_cm2 + partition.hard_stopping_eV_cm2
    assert stopping == pytest.approx(closure.stopping_eV_cm2, rel=1e-12, abs=0.0)
    assert not np.any(partition.soft.per_shell[closure.inner])
    assert np.allclose(
        partition.vacancy_cross_sections_cm2, closure.adopted_inner_cm2, rtol=1e-14, atol=0.0
    )
    probabilities = partition.hard_channel_probabilities
    assert np.all(probabilities >= 0.0)
    assert probabilities.sum() == pytest.approx(1.0, rel=1e-14, abs=0.0)


def test_partition_limits():
    material, closure = _fixture_closure(1e4)
    zero = sp.partition_shell_rates(material, closure, 0.0)
    assert not np.any(zero.soft.per_shell)
    assert zero.hard_cross_section_cm2 == pytest.approx(closure.moments.total[0], rel=1e-12)
    top = sp.partition_shell_rates(material, closure, 1e9)
    assert not np.any(top.hard.per_shell[~closure.inner])
    assert top.hard_cross_section_cm2 == pytest.approx(
        closure.adopted_inner_cm2.sum(), rel=1e-14, abs=0.0
    )


def test_hard_rate_falls_and_soft_moments_rise_with_the_cutoff():
    material, closure = _fixture_closure(1e4)
    cutoffs = np.geomspace(1.0, 6e3, 60)
    parts = [sp.partition_shell_rates(material, closure, w) for w in cutoffs]
    hard = np.array([p.hard_cross_section_cm2 for p in parts])
    soft = np.array([[p.soft_stopping_eV_cm2, p.soft_straggling_eV2_cm2] for p in parts])
    assert np.all(np.diff(hard) <= 0.0) and hard[-1] < hard[0]
    assert np.all(np.diff(soft, axis=0) >= 0.0)


def test_partition_rejects_bad_cutoff_and_foreign_closure():
    material, closure = _fixture_closure(1e4)
    for cutoff in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="cutoff"):
            sp.partition_shell_rates(material, closure, cutoff)
    other = so.build_shell_oscillators(
        {1: 1.0}, 19.2, 0.3, {1: (_shell(1, 1, "K", "1s1/2", 1, 13.6),)}, default_threshold_eV=1.0
    )
    with pytest.raises(ValueError, match="oscillators"):
        sp.partition_shell_rates(other, closure, 50.0)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("energy", [1e3, 1e4, 1e5])
@pytest.mark.parametrize("cutoff", [10.0, 50.0, 1e3])
def test_catalog_partition_reproduces_stp_and_eedl_vacancies(key, energy, cutoff):
    _require_pdatconf()
    partition = sp.catalog_shell_partition(key, energy, cutoff)
    closure = partition.closure
    total = partition.soft_stopping_eV_cm2 + partition.hard_stopping_eV_cm2
    assert total == pytest.approx(sr.adopted_stopping_cs(key, energy), rel=1e-12, abs=0.0)
    assert np.allclose(
        partition.vacancy_cross_sections_cm2, closure.adopted_inner_cm2, rtol=1e-14, atol=0.0
    )
    assert np.all(partition.soft.per_shell >= 0.0) and np.all(partition.hard.per_shell >= 0.0)
    assert 0.0 < partition.hard_cross_section_cm2 <= closure.moments.total[0]


def test_hard_loss_sampler_quantiles_and_energy_accounting():
    material, closure = _fixture_closure(1e4)
    partition = sp.partition_shell_rates(material, closure, 50.0)
    probabilities = partition.hard_channel_probabilities.ravel()
    cumulative = np.cumsum(probabilities)
    for flat_index in np.flatnonzero(probabilities > 0.0):
        channel_u = (cumulative[flat_index] - probabilities[flat_index] / 2.0) / cumulative[-1]
        index, branch = divmod(flat_index, 3)
        osc = material.oscillators[index]
        for quantile in (0.1, 0.5, 0.9):
            event = ss.sample_shell_hard_loss(material, partition, channel_u, quantile)
            assert event.oscillator_index == index
            assert event.branch == ss.BRANCHES[branch]
            assert event.transfer_eV >= osc.ionization_energy_eV
            accounted = (
                event.local_deposit_eV
                + (event.secondary_energy_eV or 0.0)
                + event.binding_reserve_eV
            )
            assert accounted == pytest.approx(event.transfer_eV, abs=1e-12)
            assert (event.vacancy is not None) == bool(closure.inner[index])
            assert event.binding_reserve_eV == (
                osc.ionization_energy_eV if closure.inner[index] else 0.0
            )
            assert event.secondary_energy_eV == pytest.approx(
                event.transfer_eV - osc.ionization_energy_eV
                if closure.inner[index]
                else event.transfer_eV
            )
            if branch != 2 and osc.ionization_energy_eV == 0.0:
                assert event.transfer_eV == osc.resonance_energy_eV
            else:
                lower = 0.0 if closure.inner[index] else partition.cutoff_eV
                selected = sg.windowed_shell_gos_moments(material, 1e4, lower, event.transfer_eV)
                window = getattr(selected, ss.BRANCHES[branch])[index, 0]
                full = getattr(partition.hard, ss.BRANCHES[branch])[index, 0]
                assert window * closure.scale[index] / full == pytest.approx(quantile, abs=1e-9)


def test_hard_loss_sampler_threshold_and_invalid_inputs():
    material, closure = _fixture_closure(1e4)
    partition = sp.partition_shell_rates(material, closure, 50.0)
    event = ss.sample_shell_hard_loss(material, partition, 0.5, 0.5, production_threshold_eV=1e5)
    assert event.secondary_energy_eV is None
    assert event.local_deposit_eV + event.binding_reserve_eV == event.transfer_eV
    for bad in (-1.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="uniform"):
            ss.sample_shell_hard_loss(material, partition, bad, 0.5)
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="production threshold"):
            ss.sample_shell_hard_loss(material, partition, 0.5, 0.5, production_threshold_eV=bad)


def test_seeded_hard_channel_sampling_matches_partition_probabilities():
    material, closure = _fixture_closure(1e4)
    partition = sp.partition_shell_rates(material, closure, 50.0)
    rng = np.random.default_rng(93)
    uniforms = rng.random((1200, 2))
    counts = np.zeros_like(partition.hard_channel_probabilities)
    for channel_u, loss_u in uniforms:
        event = ss.sample_shell_hard_loss(material, partition, channel_u, loss_u)
        counts[event.oscillator_index, ss.BRANCHES.index(event.branch)] += 1
    expected = counts.sum() * partition.hard_channel_probabilities
    assert np.all(abs(counts - expected) <= 6 * np.sqrt(expected) + 2)


def test_hard_recoil_matches_longitudinal_density_and_branch_limits():
    material, closure = _fixture_closure(1e4)
    partition = sp.partition_shell_rates(material, closure, 50.0)
    probabilities = partition.hard_channel_probabilities.ravel()
    cumulative = np.cumsum(probabilities)
    for flat_index in np.flatnonzero(probabilities > 0.0):
        channel_u = (cumulative[flat_index] - probabilities[flat_index] / 2.0) / cumulative[-1]
        index, branch = divmod(flat_index, 3)
        osc = material.oscillators[index]
        for recoil_u in (0.0, 0.25, 0.5, 0.9):
            event = ss.sample_shell_hard_collision(
                material, partition, channel_u, 0.5, recoil_u, 0.375
            )
            assert event.azimuth_rad == pytest.approx(0.75 * np.pi)
            assert -1.0 <= event.cos_primary <= 1.0
            if branch == 1:
                assert event.recoil_energy_eV is None and event.cos_primary == 1.0
            elif branch == 2:
                w = event.loss.transfer_eV
                assert event.recoil_energy_eV == w
                expected = np.sqrt((1e4 - w) / 1e4 * (1e4 + 2 * MC2) / (1e4 - w + 2 * MC2))
                assert event.cos_primary == pytest.approx(expected, rel=1e-14)
            else:
                u, w = osc.ionization_energy_eV, osc.resonance_energy_eV
                if u > 0 and 1e4 <= 3 * w - 2 * u:
                    w_mod, q_upper = (1e4 + 2 * u) / 3, u * 1e4 / (3 * w - 2 * u)
                else:
                    w_mod, q_upper = w, u if u > 0 else w
                q_lower = float(sg._qmin_ev(1e4, w_mod))
                q = event.recoil_energy_eV
                assert q_lower <= q <= q_upper

                def density(x):
                    return 1.0 / (x * (1.0 + x / (2.0 * MC2)))

                partial = quad(density, q_lower, q, epsrel=1e-12)[0]
                whole = quad(density, q_lower, q_upper, epsrel=1e-12)[0]
                assert partial / whole == pytest.approx(recoil_u, abs=1e-9)
                p0_sq = 1e4 * (1e4 + 2 * MC2)
                p1_sq = (1e4 - w_mod) * (1e4 - w_mod + 2 * MC2)
                expected = (p0_sq + p1_sq - q * (q + 2 * MC2)) / (2 * np.sqrt(p0_sq * p1_sq))
                assert event.cos_primary == pytest.approx(expected, abs=1e-12)


def test_hard_recoil_rejects_invalid_uniforms():
    material, closure = _fixture_closure(1e4)
    partition = sp.partition_shell_rates(material, closure, 50.0)
    for bad in (-1.0, 1.0, np.nan):
        with pytest.raises(ValueError, match="recoil uniforms"):
            ss.sample_shell_hard_collision(material, partition, 0.5, 0.5, bad, 0.5)


def test_secondary_direction_matches_transfer_geometry_and_threshold():
    material, closure = _fixture_closure(1e4)
    partition = sp.partition_shell_rates(material, closure, 50.0)
    probabilities = partition.hard_channel_probabilities.ravel()
    cumulative = np.cumsum(probabilities)
    for flat_index in np.flatnonzero(probabilities > 0.0):
        channel_u = (cumulative[flat_index] - probabilities[flat_index] / 2.0) / cumulative[-1]
        index, branch = divmod(flat_index, 3)
        for recoil_u in (0.0, 0.25, 0.75):
            event = ss.sample_shell_hard_collision(
                material, partition, channel_u, 0.5, recoil_u, 0.375
            )
            if event.loss.secondary_energy_eV is None:
                assert event.cos_secondary is None and event.secondary_azimuth_rad is None
                continue
            assert event.secondary_azimuth_rad == pytest.approx(1.75 * np.pi)
            assert event.cos_secondary is not None
            assert 0.0 <= event.cos_secondary <= 1.0
            if branch == 1:
                assert event.cos_secondary == 0.5
            elif branch == 2:
                w = event.loss.transfer_eV
                expected = np.sqrt(w / 1e4 * (1e4 + 2 * MC2) / (w + 2 * MC2))
                assert event.cos_secondary == pytest.approx(expected, rel=1e-14)
            else:
                osc = material.oscillators[index]
                u, w = osc.ionization_energy_eV, osc.resonance_energy_eV
                w_mod = (1e4 + 2 * u) / 3 if u > 0 and 1e4 <= 3 * w - 2 * u else w
                q = event.recoil_energy_eV
                assert q is not None
                q_sq = q * (q + 2 * MC2)
                p0_sq = 1e4 * (1e4 + 2 * MC2)
                p1_sq = (1e4 - w_mod) * (1e4 - w_mod + 2 * MC2)
                expected = (p0_sq + q_sq - p1_sq) / (2 * np.sqrt(p0_sq * q_sq))
                assert event.cos_secondary == pytest.approx(expected, abs=1e-10)

        suppressed = ss.sample_shell_hard_collision(
            material, partition, channel_u, 0.5, 0.5, 0.375, production_threshold_eV=1e5
        )
        assert suppressed.loss.secondary_energy_eV is None
        assert suppressed.cos_secondary is None and suppressed.secondary_azimuth_rad is None


def test_collision_directions_share_the_incoming_world_frame():
    material, closure = _fixture_closure(1e4)
    partition = sp.partition_shell_rates(material, closure, 50.0)
    probabilities = partition.hard_channel_probabilities.ravel()
    close = next(i for i in np.flatnonzero(probabilities > 0.0) if i % 3 == 2)
    channel_u = (np.cumsum(probabilities)[close] - probabilities[close] / 2) / probabilities.sum()
    event = ss.sample_shell_hard_collision(material, partition, channel_u, 0.5, 0.5, 0.125)
    assert event.cos_secondary is not None

    for incoming in ((0.0, 0.0, 1.0), tuple(np.ones(3) / np.sqrt(3))):
        world = ss.shell_collision_world_directions(event, incoming)
        assert world.secondary is not None
        for direction, cosine in (
            (world.primary, event.cos_primary),
            (world.secondary, event.cos_secondary),
        ):
            assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-14)
            assert np.dot(direction, incoming) == pytest.approx(cosine, abs=1e-14)
        primary_side = np.asarray(world.primary) - event.cos_primary * np.asarray(incoming)
        secondary_side = np.asarray(world.secondary) - event.cos_secondary * np.asarray(incoming)
        assert np.dot(primary_side, secondary_side) == pytest.approx(
            -np.linalg.norm(primary_side) * np.linalg.norm(secondary_side), abs=1e-14
        )

    suppressed = ss.sample_shell_hard_collision(
        material, partition, channel_u, 0.5, 0.5, 0.125, production_threshold_eV=1e5
    )
    world = ss.shell_collision_world_directions(suppressed, (0.0, 0.0, 1.0))
    assert world.secondary is None
    for bad in ((0.0, 0.0, 0.0), (1.0, 0.0, np.nan), (1.0, 1.0, 0.0)):
        with pytest.raises(ValueError, match="finite unit vector"):
            ss.shell_collision_world_directions(event, bad)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("energy", [1e3, 1e4, 1e5])
def test_catalog_hard_recoil_all_active_channels(key, energy):
    _require_pdatconf()
    material = sr.catalog_shell_oscillators(key)
    partition = sp.catalog_shell_partition(key, energy, 50.0)
    probabilities = partition.hard_channel_probabilities.ravel()
    cumulative = np.cumsum(probabilities)
    for flat_index in np.flatnonzero(probabilities > 0.0):
        channel_u = (cumulative[flat_index] - probabilities[flat_index] / 2.0) / cumulative[-1]
        event = ss.sample_shell_hard_collision(material, partition, channel_u, 0.5, 0.5, 0.5)
        assert -1.0 <= event.cos_primary <= 1.0
        if event.loss.secondary_energy_eV is not None:
            assert event.cos_secondary is not None
            assert 0.0 <= event.cos_secondary <= 1.0
            assert event.secondary_azimuth_rad == pytest.approx(0.0)
        assert event.loss.transfer_eV <= energy
        assert event.loss.local_deposit_eV >= 0.0
