"""Independent shape checks for the host-side Si bulk dielectric candidate."""

import numpy as np
import pytest
from scipy.constants import physical_constants
from scipy.integrate import quad

from pyrite.montecarlo.transport.dielectric import (
    _elf,
    build_bulk_valence_partition,
    bulk_dielectric_diimfp,
    sample_bulk_dielectric_recoil,
    sample_bulk_valence_loss,
)

# Yubero et al., Surf. Interface Anal. 20 (1993), 719; parameters released by
# Pauly, Yubero and Tougaard, Zenodo 6024064, ELF_Si.txt.
_SI_OSCILLATORS = np.array(
    [[10.0, 6.08458, 5.0, 0.5], [14.0, 30.4229, 5.0, 0.5], [16.8, 212.9603, 3.8, 0.5]]
)
_SI_GAP_EV = 1.12

# Independent optical ELF: Yang et al., Phys. Rev. B 100 (2019) 245209,
# https://micro.ustc.edu.cn/database/ELF/Si.html


def test_si_optical_valence_peak_against_independent_elf():
    """Yang et al. (2019) optical Si ELF table: 16 eV is 3.453856004."""
    optical = float(_elf(np.asarray(16.0), np.asarray(0.0), _SI_OSCILLATORS))
    assert optical == pytest.approx(3.453856004, rel=0.15)


@pytest.mark.xfail(strict=True, reason="valence-only fit omits the Si L-edge response")
def test_si_optical_core_edge_against_independent_elf():
    """Yang et al. (2019) optical Si ELF table: 110 eV is 0.047536284."""
    optical = float(_elf(np.asarray(110.0), np.asarray(0.0), _SI_OSCILLATORS))
    assert optical >= 0.5 * 0.047536284


@pytest.mark.parametrize("energy_ev", [1000.0, 3000.0])
def test_si_bulk_dielectric_plasmon_shape_against_reels(energy_ev):
    """Werner Fig. 3b: 14–20 eV mass >0.35 and peak >0.08/eV."""
    loss = np.linspace(0.0, 50.0, 1001)
    diimfp = bulk_dielectric_diimfp(energy_ev, loss, _SI_OSCILLATORS, _SI_GAP_EV)
    in_window = (loss >= 14.0) & (loss <= 20.0)
    area_50 = np.trapezoid(diimfp, loss)
    area_plasmon = np.trapezoid(diimfp[in_window], loss[in_window])

    assert area_50 > 0.0
    assert area_plasmon / area_50 >= 0.35
    assert np.max(diimfp / area_50) >= 0.08


def test_bulk_dielectric_gap_endpoints_and_quadrature():
    loss = np.array([0.0, 1.0, 1.12, 10.0, 16.0, 20.0, 1000.0])
    coarse = bulk_dielectric_diimfp(1000.0, loss, _SI_OSCILLATORS, _SI_GAP_EV, quadrature_order=64)
    fine = bulk_dielectric_diimfp(1000.0, loss, _SI_OSCILLATORS, _SI_GAP_EV)

    assert np.all(fine[[0, 1, 2, -1]] == 0.0)
    assert np.all(fine[3:6] > 0.0)
    np.testing.assert_allclose(coarse, fine, rtol=1e-4, atol=0.0)


def test_bulk_dielectric_recoil_quadrature_matches_direct_k_integral():
    """Independent log-k integration checks the dk/k to dQ/(2Q) factor."""
    energy_ev = 1000.0
    loss_ev = 16.0
    root_e = np.sqrt(energy_ev)
    root_remaining = np.sqrt(energy_ev - loss_ev)
    k_min = loss_ev / (root_e + root_remaining)
    k_max = root_e + root_remaining

    def elf_at_log_k(log_k):
        recoil_ev = np.exp(2.0 * log_k)
        total = 0.0
        for center, strength, width, dispersion in _SI_OSCILLATORS:
            shifted = center + dispersion * recoil_ev
            total += (
                strength
                * width
                * loss_ev
                / ((shifted**2 - loss_ev**2) ** 2 + (width * loss_ev) ** 2)
            )
        return total

    integral = quad(elf_at_log_k, np.log(k_min), np.log(k_max), epsrel=1e-9)[0]
    a0_ang = 1e10 * physical_constants["Bohr radius"][0]
    expected = integral / (np.pi * energy_ev * a0_ang)
    actual = bulk_dielectric_diimfp(energy_ev, np.array([loss_ev]), _SI_OSCILLATORS, _SI_GAP_EV)[0]
    assert actual == pytest.approx(expected, rel=1e-7)


def test_bulk_dielectric_rejects_nonphysical_inputs():
    with pytest.raises(ValueError, match="within"):
        bulk_dielectric_diimfp(1000.0, np.array([1001.0]), _SI_OSCILLATORS, _SI_GAP_EV)
    with pytest.raises(ValueError, match="physical"):
        invalid = _SI_OSCILLATORS.copy()
        invalid[0, 1] = -1.0
        bulk_dielectric_diimfp(1000.0, np.array([16.0]), invalid, _SI_GAP_EV)


def test_dielectric_recoil_quantiles_and_primary_angle():
    """Independent log-k quadrature checks the conditional momentum CDF."""
    energy_ev, loss_ev = 1000.0, 16.0
    remaining = energy_ev - loss_ev
    k_min = loss_ev / (np.sqrt(energy_ev) + np.sqrt(remaining))
    k_max = np.sqrt(energy_ev) + np.sqrt(remaining)

    def elf_at_log_k(log_k):
        q = np.exp(2.0 * log_k)
        return sum(
            strength
            * width
            * loss_ev
            / (((center + dispersion * q) ** 2 - loss_ev**2) ** 2 + (width * loss_ev) ** 2)
            for center, strength, width, dispersion in _SI_OSCILLATORS
        )

    total = quad(elf_at_log_k, np.log(k_min), np.log(k_max), epsrel=1e-10)[0]
    for uniform in (0.0, 0.1, 0.5, 0.9):
        recoil, cosine = sample_bulk_dielectric_recoil(
            energy_ev, loss_ev, _SI_OSCILLATORS, _SI_GAP_EV, uniform
        )
        fraction = quad(elf_at_log_k, np.log(k_min), 0.5 * np.log(recoil), epsrel=1e-10)[0] / total
        assert fraction == pytest.approx(uniform, abs=3e-4)
        expected_cosine = (energy_ev + remaining - recoil) / (2 * np.sqrt(energy_ev * remaining))
        assert cosine == pytest.approx(expected_cosine, abs=1e-12)
        assert -1.0 <= cosine <= 1.0


def test_dielectric_recoil_seeded_sampling_and_invalid_inputs():
    rng = np.random.default_rng(93)
    draws = [
        sample_bulk_dielectric_recoil(1000.0, 16.0, _SI_OSCILLATORS, _SI_GAP_EV, float(u))[0]
        for u in rng.random(128)
    ]
    median = sample_bulk_dielectric_recoil(1000.0, 16.0, _SI_OSCILLATORS, _SI_GAP_EV, 0.5)[0]
    assert abs(np.mean(np.asarray(draws) <= median) - 0.5) < 0.1
    with pytest.raises(ValueError, match="no positive"):
        sample_bulk_dielectric_recoil(1000.0, 1.0, _SI_OSCILLATORS, _SI_GAP_EV, 0.5)
    with pytest.raises(ValueError, match="uniform"):
        sample_bulk_dielectric_recoil(1000.0, 16.0, _SI_OSCILLATORS, _SI_GAP_EV, 1.0)


def test_valence_partition_cutoff_preserves_one_loss_spectrum():
    energy = 1000.0
    grid = np.linspace(0.0, 100.0, 2001)
    effective_grid = np.union1d(grid, _SI_GAP_EV)
    rate = bulk_dielectric_diimfp(energy, effective_grid, _SI_OSCILLATORS, _SI_GAP_EV)
    total_rate = np.trapezoid(rate, effective_grid)
    total_stopping = np.trapezoid(effective_grid * rate, effective_grid)
    partitions = [
        build_bulk_valence_partition(energy, grid, _SI_OSCILLATORS, _SI_GAP_EV, cutoff)
        for cutoff in (1.0, 15.123, 50.0, 100.0)
    ]
    for partition in partitions:
        assert partition.total_rate_per_ang == pytest.approx(total_rate)
        assert partition.total_stopping_ev_per_ang == pytest.approx(total_stopping)
        assert partition.soft_stopping_ev_per_ang + partition.hard_stopping_ev_per_ang == (
            pytest.approx(total_stopping)
        )
        assert partition.soft_stopping_ev_per_ang >= 0.0
    assert [p.hard_rate_per_ang for p in partitions] == sorted(
        (p.hard_rate_per_ang for p in partitions), reverse=True
    )
    assert partitions[-1].hard_rate_per_ang == 0.0
    assert partitions[-1].hard_stopping_ev_per_ang == 0.0
    assert partitions[-1].soft_stopping_ev_per_ang == pytest.approx(total_stopping)
    below_gap = build_bulk_valence_partition(energy, grid, _SI_OSCILLATORS, _SI_GAP_EV, 1.0)
    assert np.min(below_gap.lower_ev) >= _SI_GAP_EV
    with pytest.raises(ValueError, match="no hard"):
        sample_bulk_valence_loss(partitions[-1], 0.5)


@pytest.mark.parametrize("energy_ev", [1000.0, 3000.0])
def test_valence_partition_loss_grid_converges_to_adaptive_integral(energy_ev):
    """Loss-grid interpolation must resolve both partial Si valence moments."""

    def integrand(loss_ev, moment):
        rate = bulk_dielectric_diimfp(energy_ev, np.array([loss_ev]), _SI_OSCILLATORS, _SI_GAP_EV)[
            0
        ]
        return loss_ev**moment * rate

    references = [
        quad(integrand, _SI_GAP_EV, 100.0, args=(moment,), epsrel=1e-8)[0] for moment in (0, 1)
    ]
    errors = []
    for spacing_ev in (0.5, 0.25, 0.125):
        grid = np.linspace(0.0, 100.0, round(100.0 / spacing_ev) + 1)
        partition = build_bulk_valence_partition(
            energy_ev, grid, _SI_OSCILLATORS, _SI_GAP_EV, 15.123
        )
        moments = (partition.total_rate_per_ang, partition.total_stopping_ev_per_ang)
        errors.append(
            [
                abs(actual / expected - 1.0)
                for actual, expected in zip(moments, references, strict=True)
            ]
        )

    assert np.all(np.diff(errors, axis=0) < 0.0)
    assert np.max(errors[-1]) < 2e-5


def test_valence_loss_inverse_cdf_and_seeded_samples():
    grid = np.linspace(0.0, 100.0, 2001)
    partition = build_bulk_valence_partition(1000.0, grid, _SI_OSCILLATORS, _SI_GAP_EV, 15.123)
    assert partition.lower_ev[0] == pytest.approx(15.123)
    for uniform in (0.0, 0.1, 0.5, 0.9, 0.999):
        loss = sample_bulk_valence_loss(partition, uniform)
        assert 15.123 <= loss <= 100.0
        # Integrate the piecewise-linear density to the sampled energy.
        clipped = np.clip(loss - partition.lower_ev, 0.0, partition.upper_ev - partition.lower_ev)
        width = partition.upper_ev - partition.lower_ev
        y0 = partition.density_lower_per_ang_ev
        dy = partition.density_upper_per_ang_ev - y0
        cdf = np.sum(y0 * clipped + 0.5 * dy * clipped**2 / width) / partition.hard_rate_per_ang
        assert cdf == pytest.approx(uniform, abs=1e-12)

    rng = np.random.default_rng(93)
    draws = np.array([sample_bulk_valence_loss(partition, float(u)) for u in rng.random(2000)])
    median = sample_bulk_valence_loss(partition, 0.5)
    assert abs(np.mean(draws <= median) - 0.5) < 0.04
    with pytest.raises(ValueError, match="uniform"):
        sample_bulk_valence_loss(partition, 1.0)
    with pytest.raises(ValueError, match="loss grid"):
        build_bulk_valence_partition(1000.0, np.array([0.0, 1000.0]), _SI_OSCILLATORS, 1.12, 10.0)
