"""BremsLib soft/hard radiative moment and photon checks.

Validation: bremslib-radiative-partition
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import (
    bremslib_segment_state,
    evaluate_bremslib,
    prepare_bremslib_table,
    stage_bremslib_table,
)
from pyrite.montecarlo.transport._jit_radiative import (
    radiative_moments_scalar,
    sample_hard_photon_energy_scalar,
)
from pyrite.montecarlo.transport.hard_radiative import (
    build_radiative_partition,
    sample_hard_radiative_photon,
)
from tests.helpers.bremslib import synthetic_bremslib_arrays


@pytest.fixture
def table():
    return prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)


def test_cutoff_partitions_one_fixed_sdcs_without_double_counting(table):
    partitions = [
        build_radiative_partition(table, 60_000.0, k) for k in (100.0, 3_330.0, 30_000.0, 60_000.0)
    ]
    reference = partitions[0].total_stopping_cs_eV_cm2
    for partition in partitions:
        np.testing.assert_allclose(partition.total_stopping_cs_eV_cm2, reference, rtol=2e-14)
        np.testing.assert_allclose(
            partition.soft_stopping_cs_eV_cm2 + partition.hard_stopping_cs_eV_cm2,
            reference,
            rtol=2e-14,
        )
    assert all(
        a.hard_rate_cs_cm2 > b.hard_rate_cs_cm2
        for a, b in zip(partitions[:-1], partitions[1:], strict=True)
    )
    assert partitions[-1].hard_rate_cs_cm2 == 0.0
    assert partitions[-1].soft_stopping_cs_eV_cm2 == reference

    # Independent dense quadrature of the public BremsLib evaluation checks
    # the units and exact-cell integration against the same physical SDCS.
    k = np.linspace(1.0, 60_000.0, 20_000)
    staged = stage_bremslib_table(table)
    state = bremslib_segment_state(staged, [60.0])
    sdcs = np.asarray(evaluate_bremslib(staged, state, k))[0]
    quadrature = np.trapezoid(k * sdcs, k)
    np.testing.assert_allclose(reference, quadrature, rtol=2e-4)


def test_seeded_hard_energy_samples_match_exact_first_moment(table):
    partition = build_radiative_partition(table, 60_000.0, 2_500.0)
    rng = np.random.default_rng(93)
    energies = np.array([partition.sample_photon_energy(float(u)) for u in rng.random(4000)])
    assert np.all((energies >= partition.cutoff_eV) & (energies <= partition.incident_energy_eV))
    expected = partition.hard_stopping_cs_eV_cm2 / partition.hard_rate_cs_cm2
    assert abs(energies.mean() - expected) < 0.04 * expected


def test_photon_and_recoiled_electron_have_valid_states(table):
    partition = build_radiative_partition(table, 60_000.0, 2_500.0)
    incoming = np.array([0.6, 0.0, 0.8])
    photon = sample_hard_radiative_photon(partition, incoming, 0.4, 0.3, 0.2)
    np.testing.assert_allclose(photon.energy_eV + photon.electron_energy_eV, 60_000.0)
    np.testing.assert_allclose(np.linalg.norm(photon.direction), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(photon.electron_direction), 1.0, atol=1e-12)
    np.testing.assert_array_equal(photon.electron_direction, incoming)
    assert photon.energy_eV > partition.cutoff_eV
    rest = 510_998.95
    p_in = np.sqrt(60_000.0 * (60_000.0 + 2.0 * rest))
    p_out = np.sqrt(photon.electron_energy_eV * (photon.electron_energy_eV + 2.0 * rest))
    np.testing.assert_allclose(
        p_in * incoming,
        photon.energy_eV * photon.direction
        + p_out * photon.electron_direction
        + photon.target_momentum_eV_c,
        atol=1e-10,
    )


def test_invalid_partition_and_empty_hard_range(table):
    with pytest.raises(ValueError, match="positive"):
        build_radiative_partition(table, 60_000.0, 0.0)
    with pytest.raises(ValueError, match="outside"):
        build_radiative_partition(table, 1_000.0, 100.0)
    with pytest.raises(ValueError, match="no hard"):
        build_radiative_partition(table, 60_000.0, 60_000.0).sample_photon_energy(0.5)


@pytest.mark.parametrize("energy", [10_000.0, 13_700.0, 50_000.0, 71_000.0, 200_000.0])
def test_numba_scalar_partition_and_sampler_match_host(table, energy):
    arrays = (
        table.incident_energy_keV,
        table.nominal_reduced_energy,
        table.top_reduced_energy,
        table.scaled_sdcs_mb,
        table.atomic_number,
    )
    for cutoff in (100.0, 3_330.0, 0.5 * energy, energy):
        host = build_radiative_partition(table, energy, cutoff)
        got = radiative_moments_scalar(*arrays, energy, cutoff)
        np.testing.assert_allclose(
            got,
            (host.soft_stopping_cs_eV_cm2, host.hard_rate_cs_cm2, host.hard_stopping_cs_eV_cm2),
            rtol=2e-12,
            atol=1e-30,
        )
        if cutoff == energy:
            continue
        for uniform in (0.0, 0.01, 0.5, 0.99, np.nextafter(1.0, 0.0)):
            sampled = sample_hard_photon_energy_scalar(*arrays, energy, cutoff, uniform)
            np.testing.assert_allclose(sampled, host.sample_photon_energy(uniform), rtol=2e-12)
