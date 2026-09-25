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


def test_cutoff_node_rounding_keeps_scalar_sampler_on_the_hard_cells(table):
    # (kc / T) * T rounds below kc at these energies. The moment kernel and
    # sampler must still agree on which cell is the first hard one.
    arrays = (
        table.incident_energy_keV,
        table.nominal_reduced_energy,
        table.top_reduced_energy,
        table.scaled_sdcs_mb,
        table.atomic_number,
    )
    cutoff = 1_000.0
    energies = [e for e in np.linspace(20_000.0, 60_000.0, 401) if (cutoff / e) * e < cutoff]
    assert energies
    for energy in energies[:20]:
        host = build_radiative_partition(table, energy, cutoff)
        assert cutoff in host.photon_grid_eV
        for uniform in (0.0, 0.01, 0.5):
            sampled = sample_hard_photon_energy_scalar(*arrays, energy, cutoff, uniform)
            assert sampled >= cutoff
            np.testing.assert_allclose(sampled, host.sample_photon_energy(uniform), rtol=2e-12)


# ESTAR radiative stopping powers, MeV cm^2/g (NIST SRD 124, Seltzer-Berger;
# retrieved 2026-09-25) at 0.01, 0.03, 0.1, 0.3, 1, 3, 10 and 30 MeV.
_ESTAR_ENERGIES_MEV = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
_ESTAR_RADIATIVE_MEV_CM2_G = {
    "C": (0.00315, 0.003194, 0.003414, 0.004489, 0.01053, 0.03561, 0.1513, 0.5435),
    "Al": (0.006559, 0.007059, 0.007476, 0.009487, 0.02119, 0.06924, 0.2858, 1.003),
    "Ge": (0.01267, 0.01575, 0.01837, 0.02344, 0.04926, 0.1512, 0.5926, 2.029),
    "W": (0.01977, 0.02908, 0.04084, 0.05797, 0.1159, 0.3158, 1.132, 3.735),
    "Bi": (0.02064, 0.03125, 0.04524, 0.06579, 0.1313, 0.3478, 1.222, 4.01),
}


@pytest.mark.parametrize("element", sorted(_ESTAR_RADIATIVE_MEV_CM2_G))
def test_released_bremslib_total_moment_matches_seltzer_berger_nuclear_share(element):
    # Independent benchmark of the integrated first moment. BremsLib is
    # electron-atom (screened-nucleus) partial-wave bremsstrahlung; ESTAR adds
    # electron-electron bremsstrahlung, which approaches 1/Z of the nuclear
    # term at high energy. BremsLib may therefore fall short by at most that
    # share, plus the 3 % that also bounds the high-Z agreement.
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
    from pyrite.xsgen._errors import TableNotFoundError
    from pyrite.xsgen.bremslib import tables as bremslib_tables

    try:
        table = bremslib_tables.load_bremsstrahlung_tables([element])[element]
    except TableNotFoundError as exc:
        pytest.skip(f"released BremsLib tables are not installed: {exc}")
    Z = TRANSPORT_ELEMENTS[element]["Z"]
    grams_per_atom = TRANSPORT_ELEMENTS[element]["A"] / 6.02214076e23
    ratio = np.array(
        [
            build_radiative_partition(table, energy * 1e6, 1.0).total_stopping_cs_eV_cm2
            * 1e-6
            / grams_per_atom
            / estar
            for energy, estar in zip(
                _ESTAR_ENERGIES_MEV, _ESTAR_RADIATIVE_MEV_CM2_G[element], strict=True
            )
        ]
    )
    assert np.all(ratio <= 1.03)
    assert np.all(ratio >= 1.0 / (1.0 + 1.0 / Z) - 0.03)
    if Z >= 30:
        np.testing.assert_allclose(ratio, 1.0, atol=0.03)
