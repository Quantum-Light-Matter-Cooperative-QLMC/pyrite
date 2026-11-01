"""Host/JIT analytic radiative-loss partition with dielectric suppression (#439).

Synthetic BremsLib tables avoid a remote host, external tables and Geant4.
These tests establish computational parity, not a validated production model.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport._jit_radiative import (
    radiative_moments_scalar,
    sample_hard_photon_energy_scalar,
)
from pyrite.montecarlo.transport.dielectric_radiative import (
    dielectric_cell_moments,
    dielectric_scale_eV,
)
from pyrite.montecarlo.transport.hard_radiative import build_radiative_partition
from tests.helpers.bremslib import synthetic_bremslib_arrays


@pytest.mark.parametrize("energy", [30e6, 100e6])
@pytest.mark.parametrize("plasma", [0.0, 31.05, 80.0])
def test_analytic_cell_matches_independent_quadrature(energy, plasma):
    scale = dielectric_scale_eV(energy, plasma)
    for left, right in [
        (0.0, 0.1),
        (0.01, 100.0),
        (30.0, 800.0),
        (1000.0, 10000.0),
        (10000.0, 1e6),
        (1e6, 0.95 * energy),
    ]:
        c0, c1 = 5.0, 1.5
        if plasma == 0 and left == 0:
            continue  # unsuppressed infrared zeroth moment is divergent
        rate, first = dielectric_cell_moments(left, right, c0, c1, scale)

        def chi(k, c0=c0, c1=c1, left=left, right=right):
            return c0 + (c1 - c0) * (k - left) / (right - left)

        def factor(k):
            return (k / np.hypot(k, scale)) ** 2 if scale else 1.0

        expected_rate = quad(lambda k: chi(k) * factor(k) / k, left, right, epsrel=1e-10)[0]
        expected_first = quad(lambda k: chi(k) * factor(k), left, right, epsrel=1e-10)[0]
        assert rate == pytest.approx(expected_rate, rel=2e-9, abs=1e-15)
        assert first == pytest.approx(expected_first, rel=2e-9, abs=1e-13)


@pytest.fixture(scope="module")
def table():
    return prepare_bremslib_table(
        synthetic_bremslib_arrays(t1_MeV=np.array([10.0, 30.0, 50.0, 100.0])),
        atomic_number=14,
        key="synthetic/Si",
        digest="0" * 64,
    )


@pytest.mark.parametrize("energy", [30e6, 50e6, 100e6])
@pytest.mark.parametrize("cutoff", [500.0, 1000.0, 10000.0, 1e5])
@pytest.mark.parametrize("plasma", [0.0, 31.05])
def test_host_and_jit_partition_and_sampler_match(table, energy, cutoff, plasma):
    partition = build_radiative_partition(table, energy, cutoff, plasma_energy_eV=plasma)
    args = (
        table.incident_energy_keV,
        table.nominal_reduced_energy,
        table.top_reduced_energy,
        table.scaled_sdcs_mb,
        table.atomic_number,
        energy,
        cutoff,
    )
    soft, hard_rate, hard_first = radiative_moments_scalar(*args, plasma)
    np.testing.assert_allclose(
        [soft, hard_rate, hard_first],
        [
            partition.soft_stopping_cs_eV_cm2,
            partition.hard_rate_cs_cm2,
            partition.hard_stopping_cs_eV_cm2,
        ],
        rtol=2e-6,
        atol=0.0,
    )
    for uniform in [0.0, 1e-4, 0.05, 0.25, 0.5, 0.9, 0.9999]:
        host_energy = partition.sample_photon_energy(uniform)
        jit_energy = sample_hard_photon_energy_scalar(*args, uniform, plasma)
        assert cutoff <= host_energy <= energy
        assert jit_energy == pytest.approx(host_energy, rel=2e-6, abs=1e-4)


@pytest.mark.parametrize("energy", [30e6, 100e6])
def test_soft_plus_hard_is_cutoff_independent_and_suppressed(table, energy):
    cutoffs = [100.0, 500.0, 1000.0, 5000.0, 25000.0]
    suppressed = [
        build_radiative_partition(table, energy, kc, plasma_energy_eV=31.05) for kc in cutoffs
    ]
    uncorrected = build_radiative_partition(table, energy, cutoffs[0])
    np.testing.assert_allclose(
        [p.total_stopping_cs_eV_cm2 for p in suppressed],
        suppressed[0].total_stopping_cs_eV_cm2,
        rtol=5e-12,
    )
    assert 0 < suppressed[0].total_stopping_cs_eV_cm2 < uncorrected.total_stopping_cs_eV_cm2
    assert suppressed[0].hard_rate_cs_cm2 < uncorrected.hard_rate_cs_cm2
    assert suppressed[0].plasma_energy_eV == 31.05


def test_nonphysical_plasma_rejected(table):
    for plasma in [-1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="plasma_energy_eV"):
            build_radiative_partition(table, 30e6, 1000.0, plasma_energy_eV=plasma)
