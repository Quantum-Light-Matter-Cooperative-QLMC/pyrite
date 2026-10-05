"""Positron bremsstrahlung by PENELOPE ``F_p`` scaling of BremsLib tables (#276).

Validation: positron-brems-scaling
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import BremsLibBremsstrahlungTable
from pyrite.montecarlo.transport.hard_radiative import (
    positron_brems_factor,
    positron_bremslib_tables,
)

REV = 510998.95


def _sbethe_fpos(z, energy_eV):
    """Transcription of SBETHE ``RSTP``'s positron factor (sbethe.f, IPROJ=2)."""
    t = np.log(1.0 + 1.0e6 * energy_eV / (REV * z * z))
    return 1.0 - np.exp(
        -t
        * (
            1.2359e-1
            - t
            * (
                6.1274e-2
                - t
                * (3.1516e-2 - t * (7.7446e-3 - t * (1.0595e-3 - t * (7.0568e-5 - t * 1.8080e-6))))
            )
        )
    )


@pytest.mark.parametrize("z", [6, 14, 42, 74, 92])
def test_factor_matches_the_sbethe_positron_correction(z):
    energy = np.geomspace(1e2, 1e9, 57)
    np.testing.assert_allclose(
        positron_brems_factor(z, energy), _sbethe_fpos(z, energy), rtol=1e-13
    )


@pytest.mark.parametrize("z", [6, 14, 74])
def test_factor_limits_and_monotonic_rise(z):
    energy = np.geomspace(1.0, 1e10, 400)
    f = positron_brems_factor(z, energy)
    assert np.all((f > 0.0) & (f <= 1.0))
    assert np.all(np.diff(f) >= 0.0)
    assert positron_brems_factor(z, 1e-6) == pytest.approx(0.0, abs=1e-6)
    assert positron_brems_factor(z, 1e10) == pytest.approx(1.0, abs=2e-3)


def test_factor_is_smaller_for_heavier_atoms():
    """Nuclear repulsion suppresses positron bremsstrahlung more at high Z."""
    for energy in (1e4, 1e5, 1e6):
        assert positron_brems_factor(74, energy) < positron_brems_factor(14, energy)


def _table():
    t1 = np.array([10.0, 100.0, 1000.0, 1.0e4])
    nominal = np.linspace(0.0, 1.0, 13)
    rng = np.random.default_rng(1)
    sdcs = rng.uniform(1.0, 2.0, (t1.size, 13))
    theta = np.linspace(0.0, np.pi, 5)
    ddcs = rng.uniform(0.1, 0.2, (t1.size, 13, theta.size))
    return BremsLibBremsstrahlungTable(
        14, "k", "d", t1, nominal, nominal * 0.999, sdcs, theta, ddcs
    )


def test_positron_tables_scale_every_node_and_keep_shapes():
    electron = {"Si": _table()}
    positron = positron_bremslib_tables(electron)["Si"]
    factor = positron_brems_factor(14, electron["Si"].incident_energy_keV * 1e3)
    np.testing.assert_allclose(
        positron.scaled_sdcs_mb / electron["Si"].scaled_sdcs_mb,
        np.broadcast_to(factor[:, None], positron.scaled_sdcs_mb.shape),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        positron.scaled_ddcs_mb_sr / electron["Si"].scaled_ddcs_mb_sr,
        np.broadcast_to(factor[:, None, None], positron.scaled_ddcs_mb_sr.shape),
        rtol=1e-15,
    )
    assert positron.key == "k+positron-fp" and positron.digest == "d"
    np.testing.assert_array_equal(positron.incident_energy_keV, electron["Si"].incident_energy_keV)
    assert electron["Si"].key == "k"
