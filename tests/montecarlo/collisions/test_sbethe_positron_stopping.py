"""SBETHE positron stopping against the positron Bethe formula (#276).

Validation: sbethe-positron-stopping
"""

import pytest

from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport.shell_gos import bethe_stopping_cs
from pyrite.montecarlo.transport.shell_rates import (
    adopted_stopping_cs,
    catalog_shell_oscillators,
)
from tests.helpers.positron_tables import require_positron_sbethe_tables

KEYS = ("silicon", "mos2", "ws2")
HIGH_ENERGIES_EV = (1.0e6, 3.0e6, 1.0e7, 3.0e7, 1.0e8)


@pytest.fixture(autouse=True)
def _require_tables():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    require_positron_sbethe_tables()


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("energy_eV", HIGH_ENERGIES_EV)
def test_positron_stopping_reaches_the_positron_bethe_formula(key, energy_eV):
    """PENELOPE-2024 Eqs. 3.120 and 3.122 (f^(+)) with the catalog I and delta_F.

    Above 1 MeV shell corrections are negligible, so SBETHE's positron mode
    must reach Bethe; measured within 0.96% (WS2, 3 MeV).
    """
    material = catalog_shell_oscillators(key)
    sbethe = adopted_stopping_cs(key, energy_eV, projectile="positron")
    bethe = bethe_stopping_cs(material, energy_eV, projectile="positron")
    assert sbethe == pytest.approx(bethe, rel=1e-2)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("energy_eV", (1.0e6, 1.0e7, 1.0e8))
def test_positron_to_electron_ratio_follows_the_bethe_ratio(key, energy_eV):
    """Common I, density effect and table offsets cancel; only f^(+) vs f^(-) remains.

    Measured within 0.53% (WS2, 1 MeV); the row's tolerance is 1%.
    """
    material = catalog_shell_oscillators(key)
    sbethe = adopted_stopping_cs(key, energy_eV, projectile="positron") / adopted_stopping_cs(
        key, energy_eV
    )
    bethe = bethe_stopping_cs(material, energy_eV, projectile="positron") / bethe_stopping_cs(
        material, energy_eV
    )
    assert sbethe == pytest.approx(bethe, rel=1e-2)
