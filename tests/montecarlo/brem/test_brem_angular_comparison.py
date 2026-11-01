"""BremsLib angular shape against the Schiff formula (Koch-Motz 2BS).

Validation: bremslib-angular-schiff
"""

import numpy as np
import pytest

from pyrite.validation.brem_angular import (
    ELECTRON_REST_ENERGY_MEV,
    angular_window_rad,
    compare_angular_shape,
    enclosed_angle,
    schiff_density,
)
from pyrite.xsgen._errors import TableNotFoundError
from pyrite.xsgen.bremslib import tables as bremslib_tables


def test_enclosed_angle_of_isotropic_density_is_the_cap_angle():
    theta = np.linspace(0.0, np.pi, 20001)
    # (1 - cos(theta)) / 2 of the flux lies inside theta.
    for fraction in (0.25, 0.5, 0.9):
        expected = np.arccos(1.0 - 2.0 * fraction)
        assert enclosed_angle(theta, np.ones_like(theta), fraction) == pytest.approx(
            expected, rel=1e-6
        )


def test_schiff_density_is_positive_and_peaks_near_one_over_gamma():
    T_MeV, kappa = 10.0, 0.5
    E0 = 1.0 + T_MeV / ELECTRON_REST_ENERGY_MEV
    theta = np.linspace(1e-5, angular_window_rad(T_MeV), 4000)
    density = schiff_density(29, T_MeV, kappa, theta)
    assert np.all(density > 0.0)
    peak = theta[np.argmax(2.0 * np.pi * np.sin(theta) * density)]
    assert 0.3 / E0 < peak < 2.0 / E0


def test_schiff_enclosed_angle_scales_as_inverse_energy():
    # y = E0 theta is the only angular variable, so at fixed kappa and
    # E0 >> 1 the enclosed angle times E0 is nearly energy independent.
    values = []
    for T_MeV in (20.0, 40.0):
        E0 = 1.0 + T_MeV / ELECTRON_REST_ENERGY_MEV
        theta = np.linspace(1e-6, 60.0 / E0, 6000)
        values.append(E0 * enclosed_angle(theta, schiff_density(13, T_MeV, 0.3, theta), 0.5))
    assert values[1] == pytest.approx(values[0], rel=0.05)


@pytest.mark.parametrize(("element", "Z", "T_keV"), [("C", 6, 10000.0), ("Al", 13, 30000.0)])
def test_installed_bremslib_shape_matches_schiff_at_low_z(element, Z, T_keV):
    try:
        table = bremslib_tables.load_bremsstrahlung_tables([element]).get(element)
    except TableNotFoundError:
        pytest.skip("BremsLib catalogue tables are not installed")
    if table is None:
        pytest.skip("BremsLib catalogue tables are not installed")
    assert table.atomic_number == Z
    i = int(np.argmin(abs(table.incident_energy_keV - T_keV)))
    for j in (2, 4, 6):
        for ratio in compare_angular_shape(table, i, j).values():
            assert ratio == pytest.approx(1.0, abs=0.06)
