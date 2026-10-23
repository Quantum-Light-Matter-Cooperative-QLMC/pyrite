"""BremsLib staging past its complete grid: SDCS tip and x > 0 angular fill.

Validation: bremslib-high-energy-extension
"""

import numpy as np
import pytest

from pyrite._bremslib_extension import schiff_density
from pyrite._bremslib_table import prepare_bremslib_table, solid_angle_integral
from tests.helpers.bremslib import (
    extend_synthetic_arrays,
    synthetic_bremslib_arrays,
    synthetic_shape,
)

COMPLETE_T1_MEV = np.array([10.0, 20.0, 30.0])
EXTENSION_T1_MEV = np.array([50.0, 100.0])

#: Library ``k/T1`` columns among the staged ones (the rest are sub-nodes).
LIBRARY_COLUMNS = np.r_[0, 6:18]


@pytest.fixture(scope="module")
def tables():
    complete = synthetic_bremslib_arrays(COMPLETE_T1_MEV)
    extended = extend_synthetic_arrays(complete, EXTENSION_T1_MEV)
    # Electron-nucleus only: the extension is checked against library data.
    return (
        prepare_bremslib_table(complete, atomic_number=14, electron_electron=False),
        prepare_bremslib_table(extended, atomic_number=14, electron_electron=False),
    )


def _row(table, t1_MeV):
    return int(np.flatnonzero(np.isclose(table.incident_energy_keV, t1_MeV * 1e3))[0])


def test_extension_reaches_the_last_library_energy(tables):
    complete, extended = tables
    assert complete.maximum_incident_energy_keV == 30.0e3
    assert extended.maximum_incident_energy_keV == 100.0e3


def test_complete_rows_are_unchanged_by_the_extension(tables):
    complete, extended = tables
    rows = complete.incident_energy_keV.size
    np.testing.assert_allclose(
        extended.incident_energy_keV[:rows], complete.incident_energy_keV, rtol=1e-15
    )
    np.testing.assert_allclose(extended.scaled_sdcs_mb[:rows], complete.scaled_sdcs_mb, rtol=1e-14)
    np.testing.assert_allclose(
        extended.scaled_ddcs_mb_sr[:rows], complete.scaled_ddcs_mb_sr, rtol=1e-12, atol=0.0
    )


def test_library_sdcs_is_kept_and_the_tip_ratio_is_held_from_the_last_complete_row(tables):
    _, extended = tables
    library = extended.scaled_sdcs_mb[:, LIBRARY_COLUMNS]
    anchor = library[_row(extended, 30.0)]
    x = extended.nominal_reduced_energy[LIBRARY_COLUMNS]
    for t1 in EXTENSION_T1_MEV:
        row = library[_row(extended, t1)]
        expected = 10.0 * (1.0 - 0.5 * x[:11]) * t1**-0.1
        np.testing.assert_allclose(row[:11], expected, rtol=1e-14)
        np.testing.assert_allclose(row[11:] / row[10], anchor[11:] / anchor[10], rtol=1e-14)


def test_every_extension_node_integrates_to_its_sdcs(tables):
    _, extended = tables
    for t1 in EXTENSION_T1_MEV:
        row = _row(extended, t1)
        np.testing.assert_allclose(
            solid_angle_integral(extended.theta_rad, extended.scaled_ddcs_mb_sr[row]),
            extended.scaled_sdcs_mb[row],
            rtol=1e-12,
        )


def test_k0_extension_node_is_the_library_shape(tables):
    _, extended = tables
    theta = extended.theta_rad
    for t1 in EXTENSION_T1_MEV:
        row = _row(extended, t1)
        library = synthetic_shape(t1, theta).astype(np.float32).astype(float)
        library /= solid_angle_integral(theta, library)
        shape = extended.scaled_ddcs_mb_sr[row, 0] / extended.scaled_sdcs_mb[row, 0]
        np.testing.assert_allclose(shape, library, rtol=1e-6)


def test_x_dependence_follows_the_schiff_ratio_times_the_complete_row_residual(tables):
    # Synthetic library shapes do not depend on x, so the residual at 30 MeV
    # is the inverse Schiff ratio there and the filled shape is
    # S(0) * R(T) / R(T_c), renormalized.
    _, extended = tables
    theta = extended.theta_rad
    x = 0.5
    expected = (
        synthetic_shape(100.0, theta).astype(np.float32).astype(float)
        * schiff_density(14, 100.0, x, theta)
        / schiff_density(14, 100.0, 0.0, theta)
        * schiff_density(14, 30.0, 0.0, theta)
        / schiff_density(14, 30.0, x, theta)
    )
    expected /= solid_angle_integral(theta, expected)
    row = _row(extended, 100.0)
    column = int(np.flatnonzero(extended.nominal_reduced_energy == x)[0])
    shape = extended.scaled_ddcs_mb_sr[row, column] / extended.scaled_sdcs_mb[row, column]
    np.testing.assert_allclose(shape, expected, rtol=1e-6)


def test_extension_without_its_library_columns_is_refused():
    arrays = extend_synthetic_arrays(synthetic_bremslib_arrays(COMPLETE_T1_MEV), [50.0])
    arrays["sdcs_mb"][-1, 5] = np.nan
    with pytest.raises(ValueError, match=r"k/T1 <= 0.95"):
        prepare_bremslib_table(arrays, atomic_number=14)


def test_extension_rows_need_only_their_k0_node_in_order():
    arrays = extend_synthetic_arrays(synthetic_bremslib_arrays(COMPLETE_T1_MEV), [50.0])
    arrays["node_k_index"] = arrays["node_k_index"].copy()
    arrays["node_k_index"][-1] = 1
    with pytest.raises(ValueError, match="every"):
        prepare_bremslib_table(arrays, atomic_number=14)
