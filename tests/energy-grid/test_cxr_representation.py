"""Measurement semantics of the fixed-line qualification harness (#1230)."""

import numpy as np

from checks.cxr_representation_944_945 import local_shape_errors, peak_rows


def test_peak_window_requires_an_interior_maximum():
    grid = np.arange(5.0)
    assert peak_rows(grid, grid, [(0.0, 4.0)]) == [None]
    peak = peak_rows(grid, np.array([0.0, 0.0, 2.0, 0.0, 0.0]), [(0.0, 4.0)])[0]
    assert peak == {"position_eV": 2.0, "height": 2.0, "fwhm_eV": 1.0}


def test_windowed_shape_detects_a_mass_preserving_shoulder_change():
    grid = np.array([1.0, 2.0, 3.0])
    reference = [(np.array([0.5, 1.5, 2.5, 3.5]), np.array([1.0, 4.0, 1.0]))]
    error = local_shape_errors(grid, np.array([1.0, 3.0, 2.0]), reference)
    assert error["worst_l1"] == 1 / 3
    assert error["worst_linf"] == 1 / 4
