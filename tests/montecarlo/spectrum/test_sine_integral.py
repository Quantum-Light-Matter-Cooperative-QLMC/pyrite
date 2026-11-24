"""Accuracy at every device Si branch, independent of the approximation (#1407)."""

import mpmath as mp
import numpy as np

from pyrite.montecarlo.spectrum.cxr._bin_mean_jit import _si_minus_half_pi


def si_reference_points():
    """Include one-ulp neighbours on each side of 4, the rational split 8, and 48."""
    boundaries = [np.nextafter(t, toward) for t in (4.0, 8.0, 48.0) for toward in (0, np.inf)]
    return np.unique(np.r_[0.0, np.linspace(0.01, 48.0, 257), boundaries, 4.0, 8.0, 48.0])


def si_reference(t):
    """Round the complement once, after 60-digit sine integration."""
    with mp.workdps(60):
        return np.array([float(mp.si(float(x)) - mp.pi / 2) for x in t])


def test_host_si_complement_retains_the_envelope_accuracy_at_branch_boundaries():
    """Validation: sinc-bin-integration -- original 1e-15 and 2.1e-15/t bounds."""
    t = si_reference_points()
    actual = np.array([_si_minus_half_pi(x) for x in t])
    error = np.abs(actual - si_reference(t))
    bound = np.where(t <= 4.0, 1e-15, 2.1e-15 / np.maximum(t, 1.0))
    assert np.all(error <= bound)
