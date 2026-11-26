"""``fast`` math is statistically equivalent to ``reproducible`` (#1468).

One representative incoherent case (silicon slab, 30 keV, CXR lines plus the
continuum) runs under both modes from the same seed. Tolerances are explicit:

- spectra: integrated CXR and continuum yields within ``YIELD_RTOL`` and binned
  spectra within ``SPECTRUM_L1`` relative L1. This is the spectrum-kernel
  budget of #1469/#1463 adoption (approximate transcendentals, no RNG change);
- transport tallies: backscatter fraction within ``K_SIGMA`` binomial standard
  errors, and transmitted fraction and segment count within ``TALLY_RTOL``.

A kernel adoption that changes RNG consumption (transport accept/reject) must
widen these to Monte Carlo error with validation evidence, not loosen them
silently. The CPU node-window reducer differs between modes (#1469); its
own bound is tested in ``test_node_support_window.py``. The
``math-fast`` CI job runs this module with ``PYRITE_TEST_MATH=fast``.
"""

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite._math_mode import case_math_mode, resolve_math_mode
from pyrite.config import FixedSampling
from pyrite.montecarlo import run_case

HISTORIES = 24
LINE_GRID = np.linspace(1.0e3, 6.0e3, 120)
BREM_GRID = np.arange(1.0e3, 30.1e3, 500.0)
YIELD_RTOL = 2.0e-2
SPECTRUM_L1 = 5.0e-2
TALLY_RTOL = 5.0e-2
K_SIGMA = 3.0

pytestmark = pytest.mark.slow


def _case(math):
    scene = pr.Scene(
        pr.Beam(30.0, transverse=pr.beams.PointSpot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab("silicon", thickness_ang=2.0e4, tilt_deg=30.0, width=None, height=None),
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=HISTORIES, brem_histories=HISTORIES),
        seed=7,
        energy_bins=pr.detectors.EnergyBins(cxr=LINE_GRID, brem=BREM_GRID),
        execution=pr.ExecutionConfig(math=math),
    )
    return api.build_case(scene, numerics)


@pytest.fixture(scope="module")
def runs():
    return {mode: run_case(_case(mode)) for mode in ("reproducible", "fast")}


def _l1(a, b):
    return float(np.sum(np.abs(a - b)) / max(np.sum(np.abs(b)), np.finfo(float).tiny))


def test_spectra_agree_within_the_fast_budget(runs):
    fast, exact = runs["fast"], runs["reproducible"]
    for key, grid in (("spec", "E_grid"), ("brem_wide", "E_grid_brem")):
        np.testing.assert_array_equal(fast[grid], exact[grid])
        integral = np.trapezoid(fast[key], fast[grid])
        reference = np.trapezoid(exact[key], exact[grid])
        assert reference > 0
        assert integral == pytest.approx(reference, rel=YIELD_RTOL), key
        assert _l1(fast[key], exact[key]) <= SPECTRUM_L1, key


def test_transport_tallies_agree_within_monte_carlo_error(runs):
    fast, exact = runs["fast"], runs["reproducible"]
    eta = exact["eta"]
    sigma = np.sqrt(max(eta * (1.0 - eta), 1.0 / HISTORIES) / HISTORIES)
    assert abs(fast["eta"] - eta) <= K_SIGMA * sigma
    assert fast["hit_frac"] == pytest.approx(exact["hit_frac"], rel=TALLY_RTOL)
    assert fast["n_segments"] == pytest.approx(exact["n_segments"], rel=TALLY_RTOL)


def test_unset_mode_follows_the_process_resolution():
    assert case_math_mode(_case(None)) == resolve_math_mode().mode
