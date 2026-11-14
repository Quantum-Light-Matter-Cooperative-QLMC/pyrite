"""Measured anchor: GaAs (400) systematic-row X-ray ratio shapes (#1178).

Nuchter & Sigle, Philos. Mag. A 71, 165 (1995), Fig. 6 (digitized in
``tests/data/channeling/nuchter-sigle-1995-gaas400-ratios.json``): 400 keV,
300 K, GaAs (100) wedge 0-303 nm, ratios against ``k_x/g400``, each with one
free scale. The plane-wave (``sigma_xi = 0``) zone prediction is
parameter-free; the computation lives in
``checks/channeling_gaas_systematic_row.py``.

Pass criterion: ``chi^2`` per point <= 1.5 against the drawn error bars, read
as +-1 sigma (the paper does not say), with one fitted scale per ratio
(``N - 1`` degrees of freedom).

- Same-element ratios (Ga-K/Ga-L, As-K/As-L) pass: K-versus-L localization and
  the 400 Bragg dip near ``k_x/g400 = 0.55-0.6`` are reproduced.
- Cross-element ratios (Ga-K/As-L, Ga-K/As-K) fail (chi^2/N 10.8 and 13.1):
  the plane-wave prediction rises too steeply through the weak 200 position
  (``k_x/g400 = 0.25``) and overshoots the peak. An incoherent Gaussian tilt
  spread (the paper states no convergence) lowers them to 2.6-3.2 near
  ``sigma_xi = 0.18-0.21`` (2.1-2.4 mrad rms) without reaching 1.5. Those
  numbers are pinned (+-0.5) so the ledger statement cannot go stale.

Convergence (basis 8/16, cells 16/32, dz 10/5 A) moves chi^2/N by <= 0.3 in
the cross ratios and <= 0.15 in the same-element ratios
(``docs/validation/check-records/planar-zone/gaas400-ratios-1178.json``).

Validation: cr-zone-incoherent-flux-weight
"""

import runpy
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[3] / "checks/channeling_gaas_systematic_row.py"


@pytest.fixture(scope="module")
def row():
    check = runpy.run_path(str(CHECK))
    xis = check["tilt_grid"](16)
    keys, weights, _ = check["row_weights"](8, 16, 10.0, xis)
    ratios = check["load_ratios"]()

    def chi2_per_point(name, sigma=0.0):
        chi2, _, curve = check["ratio_chi2"](xis, keys, weights, name, ratios[name], sigma)
        return chi2 / len(ratios[name]), curve

    return xis, chi2_per_point


@pytest.mark.slow
def test_same_element_ratio_shapes_match_measurement(row):
    xis, chi2 = row
    for name in ("Ga-K/Ga-L", "As-K/As-L"):
        value, curve = chi2(name)
        assert value <= 1.5, (name, value)
        dip = xis[xis < 1.0][curve[xis < 1.0].argmin()]
        assert 0.55 <= dip <= 0.62, (name, dip)


@pytest.mark.slow
def test_cross_element_ratio_shapes_record_the_measured_discrepancy(row):
    _, chi2 = row
    for name, plane_wave, smeared in (("Ga-K/As-L", 10.79, 2.61), ("Ga-K/As-K", 13.13, 3.18)):
        assert chi2(name)[0] == pytest.approx(plane_wave, abs=0.5), name
        best = min(chi2(name, sigma)[0] for sigma in (0.15, 0.18, 0.21, 0.24))
        assert best == pytest.approx(smeared, abs=0.5), name
