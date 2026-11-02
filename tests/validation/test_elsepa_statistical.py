"""Independent distribution checks for the #895 conditional-uniform mapping."""

import runpy
from pathlib import Path

import numpy as np
import pytest

SAMPLERS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "checks/elsepa_statistical/samplers.py")
)


@pytest.mark.parametrize("fraction", [0.0, 2**-30, 0.125, 0.5, 0.875, 1 - 2**-30, 1.0])
def test_selected_node_and_conditional_cdf_match_mixture(fraction):
    choose = SAMPLERS["node_uniform"]
    # Midpoint quadrature on a uniform grid: each branch CDF has count error
    # at most one input point. Check the JOINT law, not just branch frequency.
    n = 4096
    selected, polar = np.array([choose(7, fraction, (i + 0.5) / n) for i in range(n)]).T
    for node, probability in ((7, 1 - fraction), (8, fraction)):
        for cut in (0.0, 0.1, 0.5, 0.9, 1.0):
            observed = np.mean((selected == node) & (polar <= cut))
            assert abs(observed - probability * cut) <= 1 / n
    assert np.all((polar >= 0) & (polar <= 1))
    if fraction in (0, 1):
        np.testing.assert_array_equal(polar, (np.arange(n) + 0.5) / n)


def test_sampler_matches_analytic_uniform_triangular_mixture():
    from pyrite.montecarlo.transport.scattering import elsepa_guide_table

    mu = np.array([0.0, 1.0])
    pdf = np.array([[1.0, 1.0], [0.0, 2.0]])
    cdf = np.array([[0.0, 1.0], [0.0, 1.0]])
    guide = elsepa_guide_table(cdf)
    energy = np.log(np.array([1000.0, 4000.0]))
    n = 4096
    sample = SAMPLERS["sample_cpu"]
    y = np.array(
        [1 - sample(2.0, (i + 0.5) / n, energy, cdf, pdf, mu, guide, 0, 2) for i in range(n)]
    )
    # At f=1/2 the node mixture has E[Y]=7/6 and E[Y²]=5/3.
    assert y.mean() == pytest.approx(7 / 6, abs=2 / n)
    assert np.mean(y * y) == pytest.approx(5 / 3, abs=4 / n)
    for cut in (0.1, 0.4, 0.8):
        assert np.mean(y <= 2 * cut) == pytest.approx((cut + cut * cut) / 2, abs=2 / n)
    # Check the complete sampler at both energy endpoints, not only its
    # branch selector: the upper endpoint must use the last node's law.
    for e, inverse in ((1.0, lambda u: u), (4.0, np.sqrt)):
        for u in (0.001, 0.1, 0.5, 0.9, 0.999):
            actual = sample(e, u, energy, cdf, pdf, mu, guide, 0, 2)
            assert actual == pytest.approx(1 - 2 * inverse(u), abs=1e-14)
