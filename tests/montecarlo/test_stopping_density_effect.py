"""How much the omitted density-effect correction delta costs.

eq-stopping-bs carries delta as a parameter and every transport call site
passes 0.0. That omission used to be justified by a bound rather than a
measurement -- and the bound was stated against the wrong threshold: it read
``beta gamma <= 1.24`` as sitting below the Sternheimer onset, but the onset is
``x_0``, not ``x_1``, and graphite's ``x_0`` is -0.009, so the swept range is
*above* onset for the repository's primary material. These tests replace that
reasoning with numbers from the PDG's fitted Sternheimer coefficients.

The conclusion survives -- delta is small -- but it is now measured, and the
worst case is quantified rather than assumed away.

Validation: relativistic-bethe-stopping
"""

import numpy as np
import pytest

from pyrite.materials._transport_data import STERNHEIMER_DENSITY_EFFECT, TRANSPORT_ELEMENTS
from pyrite.montecarlo.transport import (
    _LN2,
    _MC2_KEV,
    sternheimer_delta,
)

# The energies the repository sweeps. delta rises monotonically with beta*gamma,
# so 300 keV -- the model ceiling -- is the worst case throughout.
SWEPT_keV = (10.0, 25.0, 50.0, 100.0, 200.0, 300.0)


def _bracket(J_keV, E_keV):
    """The [...] of eq-stopping-bs, which delta is subtracted from."""
    tau = E_keV / _MC2_KEV
    gamma = 1.0 + tau
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    f_minus = 1.0 - beta_sq + (tau * tau / 8.0 - (2.0 * tau + 1.0) * _LN2) / (gamma * gamma)
    I_rel = J_keV / _MC2_KEV
    return np.log(tau * tau * (tau + 2.0) / (2.0 * I_rel * I_rel)) + f_minus


def test_every_transport_element_has_sternheimer_coefficients():
    assert set(STERNHEIMER_DENSITY_EFFECT) == set(TRANSPORT_ELEMENTS)


def test_sternheimer_branches_join_continuously():
    """delta is C0 across both breakpoints. At x_0 the conductor ramp must meet
    delta_0, and the a(x_1-x)^k term must vanish at x_1 -- if either failed the
    measured bound below would be reading a discontinuous function."""
    for element, p in STERNHEIMER_DENSITY_EFFECT.items():
        below = 2.0 * np.log(10.0) * p["x0"] - p["C_bar"] + p["a"] * (p["x1"] - p["x0"]) ** p["k"]
        assert below == pytest.approx(p["delta0"], abs=5.0e-3), element


def test_delta_is_monotone_in_energy():
    """Monotone in beta*gamma, so the 300 keV value bounds the whole range.

    Non-decreasing rather than strictly increasing: N and O are non-conductors
    (delta_0 = 0) whose x_0 sits above the top of the swept range, so their
    delta is identically zero across it.
    """
    grid = np.geomspace(1.0, 300.0, 200)
    for element in STERNHEIMER_DENSITY_EFFECT:
        delta = sternheimer_delta(element, grid)
        assert np.all(np.diff(delta) >= 0.0), element
        strict = STERNHEIMER_DENSITY_EFFECT[element]["delta0"] > 0.0
        assert (delta[-1] > delta[0]) == strict, element


def test_omitting_delta_costs_under_two_percent_of_stopping_power():
    """The measurement slice B owed.

    The fractional error in |dE/ds| from dropping delta is delta/[...]. Measured
    over all 24 catalog elements: worst 0.14% at 25 keV, 0.44% at 100 keV, and
    1.51% at 300 keV, all in low-Z solids (graphite worst at 300 keV, which is
    the repository's primary material and exactly the case the task doc said to
    measure rather than assume).

    Against the ~6% Joy-Luo error at 25 keV that this branch exists to remove,
    the omission is ~45x smaller at that operating point. It is retained, and
    stated, rather than corrected.
    """
    worst = {E: (0.0, "") for E in SWEPT_keV}
    for element, params in TRANSPORT_ELEMENTS.items():
        for E in SWEPT_keV:
            fraction = float(sternheimer_delta(element, E)) / _bracket(params["J_keV"], E)
            assert fraction >= 0.0, (element, E)
            if fraction > worst[E][0]:
                worst[E] = (fraction, element)

    assert worst[25.0][0] == pytest.approx(0.00133, abs=5.0e-5)
    assert worst[100.0][0] == pytest.approx(0.00438, abs=5.0e-5)
    assert worst[300.0][0] == pytest.approx(0.01504, abs=5.0e-5)
    assert worst[300.0][1] == "C"
    # The headline claim in stopping-power.md's assumptions list.
    assert max(value for value, _ in worst.values()) < 0.02


def test_delta_stays_far_below_the_joy_luo_error_it_replaces():
    """At 25 keV the ratio tabulated as tbl-stopping-validity-ceiling puts
    Joy-Luo 6% below ICRU-37. The delta omission must be small compared with the
    error the splice was introduced to remove, or omitting it would not be
    defensible."""
    joy_luo_error_25keV = 0.06
    worst_25keV = max(
        float(sternheimer_delta(element, 25.0)) / _bracket(params["J_keV"], 25.0)
        for element, params in TRANSPORT_ELEMENTS.items()
    )

    assert worst_25keV < joy_luo_error_25keV / 20.0
