"""Analytic energy-window yields, independent of Monte Carlo or Optuna."""

import numpy as np
import pytest

from pyrite.optimization import FixedWindowYield


def store(energy=(100, 130, 200, 300), density=(1, 1, 1, 1), **fields):
    return {"case": {30: {"E_grid": energy, "spec": density, **fields}}}


def test_exact_bounds_constant_and_linear_density_on_nonuniform_nodes():
    score = FixedWindowYield(115, 240)
    assert score(store()) == {"window_yield": 125}
    # f(E)=2E+3 => integral_115^240 = (240^2+3*240)-(115^2+3*115).
    expected = 44750.0
    assert score(store(density=(203, 263, 403, 603)))["window_yield"] == expected
    assert score(store(density=(0, 0, 0, 0))) == {"window_yield": 0}
    # Signed numerical estimates are retained, never clipped or rewarded.
    assert score(store(density=(-1, -1, -1, -1))) == {"window_yield": -125}
    assert score.units == "photons/(sr incident electron)"


def test_only_requested_component_no_population_acceptance_or_current_scaling():
    value = FixedWindowYield(100, 300)(
        store(
            spec_characteristic=np.full(4, 1e6),
            spec_coherent=np.full(4, 1e6),
            spec_transition=np.full(4, 1e6),
            spec_channeling=np.full(4, 1e6),
            brem=np.full(4, 1e6),
            scale=999,
            case={"Ne": 999, "Ne_brem": 999, "domega_sr": 99, "current_nA": 99},
        )
    )
    assert value == {"window_yield": 200}


@pytest.mark.parametrize("bounds", [(0, 200), (200, 100), (100, 100), (100, np.inf)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="window"):
        FixedWindowYield(*bounds)


@pytest.mark.parametrize("bounds", [(90, 200), (100, 310)])
def test_uncovered_window_is_an_error_not_silently_truncated(bounds):
    with pytest.raises(ValueError, match="inside"):
        FixedWindowYield(*bounds)(store())


@pytest.mark.parametrize(
    "energy,density",
    [
        ((100, 200), (1, np.nan)),
        ((100, 200), (1, np.inf)),
        ((100, 200), (1,)),
        (((100, 200),), ((1, 1),)),
        ((100, 100, 300), (1, 1, 1)),
        ((100, np.nan, 300), (1, 1, 1)),
        ((100,), (1,)),
    ],
)
def test_invalid_samples_fail_explicitly(energy, density):
    with pytest.raises(ValueError):
        FixedWindowYield(100, 200)(store(energy, density))


def test_requires_exactly_one_complete_record():
    scorer = FixedWindowYield(100, 200)
    for results in ({}, {"case": {30: {}, 40: {}}}):
        with pytest.raises(ValueError, match="exactly one"):
            scorer(results)
    with pytest.raises(KeyError, match="spec"):
        scorer({"case": {30: {"E_grid": [100, 200]}}})


def test_refuses_bin_means_and_joint_field_components():
    with pytest.raises(ValueError, match="node-sampled"):
        FixedWindowYield(100, 200)(store(case={"line_quadrature": "bin-mean"}))
    for mode in ("two-beam", {"model": "two-beam"}):
        with pytest.raises(ValueError, match="joint-field"):
            FixedWindowYield(100, 200)(store(case={"transition_radiation": mode}))


def test_scorer_identity_changes_with_bounds_and_name():
    identity = FixedWindowYield(100, 200).evaluation_id
    assert identity == FixedWindowYield(100.0, 200.0).evaluation_id
    assert identity != FixedWindowYield(100, 201).evaluation_id
    assert identity != FixedWindowYield(101, 200).evaluation_id
    assert identity != FixedWindowYield(100, 200, "another").evaluation_id
    with pytest.raises(ValueError, match="name"):
        FixedWindowYield(100, 200, " ")
