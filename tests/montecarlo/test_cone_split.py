"""Detector-cone splitting prototype (#203): RNG separation, unbiasedness, statistics.

Validation: detector-cone-variance-reduction
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport import simulate_trajectories
from pyrite.montecarlo.transport.cone_split import (
    ConeSplit,
    history_sums,
    weighted_history_statistics,
)

N_HAT = (1.0, 0.0, 0.0)
THETA_C = 0.4


def _kw(seed, ne, **extra):
    kw = dict(
        E0_keV=50.0,
        Ne=ne,
        thickness_ang=5.0e4,
        composition=[("B", 0.0553), ("N", 0.0553)],
        seed=seed,
        transport_core="per-electron",
        energy_model="midpoint",
        elastic_model="elsepa",
        E_cut_keV=1.0,
        crystal_width_mm=0.05,
        crystal_height_mm=0.05,
    )
    kw.update(extra)
    return kw


def _tallies(result, n):
    inside = np.asarray(result["v_hat"]) @ np.asarray(N_HAT) >= np.cos(THETA_C)
    owner = np.asarray(result["electron_id"])
    length = np.asarray(result["L_ang"])
    return (
        np.bincount(owner, weights=length * inside, minlength=n),
        np.bincount(owner, weights=length, minlength=n),
    )


def _split(seed, ne, target=0.05):
    return simulate_trajectories(
        **_kw(seed, ne), _cone_split=ConeSplit(N_HAT, THETA_C, target)
    )


def test_primaries_follow_the_analog_streams_until_cone_entry():
    analog = simulate_trajectories(**_kw(3, 300))
    split = _split(3, 300)
    record = split["cone_split"]
    assert record["killed"].any() and record["parent"].size
    for e in range(300):
        a = analog["r_mid"][analog["electron_id"] == e]
        b = split["r_mid"][split["electron_id"] == e]
        if record["killed"][e]:
            assert b.shape[0] <= a.shape[0]
            np.testing.assert_array_equal(a[: b.shape[0]], b)
        else:
            np.testing.assert_array_equal(a, b)
    # Copies start inside the cone; only cone-entering primaries were stopped.
    assert np.all(record["v_hat"] @ np.asarray(N_HAT) >= np.cos(THETA_C) - 1e-12)
    assert np.all(record["weight"] > 0.0)


def test_copy_records_are_prefix_stable_in_the_electron_count():
    small, large = _split(5, 120)["cone_split"], _split(5, 240)["cone_split"]
    rows = large["parent"] < 120
    for key in ("parent", "flight", "r_ang", "v_hat", "E_keV", "weight", "stream_keys"):
        np.testing.assert_array_equal(small[key], large[key][rows])


def test_split_tallies_match_analog_within_statistical_error():
    means = {"analog": [], "split": []}
    variances = {"analog": [], "split": []}
    ne = 2000
    for seed in range(3):
        cone_a, length_a = _tallies(simulate_trajectories(**_kw(seed, ne)), ne)
        result = _split(seed, ne)
        record = result["cone_split"]
        cone_b, length_b = _tallies(result, ne)
        cone_c, length_c = _tallies(record["copies"], record["parent"].size)
        cone_b = history_sums(cone_b, cone_c, record["parent"], record["weight"], ne)
        length_b = history_sums(length_b, length_c, record["parent"], record["weight"], ne)
        for label, values in (("analog", (cone_a, length_a)), ("split", (cone_b, length_b))):
            means[label].append([v.mean() for v in values])
            variances[label].append([v.var(ddof=1) / ne for v in values])
    mean_a, mean_b = np.mean(means["analog"], axis=0), np.mean(means["split"], axis=0)
    se = np.sqrt((np.sum(variances["analog"], 0) + np.sum(variances["split"], 0)) / 9.0)
    assert np.all(np.abs(mean_a - mean_b) < 4.0 * se), (mean_a, mean_b, se)
    # The in-cone tally is the rare one splitting targets: its error shrinks.
    assert np.sum(variances["split"], 0)[0] < np.sum(variances["analog"], 0)[0]


def test_history_sums_and_weighted_statistics():
    values = history_sums(
        np.array([1.0, 0.0, 2.0]), np.array([4.0, 2.0]), np.array([1, 1]), np.array([0.25, 0.5]), 3
    )
    np.testing.assert_allclose(values, [1.0, 2.0, 2.0])
    stats = weighted_history_statistics(values)
    assert stats["mean"] == pytest.approx(5.0 / 3.0)
    assert stats["max_share"] == pytest.approx(0.4)
    assert stats["effective_sample_size"] == pytest.approx(25.0 / 9.0)
    sd = np.std(values, ddof=1)
    assert stats["relative_se"] == pytest.approx(sd / (np.sqrt(3) * 5.0 / 3.0))


@pytest.mark.parametrize(
    "extra",
    [{"transport_core": "lockstep"}, {"elastic_model": "sr"}],
)
def test_refuses_cores_and_models_without_splitting(extra):
    with pytest.raises(ValueError, match="detector-cone splitting"):
        simulate_trajectories(**_kw(0, 4, **extra), _cone_split=ConeSplit(N_HAT, THETA_C, 0.05))


def test_cone_split_validates_its_request():
    with pytest.raises(ValueError):
        ConeSplit((0.0, 0.0, 0.0), THETA_C, 0.05)
    with pytest.raises(ValueError):
        ConeSplit(N_HAT, 2.0, 0.05)
    with pytest.raises(ValueError):
        ConeSplit(N_HAT, THETA_C, 0.0)
