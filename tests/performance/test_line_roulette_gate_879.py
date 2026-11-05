"""Reference grouping, spectral conditional error and held-out predictor contract."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pyrite._backend import BACKEND, REAL
from tests.helpers import scaled_rtol
from tests.helpers.segments import fake_segments, runner_transport_payload


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[2] / "checks" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load("roulette879_analysis")
gate = _load("line_roulette_spectral_gate_879")


def test_exact_bin_variance_catches_tail_that_total_yield_misses():
    f = np.array([[100.0, 0.0], [0.0, 1.0]])
    report, variance = analysis.conditional_errors(
        f,
        np.array([1.0, 0.1]),
        np.ones(2),
        np.eye(2),
        repeats=200,
        seed=879,
    )
    # Second primary's rare retention is 10*1 with prob .1, else zero:
    # variance = .1*9**2 + .9*(-1)**2 = 9. First bin is deterministic.
    np.testing.assert_array_equal(variance, [0.0, 9.0])
    assert report["conditional_bin_rse_max"] == 3.0
    assert np.sqrt(variance.sum()) / f.sum() < 0.03
    assert report["bands"][1]["conditional_rse"] == 3.0
    assert report["resampled_max_share_q50_q95"][1] == 1.0


def test_analog_limit_is_exact_in_every_bin_and_band():
    f = np.arange(1.0, 13.0).reshape(3, 4)
    report, variance = analysis.conditional_errors(
        f,
        np.ones(3),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.ones((1, 4)),
        repeats=17,
    )
    np.testing.assert_array_equal(variance, np.zeros(4))
    assert report["resampled_l1_relative_q50_q95_max"] == [0.0, 0.0, 0.0]
    assert report["conditional_bin_rse_max"] == 0.0
    assert report["resampled_bin_error_peak_normalized_q95"] == 0.0


@pytest.mark.parametrize("p", [[0.0, 1.0], [-0.1, 1.0], [1.1, 1.0]])
def test_zero_support_and_invalid_probability_are_rejected(p):
    with pytest.raises(ValueError, match="strictly positive"):
        analysis.conditional_errors(np.eye(2), p, np.ones(2), np.eye(2))


def test_band_overlap_keeps_partial_cells_instead_of_node_masking():
    weights = analysis.band_weights([0.0, 2.0, 5.0], [(1.0, 4.0), (4.0, 5.0)])
    np.testing.assert_array_equal(weights, [[1.0, 2.0], [0.0, 1.0]])


def test_frozen_pilot_threshold_does_not_rerank_heldout_histories():
    x = np.column_stack((np.arange(12.0), np.arange(12.0) ** 2, np.ones(12)))
    pilot = 8
    model = analysis.fit_predictor(x[:pilot], np.exp(np.arange(pilot) / 10))
    pilot_scores = analysis.predict(x[:pilot], model)
    _, threshold = analysis.positive_probabilities(pilot_scores, 0.25, 0.1)
    heldout = analysis.predict(x[pilot:], model)
    original, _ = analysis.positive_probabilities(heldout, 0.25, 0.1, threshold=threshold)
    appended, _ = analysis.positive_probabilities(
        np.r_[heldout, 1e8], 0.25, 0.1, threshold=threshold
    )
    np.testing.assert_array_equal(original, appended[:-1])
    assert np.all(original > 0)


def test_angular_screen_never_discards_off_cone_primary():
    segs = fake_segments(3, ne=4, elec_id=[2, 0, 2])
    features, rows = analysis.angle_features(segs, [1.0, 0.0, 0.0], 4)
    np.testing.assert_array_equal(rows, [1, 0, 2, 0])
    assert np.all(np.isfinite(features))
    p, _ = analysis.positive_probabilities(features[:, 0], 0.25, 0.1)
    assert p.min() > 0


def test_batched_strata_preserve_primary_ids_and_original_normalization():
    segs = fake_segments(6, ne=4, elec_id=[3, 0, 3, 1, 0, 2])
    segs["track_id"] = np.arange(6) + 10
    p = np.array([0.1, 1.0, 0.1, 1.0])
    selected = np.array([True, True, False, True])
    calls = []

    def score(part):
        assert part["Ne"] == 4
        assert len(part["track_id"]) == len(part["elec_id"])
        calls.append(set(part["elec_id"]))
        return np.bincount(part["elec_id"], minlength=4).astype(float) / part["Ne"]

    value = gate.stratum_score(segs, p, selected, score)
    np.testing.assert_array_equal(value, [5.0, 0.25, 0.0, 0.5])
    assert calls == [{0}, {1, 3}]


def test_per_primary_reference_adds_to_production_bin_mean_spectrum():
    print(f"roulette879 test backend={BACKEND.name} dtype={np.dtype(REAL).name}")
    assert BACKEND.name == (os.environ.get("PYRITE_TEST_BACKEND") or "cpu")
    if os.environ.get("PYRITE_FP64") == "1":
        assert np.dtype(REAL) == np.dtype(np.float64)
    segs = fake_segments(4, ne=4)
    segs["L_ang"][:] = 100.0
    segs["thickness_ang"] = 4000.0
    grid = np.linspace(700.0, 1500.0, 128)
    n_hat = np.array([1.0, 0.0, -0.05]) / np.linalg.norm([1.0, 0.0, -0.05])
    case = dict(
        crystal="hopg",
        hkl_list=[(0, 0, 2)],
        B_ang2=0.8,
        composition=[("C", 0.113)],
        line_quadrature="bin-mean",
    )
    tp = runner_transport_payload(segs, E_grid=grid, n_hat=n_hat)
    whole = gate._lines(case, tp, segs)
    contributions = np.stack(
        [gate._lines(case, tp, gate.primary_subset(segs, np.arange(4) == i)) for i in range(4)]
    )
    assert whole.max() > 0
    np.testing.assert_allclose(
        contributions.sum(axis=0),
        whole,
        rtol=scaled_rtol(1e-10, eps_multiple=32),
        atol=whole.max() * 1e-12,
    )
    p = np.array([0.1, 1.0, 0.1, 1.0])
    selected = np.array([True, True, False, True])
    candidate = gate.stratum_score(segs, p, selected, lambda s: gate._lines(case, tp, s))
    np.testing.assert_allclose(
        candidate,
        (selected / p) @ contributions,
        rtol=scaled_rtol(1e-10, eps_multiple=32),
        atol=whole.max() * 1e-12,
    )


def test_yield_variance_summary_flags_rare_history_missed_by_predictor():
    mass = np.array([100.0, 1.0, 1.0, 1.0])
    kept = analysis.yield_variance_summary(mass, np.array([1.0, 0.5, 0.5, 0.5]))
    missed = analysis.yield_variance_summary(mass, np.array([0.5, 1.0, 1.0, 1.0]))
    assert kept["added_variance"] == pytest.approx(3.0)
    assert kept["top_primary_probability"] == 1.0
    assert missed["added_variance"] == pytest.approx(1e4)
    assert missed["top_primary_probability"] == 0.5
    assert missed["floor_stratum_mass_share"] == pytest.approx(100 / 103)
    analog = analysis.yield_variance_summary(mass, np.ones(4))
    assert analog["added_variance"] == 0.0
    assert analog["added_over_analog"] == 0.0
