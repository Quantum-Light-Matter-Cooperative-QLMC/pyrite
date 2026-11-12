import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.diagnostics import sinc_feature_spacing
from pyrite.montecarlo.transport import beta_from_keV


def test_sinc_feature_spacing_matches_first_zero_for_single_flight():
    segments = {
        "E_keV": np.array([100.0]),
        "L_ang": np.array([1000.0]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "elec_id": np.array([0]),
    }
    beta = float(beta_from_keV(100.0))
    expected = 2.0 * np.pi * HBARC_EV_ANG / ((1.0 - beta) * (1000.0 / beta))

    spacing, aliased, count = sinc_feature_spacing(
        segments, np.array([0.0, 0.0, 1.0]), electron_limit=1
    )

    assert spacing == pytest.approx(expected)
    assert aliased == 0.0
    assert count == 1


def test_sinc_feature_spacing_uses_t_squared_weighted_alias_tail():
    lengths = np.concatenate([np.full(100, 1.0), np.array([10.0])])
    segments = {
        "E_keV": np.full(lengths.size, 100.0),
        "L_ang": lengths,
        "v_hat": np.tile([1.0, 0.0, 0.0], (lengths.size, 1)),
        "elec_id": np.zeros(lengths.size, dtype=int),
    }

    spacing, aliased, _ = sinc_feature_spacing(
        segments,
        np.array([0.0, 0.0, 1.0]),
        aliased_weight_limit=0.51,
    )

    beta = float(beta_from_keV(100.0))
    expected_short_flight = 2.0 * np.pi * HBARC_EV_ANG / (1.0 / beta)
    assert spacing == pytest.approx(expected_short_flight)
    assert aliased == pytest.approx(0.5)


def test_sample_reduction_consumes_its_input_and_matches_the_spacing():
    """The memory-bounded reduction keeps the joined arithmetic (#1001)."""
    from pyrite.montecarlo.spectrum.diagnostics import (
        sinc_feature_samples,
        sinc_spacing_from_samples,
    )

    rng = np.random.default_rng(3)
    n = 500
    v_hat = rng.normal(size=(n, 3))
    v_hat /= np.linalg.norm(v_hat, axis=1)[:, None]
    segments = {
        "E_keV": rng.uniform(20.0, 300.0, n),
        "L_ang": rng.uniform(10.0, 1e4, n),
        "v_hat": v_hat,
        "elec_id": rng.integers(0, 40, n),
    }
    n_hat = np.array([0.0, 0.0, 1.0])
    expected = sinc_feature_spacing(segments, n_hat, electron_limit=30, aliased_weight_limit=1e-3)
    samples = list(sinc_feature_samples(segments, n_hat, electron_limit=30))
    assert sinc_spacing_from_samples(samples, aliased_weight_limit=1e-3) == expected
    assert samples == []
