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


def test_flight_grouped_substeps_give_one_flight_width():
    """Substeps of one flight score one ``sinc^2`` line, so they give one sample (#1214).

    The grouped reduction sums a flight's substep fields; its first zero is at
    ``2 pi hbar c / sum_i kappa_i t_i``, weighted ``(sum_i t_i)**2``. Two
    half-length substeps then reproduce the uncapped single row exactly; a
    one-row flight beside them keeps its own width; an invalid substep drops
    before the flight sums it.
    """
    from pyrite.montecarlo.spectrum.diagnostics import sinc_feature_samples

    n_hat = np.array([0.0, 0.0, 1.0])
    whole = {
        "E_keV": np.array([100.0, 100.0]),
        "L_ang": np.array([1000.0, 300.0]),
        "v_hat": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "elec_id": np.array([0, 1]),
        "flight_id": np.array([0, 0]),
    }
    split = {
        "E_keV": np.array([100.0, 100.0, 100.0, np.nan]),
        "L_ang": np.array([500.0, 500.0, 300.0, 50.0]),
        "v_hat": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "elec_id": np.array([0, 0, 1, 0]),
        "flight_id": np.array([0, 0, 0, 0]),
    }
    expected = sorted(zip(*sinc_feature_samples(whole, n_hat), strict=True))
    grouped = sorted(zip(*sinc_feature_samples(split, n_hat), strict=True))
    assert len(grouped) == 2
    np.testing.assert_allclose(grouped, expected, rtol=1e-12)
    beta = float(beta_from_keV(100.0))
    assert expected[0][0] == pytest.approx(2.0 * np.pi * HBARC_EV_ANG * beta / 1000.0 / (1 - beta))


def test_one_row_flights_keep_the_per_row_samples():
    from pyrite.montecarlo.spectrum.diagnostics import sinc_feature_samples

    rng = np.random.default_rng(5)
    n = 40
    v_hat = rng.normal(size=(n, 3))
    v_hat /= np.linalg.norm(v_hat, axis=1)[:, None]
    rows = {
        "E_keV": rng.uniform(20.0, 300.0, n),
        "L_ang": rng.uniform(10.0, 1000.0, n),
        "v_hat": v_hat,
        "elec_id": np.repeat(np.arange(4), 10),
    }
    with_flights = {**rows, "flight_id": np.tile(np.arange(10), 4)}
    n_hat = np.array([0.0, 0.6, 0.8])
    for plain, flights in zip(
        sinc_feature_samples(rows, n_hat), sinc_feature_samples(with_flights, n_hat), strict=True
    ):
        np.testing.assert_array_equal(plain, flights)
