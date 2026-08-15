import numpy as np
import pytest
from scipy.constants import elementary_charge

from pyrite.detectors import Detector, IdealPhotonCounter, NativeSpectrum, Timepix3
from pyrite.instrument import (
    Acquisition,
    combine_acquisitions,
    electron_count,
    score_acquisition,
)


def _score(
    acquisition: Acquisition,
    *,
    coordinates=((0, 0),),
    component="line",
):
    energy = np.array([0.5, 1.5, 2.5])
    accepted = np.full((len(coordinates), 3), 2.0)
    return score_acquisition(
        Detector(response=IdealPhotonCounter()),
        energy,
        accepted,
        acquisition,
        rep_rate_hz=1.0,
        bunch_charge_pc=elementary_charge * 1.0e12,
        observation_digest="a" * 64,
        coordinates=np.asarray(coordinates),
        component=component,
    )


def test_ideal_native_response_preserves_batched_event_mass() -> None:
    native = Detector(response=IdealPhotonCounter()).native_score(
        np.array([0.5, 1.5, 2.5]),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    np.testing.assert_array_equal(native.edges_eV, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(native.events, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert not native.edges_eV.flags.writeable
    assert not native.events.flags.writeable


def test_native_response_requires_explicit_supported_response() -> None:
    energy = np.array([0.5, 1.5])
    density = np.ones(2)

    with pytest.raises(ValueError, match="explicit detector response"):
        Detector().native_score(energy, density)


def test_timepix_native_response_batches_on_explicit_measured_edges() -> None:
    energy = np.arange(1_000.0, 4_100.0, 100.0)
    density = np.stack((np.ones(energy.size), np.linspace(0.5, 1.5, energy.size)))
    detector = Detector(response=Timepix3(dE_mc=200.0, dE_out=100.0, n_mc=2_000, seed=7))

    batch = detector.native_score(energy, density)
    first = detector.native_score(energy, density[0])

    assert batch.events.shape == (2, batch.edges_eV.size - 1)
    np.testing.assert_allclose(np.diff(batch.edges_eV), 100.0)
    np.testing.assert_allclose(batch.events[0], first.events, rtol=2.0e-15, atol=0.0)
    assert np.all(batch.events >= 0.0)
    assert np.all(batch.events.sum(axis=-1) <= np.sum(density, axis=-1) * 100.0)


def test_timepix_native_response_accepts_nonuniform_true_energy_samples() -> None:
    """#100: TimepixResponse resolves local input widths from node_bin_edges_and_widths
    instead of requiring a uniform true-energy grid, so native_score's rebinning also
    accepts one. Bound event mass against the incident density integrated on its own
    local widths, not a fixed dE."""
    from pyrite.energy_grid.semantics import node_bin_edges_and_widths

    energy = np.logspace(3.0, np.log10(4_000.0), 24)
    density = np.stack((np.ones(energy.size), np.linspace(0.5, 1.5, energy.size)))
    detector = Detector(response=Timepix3(n_mc=2_000, seed=7))

    batch = detector.native_score(energy, density)

    _, widths = node_bin_edges_and_widths(energy)
    incident_mass = np.sum(density * widths, axis=-1)

    assert batch.events.shape == (2, batch.edges_eV.size - 1)
    assert np.all(batch.events >= 0.0)
    assert np.all(np.isfinite(batch.events))
    assert np.all(batch.events.sum(axis=-1) <= incident_mass + 1.0e-9)


def test_timepix_native_response_rejects_non_monotonic_true_energy_samples() -> None:
    detector = Detector(response=Timepix3(n_mc=100, seed=7))

    with pytest.raises(ValueError, match="finite increasing energy grid"):
        detector.native_score(np.array([1_000.0, 900.0, 1_300.0]), np.ones(3))


def test_acquisition_partitions_event_mass_without_double_counting() -> None:
    acquisition = Acquisition(
        exposure_s=1.0,
        measured_edges_eV=(0.5, 1.5, 2.5),
        hit_threshold_eV=1.25,
    )

    scored = _score(acquisition)

    np.testing.assert_allclose(scored.expected, [[0.5, 2.0]], rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(scored.underflow_expected, [1.0])
    np.testing.assert_allclose(scored.overflow_expected, [1.0])
    np.testing.assert_allclose(scored.below_cut_expected, [1.5])
    np.testing.assert_allclose(
        scored.expected.sum(axis=-1)
        + scored.underflow_expected
        + scored.overflow_expected
        + scored.below_cut_expected,
        [6.0],
    )
    assert scored.realized is None
    np.testing.assert_array_equal(scored.counts, scored.expected)
    np.testing.assert_allclose(scored.total_counts, [2.5])
    np.testing.assert_allclose(scored.window_counts((1.5, 2.5)), [2.0])


def test_electron_count_uses_charge_cadence_and_exposure() -> None:
    acquisition = Acquisition(exposure_s=2.0, measured_edges_eV=(0.0, 1.0))

    assert electron_count(
        acquisition,
        rep_rate_hz=3.0,
        bunch_charge_pc=elementary_charge * 1.0e12,
    ) == pytest.approx(6.0)
    assert electron_count(acquisition, rep_rate_hz=3.0, bunch_charge_pc=0.0) == 0.0


def test_poisson_realization_is_coordinate_and_component_stable() -> None:
    acquisition = Acquisition(
        exposure_s=1.0,
        measured_edges_eV=(0.0, 1.0, 2.0, 3.0),
        mode="poisson",
        seed=17,
    )
    coordinates = ((4, 2), (0, 1), (9, 3))

    full = _score(acquisition, coordinates=coordinates)
    reversed_selection = _score(acquisition, coordinates=coordinates[::-1])
    singleton = _score(acquisition, coordinates=(coordinates[1],))
    background = _score(acquisition, coordinates=coordinates, component="background")

    by_coordinate = dict(zip(map(tuple, full.coordinates), full.realized, strict=True))
    reversed_by_coordinate = dict(
        zip(map(tuple, reversed_selection.coordinates), reversed_selection.realized, strict=True)
    )
    assert by_coordinate.keys() == reversed_by_coordinate.keys()
    for coordinate in by_coordinate:
        np.testing.assert_array_equal(by_coordinate[coordinate], reversed_by_coordinate[coordinate])
    np.testing.assert_array_equal(singleton.realized[0], by_coordinate[coordinates[1]])
    assert np.any(background.realized != full.realized)


def test_component_combination_sums_draws_instead_of_redrawing_total() -> None:
    acquisition = Acquisition(
        exposure_s=1.0,
        measured_edges_eV=(0.0, 1.0, 2.0, 3.0),
        mode="poisson",
        seed=17,
    )
    line = _score(acquisition, component="line")
    background = _score(acquisition, component="background")

    combined = combine_acquisitions((line, background))
    reverse = combine_acquisitions((background, line))

    np.testing.assert_array_equal(combined.realized, line.realized + background.realized)
    np.testing.assert_array_equal(combined.realized, reverse.realized)
    np.testing.assert_allclose(combined.expected, line.expected + background.expected)
    assert combined.components == ("background", "line")


def test_window_counts_requires_reporting_bin_edges() -> None:
    scored = _score(Acquisition(exposure_s=1.0, measured_edges_eV=(0.0, 1.0, 2.0)))

    with pytest.raises(ValueError, match="reporting edges"):
        scored.window_counts((0.5, 2.0))


def test_native_spectrum_rejects_negative_or_mismatched_events() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        NativeSpectrum((0.0, 1.0), np.array([-1.0]))
    with pytest.raises(ValueError, match="last dimension"):
        NativeSpectrum((0.0, 1.0, 2.0), np.array([1.0]))
