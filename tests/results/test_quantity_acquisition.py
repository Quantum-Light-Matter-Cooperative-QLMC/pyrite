"""#755: detector acquisition quantity views preserve counts and provenance."""

import subprocess
import sys

import numpy as np
import pytest

from pyrite.instrument.acquisition import AcquisitionBatch
from pyrite.results import QuantifiedAcquisitionBatch
from pyrite.units import ureg


def _batch(*, realized: bool) -> AcquisitionBatch:
    return AcquisitionBatch(
        coordinates=np.array([[0, 1], [1, 0]]),
        measured_edges_eV=np.array([1000.0, 1500.0, 2000.0]),
        expected=np.array([[2.5, 3.25], [4.5, 5.75]]),
        underflow_expected=np.array([1.0, 2.0]),
        overflow_expected=np.array([3.0, 4.0]),
        below_cut_expected=np.array([5.0, 6.0]),
        realized=np.array([[7, 11], [13, 17]]) if realized else None,
        underflow_realized=np.array([0, 1]) if realized else None,
        overflow_realized=np.array([2, 3]) if realized else None,
        below_cut_realized=np.array([4, 5]) if realized else None,
        components=("brem", "cxr"),
        observation_digest="a" * 64,
    )


@pytest.mark.parametrize("realized", [False, True])
def test_edges_convert_but_counts_and_identity_do_not(realized: bool) -> None:
    batch = _batch(realized=realized)
    view = batch.quantify()
    kev = view.to_energy("keV")

    assert isinstance(kev, QuantifiedAcquisitionBatch)
    assert kev.sampling == "bin_counts"
    assert kev.measure == "per_bin"
    assert kev.normalization == "acquisition_exposure"
    assert kev.count_basis == "detector_events_per_exposure"
    assert kev.axis_names == ("pixel", "measured_photon_energy")
    assert kev.components == batch.components
    assert kev.observation_digest == batch.observation_digest
    assert kev.measured_energy_edges.units == ureg.keV
    assert kev.bin_widths.units == ureg.keV
    assert kev.counts.units == ureg.dimensionless
    np.testing.assert_allclose(kev.measured_energy_edges.magnitude, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(kev.bin_widths.magnitude, [0.5, 0.5])
    np.testing.assert_array_equal(kev.pixel_coordinates, batch.coordinates)
    np.testing.assert_array_equal(kev.counts.magnitude, batch.counts)
    np.testing.assert_array_equal(kev.expected_counts.magnitude, batch.expected)
    np.testing.assert_array_equal(kev.total_counts.magnitude, batch.total_counts)
    for channel in ("underflow", "overflow", "below_cut"):
        original = getattr(batch, f"{channel}_{'realized' if realized else 'expected'}")
        np.testing.assert_array_equal(getattr(kev, f"{channel}_counts").magnitude, original)
    assert (kev.realized_counts is None) == (not realized)
    for channel in ("underflow", "overflow", "below_cut"):
        np.testing.assert_array_equal(
            getattr(kev, f"{channel}_expected").magnitude,
            getattr(batch, f"{channel}_expected"),
        )
    np.testing.assert_array_equal(
        kev.to_energy("eV").measured_energy_edges.magnitude, batch.measured_edges_eV
    )


def test_snapshot_and_mutable_readback_are_isolated() -> None:
    batch = _batch(realized=True)
    view = batch.quantify()
    view.counts.magnitude[:] = -1
    view.expected_counts.magnitude[:] = -1
    view.measured_energy_edges.magnitude[:] = -1
    view.pixel_coordinates[:] = 99
    np.testing.assert_array_equal(view.counts.magnitude, batch.realized)
    np.testing.assert_array_equal(view.expected_counts.magnitude, batch.expected)
    np.testing.assert_array_equal(view.pixel_coordinates, batch.coordinates)
    object.__setattr__(batch, "expected", np.zeros((2, 2)))
    np.testing.assert_array_equal(view.expected_counts.magnitude, [[2.5, 3.25], [4.5, 5.75]])


def test_public_alias_and_pickle_identity_survive_owner_move() -> None:
    import pickle
    from typing import get_type_hints

    from pyrite.instrument.quantified_acquisition import QuantifiedAcquisitionBatch as Owner
    from pyrite.results.quantified_acquisition import QuantifiedAcquisitionBatch as Legacy

    assert Owner is Legacy is QuantifiedAcquisitionBatch
    assert Owner.__module__ == "pyrite.results.quantified_acquisition"
    assert get_type_hints(Owner)["_coordinates"] is np.ndarray
    view = _batch(realized=True).quantify().to_energy("keV")
    restored = pickle.loads(pickle.dumps(view))
    assert type(restored) is Owner
    assert restored.observation_digest == view.observation_digest
    np.testing.assert_array_equal(restored.counts.magnitude, view.counts.magnitude)
    np.testing.assert_array_equal(
        restored.measured_energy_edges.magnitude, view.measured_energy_edges.magnitude
    )


@pytest.mark.parametrize("unit", ["s", "Hz", "mm", "radian", "dimensionless"])
def test_bad_energy_unit_refused(unit: str) -> None:
    with pytest.raises(ValueError, match="energy_unit.*energy"):
        _batch(realized=False).quantify().to_energy(unit)


def test_import_is_pint_lazy() -> None:
    code = (
        "import sys; import pyrite; "
        "from pyrite.instrument.acquisition import AcquisitionBatch; "
        "from pyrite.results import QuantifiedAcquisitionBatch; "
        "assert 'pint' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"_expected": np.ones((2, 3))}, "_expected.*shape"),
        ({"_expected": -np.ones((2, 2))}, "_expected.*nonnegative"),
        ({"_realized": np.ones((2, 3), dtype=int)}, "_realized.*shape"),
        ({"_underflow_expected": -np.ones(2)}, "_underflow_expected.*nonnegative"),
        ({"_overflow_realized": np.ones(3, dtype=int)}, "_overflow_realized.*shape"),
        ({"_below_cut_realized": None}, "all present or absent"),
        ({"_edges_eV": np.array([2.0, 1.0, 3.0])}, "strictly increasing"),
        ({"_coordinates": np.array([[0, -1], [1, 0]])}, "nonnegative"),
        ({"_expected": np.full((2, 2), np.nan)}, "finite"),
        ({"_realized": np.ones((2, 2))}, "dtype"),
        ({"components": ()}, "named source components"),
        ({"observation_digest": "invalid"}, "SHA-256"),
    ],
)
def test_invalid_quantity_channels_refused(changes: dict, match: str) -> None:
    from dataclasses import replace

    with pytest.raises((TypeError, ValueError), match=match):
        replace(_batch(realized=True).quantify(), **changes)
