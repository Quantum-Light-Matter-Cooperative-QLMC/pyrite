"""Keep compatibility snapshots sensitive to changes beyond fp64 roundoff."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.physics_baseline import assert_output_baseline


@pytest.fixture
def legacy_output():
    return json.loads(
        (Path(__file__).parents[1] / "data/physics-baseline/outputs.json").read_text()
    )["legacy"]


def test_output_baseline_accepts_fp64_roundoff(legacy_output):
    legacy_output["transport"]["path_ang"] = np.nextafter(
        legacy_output["transport"]["path_ang"], np.inf
    )
    legacy_output["spectra"]["brem"][0] = np.nextafter(
        legacy_output["spectra"]["brem"][0], np.inf
    )
    assert_output_baseline("legacy", legacy_output)


@pytest.mark.parametrize(
    "field", ["rows", "n_backscattered", "n_transmitted", "electron_id", "event_kind"]
)
def test_output_baseline_rejects_changed_counts_and_ids(legacy_output, field):
    if field in ("electron_id", "event_kind"):
        legacy_output["transport"][field][0] += 1
    else:
        legacy_output["transport"][field] += 1
    with pytest.raises(AssertionError, match=field):
        assert_output_baseline("legacy", legacy_output)


def test_output_baseline_rejects_missing_component(legacy_output):
    del legacy_output["spectra"]["brem"]
    with pytest.raises(AssertionError, match="spectra"):
        assert_output_baseline("legacy", legacy_output)


def test_output_baseline_rejects_broadcastable_shape_change(legacy_output):
    legacy_output["spectra"]["brem"] = legacy_output["spectra"]["brem"][:1]
    with pytest.raises(AssertionError, match="brem"):
        assert_output_baseline("legacy", legacy_output)


@pytest.mark.parametrize("field", ["path_ang", "brem"])
def test_output_baseline_rejects_numerical_change(legacy_output, field):
    if field == "path_ang":
        legacy_output["transport"][field] *= 1 + 1e-8
    else:
        # This spectrum is below 1e-8: an absolute tolerance would hide changes.
        legacy_output["spectra"][field][0] *= 1 + 1e-8
    with pytest.raises(AssertionError, match=field):
        assert_output_baseline("legacy", legacy_output)
