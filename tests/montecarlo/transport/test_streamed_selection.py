"""Automatic reflection selection on a streamed electron prefix (#1281).

Streamed runs hold the leading blocks until the prefix ``elec_id <
SELECTION_HISTORIES`` is complete, select on it, then score every block with
the selected set. The selection record must equal the retained route's, and
the spectra must match it up to block summation order, for any block size.

Validation: reflection-yield-tolerance-selector
"""

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import reflection_cutoff
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
from tests.montecarlo.runner.test_reflection_selection_spectrum import SELECTION, _transported

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


def assert_same_selection(record, expected, path="record"):
    """Equal records, floats to summation order: the streamed selection sums its
    held blocks one at a time, the retained route scores the merged prefix."""
    assert type(record) is type(expected), path
    if isinstance(record, dict):
        assert record.keys() == expected.keys(), path
        for key in record:
            assert_same_selection(record[key], expected[key], f"{path}.{key}")
    elif isinstance(record, list | tuple):
        assert len(record) == len(expected), path
        for i, (a, b) in enumerate(zip(record, expected, strict=True)):
            assert_same_selection(a, b, f"{path}[{i}]")
    elif isinstance(record, float):
        np.testing.assert_allclose(record, expected, rtol=1e-12, atol=1e-300, err_msg=path)
    else:
        assert record == expected, path


@pytest.fixture
def automatic(monkeypatch):
    # Diffracted brem keeps the build-time set on both routes, so spectra compare whole.
    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 2)
    case, _ = _transported()
    case = {k: v for k, v in case.items() if k != "line_grid_policy"}
    return {**case, "reflection_selection": {**SELECTION, "automatic": True}}


@pytest.fixture
def retained(automatic):
    transport = runner._transport_case(automatic, transport_core="per-electron")
    return runner._spectrum_case(automatic, transport)


def test_automatic_selection_is_live_streamable(automatic):
    assert streaming_refusal(automatic, live=True) is None
    explicit = {**automatic, "reflection_selection": SELECTION}
    assert "reflection_selection" in streaming_refusal(explicit, live=True)


@pytest.mark.parametrize("block", [1, 3, 1000])
def test_streamed_prefix_selection_matches_retained(automatic, retained, block):
    streamed = run_case_streamed(automatic, block, transport_core="per-electron")

    record, expected = streamed["reflection_selection"], retained["reflection_selection"]
    assert record["selection_histories"] == {"lines": 2, "continuum": 2}
    assert_same_selection(record, expected)
    # The streamed scorer used the selected set, not the build-time families.
    (radiator,) = record["radiators"]
    assert radiator["reflections"] > len(automatic["hkl_list"])
    for key in SPECTRA:
        scale = np.abs(retained[key]).max()
        np.testing.assert_allclose(
            streamed[key], retained[key], rtol=1e-12, atol=1e-14 * scale, err_msg=key
        )
    assert streamed["n_segments"] == retained["n_segments"]


def test_held_prefix_block_replay_commits_once(automatic, retained, monkeypatch):
    """A replayed held block replaces its first attempt (resident OOM recovery)."""
    from pyrite.montecarlo.runner import streamed_selection

    adds = []
    original = streamed_selection.PrefixSelectingScorer.add

    def add_twice(self, segs):
        original(self, segs)
        if self.scorer is None:  # still held: replay it as recovery would
            adds.append(1)
            original(self, segs)

    monkeypatch.setattr(streamed_selection.PrefixSelectingScorer, "add", add_twice)
    streamed = run_case_streamed(automatic, 1, transport_core="per-electron")
    assert len(adds) > 1
    assert_same_selection(streamed["reflection_selection"], retained["reflection_selection"])
    np.testing.assert_allclose(streamed["spec"], retained["spec"], rtol=1e-12)
    assert streamed["n_segments"] == retained["n_segments"]


def test_held_prefix_is_scored_block_by_block_on_host(automatic, retained, monkeypatch):
    """The held prefix waits on the host and the selection never merges it (#1295).

    A streamed case's prefix can be the whole population, which by construction
    does not fit the device: merging and staging it ran out of device memory.
    """
    from pyrite.montecarlo.runner import block_transport, recovery

    def no_merge(blocks):
        raise AssertionError("the streamed selection must not merge its held prefix")

    downloads = []
    monkeypatch.setattr(block_transport, "merge_electron_blocks", no_merge)
    monkeypatch.setattr("pyrite._backend.is_device_array", lambda value: hasattr(value, "shape"))
    monkeypatch.setattr(recovery, "_host_block", lambda block: downloads.append(1) or dict(block))

    streamed = run_case_streamed(automatic, 1, transport_core="per-electron")

    assert downloads, "held blocks are downloaded to the host"
    assert_same_selection(streamed["reflection_selection"], retained["reflection_selection"])
    np.testing.assert_allclose(streamed["spec"], retained["spec"], rtol=1e-12)
