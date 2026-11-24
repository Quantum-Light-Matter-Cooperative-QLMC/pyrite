"""Automatic reflection selection on a streamed electron prefix (#1281).

Streamed runs hold the leading blocks until the prefix ``elec_id < k`` is
complete (``k`` capped at ``SELECTION_HISTORIES``, or ended earlier by the
``SELECTION_ROWS`` row budget, #1368), select on it, then score every block
with the selected set. The selection record must equal the retained route's, and
the spectra must match it up to block summation order, for any block size.

Validation: reflection-yield-tolerance-selector
"""

import numpy as np
import pytest

from pyrite._backend import _to_cpu
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import reflection_cutoff
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
from tests.montecarlo.runner.test_reflection_selection_spectrum import SELECTION, _transported

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


@pytest.mark.parametrize("route", ["spectrum", "mass", "continuum"])
def test_selection_releases_each_staged_block_before_uploading_next(monkeypatch, route):
    """Many held blocks use one device row buffer, independent of prefix size (#1416)."""
    import weakref
    from types import SimpleNamespace

    from pyrite.montecarlo.runner.emission_adapters import EmissionBatch
    from pyrite.montecarlo.runner.lattice_share import SelectionContinuum

    refs = []

    def upload(segments):
        assert not any(ref() is not None for ref in refs), "previous device block is still live"
        out = {**segments, "L_ang": segments["L_ang"].copy()}
        refs.append(weakref.ref(out["L_ang"]))
        return out

    monkeypatch.setattr(runner, "_segments_on_device", upload)
    blocks = [dict(L_ang=np.ones(4), elec_id=np.full(4, i), Ne=64) for i in range(64)]

    def parts():
        return reflection_cutoff._staged_parts(lambda: iter(blocks), {"Ne": 64}, None)

    batch = EmissionBatch({}, np.array([1000.0, 2000.0]), {}, None, None, None)

    def score(batch, _key=None):
        if batch.truncation_audit is not None:
            batch.truncation_audit["band_mass"] = float(batch.segments["L_ang"].sum())
        return SimpleNamespace(values=np.full(2, batch.segments["L_ang"].sum()))

    if route == "spectrum":
        values = reflection_cutoff._summed_parts(score, batch, parts)
    elif route == "mass":
        emission = SimpleNamespace(cxr=SimpleNamespace(evaluate=score))
        evaluate = reflection_cutoff._band_evaluator(emission, batch, 64, parts=parts)
        values, mass = evaluate({}, batch.energy_eV, (1000.0, 2000.0))
        assert mass == 4.0
    else:
        monkeypatch.setattr(
            "pyrite.montecarlo.runner.case_tables._case_bremslib_tables", lambda _: None
        )
        monkeypatch.setattr(
            "pyrite.montecarlo.runner.emission._layer_brem_from_segments",
            lambda segments, grid, *a, **kw: np.full(grid.size, segments["L_ang"].sum()),
        )
        continuum = SelectionContinuum({}, {}, None, None, (1000.0, 2000.0), parts=parts)
        assert continuum._yield(None, None, False) == pytest.approx(256000.0)
        values = np.full(2, 256.0)
    np.testing.assert_array_equal(values, [256.0, 256.0])
    assert len(refs) == 64 and all(ref() is None for ref in refs)


def test_selection_splits_held_blocks_before_device_upload(monkeypatch):
    """A host transport block may exceed the selection's available device memory (#1416)."""
    from pyrite.montecarlo.runner import electron_blocks

    monkeypatch.setattr(electron_blocks, "device_headroom_bytes", lambda: 16 * 1024)
    uploaded = []

    def upload(segments):
        uploaded.append(segments["L_ang"].size)
        assert uploaded[-1] <= 16, "selection staged an over-budget transport block"
        return segments

    monkeypatch.setattr(runner, "_segments_on_device", upload)
    block = dict(L_ang=np.ones(64), elec_id=np.arange(64), Ne=64)
    parts = reflection_cutoff._staged_parts(lambda: iter([block]), {"Ne": 64}, 48)
    assert sum(part["L_ang"].size for part in parts) == 48
    assert len(uploaded) == 3


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


@pytest.fixture(scope="module")
def automatic():
    # Diffracted brem keeps the build-time set on both routes, so spectra compare whole.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 2)
        case, _ = _transported()
        case = {k: v for k, v in case.items() if k != "line_grid_policy"}
        yield {**case, "reflection_selection": {**SELECTION, "automatic": True}}


@pytest.fixture(scope="module")
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
    assert record["selection_histories"]["lines"] == 2
    assert record["selection_histories"]["continuum"] == 2
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


@pytest.mark.slow
def test_selection_subdivided_for_device_headroom_matches_retained(
    automatic, retained, monkeypatch
):
    """Force whole-electron subblocks inside a held transport block (#1416)."""
    from pyrite.montecarlo.runner import electron_blocks

    monkeypatch.setattr(electron_blocks, "device_headroom_bytes", lambda: 1024)
    streamed = run_case_streamed(automatic, 3, transport_core="per-electron")
    assert_same_selection(streamed["reflection_selection"], retained["reflection_selection"])
    for key in SPECTRA:
        scale = np.abs(retained[key]).max()
        np.testing.assert_allclose(streamed[key], retained[key], rtol=1e-12, atol=1e-14 * scale)


@pytest.mark.parametrize("block", [1, 3])
def test_streamed_row_budget_prefix_matches_retained(automatic, block, monkeypatch):
    """With the row budget binding before the history cap, the streamed scorer
    stops holding blocks once the held rows reach it and selects on the same
    prefix as the retained route (#1368)."""
    from pyrite.montecarlo.runner import streamed_selection

    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 4096)
    transport = runner._transport_case(automatic, transport_core="per-electron")
    counts = np.bincount(np.asarray(_to_cpu(transport["segs"]["elec_id"])))
    budget = int(counts[:2].sum())  # k = 2 of 6 line histories
    assert counts[1] > 0 and budget < counts.sum()
    monkeypatch.setattr(reflection_cutoff, "SELECTION_ROWS", budget)
    retained = runner._spectrum_case(automatic, transport)

    held = []
    original = streamed_selection.PrefixSelectingScorer._select

    def select(self):
        held.append(sum(int(b["L_ang"].shape[0]) for b in self._held.values()))
        return original(self)

    monkeypatch.setattr(streamed_selection.PrefixSelectingScorer, "_select", select)
    streamed = run_case_streamed(automatic, block, transport_core="per-electron")

    record = streamed["reflection_selection"]
    assert record["selection_histories"] == {
        "lines": 2,
        "continuum": 2,
        "rows": budget,
        "row_budget": budget,
    }
    assert_same_selection(record, retained["reflection_selection"])
    assert held and held[0] < counts.sum(), "selection ran before the population was held"
    for key in SPECTRA:
        scale = np.abs(retained[key]).max()
        np.testing.assert_allclose(
            streamed[key], retained[key], rtol=1e-12, atol=1e-14 * scale, err_msg=key
        )


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


@pytest.mark.slow
def test_streamed_band_mass_selection_matches_retained(monkeypatch):
    """With an automatic bandwidth the shells are decided on the exact band mass
    (#1326), summed over the held blocks one at a time; the record equals the
    retained route's to summation order."""
    from tests.montecarlo.runner.test_reflection_selection_spectrum import _automatic_scene

    monkeypatch.setattr(reflection_cutoff, "SELECTION_HISTORIES", 2)
    case, _ = _automatic_scene("kinematic-ceiling", 30000.0, monkeypatch)
    case = {**case, "reflection_selection": {**SELECTION, "automatic": True}}
    transport = runner._transport_case(case, transport_core="per-electron")
    retained = runner._spectrum_case(case, transport)
    streamed = run_case_streamed(case, 1, transport_core="per-electron")

    record = streamed["reflection_selection"]
    assert record["reference_quadrature"] == "band-mass"
    assert record["selection_histories"]["lines"] == 2
    assert_same_selection(record, retained["reflection_selection"])
