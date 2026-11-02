"""Live streamed spectrum scoring over electron blocks (#611).

Streaming scores each transport block as it finishes and keeps none of them.
Spectra must equal the joined full-trajectory path up to summation order,
transport counts must be exact, and unsupported cases must fail before any
transport.
"""

import gc
import weakref

import numpy as np
import pytest

from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import block_transport
from pyrite.montecarlo.runner.streamed_scoring import (
    LIVE_STREAM_REFUSED_KEYS,
    STREAM_REFUSED_KEYS,
    run_case_streamed,
    streaming_refusal,
)

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


@pytest.fixture(scope="module")
def case():
    sweep = Sweep(
        material="hopg",
        thickness_ang=1e4,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=30.0,
        n_electrons=8,
        n_electrons_brem=8,
    )
    base = dict(build_cases(sweep)[0])
    # A final line grid (no case-local resolution) and unequal populations,
    # which exercise the per-estimator electron limits.
    base.pop("line_grid_policy")
    return {**base, "Ne": 40, "Ne_brem": 52}


@pytest.fixture(scope="module")
def reference(case):
    transport = runner._transport_case(case, transport_core="per-electron")
    return runner._spectrum_case(case, transport)


def test_case_is_streamable(case):
    assert case.get("line_grid_policy") is None


@pytest.mark.parametrize("block", [7, 20, 52, 1000])
def test_streamed_spectra_match_the_full_trajectory_path(case, reference, block):
    streamed = run_case_streamed(case, block, transport_core="per-electron")
    for key in SPECTRA:
        np.testing.assert_allclose(streamed[key], reference[key], rtol=1e-12, atol=0, err_msg=key)
    np.testing.assert_array_equal(streamed["E_grid"], reference["E_grid"])
    np.testing.assert_array_equal(streamed["E_grid_brem"], reference["E_grid_brem"])
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == reference[key], key


@pytest.mark.parametrize("key", STREAM_REFUSED_KEYS + LIVE_STREAM_REFUSED_KEYS)
def test_unsupported_cases_fail_before_transport(monkeypatch, case, key):
    def no_transport(*args, **kwargs):
        raise AssertionError("transport ran")

    monkeypatch.setattr(runner, "_transport_case", no_transport)
    with pytest.raises(ValueError, match="cannot stream"):
        run_case_streamed({**case, key: {"coherence": "x"}}, 10)


def test_bunch_transition_radiation_is_refused(monkeypatch, case):
    monkeypatch.setattr(runner, "_transport_case", None)
    with pytest.raises(ValueError, match="bunch transition radiation"):
        run_case_streamed({**case, "transition_radiation": {"coherence": "bunch"}}, 10)


@pytest.mark.parametrize("block", [1, 3, 8, 100])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("secondary_threshold_eV", 1000.0),
        ("pair_production_model", "penelope-2024"),
        ("positron_transport", True),
    ],
)
def test_cascades_fail_before_transport_with_the_locality_reason(monkeypatch, block, key, value):
    def no_transport(*args, **kwargs):
        raise AssertionError("transport ran")

    monkeypatch.setattr(runner, "_transport_case", no_transport)
    # Isolate each gate, including pair/positron options whose valid cases also
    # set secondary_threshold_eV and therefore encounter its gate first.
    case = {key: value}
    reason = f"{key}: cascade RNG uses population-local parent track IDs"
    assert streaming_refusal(case, live=True) == reason
    # Saved complete histories can still be scored in whole-electron blocks.
    assert streaming_refusal(case) is None
    with pytest.raises(ValueError, match=reason):
        run_case_streamed(case, block, transport_core="per-electron")


def test_lockstep_core_is_refused(case):
    with pytest.raises(ValueError, match="per-electron or CUDA"):
        run_case_streamed(case, 10, transport_core="lockstep")


def test_consumed_blocks_are_released_before_the_next_block(monkeypatch):
    live = []

    def simulate_block(start, stop):
        # The previous block must already be unreachable when the next starts.
        gc.collect()
        assert all(ref() is None for ref in live)
        L = np.ones(3 * (stop - start))
        live.append(weakref.ref(L))
        ids = np.repeat(np.arange(stop - start), 3)
        return {"L_ang": L, "electron_id": ids, "Ne": stop - start, "n_backscattered": 1}

    released = []

    def release():
        gc.collect()
        assert all(ref() is None for ref in live)
        released.append(True)

    monkeypatch.setattr(block_transport.BACKEND, "release_memory", release)
    seen = []
    counts = block_transport.transport_electron_blocks(
        simulate_block,
        10,
        4,
        seed=0,
        bunch={},
        consume=lambda block: seen.append(int(block["electron_id"].max())),
    )
    assert len(released) == 3
    assert seen == [3, 7, 9]  # globalized indices
    assert counts["Ne"] == 10 and counts["n_backscattered"] == 3


@pytest.mark.parametrize("phase", ["transport", "spec_characteristic", "brem_wide"])
def test_streamed_oom_replays_only_failed_block(case, monkeypatch, phase):
    """A late OOM must neither re-score earlier blocks nor count failed rows."""
    from pyrite.montecarlo.runner import case_driver, pool
    from pyrite.montecarlo.runner.emission_adapters import EmissionAdapter

    expected = run_case_streamed(case, 20, transport_core="per-electron")
    simulate, evaluate = case_driver.simulate_trajectories, EmissionAdapter.evaluate
    calls, failed = [], []

    class AllocatorOOM(RuntimeError):
        pass

    def transport(*args, **kwargs):
        start, resident = kwargs["_electron_start"], kwargs["keep_segments_on_device"]
        calls.append((start, resident))
        if phase == "transport" and start == 20 and resident and not failed:
            failed.append(True)
            raise AllocatorOOM()
        # Exercise the exact driver/recovery path without requiring hardware.
        kwargs.update(transport_core="per-electron", keep_segments_on_device=False)
        return simulate(*args, **kwargs)

    def score(self, batch, slot, **kwargs):
        if slot == phase and int(batch.segments["electron_id"].min()) == 20 and not failed:
            failed.append(True)
            raise AllocatorOOM()
        return evaluate(self, batch, slot, **kwargs)

    monkeypatch.setattr(runner, "_case_transport_core", lambda *_: "cuda")
    monkeypatch.setattr(case_driver, "simulate_trajectories", transport)
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda e: isinstance(e, AllocatorOOM))
    monkeypatch.setattr(pool, "_admit_host_segments", lambda: None)
    monkeypatch.setattr(EmissionAdapter, "evaluate", score)
    actual = run_case_streamed(case, 20, transport_core="cuda")
    assert calls == (
        [(0, True), (20, True), (20, False), (40, True)]
        if phase == "transport"
        else [(0, True), (20, True), (40, True)]
    )
    if phase == "transport":
        assert actual["_transport_gpu_oom_retries"] == 1
        assert actual["_recovery_path"] == "host_segments"
    for key in SPECTRA:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=16 * np.finfo(np.float32).eps, atol=0, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments", "line_data_coverage"):
        np.testing.assert_equal(actual[key], expected[key], err_msg=key)


def _cuda_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.hardware
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize("oom_phase", [None, "transport", "spec_characteristic", "brem_wide"])
def test_cuda_streamed_residency_and_replay(case, monkeypatch, oom_phase):
    """Resident, host and recovered CUDA blocks have identical spectra (#914)."""
    import cupy

    from pyrite.montecarlo.runner import case_driver
    from pyrite.montecarlo.runner.emission_adapters import EmissionAdapter

    original_transport = runner._transport_case

    def host_transport(*args, **kwargs):
        return original_transport(*args, **{**kwargs, "keep_segments_on_device": False})

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_transport_case", host_transport)
        expected = run_case_streamed(case, 20, transport_core="cuda")

    simulate, evaluate = case_driver.simulate_trajectories, EmissionAdapter.evaluate
    attempts, residency, failed = [], [], []

    def transport(*args, **kwargs):
        start, keep = kwargs.get("_electron_start", 0), kwargs["keep_segments_on_device"]
        attempts.append((start, keep))
        if oom_phase == "transport" and start == 20 and keep and not failed:
            failed.append(True)
            raise cupy.cuda.memory.OutOfMemoryError(1, 1)
        return simulate(*args, **kwargs)

    def score(self, batch, slot, **kwargs):
        start = int(batch.segments["electron_id"].min())
        resident = isinstance(batch.segments["r_mid"], cupy.ndarray)
        residency.append(resident)
        if slot == oom_phase and start == 20 and not failed:
            failed.append(True)
            raise cupy.cuda.memory.OutOfMemoryError(1, 1)
        return evaluate(self, batch, slot, **kwargs)

    raw_residency = []
    original_add = runner.streamed_scoring.BlockSpectrumScorer.add

    def add(self, block):
        raw_residency.append(isinstance(block["r_mid"], cupy.ndarray))
        return original_add(self, block)

    monkeypatch.setattr(case_driver, "simulate_trajectories", transport)
    monkeypatch.setattr(EmissionAdapter, "evaluate", score)
    monkeypatch.setattr(runner.streamed_scoring.BlockSpectrumScorer, "add", add)
    actual = run_case_streamed(case, 20, transport_core="cuda")
    assert residency and all(residency)
    assert attempts == (
        [(0, True), (20, True), (40, True)]
        if oom_phase != "transport"
        else [(0, True), (20, True), (20, False), (40, True)]
    )
    assert raw_residency == (
        [True, True, True] if oom_phase != "transport" else [True, False, True]
    )
    for key in SPECTRA:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=16 * np.finfo(np.float32).eps, atol=0, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments", "line_data_coverage"):
        np.testing.assert_equal(actual[key], expected[key], err_msg=key)
    if oom_phase is None:
        joined = runner._spectrum_case(case, original_transport(case, transport_core="cuda"))
        # Whole-electron regrouping in fp32: the #611 16-epsilon sum allowance.
        for key in SPECTRA:
            np.testing.assert_allclose(
                actual[key], joined[key], rtol=16 * np.finfo(np.float32).eps, atol=0, err_msg=key
            )
        for key in ("eta", "hit_frac", "n_segments"):
            assert actual[key] == joined[key], key


@pytest.mark.parametrize("component", ["characteristic", "brem"])
def test_escape_oom_chunks_rows_without_replaying_transport(case, monkeypatch, component):
    """Scoring scratch must shrink even when transport's entire block fits (#915)."""
    from pyrite.montecarlo.runner import case_driver
    from pyrite.montecarlo.spectrum import brem, characteristic

    expected = run_case_streamed(case, 20, transport_core="per-electron")
    module = characteristic if component == "characteristic" else brem
    escape = module.segment_escape_paths
    simulate = case_driver.simulate_trajectories
    starts, failures = [], []
    row_budget = []

    class AllocatorOOM(RuntimeError):
        pass

    def transport(*args, **kwargs):
        starts.append(kwargs["_electron_start"])
        return simulate(*args, **kwargs)

    def bounded_escape(segments, index, *args, **kwargs):
        if not row_budget:
            row_budget.append(max(1, int(index.size) // 4))
        if index.size > row_budget[0]:
            failures.append(int(index.size))
            raise AllocatorOOM("escape scratch exceeds available memory")
        return escape(segments, index, *args, **kwargs)

    monkeypatch.setattr(case_driver, "simulate_trajectories", transport)
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda e: isinstance(e, AllocatorOOM))
    monkeypatch.setattr(module, "segment_escape_paths", bounded_escape)
    actual = run_case_streamed(case, 20, transport_core="per-electron")
    assert failures
    assert starts == [0, 20, 40]
    assert np.any(expected["spec_characteristic"] > 0)
    for key in SPECTRA:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=0, err_msg=key)
    for key in ("eta", "hit_frac", "n_segments", "line_data_coverage"):
        assert actual[key] == expected[key], key
