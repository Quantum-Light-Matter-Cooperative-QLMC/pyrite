"""Spectrum-phase ``host_segments`` recovery for device-resident rows (#905)."""

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import pool, recovery
from pyrite.montecarlo.runner.oom import _SpectrumPhaseOOM


class AllocationError(Exception):
    pass


class DeviceRows:
    """Stand-in for a CuPy row array: array-shaped, not a NumPy ndarray."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape, self.dtype = self.values.shape, self.values.dtype


def _host(value):
    return value.values if isinstance(value, DeviceRows) else value


@pytest.fixture
def policy(monkeypatch):
    calls = []
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (AllocationError,))
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom_retries", 2)
    monkeypatch.setattr(runner, "_admit_chunk", lambda chunk, *_args, **_kw: chunk)
    monkeypatch.setattr("pyrite.montecarlo.runner.stage_chunks._admit_chunk", runner._admit_chunk)
    monkeypatch.setattr(runner, "_ensure_pool_limit", lambda: None)
    monkeypatch.setattr(runner.BACKEND, "release_memory", lambda: calls.append("release"))
    monkeypatch.setattr(
        recovery, "_host_arrays", lambda segs: {key: _host(v) for key, v in segs.items()}
    )
    # Explicit backends bind: no CPU fallback may hide the device-side recovery.
    monkeypatch.setenv("PYRITE_MC_BACKEND", "cuda")
    return calls


def _transport():
    rows = {"L_ang": DeviceRows([1.0, 2.0]), "E_keV": DeviceRows([3.0, 4.0]), "Ne": 1}
    return {"E_grid": np.arange(2), "E_brem": np.arange(2), "segs": rows}


def test_resident_rows_move_to_host_before_chunks_shrink(policy, monkeypatch):
    calls = policy
    owner = _transport()
    seen = []
    monkeypatch.setattr(pool, "_admit_host_segments", lambda: calls.append("admit"))

    def spectrum(work, tp, record_timing=False):
        seen.append((type(tp["segs"]["L_ang"]), work["spec_chunk"], work["brem_chunk"]))
        if isinstance(tp["segs"]["L_ang"], DeviceRows):
            raise _SpectrumPhaseOOM("characteristic", AllocationError("allocation failed"))
        return {"spec": np.array([1.0])}

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    case = {"spec_chunk": 8, "brem_chunk": 8, "seed": 42}
    out = recovery._spectrum_with_recovery(case, dict(owner), segs_owner=owner)
    assert seen == [(DeviceRows, 8, 8), (np.ndarray, 8, 8)]
    assert out["_recovery_path"] == "host_segments"
    assert out["_recovery_events"] == [{"phase": "characteristic", "mitigation": "host_segments"}]
    assert out["_gpu_oom_retries"] == 0
    # The caller's transport now holds the same host rows, freeing the device copy.
    np.testing.assert_array_equal(owner["segs"]["L_ang"], [1.0, 2.0])
    assert owner["segs"]["Ne"] == 1
    assert calls[0] == "admit" and "release" in calls


def test_refused_admission_keeps_rows_and_halves_chunks(policy, monkeypatch):
    def refuse():
        raise recovery.BackendResourceError("cannot admit host segments")

    monkeypatch.setattr(pool, "_admit_host_segments", refuse)
    seen = []

    def spectrum(work, tp, record_timing=False):
        seen.append((type(tp["segs"]["L_ang"]), work["brem_chunk"]))
        if len(seen) == 1:
            raise _SpectrumPhaseOOM("brem", AllocationError("allocation failed"))
        return {"spec": np.array([1.0])}

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    owner = _transport()
    out = recovery._spectrum_with_recovery(
        {"spec_chunk": 8, "brem_chunk": 8}, dict(owner), segs_owner=owner
    )
    assert seen == [(DeviceRows, 8), (DeviceRows, 4)]
    assert out["_recovery_path"] == "device_spectrum"
    assert out["_recovery_events"] == [{"phase": "brem", "mitigation": "halve_chunks"}]
    assert isinstance(owner["segs"]["L_ang"], DeviceRows)


def test_host_rows_skip_the_mitigation(policy, monkeypatch):
    monkeypatch.setattr(pool, "_admit_host_segments", lambda: pytest.fail("no host move"))
    seen = []

    def spectrum(work, tp, record_timing=False):
        seen.append(work["spec_chunk"])
        if len(seen) == 1:
            raise _SpectrumPhaseOOM("line", AllocationError("allocation failed"))
        return {"spec": np.array([1.0])}

    monkeypatch.setattr(runner, "_spectrum_case", spectrum)
    tp = {"E_grid": np.arange(2), "E_brem": np.arange(2), "segs": {"L_ang": np.ones(2)}}
    out = recovery._spectrum_with_recovery({"spec_chunk": 8, "brem_chunk": 8}, tp)
    assert seen == [8, 4]
    assert out["_recovery_events"] == [{"phase": "line", "mitigation": "halve_chunks"}]
