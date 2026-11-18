"""Admission charges aliases once and refuses before device downloads."""

import numpy as np
import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import recovery
from pyrite.montecarlo.runner.host_admission import AdaptiveHostAdmission, block_bytes
from pyrite.montecarlo.transport import host_memory


def test_projection_counts_aliases_join_and_staging(monkeypatch):
    row = np.arange(4, dtype=np.float64)
    blocks = [{"Ne": 4, "L_ang": row, "alias": row}]
    assert block_bytes(blocks) == 32
    # 10 histories * 8 bytes * 3 copies minus the live 32 bytes.
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 207)
    with pytest.raises(host_memory.HostMemoryResourceError) as caught:
        AdaptiveHostAdmission(10).check(blocks, 4)
    assert caught.value.required_bytes == 208
    assert caught.value.available_bytes == 207


def test_resident_metadata_does_not_trigger_host_projection(monkeypatch):
    class DeviceRows:
        nbytes = 1024

    blocks = [{"L_ang": DeviceRows(), "initial_E_keV": np.ones(4)}]
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 0)
    admission = AdaptiveHostAdmission(20_000_000)
    admission.check(blocks, 4)
    with pytest.raises(host_memory.HostMemoryResourceError):
        admission.check(blocks, 4, force=True)


def test_resident_recovery_admits_before_downloading(monkeypatch):
    class OOM(Exception):
        pass

    def simulate(*_args, **_kwargs):
        raise OOM("device full")

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_oom", (OOM,))
    monkeypatch.setattr(runner.BACKEND, "release_memory", lambda: None)
    monkeypatch.setattr(runner.pool, "_admit_host_segments", lambda: None)
    monkeypatch.setattr(recovery, "_host_block", lambda _: pytest.fail("download before admission"))
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 0)
    kept = [{"L_ang": np.ones(4)}]
    admission = AdaptiveHostAdmission(20_000_000)
    recover = recovery.ResidentBlockRecovery(simulate, True)
    with pytest.raises(host_memory.HostMemoryResourceError):
        recover.simulate(4, 8, kept, admit_host=lambda: admission.check(kept, 4, force=True))
    assert recover.resident and not recover.events


def test_streamed_route_switches_before_refusing_device_rows(monkeypatch):
    """#1280: projected kept bytes over headroom switch route, device rows included."""
    import pickle

    from pyrite.montecarlo.runner.host_admission import StreamedAdaptiveRoute

    class DeviceRows:
        nbytes = 1024

    blocks = [{"L_ang": DeviceRows()}]
    monkeypatch.setattr(host_memory, "host_headroom_bytes", lambda: 0)
    with pytest.raises(StreamedAdaptiveRoute) as caught:
        AdaptiveHostAdmission(20_000_000, stream=True).check(blocks, 4)
    route = pickle.loads(pickle.dumps(caught.value))  # crosses the transport pool
    assert route.block_electrons == 2 * 1024**3 // 256
    assert route.required == 3 * 1024 * 5_000_000 and route.available == 0
