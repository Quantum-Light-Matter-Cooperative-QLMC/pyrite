"""The device pool cap is in force before CUDA transport allocates (#1303).

Uncapped, CuPy keeps every freed block cached: streamed adaptive transport of
hbn 300 keV/20M reserved 21.7 GB for 6.2 GB in use and oversubscribed a 16 GB
card before the spectrum phase applied the cap. Each transport call on the CUDA
core must therefore see the cap already set; the CPU cores never touch it.
"""

import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import case_driver, oom
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed
from tests.montecarlo.transport.test_streamed_adaptive import _requested


@pytest.fixture
def pool(monkeypatch):
    """A GPU resource policy whose cap is not yet set; records each cap set."""
    limits = []
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_pool_fraction", 0.7)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_pool_share", 1)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "pool_limit_set", False)
    monkeypatch.setattr(oom.BACKEND, "set_memory_limit", limits.append, raising=False)
    return limits


def _fake_cuda(monkeypatch, capped_at_call):
    """Resolve the CUDA core but transport on the host, noting the cap per call."""
    simulate = case_driver.simulate_trajectories

    def transport(*args, **kwargs):
        capped_at_call.append(runner._RESOURCE_POLICY.pool_limit_set)
        kwargs.update(transport_core="per-electron", keep_segments_on_device=False)
        return simulate(*args, **kwargs)

    monkeypatch.setattr(runner, "_case_transport_core", lambda *_: "cuda")
    monkeypatch.setattr(case_driver, "simulate_trajectories", transport)


def test_cuda_transport_sets_the_pool_cap_before_its_first_call(monkeypatch, pool):
    capped = []
    _fake_cuda(monkeypatch, capped)
    case = {key: value for key, value in _requested().items() if key != "adaptive_precision"}
    runner._transport_case(case, transport_core="cuda")
    assert capped and all(capped)
    assert pool == [pytest.approx(0.7)]  # once per process


def test_streamed_adaptive_measure_pass_runs_capped(monkeypatch, pool):
    """The measure-only pass precedes the streamed pass's own cap (#1280)."""
    capped = []
    _fake_cuda(monkeypatch, capped)
    run_case_streamed(_requested(), 40, transport_core="cuda")
    assert len(capped) > 2 and all(capped)
    assert pool == [pytest.approx(0.7)]


def test_cpu_transport_leaves_the_pool_uncapped(pool):
    case = {key: value for key, value in _requested().items() if key != "adaptive_precision"}
    runner._transport_case(case, transport_core="per-electron")
    assert pool == []
    assert runner._RESOURCE_POLICY.pool_limit_set is False
