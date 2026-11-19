"""The device pool is capped before CUDA transport allocates (#1303, #1315).

Uncapped, CuPy keeps every freed block cached: streamed adaptive transport of
hbn 300 keV/20M reserved 21.7 GB for 6.2 GB in use and oversubscribed a 16 GB
card before the spectrum phase applied the cap. Each transport call on the CUDA
core must therefore see a cap already set; the CPU cores never touch it.

Transport is capped at the card less memory outside the pool, not at the
spectrum budget (#1315): the resident adaptive join holds about twice the kept
rows, and at the budget hopg 1000 keV/20M fell back to host rows. The spectrum
phase lowers the cap to the budget again, and a scoped cap (streamed block
sizing) is never raised.
"""

from types import SimpleNamespace

import pytest

from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import case_driver, oom
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed
from tests.montecarlo.transport.test_streamed_adaptive import _requested

GIB = 1 << 30
CARD = 16 * GIB
OUTSIDE = 1 * GIB
BUDGET = int(0.7 * CARD)
CEILING = CARD - OUTSIDE - oom.TRANSPORT_POOL_HEADROOM_BYTES


class _Pool:
    """A fake device pool: the cap in force and every cap set, in bytes."""

    def __init__(self):
        self.limit = 0
        self.sets = []

    def set_bytes(self, size):
        self.limit = int(size)
        self.sets.append(self.limit)

    def set_fraction(self, fraction):
        self.set_bytes(fraction * CARD)


@pytest.fixture
def pool(monkeypatch):
    """A GPU resource policy whose cap is not yet set, on a 16 GiB card."""
    fake = _Pool()
    policy = runner._RESOURCE_POLICY
    monkeypatch.setattr(policy, "gpu", True)
    monkeypatch.setattr(policy, "gpu_pool_fraction", 0.7)
    monkeypatch.setattr(policy, "gpu_pool_share", 1)
    monkeypatch.setattr(policy, "transport_ceiling", True)
    monkeypatch.setattr(policy, "pool_limit_bytes", None)
    monkeypatch.setattr(policy, "pool_limit_phase", None)
    backend = oom.BACKEND
    monkeypatch.setattr(backend, "device", SimpleNamespace(total_memory_bytes=CARD))
    monkeypatch.setattr(backend, "set_memory_limit", fake.set_fraction, raising=False)
    monkeypatch.setattr(backend, "set_memory_limit_bytes", fake.set_bytes, raising=False)
    monkeypatch.setattr(backend, "memory_limit_bytes", lambda: fake.limit, raising=False)
    monkeypatch.setattr(backend, "device_bytes_outside_pool", lambda: OUTSIDE, raising=False)
    return fake


def _fake_cuda(monkeypatch, cap_at_call, pool):
    """Resolve the CUDA core but transport on the host, noting the cap per call."""
    simulate = case_driver.simulate_trajectories

    def transport(*args, **kwargs):
        cap_at_call.append(pool.limit)
        kwargs.update(transport_core="per-electron", keep_segments_on_device=False)
        return simulate(*args, **kwargs)

    monkeypatch.setattr(runner, "_case_transport_core", lambda *_: "cuda")
    monkeypatch.setattr(case_driver, "simulate_trajectories", transport)


def _fixed_case():
    return {key: value for key, value in _requested().items() if key != "adaptive_precision"}


def test_cuda_transport_sets_the_ceiling_before_its_first_call(monkeypatch, pool):
    caps = []
    _fake_cuda(monkeypatch, caps, pool)
    runner._transport_case(_fixed_case(), transport_core="cuda")
    assert caps and set(caps) == {CEILING}
    assert CEILING > BUDGET
    assert pool.sets == [CEILING]  # once per phase change


def test_spectrum_lowers_to_the_budget_and_transport_raises_again(pool):
    oom._ensure_transport_pool_limit()
    oom._ensure_pool_limit()
    oom._ensure_pool_limit()
    oom._ensure_transport_pool_limit()
    assert pool.sets == [CEILING, BUDGET, CEILING]


def test_streamed_adaptive_measure_pass_runs_capped(monkeypatch, pool):
    """The measure-only pass precedes the streamed pass's own cap (#1280)."""
    caps = []
    _fake_cuda(monkeypatch, caps, pool)
    run_case_streamed(_requested(), 40, transport_core="cuda")
    assert len(caps) > 2 and all(caps)
    assert set(caps) <= {CEILING, BUDGET}


def test_phase_caps_leave_a_scoped_cap_alone(pool):
    """Streamed block sizing caps the pool for its run; transport must not raise it."""
    oom._ensure_pool_limit()
    pool.set_bytes(4 * GIB)  # MemoryBlockSizer.pool_limit
    oom._ensure_transport_pool_limit()
    oom._ensure_pool_limit()
    assert pool.limit == 4 * GIB
    pool.set_bytes(BUDGET)  # the scope restores the previous cap
    oom._ensure_transport_pool_limit()
    assert pool.limit == CEILING


@pytest.mark.parametrize("share", [1, 2])
def test_explicit_pool_policy_keeps_transport_at_the_budget(monkeypatch, pool, share):
    monkeypatch.setattr(runner._RESOURCE_POLICY, "transport_ceiling", False)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_pool_share", share)
    oom._ensure_transport_pool_limit()
    assert pool.sets == [int(0.7 * CARD / share)]


def test_ceiling_is_shared_and_never_below_the_budget(monkeypatch, pool):
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu_pool_share", 2)
    assert oom._transport_pool_ceiling() == CEILING // 2
    monkeypatch.setattr(oom.BACKEND, "device_bytes_outside_pool", lambda: 12 * GIB)
    assert oom._transport_pool_ceiling() == int(0.7 * CARD / 2)


def test_cpu_transport_leaves_the_pool_uncapped(pool):
    runner._transport_case(_fixed_case(), transport_core="per-electron")
    assert pool.sets == []
    assert runner._RESOURCE_POLICY.pool_limit_phase is None
