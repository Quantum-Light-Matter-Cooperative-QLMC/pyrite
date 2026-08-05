from __future__ import annotations

import types

import numpy as np
import pytest

from cxr_mc.montecarlo import _backend, runner
from cxr_mc.montecarlo._resources import GIB, admitted_chunk, resolve_resource_policy


def test_cpu_backend_contract_round_trip():
    backend = _backend.select_backend("cpu")
    value = backend.xp.asarray([1.0, 2.0])

    assert backend.device.backend == "cpu"
    assert backend.device.supports_fp64
    assert backend.is_array(value)
    np.testing.assert_array_equal(backend.to_cpu(value), [1.0, 2.0])
    assert backend.allocator_stats() == {
        "used_mib": None,
        "reserved_mib": None,
        "peak_mib": None,
    }


def test_explicit_unavailable_backend_errors(monkeypatch):
    def unavailable(expected=None):
        raise _backend.BackendUnavailableError(f"missing {expected}")

    monkeypatch.setattr(_backend, "_load_cupy", unavailable)

    with pytest.raises(_backend.BackendUnavailableError, match="missing rocm"):
        _backend.select_backend("rocm")


def test_auto_falls_back_to_cpu(monkeypatch):
    def unavailable(*args, **kwargs):
        raise _backend.BackendUnavailableError("missing")

    monkeypatch.setattr(_backend, "_load_cupy", unavailable)
    monkeypatch.setattr(_backend, "_load_sycl", unavailable)

    backend = _backend.select_backend("auto")
    assert backend.name == "cpu"
    assert backend.fallback_reason == "accelerator_unavailable: missing; missing"


def test_explicit_fp64_incompatible_backend_errors(monkeypatch):
    backend = _backend.NumPyBackend()
    backend.name = "sycl"
    backend.vendor = "intel"
    backend.device = _backend.DeviceInfo("sycl", "intel", "Arc", 4 * GIB, False)
    monkeypatch.setattr(_backend, "_load_sycl", lambda: backend)
    monkeypatch.setenv("CXR_FP64", "1")

    with pytest.raises(_backend.BackendUnavailableError, match="lacks fp64"):
        _backend.select_backend("sycl")


def test_sycl_contract_pins_creation_queue_and_round_trips():
    queue = types.SimpleNamespace(wait=lambda: None)
    device = types.SimpleNamespace(
        name="Intel Arc fake",
        global_mem_size=4 * GIB,
        max_compute_units=128,
        has_aspect_fp64=False,
        backend="backend_type.level_zero",
    )

    class FakeArray(np.ndarray):
        pass

    calls = []

    def asarray(value, **kwargs):
        calls.append(kwargs)
        return np.asarray(value).view(FakeArray)

    fake_dpnp = types.SimpleNamespace(
        ndarray=FakeArray,
        asarray=asarray,
        asnumpy=np.asarray,
        float32=np.float32,
        float64=np.float64,
    )
    fake_dpctl = types.SimpleNamespace(
        get_devices=lambda **kwargs: [device],
        SyclQueue=lambda selected: queue,
    )
    backend = _backend.SyclBackend(fake_dpnp, fake_dpctl)

    value = backend.xp.asarray([1.0])
    assert calls == [{"sycl_queue": queue}]
    assert backend.is_array(value)
    np.testing.assert_array_equal(backend.to_cpu(value), [1.0])
    assert backend.device.name == "Intel Arc fake"
    assert not backend.device.supports_fp64


def test_four_gib_auto_policy_is_conservative():
    backend = _backend.NumPyBackend()
    backend.device = _backend.DeviceInfo("sycl", "intel", "Arc", 4 * GIB, False)

    policy = resolve_resource_policy(backend, "auto")

    assert policy.name == "conservative"
    assert policy.device_budget_bytes == 2 * GIB
    assert policy.device_reserve_bytes == 2 * GIB


def test_preallocation_admission_caps_or_errors():
    assert (
        admitted_chunk(
            requested_chunk=40_000,
            bins=2_000,
            itemsize=4,
            budget_bytes=96_000_000,
        )
        == 4_000
    )
    with pytest.raises(_backend.BackendResourceError, match="cannot admit"):
        admitted_chunk(
            requested_chunk=40_000,
            bins=20_000,
            itemsize=8,
            budget_bytes=1_000_000,
        )


def test_cpu_fallback_requires_host_ram_admission(monkeypatch):
    monkeypatch.setattr(runner, "_TOTAL_MEM", 8_000)
    monkeypatch.setattr(runner, "_available_mem_mb", lambda: 4_000)
    monkeypatch.setattr(runner, "_WORKER_MEM_MB", 6_144)

    with pytest.raises(_backend.BackendResourceError, match="cannot admit CPU fallback"):
        runner._admit_cpu_fallback()

def test_cpu_spectrum_backend_updates_active_itemsize() -> None:
    import numpy as np

    from cxr_mc.montecarlo import runner

    with runner._cpu_spectrum_backend():
        assert runner._GPU is False
        assert runner._spectrum_mod.xp is np
        assert runner._spectrum_mod.REAL is np.float64
        assert runner._real_bytes() == 8


import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.intel_sycl,
]


@pytest.mark.skipif(
    os.environ.get("CXR_RUN_INTEL_SYCL_TESTS") != "1",
    reason="set CXR_RUN_INTEL_SYCL_TESTS=1 to run Intel SYCL hardware tests",
)
def test_intel_machine_selects_sycl_backend() -> None:
    script = textwrap.dedent(
        """
        import sys

        import cxr_mc.config
        from cxr_mc.montecarlo._backend import BACKEND

        assert BACKEND.name == "sycl", (
            f"Expected SYCL backend, got {BACKEND.name!r}. "
            f"Fallback reason: {BACKEND.fallback_reason!r}"
        )

        assert not any(
            name == "cupy"
            or name.startswith("cupy.")
            or name == "cupyx"
            or name.startswith("cupyx.")
            for name in sys.modules
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Intel SYCL backend test failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_nsys_helpers_are_noops_for_sycl(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_GPU", True)
    monkeypatch.setattr(runner, "_NSYS", True)
    monkeypatch.setattr(
        runner,
        "BACKEND",
        SimpleNamespace(name="sycl"),
    )

    before = set(sys.modules)

    with runner._nsys_range("test"):
        pass

    runner._nsys_push("test")
    runner._nsys_pop()

    newly_loaded = set(sys.modules) - before

    assert "cupy" not in newly_loaded
    assert "cupyx" not in newly_loaded
    assert not any(name.startswith("cupy.") for name in newly_loaded)
    assert not any(name.startswith("cupyx.") for name in newly_loaded)

def test_cpu_spectrum_backend_restores_backend() -> None:
    from cxr_mc.montecarlo import runner

    original = (
        runner._GPU,
        runner._spectrum_mod.xp,
        runner._spectrum_mod.REAL,
    )

    with runner._cpu_spectrum_backend():
        pass

    assert (
        runner._GPU,
        runner._spectrum_mod.xp,
        runner._spectrum_mod.REAL,
    ) == original