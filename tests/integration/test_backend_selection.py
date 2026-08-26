from __future__ import annotations

import types

import numpy as np
import pytest

from pyrite import _backend
from pyrite._env import env_value
from pyrite.montecarlo import runner
from pyrite.montecarlo._resources import GIB, admitted_chunk, resolve_resource_policy


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
    monkeypatch.setenv("PYRITE_FP64", "1")

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


def test_numpy_backend_noop_device_controls():
    backend = _backend.NumPyBackend()

    assert backend.synchronize() is None
    assert backend.set_memory_limit(0.5) is None


def _fake_cupy_module(*, is_hip=False, device_count=1, name=b"Fake GPU", total_mem=8 * (1 << 30)):
    class FakeArray(np.ndarray):
        pass

    pool_calls: dict[str, object] = {}

    class FakePool:
        def set_limit(self, fraction):
            pool_calls["limit"] = fraction

        def free_all_blocks(self):
            pool_calls["freed"] = True

        def used_bytes(self):
            return 512

        def total_bytes(self):
            return 1024

    class FakeStream:
        def synchronize(self):
            pool_calls["synced"] = True

    runtime = types.SimpleNamespace(
        is_hip=is_hip,
        getDeviceCount=lambda: device_count,
        getDeviceProperties=lambda index: {"name": name},
        memGetInfo=lambda: (0, total_mem),
    )
    cuda = types.SimpleNamespace(
        runtime=runtime,
        memory=types.SimpleNamespace(OutOfMemoryError=MemoryError),
        get_current_stream=lambda: FakeStream(),
    )
    module = types.SimpleNamespace(
        cuda=cuda,
        ndarray=FakeArray,
        asnumpy=lambda value: np.asarray(value),
        get_default_memory_pool=lambda: FakePool(),
    )
    return module, FakeArray, pool_calls


def test_cupy_backend_cuda_contract_round_trip():
    module, FakeArray, pool_calls = _fake_cupy_module()
    backend = _backend.CuPyBackend(module)

    assert backend.name == "cuda"
    assert backend.vendor == "nvidia"
    assert backend.device.name == "Fake GPU"
    assert backend.device.total_memory_bytes == 8 * (1 << 30)

    value = np.asarray([1.0, 2.0]).view(FakeArray)
    assert backend.is_array(value)
    np.testing.assert_array_equal(backend.to_cpu(value), [1.0, 2.0])
    np.testing.assert_array_equal(backend.to_cpu([1.0, 2.0]), [1.0, 2.0])

    backend.synchronize()
    backend.set_memory_limit(0.5)
    backend.release_memory()
    stats = backend.allocator_stats()

    assert pool_calls == {"synced": True, "limit": 0.5, "freed": True}
    assert stats["used_mib"] == 512 / (1 << 20)
    assert stats["reserved_mib"] == 1024 / (1 << 20)
    assert stats["peak_mib"] == 1024 / (1 << 20)


def test_cupy_backend_hip_reports_rocm_vendor():
    module, _, _ = _fake_cupy_module(is_hip=True, name="Fake AMD GPU")
    backend = _backend.CuPyBackend(module)

    assert backend.name == "rocm"
    assert backend.vendor == "amd"
    assert backend.device.name == "Fake AMD GPU"


def test_cupy_backend_rejects_mismatched_expected_target():
    module, _, _ = _fake_cupy_module(is_hip=False)

    with pytest.raises(_backend.BackendUnavailableError, match="targets cuda"):
        _backend.CuPyBackend(module, expected="rocm")


def test_cupy_backend_no_device_raises():
    module, _, _ = _fake_cupy_module(device_count=0)

    with pytest.raises(_backend.BackendUnavailableError, match="no usable cuda device"):
        _backend.CuPyBackend(module)


def test_load_cupy_reraises_backend_unavailable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))

    def boom(module, expected=None):
        raise _backend.BackendUnavailableError(f"missing {expected}")

    monkeypatch.setattr(_backend, "CuPyBackend", boom)

    with pytest.raises(_backend.BackendUnavailableError, match="missing rocm"):
        _backend._load_cupy(expected="rocm")


def _fake_sycl_modules(*, devices=None, get_devices=None, sycl_device=None):
    calls: dict[str, object] = {}

    def default_get_devices(**kwargs):
        calls["get_devices_kwargs"] = kwargs
        return devices if devices is not None else []

    fake_dpctl = types.SimpleNamespace(
        get_devices=get_devices or default_get_devices,
        SyclQueue=lambda selected: types.SimpleNamespace(
            wait=lambda: calls.setdefault("waited", True)
        ),
        SyclDevice=sycl_device or (lambda selector: types.SimpleNamespace(name=selector)),
    )
    fake_dpnp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda value, **kwargs: np.asarray(value),
        asnumpy=np.asarray,
    )
    return fake_dpctl, fake_dpnp, calls


def test_sycl_backend_selects_env_pinned_device(monkeypatch):
    monkeypatch.setenv("PYRITE_MC_SYCL_DEVICE", "level_zero:0")
    fake_dpctl, fake_dpnp, calls = _fake_sycl_modules(
        sycl_device=lambda selector: types.SimpleNamespace(
            name=selector, global_mem_size=0, max_compute_units=0, has_aspect_fp64=False
        )
    )

    backend = _backend.SyclBackend(fake_dpnp, fake_dpctl)

    assert backend.device.name == "level_zero:0"
    assert "get_devices_kwargs" not in calls


def test_sycl_backend_falls_back_when_no_level_zero_device(monkeypatch):
    monkeypatch.delenv("PYRITE_MC_SYCL_DEVICE", raising=False)
    device = types.SimpleNamespace(
        name="opencl device",
        global_mem_size=0,
        max_compute_units=0,
        has_aspect_fp64=False,
        backend="backend_type.opencl",
    )
    fake_dpctl, fake_dpnp, _ = _fake_sycl_modules(devices=[device])

    backend = _backend.SyclBackend(fake_dpnp, fake_dpctl)

    assert backend.device.name == "opencl device"


def test_sycl_backend_raises_when_no_device_found(monkeypatch):
    monkeypatch.delenv("PYRITE_MC_SYCL_DEVICE", raising=False)
    fake_dpctl, fake_dpnp, _ = _fake_sycl_modules(devices=[])

    with pytest.raises(_backend.BackendUnavailableError, match="no usable SYCL GPU"):
        _backend.SyclBackend(fake_dpnp, fake_dpctl)


def test_sycl_backend_wraps_unexpected_queue_errors(monkeypatch):
    monkeypatch.delenv("PYRITE_MC_SYCL_DEVICE", raising=False)
    device = types.SimpleNamespace(
        name="broken",
        global_mem_size=0,
        max_compute_units=0,
        has_aspect_fp64=False,
        backend="backend_type.level_zero",
    )

    def raising_queue(selected):
        raise RuntimeError("queue creation failed")

    fake_dpctl, fake_dpnp, _ = _fake_sycl_modules(devices=[device])
    fake_dpctl.SyclQueue = raising_queue

    with pytest.raises(_backend.BackendUnavailableError, match="cannot create SYCL GPU queue"):
        _backend.SyclBackend(fake_dpnp, fake_dpctl)


def test_sycl_backend_synchronize_waits_on_queue(monkeypatch):
    monkeypatch.delenv("PYRITE_MC_SYCL_DEVICE", raising=False)
    device = types.SimpleNamespace(
        name="wait-device",
        global_mem_size=0,
        max_compute_units=0,
        has_aspect_fp64=False,
        backend="backend_type.level_zero",
    )
    fake_dpctl, fake_dpnp, calls = _fake_sycl_modules(devices=[device])

    backend = _backend.SyclBackend(fake_dpnp, fake_dpctl)
    backend.synchronize()

    assert calls.get("waited") is True


def test_sycl_namespace_passes_through_non_creator_attributes():
    fake_dpnp = types.SimpleNamespace(float32=np.float32)
    namespace = _backend._SyclNamespace(fake_dpnp, queue=object())

    assert namespace.float32 is np.float32


def test_load_sycl_builds_backend_from_importable_modules(monkeypatch):
    import sys

    device = types.SimpleNamespace(
        name="load-sycl-device",
        global_mem_size=0,
        max_compute_units=0,
        has_aspect_fp64=False,
        backend="backend_type.level_zero",
    )
    fake_dpctl, fake_dpnp, _ = _fake_sycl_modules(devices=[device])
    monkeypatch.setitem(sys.modules, "dpctl", fake_dpctl)
    monkeypatch.setitem(sys.modules, "dpnp", fake_dpnp)

    backend = _backend._load_sycl()

    assert isinstance(backend, _backend.SyclBackend)
    assert backend.device.name == "load-sycl-device"


def test_load_sycl_reraises_backend_unavailable(monkeypatch):
    import sys

    fake_dpctl, fake_dpnp, _ = _fake_sycl_modules(devices=[])
    monkeypatch.setitem(sys.modules, "dpctl", fake_dpctl)
    monkeypatch.setitem(sys.modules, "dpnp", fake_dpnp)

    with pytest.raises(_backend.BackendUnavailableError, match="no usable SYCL GPU"):
        _backend._load_sycl()


def test_select_backend_rejects_unknown_value():
    with pytest.raises(_backend.BackendUnavailableError, match="must be one of"):
        _backend.select_backend("bogus")


def test_select_backend_explicit_accelerator_returns_resolved_backend(monkeypatch):
    fake = _backend.NumPyBackend()
    fake.name = "cuda"
    monkeypatch.setattr(_backend, "_load_cupy", lambda expected=None: fake)

    assert _backend.select_backend("cuda") is fake


def test_select_backend_auto_returns_first_available_accelerator(monkeypatch):
    fake = _backend.NumPyBackend()
    fake.name = "cuda"
    monkeypatch.setattr(_backend, "_load_cupy", lambda expected=None: fake)

    assert _backend.select_backend("auto") is fake


def test_select_backend_auto_falls_back_when_fp64_unsupported(monkeypatch):
    fake = _backend.NumPyBackend()
    fake.name = "cuda"
    fake.device = _backend.DeviceInfo("cuda", "nvidia", "fake", None, False)
    monkeypatch.setattr(_backend, "_load_cupy", lambda expected=None: fake)
    monkeypatch.setenv("PYRITE_FP64", "1")

    result = _backend.select_backend("auto")

    assert isinstance(result, _backend.NumPyBackend)
    assert result.fallback_reason == "unsupported_fp64: fake"


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
    monkeypatch.setattr(runner._RESOURCE_POLICY, "total_mem_mb", 8_000)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "available_mem_mb", lambda: 4_000)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "worker_mem_mb", 6_144)

    with pytest.raises(_backend.BackendResourceError, match="cannot admit CPU fallback"):
        runner._admit_cpu_fallback()


def test_cpu_spectrum_backend_updates_active_itemsize() -> None:
    import numpy as np

    from pyrite.montecarlo import runner

    with runner._cpu_spectrum_backend():
        assert runner._RESOURCE_POLICY.gpu is False
        assert runner._spectrum_mod.xp is np
        assert runner._spectrum_mod.REAL is np.float64
        assert runner._real_itemsize() == 8


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
    env_value("PYRITE_RUN_INTEL_SYCL_TESTS") != "1",
    reason="set PYRITE_RUN_INTEL_SYCL_TESTS=1 to run Intel SYCL hardware tests",
)
def test_intel_machine_selects_sycl_backend() -> None:
    script = textwrap.dedent(
        """
        import sys

        import pyrite.campaign.config
        from pyrite._backend import BACKEND

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

    # Drop the test session's CPU pin so the subprocess exercises the real
    # auto-selection probe against the machine's hardware.
    env = os.environ.copy()
    env.pop("PYRITE_MC_BACKEND", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Intel SYCL backend test failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_nsys_helpers_are_noops_for_sycl(monkeypatch) -> None:
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner._RESOURCE_POLICY, "nsys", True)
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
    from pyrite.montecarlo import runner

    original = (
        runner._RESOURCE_POLICY.gpu,
        runner._spectrum_mod.xp,
        runner._spectrum_mod.REAL,
    )

    with runner._cpu_spectrum_backend():
        pass

    assert (
        runner._RESOURCE_POLICY.gpu,
        runner._spectrum_mod.xp,
        runner._spectrum_mod.REAL,
    ) == original
