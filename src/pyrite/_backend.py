"""Portable array-backend selection for every device-dispatching kernel.

Selection is controlled by ``PYRITE_MC_BACKEND=auto|cpu|cuda|rocm|sycl``.
Automatic selection probes CUDA/ROCm CuPy, then Intel SYCL through dpnp, and
finally NumPy. Explicit accelerator requests fail instead of changing device.

This sits at the package root rather than under ``montecarlo`` because the
physics core dispatches on it too: ``materials.attenuation`` returns absorption
on the same device as its input. Nothing here is Monte Carlo specific, and
keeping it below the domain packages is what lets ``materials`` stay out of the
``montecarlo`` import cycle. Importing this module runs the accelerator probe,
so callers that must stay cheap at import time reach for it inside the function
that needs a device.
"""

import logging
import warnings
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np

from ._env import env_value

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("auto", "cpu", "cuda", "rocm", "sycl")


class BackendError(RuntimeError):
    """Base error for backend selection and execution."""


class BackendUnavailableError(BackendError):
    """Requested accelerator backend is unavailable or incompatible."""


class BackendResourceError(BackendError):
    """Resolved backend cannot safely admit the requested work."""


@dataclass(frozen=True)
class DeviceInfo:
    """Stable device identity and capabilities used by runtime planning."""

    backend: str
    vendor: str
    name: str
    total_memory_bytes: int | None
    supports_fp64: bool


class ArrayBackend:
    """Small adapter contract consumed by device spectrum kernels."""

    name = "cpu"
    vendor = "numpy"
    xp: Any = np
    array_type: type[Any] = np.ndarray
    oom_exceptions: tuple[type[BaseException], ...] = ()

    def __init__(self, device: DeviceInfo, *, fallback_reason: str | None = None):
        self.device = device
        self.fallback_reason = fallback_reason

    def to_cpu(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def is_array(self, value: Any) -> bool:
        return isinstance(value, self.array_type)

    def is_oom_error(self, error: BaseException) -> bool:
        """Return whether *error* is an allocation failure for this backend."""
        return isinstance(error, self.oom_exceptions)

    def synchronize(self) -> None:
        return None

    def set_memory_limit(self, fraction: float) -> None:
        return None

    def release_memory(self) -> None:
        return None

    def allocator_stats(self) -> dict[str, float | None]:
        return {"used_mib": None, "reserved_mib": None, "peak_mib": None}


class NumPyBackend(ArrayBackend):
    def __init__(self, *, fallback_reason: str | None = None):
        super().__init__(
            DeviceInfo(
                backend="cpu",
                vendor="numpy",
                name="host CPU",
                total_memory_bytes=None,
                supports_fp64=True,
            ),
            fallback_reason=fallback_reason,
        )


class CuPyBackend(ArrayBackend):
    """CuPy adapter for either CUDA or a ROCm/HIP build."""

    def __init__(self, module: ModuleType, *, expected: str | None = None):
        self.cp = module
        is_hip = bool(getattr(module.cuda.runtime, "is_hip", False))
        self.name = "rocm" if is_hip else "cuda"
        self.vendor = "amd" if is_hip else "nvidia"
        if expected is not None and self.name != expected:
            raise BackendUnavailableError(
                f"PYRITE_MC_BACKEND={expected} requested, but installed CuPy targets {self.name}; "
                f"install pyrite-xray[{('amd' if expected == 'rocm' else 'nvidia')}] "
                "in a clean environment"
            )
        if module.cuda.runtime.getDeviceCount() < 1:
            raise BackendUnavailableError(f"no usable {self.name} device")
        props = module.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", f"{self.name} device")
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode(errors="replace")
        total = int(module.cuda.runtime.memGetInfo()[1])
        super().__init__(
            DeviceInfo(
                backend=self.name,
                vendor=self.vendor,
                name=str(raw_name),
                total_memory_bytes=total,
                supports_fp64=True,
            )
        )
        self.xp = module
        self.array_type = module.ndarray
        self.oom_exceptions = (module.cuda.memory.OutOfMemoryError,)
        self._runtime_memory_status = getattr(module.cuda.runtime, "errorMemoryAllocation", 2)
        self._peak_bytes = 0

    def to_cpu(self, value: Any) -> np.ndarray:
        return self.cp.asnumpy(value) if self.is_array(value) else np.asarray(value)

    def is_oom_error(self, error: BaseException) -> bool:
        """Recognize pool OOMs and delayed CUDA/HIP allocation failures.

        CuPy normally raises ``cuda.memory.OutOfMemoryError`` while allocating
        an array. Device work is asynchronous, however, so an allocation
        failure inside an already-launched operation can first surface at a
        later stream synchronization as ``CUDARuntimeError``. Only the runtime
        memory-allocation status/message is promoted to a retryable OOM; other
        CUDA runtime failures remain hard errors.
        """
        if super().is_oom_error(error):
            return True
        error_type = type(error)
        runtime_error = error_type.__name__ in {"CUDARuntimeError", "HIPRuntimeError"} and (
            error_type.__module__.startswith("cupy")
            or error_type is getattr(self.cp.cuda.runtime, "CUDARuntimeError", None)
        )
        if not runtime_error:
            return False
        if getattr(error, "status", None) == self._runtime_memory_status:
            return True
        message = str(error).lower().replace("_", "")
        return "errormemoryallocation" in message or "out of memory" in message

    def synchronize(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()

    def set_memory_limit(self, fraction: float) -> None:
        if fraction > 0:
            self.cp.get_default_memory_pool().set_limit(fraction=fraction)

    def release_memory(self) -> None:
        self.cp.get_default_memory_pool().free_all_blocks()

    def allocator_stats(self) -> dict[str, float | None]:
        pool = self.cp.get_default_memory_pool()
        used = int(pool.used_bytes())
        reserved = int(pool.total_bytes())
        self._peak_bytes = max(self._peak_bytes, reserved)
        return {
            "used_mib": used / (1 << 20),
            "reserved_mib": reserved / (1 << 20),
            "peak_mib": self._peak_bytes / (1 << 20),
        }


class _SyclNamespace:
    """dpnp namespace wrapper pinning array creation to one SYCL queue."""

    _CREATORS = {
        "array",
        "arange",
        "asarray",
        "empty",
        "empty_like",
        "eye",
        "full",
        "full_like",
        "linspace",
        "logspace",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }

    def __init__(self, module: ModuleType, queue: Any):
        self._module = module
        self._queue = queue

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._module, name)
        if name not in self._CREATORS:
            return value

        def create(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("sycl_queue", self._queue)
            return value(*args, **kwargs)

        return create


class SyclBackend(ArrayBackend):
    """Intel dpnp/dpctl adapter with queue-pinned device allocations."""

    name = "sycl"
    vendor = "intel"

    def __init__(self, dpnp: ModuleType, dpctl: ModuleType):
        selector = env_value("PYRITE_MC_SYCL_DEVICE")
        try:
            if selector:
                devices = [dpctl.SyclDevice(selector)]
            else:
                devices = [
                    device
                    for device in dpctl.get_devices(device_type="gpu")
                    if str(device.backend).endswith("level_zero")
                ]
                if not devices:
                    devices = list(dpctl.get_devices(device_type="gpu"))
            if not devices:
                raise BackendUnavailableError("no usable SYCL GPU")
            # Prefer compute units before reported memory: integrated adapters
            # report host RAM as global memory, while the Arc reports real VRAM.
            device = max(
                devices,
                key=lambda item: (
                    int(getattr(item, "max_compute_units", 0)),
                    int(getattr(item, "global_mem_size", 0)),
                    str(item),
                ),
            )
            queue = dpctl.SyclQueue(device)
        except BackendUnavailableError:
            raise
        except Exception as error:
            raise BackendUnavailableError(f"cannot create SYCL GPU queue: {error}") from error
        supports_fp64 = bool(getattr(device, "has_aspect_fp64", False))
        super().__init__(
            DeviceInfo(
                backend="sycl",
                vendor="intel",
                name=str(getattr(device, "name", device)),
                total_memory_bytes=int(getattr(device, "global_mem_size", 0)) or None,
                supports_fp64=supports_fp64,
            )
        )
        self.dpnp = dpnp
        self.queue = queue
        self.xp = _SyclNamespace(dpnp, queue)
        self.array_type = dpnp.ndarray
        # dpctl/dpnp do not expose one stable portable allocation exception.
        self.oom_exceptions = (MemoryError,)

    def to_cpu(self, value: Any) -> np.ndarray:
        return self.dpnp.asnumpy(value) if self.is_array(value) else np.asarray(value)

    def synchronize(self) -> None:
        self.queue.wait()


def _load_cupy(expected: str | None = None) -> CuPyBackend:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
            import cupy

        return CuPyBackend(cupy, expected=expected)
    except BackendUnavailableError:
        raise
    except Exception as error:
        raise BackendUnavailableError(f"CuPy {expected or 'GPU'} probe failed: {error}") from error


def _load_sycl() -> SyclBackend:
    try:
        dpctl = import_module("dpctl")
        dpnp = import_module("dpnp")

        return SyclBackend(dpnp, dpctl)
    except BackendUnavailableError:
        raise
    except Exception as error:
        raise BackendUnavailableError(f"Intel SYCL probe failed: {error}") from error


def select_backend(requested: str | None = None) -> ArrayBackend:
    """Resolve requested backend; ``auto`` alone may fall back to NumPy."""

    requested = (requested or env_value("PYRITE_MC_BACKEND", "auto")).strip().lower()
    if requested not in _VALID_BACKENDS:
        raise BackendUnavailableError(
            f"PYRITE_MC_BACKEND must be one of {', '.join(_VALID_BACKENDS)}; got {requested!r}"
        )
    if requested == "cpu":
        return NumPyBackend()
    if requested in ("cuda", "rocm"):
        backend = _load_cupy(requested)
    elif requested == "sycl":
        backend = _load_sycl()
    else:
        backend = None
        failures = []
        for loader in (_load_cupy, _load_sycl):
            try:
                backend = loader()
                break
            except BackendUnavailableError as error:
                failures.append(str(error))
                logger.debug("accelerator probe skipped: %s", error)
        if backend is None:
            return NumPyBackend(fallback_reason="accelerator_unavailable: " + "; ".join(failures))
    if env_value("PYRITE_FP64") == "1" and not backend.device.supports_fp64:
        if requested == "auto":
            logger.warning("%s lacks fp64; PYRITE_FP64=1 selects CPU NumPy", backend.device.name)
            return NumPyBackend(fallback_reason=f"unsupported_fp64: {backend.device.name}")
        raise BackendUnavailableError(
            f"PYRITE_MC_BACKEND={backend.name} device {backend.device.name!r} lacks fp64; "
            "unset PYRITE_FP64 or select PYRITE_MC_BACKEND=cpu"
        )
    return backend


BACKEND = select_backend()
xp = BACKEND.xp
cp: ModuleType | None = BACKEND.cp if isinstance(BACKEND, CuPyBackend) else None
_GPU = BACKEND.name != "cpu"
REAL = xp.float32 if (_GPU and env_value("PYRITE_FP64") != "1") else xp.float64


def _to_cpu(value: Any) -> np.ndarray:
    """Move an active-backend array to NumPy."""

    return BACKEND.to_cpu(value)


def is_device_array(value: Any) -> bool:
    """Return whether *value* belongs to the selected accelerator backend."""

    return _GPU and BACKEND.is_array(value)


def array_namespace(*values: Any) -> Any:
    """Array module that OWNS *values*, not the module-global one.

    ``xp`` names the SELECTED backend, which is the right namespace for a
    kernel fed by the staged device copy of a case. It is the wrong one for a
    namespace-neutral leaf helper -- pure gather/blend algebra with no device
    state -- called on host arrays while an accelerator is selected: CuPy's
    ufuncs reject NumPy operands rather than uploading them, so the helper
    would fail on data that never needed a device at all. Resolving the
    namespace from the operands keeps such a helper exact on either kind of
    input and performs no transfer in either direction; host arrays stay on the
    host and device arrays stay on the device.

    Mixed operands still resolve to the accelerator and still raise there. That
    is deliberate: a caller holding one host and one device array has an
    unstaged seam, and hiding it behind an implicit upload is what this returns
    a namespace to avoid.
    """

    return xp if any(is_device_array(value) for value in values) else np
