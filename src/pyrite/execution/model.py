"""Data-only resource, work-unit, and execution-state contracts.

Work identifiers are *scheduling* identities, not content-addressable physics
cache keys. The optional ``case_keys`` must be provided by the campaign's
existing physics identity function (including external cross-section tables).
"""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class ExecutionState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Advisory requirements for a scheduler, not an isolation mechanism.

    CPU affinity, GPU assignment, memory limits, and CUDA/ROCm visibility must
    still be enforced by the batch system or the worker environment.
    ``accelerator='any'`` means no device-specific requirement is asserted.
    """

    cpu_cores: int = 1
    memory_bytes: int = 0
    gpu_count: int = 0
    gpu_memory_bytes: int = 0
    accelerator: str = "any"

    def __post_init__(self) -> None:
        if (
            isinstance(self.cpu_cores, bool)
            or not isinstance(self.cpu_cores, int)
            or self.cpu_cores < 1
        ):
            raise ValueError("cpu_cores must be a positive integer")
        for field in ("memory_bytes", "gpu_count", "gpu_memory_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.accelerator not in ("any", "cpu", "cuda", "rocm", "sycl"):
            raise ValueError("unsupported accelerator requirement")
        if self.accelerator == "cpu" and self.gpu_count:
            raise ValueError("cpu-only work cannot request GPUs")
        if self.accelerator in ("cuda", "rocm", "sycl") and not self.gpu_count:
            raise ValueError("device-specific work must request at least one GPU")
        if self.gpu_memory_bytes and not self.gpu_count:
            raise ValueError("gpu_memory_bytes requires gpu_count > 0")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Scheduler-level attempts; numerical OOM recovery stays in run_cases."""

    max_attempts: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts must be a positive integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")


@dataclass(frozen=True, slots=True)
class WorkUnit:
    """One contiguous case interval in a particular campaign ordering.

    Half-open ``[start, stop)`` indexes are not cross-campaign cache IDs.
    The coordinator owns manifests and result consolidation; workers will
    eventually publish isolated immutable artifacts keyed by ``case_keys``.
    """

    unit_id: str
    start: int
    stop: int
    resources: ResourceRequest = ResourceRequest()
    case_keys: tuple[str, ...] | None = None
    estimated_cost: float | None = None

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("unit_id must be nonempty")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("work unit must cover at least one nonnegative index")
        if self.case_keys is not None:
            if not isinstance(self.case_keys, tuple):
                raise ValueError("case_keys must be an immutable tuple")
            if len(self.case_keys) != self.stop - self.start or not all(
                isinstance(key, str) and key for key in self.case_keys
            ):
                raise ValueError("case_keys must align with the work interval")
        if self.estimated_cost is not None:
            if not isfinite(self.estimated_cost) or self.estimated_cost < 0:
                raise ValueError("estimated_cost must be finite and nonnegative")

    @property
    def size(self) -> int:
        return self.stop - self.start
