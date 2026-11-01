"""Reusable typed numerical sections and local kernel controls (#680).

These are *values*, not schedulers. ``pyrite.execution.Executor`` and
``ResourceRequest`` (#681) orchestrate case dispatch independently. The
legacy ``Numerics`` fields remain the canonical lowering boundary during the
v1 compatibility window; no new case schema is introduced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ._energy_bins import EnergyBins
from ._numerics import CxrReflections


@dataclass(frozen=True)
class TransportNumerics:
    """Electron transport integration scheme, distinct from interaction physics.

    ``max_dE_frac=0`` disables fractional-loss substepping. These values
    affect the scientific approximation and remain part of case identity.
    """

    energy_model: Literal["frozen", "midpoint"] = "midpoint"
    max_dE_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.energy_model not in ("frozen", "midpoint"):
            raise ValueError("energy_model must be 'frozen' or 'midpoint'")
        value = self.max_dE_frac
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("max_dE_frac must be finite and non-negative")
        if value and self.energy_model != "midpoint":
            raise ValueError("max_dE_frac > 0 requires energy_model='midpoint'")


@dataclass(frozen=True)
class RadiationNumerics:
    """Radiation spectral-grid and CXR quadrature choices.

    Photon grids and truncation choices may affect scientific results; they
    must not be treated as execution-only options. ``energy_bins`` retains
    the same established :class:`EnergyBins` contract.
    """

    cxr_reflections: CxrReflections = field(default_factory=CxrReflections)
    energy_bins: EnergyBins | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cxr_reflections, CxrReflections):
            raise TypeError("cxr_reflections must be a CxrReflections")
        if self.energy_bins is not None and not isinstance(self.energy_bins, EnergyBins):
            raise TypeError("energy_bins must be an EnergyBins or None")


TRANSPORT_SCHEDULES = ("full-history", "continuation", "auto")
#: CUDA transport threads per block (mirrors ``TransportKernelConfig``).
KERNEL_THREADS = (32, 64, 128, 256, 512, 1024)
#: CXR reduction kernel variants (``cxr_jit_kernel._kernel_<k>e``).
CXR_ENERGIES_PER_BLOCK = (1, 2, 3, 4)


@dataclass(frozen=True)
class TransportKernelTuning:
    """Per-electron/CUDA transport batching and schedule controls (experimental, #776).

    ``schedule`` is ``"full-history"``, ``"continuation"`` or ``"auto"``
    (#799) and takes precedence over ``PYRITE_MC_TRANSPORT_SCHEDULE``;
    ``None`` defers to that variable, then ``"auto"`` (the #799 default).
    ``continuation_rows`` is the continuation row quantum: the trajectory rows
    each active history may emit per launch (``None``: 256). It counts emitted
    rows, not wall-clock time, and no launch ends early when its first history
    finishes; it requires ``"continuation"`` or ``"auto"``.
    ``electron_batch`` caps the electrons (or concurrent continuation
    histories) per transport launch and takes precedence over
    ``PYRITE_MC_ELECTRON_BATCH_MAX`` (``None``: that variable, then 65536).
    ``scratch_mib`` is the device scratch budget, in MiB, that full-history
    batches and continuation lanes are sized from (``None``: 512); it, not
    ``electron_batch``, usually sets how many histories share one launch.
    ``kernel_threads`` is the CUDA transport launch's threads per block, one of
    32, 64, 128, 256, 512 or 1024 (``None``: 32, #875). Both are per scene and GPU
    tuning knobs (#825). These fields leave transport bitwise-identical.
    ``precision`` is the exact CUDA kernel's arithmetic (#838): ``"mixed"``
    (FP32 table lookups and sampling, FP64 accumulation), ``"fp64"``, or
    ``None`` for ``PYRITE_MC_TRANSPORT_PRECISION``, then the default
    (``montecarlo.transport.precision.DEFAULT_TRANSPORT_PRECISION``); other
    cores always run FP64. It changes trajectories at rounding scale and moves
    observables by less than one third of the #510 statistical budget, so,
    like the backend, it is recorded in run provenance rather than case
    identity. No field enters case identity. Batch executors reject
    non-default values and use the variables.
    """

    schedule: Literal["full-history", "continuation", "auto"] | None = None
    continuation_rows: int | None = None
    electron_batch: int | None = None
    scratch_mib: int | None = None
    kernel_threads: int | None = None
    precision: Literal["fp64", "mixed"] | None = None

    def __post_init__(self) -> None:
        if self.precision is not None and self.precision not in ("fp64", "mixed"):
            raise ValueError("precision must be 'fp64', 'mixed' or None")
        if self.schedule is not None and self.schedule not in TRANSPORT_SCHEDULES:
            raise ValueError(f"schedule must be one of {TRANSPORT_SCHEDULES} or None")
        threads = self.kernel_threads
        if threads is not None and (type(threads) is not int or threads not in KERNEL_THREADS):
            raise ValueError(f"kernel_threads must be one of {KERNEL_THREADS} or None")
        for name in ("continuation_rows", "electron_batch", "scratch_mib"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        rows = self.continuation_rows
        if rows is not None and self.schedule not in TRANSPORT_SCHEDULES[1:]:
            raise ValueError("continuation_rows requires schedule='continuation' or 'auto'")


@dataclass(frozen=True)
class ExecutionConfig:
    """Local transport-kernel selection and spectrum batch sizes.

    This is **not** a ``pyrite.execution.Executor`` and does not choose a
    SLURM/local scheduler, process-pool size or resource request. The latter
    are owned by #681's execution orchestration API. These controls still
    lower through existing case fields; their effect on cache identity is
    **not** changed or presumed invariant in this compatibility slice.
    ``transport`` holds the experimental CUDA transport schedule
    (:class:`TransportKernelTuning`), which never enters case identity.
    ``cxr_threads`` (32-1024, ``None``: 512) and ``cxr_energies_per_block``
    (1-4, ``None``: 3) set the CUDA CXR line-reduction launch on the
    node-quadrature route: threads per block and how many energy nodes one
    block reduces (#825). They change summation grouping only (float
    rounding), never case identity, and their best values depend on the scene
    and GPU.
    ``stream_block_electrons`` (experimental, #865; ``None``: off) opts
    :func:`pyrite.simulate` into live streamed scoring: transport runs in
    whole-electron blocks of this many electrons, each block is scored and
    released, and peak memory follows the block size times the segments per
    electron instead of the electron count. Spectra equal the default path up
    to floating-point summation order; transport counts are exact; it never
    enters case identity. Only scalar-detector, in-process runs of additive
    spectra are supported; other cases and entry points raise ``ValueError``
    before transport. Where the joined run fits in memory it is faster.
    """

    transport_core: str = "auto"
    cxr_chunk: int | None = None
    brem_chunk: int | None = None
    transport: TransportKernelTuning = field(default_factory=TransportKernelTuning)
    cxr_threads: int | None = None
    cxr_energies_per_block: int | None = None
    stream_block_electrons: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transport_core, str) or not self.transport_core:
            raise ValueError("transport_core must be a nonempty string")
        for name in ("cxr_chunk", "brem_chunk", "stream_block_electrons"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(self.transport, TransportKernelTuning):
            raise TypeError("transport must be a TransportKernelTuning")
        for name, allowed in (
            ("cxr_threads", KERNEL_THREADS),
            ("cxr_energies_per_block", CXR_ENERGIES_PER_BLOCK),
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value not in allowed):
                raise ValueError(f"{name} must be one of {allowed} or None")

    def spectrum_launch_default(self) -> bool:
        """True when both CXR launch controls keep the built-in default."""
        return self.cxr_threads is None and self.cxr_energies_per_block is None

    def __setstate__(self, state: dict[str, object]) -> None:
        # Pickles from before #776 carry no ``transport`` entry, before #825 no
        # CXR launch controls, before #865 no streaming block.
        state.setdefault("transport", TransportKernelTuning())
        state.setdefault("cxr_threads", None)
        state.setdefault("cxr_energies_per_block", None)
        state.setdefault("stream_block_electrons", None)
        self.__dict__.update(state)


# Preferred public spelling (#702). No second dataclass or pickle schema.
KernelTuning = ExecutionConfig
