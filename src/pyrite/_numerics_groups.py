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


@dataclass(frozen=True)
class ExecutionConfig:
    """Local transport-kernel selection and spectrum batch sizes.

    This is **not** a ``pyrite.execution.Executor`` and does not choose a
    SLURM/local scheduler, process-pool size or resource request. The latter
    are owned by #681's execution orchestration API. These controls still
    lower through existing case fields; their effect on cache identity is
    **not** changed or presumed invariant in this compatibility slice.
    """

    transport_core: str = "auto"
    cxr_chunk: int | None = None
    brem_chunk: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transport_core, str) or not self.transport_core:
            raise ValueError("transport_core must be a nonempty string")
        for name in ("cxr_chunk", "brem_chunk"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
