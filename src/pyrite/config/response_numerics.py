"""Numerical construction settings for downstream detector-response models.

This value is deliberately independent of detector geometry, readout exposure,
and simulation execution. Its fields normalize units at the public boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

from .._units import normalize_unit


@dataclass(frozen=True)
class Timepix3ResponseNumerics:
    """Response-matrix approximation, not Timepix3 sensor hardware.

    ``dE_mc`` and ``dE_out`` are input and native measured energy-bin widths
    in eV; they are not acquisition reporting bins or ADC quantization.
    ``n_mc`` samples clustered events *per input energy* to construct the
    response matrix. ``seed`` selects its Monte Carlo realization and differs
    from the acquisition Poisson realization seed. Neither selects a transport
    kernel, scheduler, or angular sampling resolution.
    """

    dE_mc: float = 50.0
    dE_out: float = 25.0
    n_mc: int = 60000
    seed: int = 0

    def __post_init__(self) -> None:
        for field_name in ("dE_mc", "dE_out"):
            value = normalize_unit(
                getattr(self, field_name),
                "eV",
                field=f"Timepix3ResponseNumerics.{field_name}",
                expected="energy",
            )
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number in eV")
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
            # Keep legacy int-vs-float identity exactly as the caller supplied it.
            object.__setattr__(self, field_name, value)
        for field_name, minimum in (("n_mc", 1), ("seed", 0)):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            if value < minimum:
                raise ValueError(f"{field_name} must be >= {minimum}")
            object.__setattr__(self, field_name, int(value))


__all__ = ["Timepix3ResponseNumerics"]
