"""Numerics of the opt-in full-fiber CXR intensity average (#1480).

A :class:`FiberAverage` sits beside the mosaic controls on
:class:`pyrite.config.CxrReflections`. It selects *how* the full reciprocal
domain of a fiber-textured crystal is truncated and roll-averaged; the
physics (projected thermal tensor, intensity rather than amplitude average)
belongs to ``montecarlo.spectrum.cxr.fiber``. ``None`` on ``CxrReflections``
keeps the historical basal-only domain and adds nothing to case identity.

Validation: cxr-fiber-intensity-average
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral


def _positive_count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class FiberAverage:
    """Reciprocal cutoff and roll quadrature of the full-fiber CXR average.

    Parameters
    ----------
    cutoff_invang
        Reciprocal-vector magnitude cutoff ``|g|`` [1/Angstrom] of the full
        (non-basal included) reflection domain. Required: #1480 found the
        full-fiber yield still growing at 8-10 1/Angstrom, so no value is a
        converged default. The scorer refuses cutoffs beyond the form-factor
        range (75.4 1/Angstrom).
    azimuth_count
        Midpoint roll nodes on one 60-degree sector about the fiber (c) axis;
        each node is a distinct crystallite orientation. Integrated yields
        converged to about 0.5 % by 8 nodes in the #1480 scenes; 32 leaves
        margin for narrow line shapes.
    reflection_batch_size
        Reflections scored per coupling-table batch. Bounds memory only; it
        is not a physics truncation and changes float summation order only.

    Basal ``(00l)`` rows are roll invariant and are scored once; mosaic
    quadrature (``mosaic_nodes``, ``mosaic_route``) applies on top of each
    roll node.
    """

    cutoff_invang: float
    azimuth_count: int = 32
    reflection_batch_size: int = 256

    def __post_init__(self) -> None:
        cutoff = self.cutoff_invang
        if (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, int | float)
            or not math.isfinite(cutoff)
            or cutoff <= 0
        ):
            raise ValueError("cutoff_invang must be positive and finite")
        _positive_count(self.azimuth_count, "azimuth_count")
        _positive_count(self.reflection_batch_size, "reflection_batch_size")
