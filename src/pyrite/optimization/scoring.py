"""Fixed physical-window objectives for the standard sweep record contract."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class FixedWindowYield:
    """Score one scalar run's incoherent PXR/CBS yield in a fixed energy band.

    ``lower_eV`` and ``upper_eV`` must lie inside the stored evaluation nodes.
    The selected ``spec`` density is piecewise linearly interpolated at the
    endpoints and integrated in physical eV by ``radiation_band_metrics``.
    Source: energy-grid semantics, ``eq-grid-physical-integral``. Assumptions:
    node-sampled, separate PXR/CBS density per incident electron/eV/sr. A
    constant density c gives c * (upper_eV - lower_eV); zero density gives zero.
    Validation: optimization-fixed-window-yield

    No current, sampled-history count, detector response or solid-angle factor
    is applied. This is a directional source objective, not accepted photons.
    Characteristic, channeling, transition, coherent and continuum arrays are
    excluded. Bin-mean line records and two-beam joint-field corrections are
    refused rather than reinterpreted as sampled PXR/CBS source densities.
    """

    lower_eV: float
    upper_eV: float
    name: str = "window_yield"
    units: ClassVar[str] = "photons/(sr incident electron)"

    def __post_init__(self) -> None:
        if not np.isfinite([self.lower_eV, self.upper_eV]).all() or not (
            0 < self.lower_eV < self.upper_eV
        ):
            raise ValueError("window requires finite positive lower_eV < upper_eV")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("objective name must be nonempty")

    @property
    def evaluation_id(self) -> str:
        """Stable scorer identity; changes with either bound or observable name."""
        return (
            f"fixed-window-cxr-nodes-v1:{float(self.lower_eV).hex()}:"
            f"{float(self.upper_eV).hex()}:{self.name}"
        )

    def __call__(self, results: Mapping[str, Any]) -> dict[str, float]:
        """Reduce exactly one record in ``run_sweep``'s nested result store."""
        from ..results.radiation import radiation_band_metrics, two_beam_record
        from ..results.selection import records

        selected = records(results)
        if len(selected) != 1:
            raise ValueError("fixed-window scoring requires exactly one result record")
        record = selected[0]
        if record.get("case", {}).get("line_quadrature") is not None:
            raise ValueError("fixed-window scoring requires node-sampled line density")
        if two_beam_record(record):
            raise ValueError("two-beam joint-field correction is not a PXR/CBS source density")
        energy = np.asarray(record["E_grid"], dtype=float)
        density = np.asarray(record["spec"], dtype=float)
        if energy.ndim != 1 or density.shape != energy.shape or not np.isfinite(density).all():
            raise ValueError("line density must be finite and match a one-dimensional energy grid")
        # Pass only the requested source component: auxiliary arrays may have
        # other quadrature semantics and must not affect this objective.
        band = radiation_band_metrics(
            {"E_grid": energy, "spec": density}, (self.lower_eV, self.upper_eV)
        )
        return {self.name: band["yields"]["line"]}
