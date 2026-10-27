"""Experimental quantity view of directional source densities on native nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import numpy as np

from .quantified import _check_energy_unit

if TYPE_CHECKING:
    from pint import Quantity


@dataclass(frozen=True, slots=True, eq=False)
class QuantifiedDirectionalSpectrum:
    """Independent CPU view from :meth:`DirectionalSpectrum.quantify`.

    ``density`` has axes (angle, photon energy). Angles are periodic point
    samples, never solid-angle bins. Energy conversion scales density inversely;
    converting the returned angle quantity does not change the per-sr measure.
    ``supported`` flags valid directions, retaining unsupported density NaNs.
    Each array property returns a fresh copy. The view snapshots native NumPy
    arrays, retains component identity, and does no scoring, resampling, device
    transfer or current/acceptance scaling. Case/provenance stay on the source.
    Numeric arrays and private constructor fields use degrees, eV and per-eV/sr.
    """

    _angle_deg: np.ndarray = field(repr=False)
    _energy_eV: np.ndarray = field(repr=False)
    _density_per_eV_sr: np.ndarray = field(repr=False)
    _supported: np.ndarray = field(repr=False)
    component: str
    _energy_unit: str = field(default="eV", repr=False)
    axis_names: ClassVar[tuple[str, str]] = ("angle", "photon_energy")
    sampling: ClassVar[Literal["nodes"]] = "nodes"
    measure: ClassVar[Literal["per_energy_per_steradian"]] = "per_energy_per_steradian"
    count_basis: ClassVar[Literal["photons_per_incident_electron"]] = (
        "photons_per_incident_electron"
    )
    normalization: ClassVar[Literal["source"]] = "source"

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        for name in ("_angle_deg", "_energy_eV", "_density_per_eV_sr", "_supported"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a CPU NumPy array")
            if value.dtype.kind not in "fiu" and not (
                name == "_supported" and value.dtype.kind == "b"
            ):
                raise TypeError(f"{name} must contain real numeric values")
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if self._angle_deg.ndim != 1 or self._energy_eV.ndim != 1:
            raise ValueError("angle and photon-energy coordinates must be one-dimensional")
        if self._density_per_eV_sr.shape != (self._angle_deg.size, self._energy_eV.size):
            raise ValueError("density must have shape (angle, photon_energy)")
        if self._supported.shape != self._angle_deg.shape or self._supported.dtype.kind != "b":
            raise ValueError("supported must be a boolean mask on the angle axis")

    @property
    def angle(self) -> Quantity:
        """Fresh angular point coordinates in degrees, independent of measure."""
        from ..units import Q_

        return Q_(self._angle_deg.copy(), "degree")

    @property
    def photon_energy(self) -> Quantity:
        """Fresh photon-energy nodes in the selected display unit."""
        from ..units import Q_

        return Q_(self._energy_eV.copy(), "eV").to(self._energy_unit)

    @property
    def density(self) -> Quantity:
        """Fresh density per display energy and sr; count basis stays semantic."""
        from ..units import Q_

        return Q_(self._density_per_eV_sr.copy(), "1 / eV / steradian").to(
            f"1 / ({self._energy_unit}) / steradian"
        )

    @property
    def supported(self) -> np.ndarray:
        """Fresh boolean mask, one entry per angular point."""
        return self._supported.copy()

    def to_energy(self, unit: str) -> Self:
        """Return a new view with coupled energy/density conversion.

        Native nodes, support and per-primary/per-sr normalization are retained.
        Non-energy unit strings raise ValueError; other inputs raise TypeError.
        """
        return type(self)(
            self._angle_deg,
            self._energy_eV,
            self._density_per_eV_sr,
            self._supported,
            self.component,
            unit,
        )
