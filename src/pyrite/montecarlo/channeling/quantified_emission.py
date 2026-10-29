"""Opt-in quantity view of a diagnostic channeling-radiation spectrum (#755)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

from ..._units import _check_energy_unit

if TYPE_CHECKING:
    from pint import Quantity

    from .emission import ChannelingSpectrum

_ARRAYS = (
    "photon_energy_eV",
    "theta_rad",
    "phi_rad",
    "spectrum",
    "capture_spectrum",
    "stokes",
    "z_ang",
    "line_rates",
    "bound_population",
    "line_yield",
)


def _owned(value: object) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class QuantifiedChannelingSpectrum:
    """Quantity view of a :class:`ChannelingSpectrum` at the crystal exit.

    Obtain via :meth:`ChannelingSpectrum.quantify`. ``density`` and
    ``capture_density`` are node-sampled photons per displayed energy and
    steradian **per incident electron**, before any detector acceptance or
    response, on the source's original photon-energy nodes and broadcast
    direction shape ``S``. :meth:`to_energy` converts the energy axis and both
    densities together, so the integrated yield is unchanged; it never
    interpolates. ``line_yield`` is the depth-integrated photons per incident
    electron per catalog entry over ``4 pi`` (dimensionless count; see
    ``count_basis``). ``line_rates`` is the estimator per Å of depth,
    ``bound_population`` the bound fraction at each sampled depth and
    ``stokes`` the dimensionless ``(Q, U)``, NaN where the spectrum is zero.

    Arrays are copied at construction and every property returns a fresh
    quantity, so caller mutation cannot alter the source or this view.
    """

    _arrays: dict[str, np.ndarray] = field(repr=False)
    _energy_unit: str = field(default="eV", repr=False)
    sampling: Literal["nodes"] = "nodes"
    measure: Literal["per_energy_per_steradian"] = "per_energy_per_steradian"
    count_basis: Literal["photons_per_incident_electron"] = "photons_per_incident_electron"
    normalization: Literal["source"] = "source"

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        object.__setattr__(self, "_arrays", {name: _owned(self._arrays[name]) for name in _ARRAYS})

    @classmethod
    def from_spectrum(cls, spectrum: ChannelingSpectrum) -> Self:
        """Copy a numeric spectrum's arrays into a new quantity view."""
        return cls({name: getattr(spectrum, name) for name in _ARRAYS})

    def _quantity(self, name: str, unit: str) -> Quantity:
        from ...units import Q_

        return Q_(self._arrays[name].copy(), unit)

    @property
    def photon_energy(self) -> Quantity:
        """Fresh photon-energy nodes in the displayed energy unit."""
        return self._quantity("photon_energy_eV", "eV").to(self._energy_unit)

    @property
    def theta(self) -> Quantity:
        """Fresh polar detector angles to the beam axis in rad."""
        return self._quantity("theta_rad", "radian")

    @property
    def phi(self) -> Quantity:
        """Fresh detector azimuths from the plane normal in rad."""
        return self._quantity("phi_rad", "radian")

    @property
    def density(self) -> Quantity:
        """Total node density per displayed energy and steradian per electron."""
        return self._density("spectrum")

    @property
    def capture_density(self) -> Quantity:
        """Free-to-bound capture component of :attr:`density`."""
        return self._density("capture_spectrum")

    def _density(self, name: str) -> Quantity:
        return self._quantity(name, "1 / eV / steradian").to(
            f"1 / ({self._energy_unit}) / steradian"
        )

    @property
    def stokes(self) -> Quantity:
        """Fresh dimensionless normalized Stokes ``(Q, U)``."""
        return self._quantity("stokes", "dimensionless")

    @property
    def depth(self) -> Quantity:
        """Fresh sampled depths in Å."""
        return self._quantity("z_ang", "angstrom")

    @property
    def line_rates(self) -> Quantity:
        """Fresh per-entry estimator per Å of depth, shape ``(n_z, n_lines)``."""
        return self._quantity("line_rates", "1 / angstrom")

    @property
    def bound_population(self) -> Quantity:
        """Fresh dimensionless bound population at each sampled depth."""
        return self._quantity("bound_population", "dimensionless")

    @property
    def line_yield(self) -> Quantity:
        """Fresh depth-integrated photons per incident electron per catalog entry."""
        return self._quantity("line_yield", "dimensionless")

    def to_energy(self, unit: str) -> Self:
        """Convert energy **and** densities; preserve every node and integral."""
        return type(self)(self._arrays, unit)


__all__ = ["QuantifiedChannelingSpectrum"]
