"""Detector acceptance, photon-energy binning, and read-time scoring."""

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol

import numpy as np

from .response import convolve_detector, detector_efficiency


def _number(name: str, value: object, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < minimum or (maximum is not None and number > maximum):
        upper = "" if maximum is None else f" and <= {maximum:g}"
        raise ValueError(f"{name} must be >= {minimum:g}{upper}")
    return number


def _positive_optional(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    number = _number(name, value, minimum=0.0)
    if number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True, eq=False)
class EnergyBins:
    """The detector's line and bremsstrahlung photon-energy binnings.

    ``line`` is fine and narrow because coherent-line evaluation is expensive
    and kinematically bounded. ``brem`` is coarse and wide because the smooth,
    cheap continuum must extend to the beam energy. ``line_by_energy`` selects
    a fine line grid per beam energy when one fixed line grid is insufficient.

    Parameters
    ----------
    line
        Optional shared one-dimensional line photon-energy grid in eV.
    line_by_energy
        Optional mapping from electron energy in keV to line grids in eV.
        Mutually exclusive with ``line`` at catalog resolution.
    brem
        Optional one-dimensional bremsstrahlung photon-energy grid in eV.
    """

    line: np.ndarray | None = None
    line_by_energy: Mapping[float, np.ndarray] | None = None
    brem: np.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnergyBins):
            return NotImplemented
        if (self.line is None) != (other.line is None) or (self.brem is None) != (
            other.brem is None
        ):
            return False
        if (
            self.line is not None
            and other.line is not None
            and not np.array_equal(self.line, other.line)
        ):
            return False
        if (
            self.brem is not None
            and other.brem is not None
            and not np.array_equal(self.brem, other.brem)
        ):
            return False
        if (self.line_by_energy is None) != (other.line_by_energy is None):
            return False
        if self.line_by_energy is None:
            return True
        assert other.line_by_energy is not None
        return self.line_by_energy.keys() == other.line_by_energy.keys() and all(
            np.array_equal(grid, other.line_by_energy[energy])
            for energy, grid in self.line_by_energy.items()
        )


class DetectorResponse(Protocol):
    """Read-time detector response consumed by :meth:`Detector.score`."""

    def score(
        self,
        energy_eV: np.ndarray,
        intrinsic_density: np.ndarray,
        *,
        fwhm_eV: float | None,
        scale: float,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class Timepix3:
    """Timepix3 response configuration.

    ``thickness_um`` is the one formerly inert detector field with an existing
    physical consumer. The forward-model equations remain in
    :mod:`pyrite.detectors.timepix_response`.

    Parameters
    ----------
    thickness_um
        Active silicon thickness in micrometres; ``None`` uses hardware default.
    bias_v
        Sensor bias in volts; ``None`` uses hardware default.
    dE_mc, dE_out
        Monte Carlo input and recorded-output bin widths in eV.
    n_mc
        Simulated photons per coarse input energy.
    seed
        Response-matrix random seed.
    """

    thickness_um: float | None = None
    bias_v: float | None = None
    dE_mc: float = 50.0
    dE_out: float = 25.0
    n_mc: int = 60000
    seed: int = 0

    def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
        from .timepix_response import get_response

        incident = np.asarray(intrinsic_density) * scale
        return get_response(
            energy_eV,
            dE_mc=self.dE_mc,
            dE_out=self.dE_out,
            n_mc=self.n_mc,
            seed=self.seed,
            thickness_um=self.thickness_um,
            bias_v=self.bias_v,
        ).apply(incident)


@dataclass(frozen=True)
class EagleXO:
    """Configure the Eagle XO response forward model.

    Parameters
    ----------
    coating
        Entrance coating, ``"BN"`` or ``"BEN"``.
    resolve_energy
        Apply photon-counting energy blur after quantum efficiency.
    n_pix
        Pixels in one photon cluster for the read-noise contribution.
    """

    coating: str = "BN"
    resolve_energy: bool = False
    n_pix: int = 4

    def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
        from .eaglexo_response import get_response

        incident = np.asarray(intrinsic_density) * scale
        return get_response(
            energy_eV,
            coating=self.coating,
            resolve_energy=self.resolve_energy,
            n_pix=self.n_pix,
        ).apply(incident)


@dataclass(frozen=True)
class LegacyEDS:
    """Configure the historical EDS compatibility response.

    Parameters
    ----------
    apply_qe
        Apply the historical polymer/aluminium/silicon efficiency curve.
    convolve
        Apply a Gaussian energy blur; scoring then requires ``fwhm_eV``.
    """

    apply_qe: bool = False
    convolve: bool = False

    def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
        scored = np.asarray(intrinsic_density)
        if self.apply_qe:
            scored = scored * detector_efficiency(energy_eV)
        if self.convolve:
            if fwhm_eV is None:
                raise ValueError("LegacyEDS convolution requires fwhm_eV")
            scored = convolve_detector(energy_eV, scored, fwhm_eV)
        return scored * scale


@dataclass(frozen=True)
class Detector:
    """One detector's acceptance, photon-energy binnings, and response.

    Stored source arrays remain response-free. :meth:`score` applies the response
    only when results are read, so the same transport can be rescored. A
    ``None`` response is the identity apart from the acceptance ``scale``.

    Parameters
    ----------
    observation_angle_deg
        Polar observation angle from the beam axis in degrees.
    polar_acceptance_deg
        Optional full polar acceptance span in degrees.
    solid_angle_sr
        Optional accepted solid angle in sr.
    energy_bins
        Line and bremsstrahlung photon-energy grids.
    response
        Optional object implementing read-time ``score``.
    """

    observation_angle_deg: float = 90.0
    polar_acceptance_deg: float | None = None
    solid_angle_sr: float | None = None
    energy_bins: EnergyBins = field(default_factory=EnergyBins)
    response: DetectorResponse | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_angle_deg",
            _number(
                "observation_angle_deg",
                self.observation_angle_deg,
                minimum=0.0,
                maximum=180.0,
            ),
        )
        acceptance = _positive_optional("polar_acceptance_deg", self.polar_acceptance_deg)
        if acceptance is not None and acceptance > 180.0:
            raise ValueError("polar_acceptance_deg is a full polar span and must be <= 180 degrees")
        object.__setattr__(self, "polar_acceptance_deg", acceptance)
        solid_angle = _positive_optional("solid_angle_sr", self.solid_angle_sr)
        if solid_angle is not None and solid_angle > 4.0 * math.pi:
            raise ValueError("solid_angle_sr must be <= 4*pi sr")
        object.__setattr__(self, "solid_angle_sr", solid_angle)
        if not isinstance(self.energy_bins, EnergyBins):
            raise TypeError("energy_bins must be an EnergyBins")
        if self.response is not None and not hasattr(self.response, "score"):
            raise TypeError("response must implement score()")

    def score(
        self,
        energy_eV: np.ndarray,
        intrinsic_density: np.ndarray,
        *,
        fwhm_eV: float | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Score a response-free source density without mutating stored data.

        Parameters
        ----------
        energy_eV
            Photon-energy coordinate in eV.
        intrinsic_density
            Response-free source density on ``energy_eV``.
        fwhm_eV
            Optional detector-resolution override in eV.
        scale
            Multiplicative acceptance or rate scale.

        Returns
        -------
        numpy.ndarray
            Scored density in the input density's scaled units.
        """
        if self.response is None:
            return np.asarray(intrinsic_density) * scale
        return self.response.score(
            np.asarray(energy_eV),
            np.asarray(intrinsic_density),
            fwhm_eV=fwhm_eV,
            scale=scale,
        )


class DetectorSpec(Detector):
    """Deprecated compatibility spelling for :class:`Detector`."""

    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "DetectorSpec is deprecated; use Detector",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls)
