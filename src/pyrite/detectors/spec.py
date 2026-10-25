"""Detector acceptance, photon-energy binning, and read-time scoring."""

import math
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol

import numpy as np

from .._energy_bins import EnergyBins as EnergyBins
from .._units import normalize_unit
from .acceptance import AngularAcceptance, SensorAcceptance


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
class NativeSpectrum:
    """Measured event mass on explicit native half-open energy bins.

    ``events`` may have arbitrary leading batch dimensions; its last dimension
    corresponds to ``edges_eV`` intervals. Values retain the caller's leading
    normalization, for example events per incident electron.
    """

    edges_eV: np.ndarray
    events: np.ndarray

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges_eV, dtype=float)
        events = np.asarray(self.events, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("NativeSpectrum.edges_eV requires at least two edges")
        if not np.all(np.isfinite(edges)) or np.any(edges < 0.0):
            raise ValueError("NativeSpectrum.edges_eV must be finite and nonnegative")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("NativeSpectrum.edges_eV must be strictly increasing")
        if events.ndim == 0 or events.shape[-1] != edges.size - 1:
            raise ValueError("NativeSpectrum.events last dimension must match its bins")
        if not np.all(np.isfinite(events)) or np.any(events < 0.0):
            raise ValueError("NativeSpectrum.events must be finite and nonnegative")
        edges.setflags(write=False)
        events.setflags(write=False)
        object.__setattr__(self, "edges_eV", edges)
        object.__setattr__(self, "events", events)


def _sample_edges(energy_eV: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_eV, dtype=float)
    if energy.ndim != 1 or energy.size < 2:
        raise ValueError("native response energy_eV requires at least two samples")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0) or np.any(np.diff(energy) <= 0.0):
        raise ValueError("native response energy_eV must be finite, nonnegative, and increasing")
    midpoints = 0.5 * (energy[:-1] + energy[1:])
    first = max(0.0, energy[0] - 0.5 * (energy[1] - energy[0]))
    last = energy[-1] + 0.5 * (energy[-1] - energy[-2])
    return np.concatenate(([first], midpoints, [last]))


def _native_density(energy_eV, intrinsic_density, *, scale: float) -> NativeSpectrum:
    edges = _sample_edges(np.asarray(energy_eV, dtype=float))
    density = np.asarray(intrinsic_density, dtype=float)
    if density.ndim == 0 or density.shape[-1] != edges.size - 1:
        raise ValueError("intrinsic_density last dimension must match energy_eV")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("native response intrinsic_density must be finite and nonnegative")
    return NativeSpectrum(edges, density * np.diff(edges) * float(scale))


@dataclass(frozen=True)
class IdealPhotonCounter:
    """Explicit unit-efficiency, energy-preserving photon-event response.

    This is a mathematical reference response, not calibrated hardware. One
    accepted photon remains one measured event at the same energy.
    """

    def score(self, energy_eV, intrinsic_density, *, fwhm_eV, scale):
        del energy_eV, fwhm_eV
        return np.asarray(intrinsic_density) * scale

    def native_score(self, energy_eV, intrinsic_density, *, scale=1.0) -> NativeSpectrum:
        """Integrate accepted density into energy-preserving native bins."""
        return _native_density(energy_eV, intrinsic_density, scale=scale)


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
        Numerical response-input and measured-output histogram widths in eV.
        They approximate continuous recorded energies, not ADC/ToT quantization
        or the acquisition reporting bins.
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

    def native_score(self, energy_eV, intrinsic_density, *, scale=1.0) -> NativeSpectrum:
        """Apply the clustered-event response on its native measured bins."""
        from .timepix_response import get_response

        incident = np.asarray(intrinsic_density) * scale
        response = get_response(
            energy_eV,
            dE_mc=self.dE_mc,
            dE_out=self.dE_out,
            n_mc=self.n_mc,
            seed=self.seed,
            thickness_um=self.thickness_um,
            bias_v=self.bias_v,
        )
        return NativeSpectrum(response.out_edges_eV, response.apply_native(incident))


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
            from .response import detector_efficiency

            scored = scored * detector_efficiency(energy_eV)
        if self.convolve:
            if fwhm_eV is None:
                raise ValueError("LegacyEDS convolution requires fwhm_eV")
            from .response import convolve_detector

            scored = convolve_detector(energy_eV, scored, fwhm_eV)
        return scored * scale


@dataclass(frozen=True, init=False)
class Detector:
    """One detector's acceptance and read-time response.

    Stored source arrays remain response-free. :meth:`score` applies the response
    only when results are read, so the same transport can be rescored. A
    ``None`` response is the identity apart from the acceptance ``scale``.

    Parameters
    ----------
    observation_angle_deg
        Polar observation angle from the beam axis in degrees.
    acceptance
        AngularAcceptance or SensorAcceptance. Resolved full spans in degrees
        and solid angle in sr are available as read-only attributes. None
        retains the profile/material defaults.
    response
        Optional object implementing read-time ``score``.
    """

    observation_angle_deg: float = 90.0
    polar_acceptance_deg: float | None = field(default=None, init=False)
    solid_angle_sr: float | None = field(default=None, init=False)
    azimuthal_acceptance_deg: float | None = field(default=None, init=False, compare=False)
    acceptance: AngularAcceptance | SensorAcceptance | None = field(
        default=None, kw_only=True, compare=False
    )
    _legacy_acceptance: tuple[float | None, float | None] = field(
        default=(None, None), repr=False, compare=False, kw_only=True
    )
    response: DetectorResponse | None = None

    def __init__(
        self,
        observation_angle_deg: float = 90.0,
        *,
        acceptance: AngularAcceptance | SensorAcceptance | None = None,
        response: DetectorResponse | None = None,
        _legacy_acceptance: tuple[float | None, float | None] = (None, None),
    ) -> None:
        if acceptance is not None and not isinstance(
            acceptance, (AngularAcceptance, SensorAcceptance)
        ):
            raise TypeError("acceptance must be AngularAcceptance or SensorAcceptance")
        polar, omega = _legacy_acceptance
        object.__setattr__(self, "observation_angle_deg", observation_angle_deg)
        object.__setattr__(self, "polar_acceptance_deg", polar)
        object.__setattr__(self, "solid_angle_sr", omega)
        object.__setattr__(self, "azimuthal_acceptance_deg", None)
        object.__setattr__(self, "acceptance", acceptance)
        object.__setattr__(
            self, "_legacy_acceptance", _legacy_acceptance if acceptance is None else (None, None)
        )
        object.__setattr__(self, "response", response)
        self.__post_init__()

    def _resolve_acceptance(self, acceptance: AngularAcceptance | SensorAcceptance) -> None:
        polar, azimuthal, omega = (
            acceptance.resolve(self.observation_angle_deg)
            if isinstance(acceptance, AngularAcceptance)
            else acceptance.resolve()
        )
        object.__setattr__(self, "polar_acceptance_deg", polar)
        object.__setattr__(self, "azimuthal_acceptance_deg", azimuthal)
        object.__setattr__(self, "solid_angle_sr", omega)

    def __post_init__(self) -> None:
        observation_angle = normalize_unit(
            self.observation_angle_deg,
            "degree",
            field="Detector.observation_angle_deg",
            expected="angle",
            kind="angle",
        )
        object.__setattr__(
            self,
            "observation_angle_deg",
            _number(
                "observation_angle_deg",
                observation_angle,
                minimum=0.0,
                maximum=180.0,
            ),
        )
        polar = normalize_unit(
            self.polar_acceptance_deg,
            "degree",
            field="Detector.polar_acceptance_deg",
            expected="angle",
            kind="angle",
        )
        acceptance = _positive_optional("polar_acceptance_deg", polar)
        if acceptance is not None and acceptance > 180.0:
            raise ValueError("polar_acceptance_deg is a full polar span and must be <= 180 degrees")
        object.__setattr__(self, "polar_acceptance_deg", acceptance)
        solid_angle_value = normalize_unit(
            self.solid_angle_sr,
            "steradian",
            field="Detector.solid_angle_sr",
            expected="solid angle",
            kind="solid_angle",
        )
        solid_angle = _positive_optional("solid_angle_sr", solid_angle_value)
        if solid_angle is not None and solid_angle > 4.0 * math.pi:
            raise ValueError("solid_angle_sr must be <= 4*pi sr")
        object.__setattr__(self, "solid_angle_sr", solid_angle)
        if self.acceptance is not None:
            self._resolve_acceptance(self.acceptance)
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

    def native_score(
        self,
        energy_eV: np.ndarray,
        intrinsic_density: np.ndarray,
        *,
        scale: float = 1.0,
    ) -> NativeSpectrum:
        """Return event mass on the response's explicit native measured bins."""
        if self.response is None:
            raise ValueError("native scoring requires an explicit detector response")
        scorer = getattr(self.response, "native_score", None)
        if scorer is None:
            raise NotImplementedError(
                f"{type(self.response).__name__} does not expose native measured bins"
            )
        result = scorer(
            np.asarray(energy_eV),
            np.asarray(intrinsic_density),
            scale=scale,
        )
        if not isinstance(result, NativeSpectrum):
            raise TypeError("response native_score() must return NativeSpectrum")
        return result


class DetectorSpec(Detector):
    """Deprecated compatibility spelling for :class:`Detector`."""

    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "DetectorSpec is deprecated; use Detector",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls)
