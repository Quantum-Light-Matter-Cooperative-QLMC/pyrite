"""Explicitly calibrated, energy-integrating optical/IR single-pulse channel.

Issue #617, preliminary independent detector operator. The input is the
*already collected* photons/(eV*pulse) after any source, optical propagation,
polarization and solid-angle/area acceptance. No X-ray QE is borrowed and no
angular acceptance or finite-radiator correction is silently invented.

The scalar gain converts absorbed joules **per pulse** to an isolated pulse's
peak readout voltage. At finite repetition rate the *average optical power*
can also be reported, but this gain is never reused as DC V/W sensitivity:
that requires a separate detector/time-response model. Noise is a reported
RMS uncertainty, not random noise added to the expected signal.

Validation: optical-instrument-observable
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._backend import array_namespace

__all__ = ["CalibratedPulseEnergyDetector", "OpticalPulseReading", "OpticalPulseBatchReading"]

_ELECTRON_VOLT_J = 1.602176634e-19

if TYPE_CHECKING:
    from pint import Quantity


def _quantity(value: Any, unit: str) -> Quantity:
    """Fresh quantity wrapping a copy on the value's own NumPy/CuPy backend."""
    from ..units import Q_

    copy = getattr(value, "copy", None)
    return Q_(copy() if callable(copy) else value, unit)


def _optional_quantity(value: Any, unit: str) -> Quantity | None:
    return None if value is None else _quantity(value, unit)


class _ReadingQuantities:
    """Semantic J/V/W read-back shared by host and batch pulse readings.

    Each property returns a fresh Pint quantity wrapping a copy of the stored
    value on its own NumPy/CuPy backend (no host transfer). Values are per
    collected pulse; average powers are ``None`` without a repetition rate.
    """

    @property
    def incident_energy(self) -> Quantity:
        """Collected optical energy per pulse in J."""
        return _quantity(cast(Any, self).incident_joule_per_pulse, "J")

    @property
    def absorbed_energy(self) -> Quantity:
        """Absorbed optical energy per pulse in J."""
        return _quantity(cast(Any, self).absorbed_joule_per_pulse, "J")

    @property
    def expected_voltage(self) -> Quantity:
        """Unsaturated expected peak readout voltage in V."""
        return _quantity(cast(Any, self).expected_voltage_v, "V")

    @property
    def reported_voltage(self) -> Quantity:
        """Saturation-clipped reported peak voltage in V."""
        return _quantity(cast(Any, self).reported_voltage_v, "V")

    @property
    def noise_rms_voltage(self) -> Quantity:
        """Reported RMS readout uncertainty in V."""
        return _quantity(cast(Any, self).noise_rms_voltage_v, "V")

    @property
    def average_incident_power(self) -> Quantity | None:
        """Average collected optical power in W, or ``None``."""
        return _optional_quantity(cast(Any, self).average_incident_power_w, "W")

    @property
    def average_absorbed_power(self) -> Quantity | None:
        """Average absorbed optical power in W, or ``None``."""
        return _optional_quantity(cast(Any, self).average_absorbed_power_w, "W")


@dataclass(frozen=True, eq=False)
class OpticalPulseReading(_ReadingQuantities):
    """Deterministic prediction for one collected bunch/pulse."""

    incident_joule_per_pulse: float
    absorbed_joule_per_pulse: float
    expected_voltage_v: float
    reported_voltage_v: float
    noise_rms_voltage_v: float
    saturated: bool
    response_sha256: str
    average_incident_power_w: float | None
    average_absorbed_power_w: float | None


@dataclass(frozen=True, eq=False)
class OpticalPulseBatchReading(_ReadingQuantities):
    """Pulse predictions with NumPy/CuPy arrays over the input's batch axes.

    Every numerical member has shape ``photon_density.shape[:-1]``. Scalars
    and metadata validation may synchronize; spectral arrays and reductions
    remain on their owning device. No stochastic noise is sampled.
    """

    incident_joule_per_pulse: Any
    absorbed_joule_per_pulse: Any
    expected_voltage_v: Any
    reported_voltage_v: Any
    noise_rms_voltage_v: Any
    saturated: Any
    response_sha256: str
    average_incident_power_w: Any
    average_absorbed_power_w: Any


@dataclass(frozen=True, eq=False)
class CalibratedPulseEnergyDetector:
    """Spectral absorptance and calibrated isolated-pulse voltage gain.

    ``efficiency`` is a dimensionless energy-absorption fraction, *not* an
    incident-photon quantum efficiency. Values at the strictly increasing
    ``energy_eV`` knots must be in [0,1]. Use measured absolute collection and
    absorption data; input spectra must lie wholly inside this calibration
    interval (no extrapolation, even for zero-valued bins). The readout gain
    ``pulse_gain_v_per_j`` covers one isolated pulse under its stated
    electronics/shaping conditions, not summed exposures.

    ``calibration_id`` must identify the measurement, calibration conditions
    and provenance. ``response_sha256`` additionally fingerprints numerical
    response values and readout parameters for persistence and comparisons.

    Validation: optical-instrument-observable
    """

    energy_eV: np.ndarray
    efficiency: np.ndarray
    pulse_gain_v_per_j: float
    calibration_id: str
    noise_rms_v: float = 0.0
    saturation_v: float | None = None
    response_sha256: str = field(init=False)

    def __post_init__(self):
        energy = np.asarray(self.energy_eV, dtype="<f8")
        response = np.asarray(self.efficiency, dtype="<f8")
        if (
            energy.ndim != 1
            or energy.size < 2
            or response.shape != energy.shape
            or not np.all(np.isfinite(energy))
            or not np.all(np.diff(energy) > 0)
            or energy[0] <= 0
        ):
            raise ValueError("calibration energy_eV must be positive, finite and increasing")
        if not np.all(np.isfinite(response)) or np.any((response < 0) | (response > 1)):
            raise ValueError("calibrated energy-absorption efficiency must lie in [0,1]")
        if not self.calibration_id or not self.calibration_id.strip():
            raise ValueError("a provenance-bearing calibration_id is required")
        if not np.isfinite(self.pulse_gain_v_per_j) or self.pulse_gain_v_per_j <= 0:
            raise ValueError("pulse_gain_v_per_j must be positive and finite")
        if not np.isfinite(self.noise_rms_v) or self.noise_rms_v < 0:
            raise ValueError("noise_rms_v must be nonnegative and finite")
        if self.saturation_v is not None and (
            not np.isfinite(self.saturation_v) or self.saturation_v <= 0
        ):
            raise ValueError("saturation_v must be positive and finite")
        energy = energy.copy()
        response = response.copy()
        energy.setflags(write=False)
        response.setflags(write=False)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "efficiency", response)
        header = json.dumps(
            {
                "model": "optical-single-pulse-energy-v1",
                "calibration_id": self.calibration_id,
                "pulse_gain_v_per_j": float(self.pulse_gain_v_per_j),
                "noise_rms_v": float(self.noise_rms_v),
                "saturation_v": None if self.saturation_v is None else float(self.saturation_v),
            },
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher = hashlib.sha256(header)
        hasher.update(energy.tobytes())
        hasher.update(response.tobytes())
        object.__setattr__(self, "response_sha256", hasher.hexdigest())

    @property
    def energies(self) -> Quantity:
        """Calibration photon-energy knots as a fresh eV quantity."""
        return _quantity(self.energy_eV, "eV")

    @property
    def pulse_gain(self) -> Quantity:
        """Isolated-pulse readout gain as a fresh V/J quantity."""
        return _quantity(float(self.pulse_gain_v_per_j), "V / J")

    @property
    def noise_rms(self) -> Quantity:
        """RMS readout noise as a fresh V quantity."""
        return _quantity(float(self.noise_rms_v), "V")

    @property
    def saturation(self) -> Quantity | None:
        """Saturation voltage as a fresh V quantity, or ``None``."""
        return None if self.saturation_v is None else _quantity(float(self.saturation_v), "V")

    def predict(
        self,
        energy_eV,
        collected_photons_per_eV_per_pulse,
        *,
        repetition_hz: float | None = None,
    ) -> OpticalPulseReading:
        """Integrate one photon density on a nonuniform energy grid.

        Returns host scalars for compatibility. Use :meth:`predict_batch`
        for device-resident reductions or multiple spectra. Repetition rate
        reports optical average W only, never an arbitrary pulse-voltage
        conversion. AC coupling, pile-up and recovery are not modeled.
        """
        xp = array_namespace(energy_eV, collected_photons_per_eV_per_pulse)
        photons = xp.asarray(collected_photons_per_eV_per_pulse)
        if photons.ndim != 1:
            raise ValueError("input must be a 1-D photon density; use predict_batch for batches")
        out = self.predict_batch(energy_eV, photons, repetition_hz=repetition_hz)
        return OpticalPulseReading(
            incident_joule_per_pulse=float(out.incident_joule_per_pulse),
            absorbed_joule_per_pulse=float(out.absorbed_joule_per_pulse),
            expected_voltage_v=float(out.expected_voltage_v),
            reported_voltage_v=float(out.reported_voltage_v),
            noise_rms_voltage_v=self.noise_rms_v,
            saturated=bool(out.saturated),
            response_sha256=out.response_sha256,
            average_incident_power_w=None
            if repetition_hz is None
            else float(out.average_incident_power_w),
            average_absorbed_power_w=None
            if repetition_hz is None
            else float(out.average_absorbed_power_w),
        )

    def predict_batch(
        self,
        energy_eV: Any,
        collected_photons_per_eV_per_pulse: Any,
        *,
        repetition_hz: float | None = None,
    ) -> OpticalPulseBatchReading:
        """Reduce photon densities shaped (..., energy) on their NumPy/CuPy backend.

        The common energy grid is one-dimensional and wholly inside the measured
        calibration band. Trapezoidal quadrature integrates the final axis.
        Calibration knots are uploaded once per call; no spectrum is downloaded.
        Validation synchronizes scalar predicates, not individual batch readings.

        Validation: optical-instrument-observable
        """
        xp = array_namespace(energy_eV, collected_photons_per_eV_per_pulse)
        energy = xp.asarray(energy_eV, dtype=float)
        photons = xp.asarray(collected_photons_per_eV_per_pulse, dtype=float)
        if (
            energy.ndim != 1
            or energy.size < 2
            or photons.ndim < 1
            or photons.shape[-1] != energy.size
            or not bool(xp.all(xp.isfinite(energy)))
            or not bool(xp.all(xp.diff(energy) > 0))
            or not bool(xp.all(xp.isfinite(photons)))
            or bool(xp.any(photons < 0))
        ):
            raise ValueError("input must be finite, nonnegative photons/eV on an increasing grid")
        if bool(energy[0] < self.energy_eV[0]) or bool(energy[-1] > self.energy_eV[-1]):
            raise ValueError("optical detector spectrum is outside its calibrated energy band")
        if repetition_hz is not None and (not np.isfinite(repetition_hz) or repetition_hz < 0):
            raise ValueError("repetition_hz must be finite and nonnegative")
        absorption = xp.interp(energy, xp.asarray(self.energy_eV), xp.asarray(self.efficiency))
        density = photons * energy * _ELECTRON_VOLT_J
        widths = xp.diff(energy) / 2
        incident = xp.sum((density[..., 1:] + density[..., :-1]) * widths, axis=-1)
        absorbed_density = density * absorption
        absorbed = xp.sum(
            (absorbed_density[..., 1:] + absorbed_density[..., :-1]) * widths, axis=-1
        )
        expected = absorbed * self.pulse_gain_v_per_j
        if not bool(xp.all(xp.isfinite(incident) & xp.isfinite(absorbed) & xp.isfinite(expected))):
            raise ValueError("calculated detector energy or voltage overflowed")
        saturated = (
            xp.zeros(expected.shape, dtype=bool)
            if self.saturation_v is None
            else expected > self.saturation_v
        )
        reported = (
            expected.copy()
            if self.saturation_v is None
            else xp.minimum(expected, self.saturation_v)
        )
        return OpticalPulseBatchReading(
            incident_joule_per_pulse=incident,
            absorbed_joule_per_pulse=absorbed,
            expected_voltage_v=expected,
            reported_voltage_v=reported,
            noise_rms_voltage_v=xp.full(expected.shape, self.noise_rms_v),
            saturated=saturated,
            response_sha256=self.response_sha256,
            average_incident_power_w=None if repetition_hz is None else incident * repetition_hz,
            average_absorbed_power_w=None if repetition_hz is None else absorbed * repetition_hz,
        )
