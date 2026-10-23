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

import numpy as np

__all__ = ["CalibratedPulseEnergyDetector", "OpticalPulseReading"]

_ELECTRON_VOLT_J = 1.602176634e-19


@dataclass(frozen=True, eq=False)
class OpticalPulseReading:
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

    def predict(
        self,
        energy_eV,
        collected_photons_per_eV_per_pulse,
        *,
        repetition_hz: float | None = None,
    ) -> OpticalPulseReading:
        """Integrate a collected photon *density* in energy (not bin counts).

        Samples use trapezoidal quadrature on the physical nonuniform energy
        grid; resolution/convergence is the caller's responsibility. Repetition
        rate only reports optical average W, not an arbitrary conversion to
        pulse voltage. AC coupling, pulse pile-up and response times are not
        modeled. No uncertainty is inferred for the input spectrum or
        calibration; ``noise_rms_v`` is the declared electronics noise alone.
        """
        energy = np.asarray(energy_eV, dtype=float)
        photons = np.asarray(collected_photons_per_eV_per_pulse, dtype=float)
        if (
            energy.ndim != 1
            or energy.size < 2
            or photons.shape != energy.shape
            or not np.all(np.isfinite(energy))
            or not np.all(np.diff(energy) > 0)
            or not np.all(np.isfinite(photons))
            or np.any(photons < 0)
        ):
            raise ValueError("input must be finite, nonnegative photons/eV on an increasing grid")
        if energy[0] < self.energy_eV[0] or energy[-1] > self.energy_eV[-1]:
            raise ValueError("optical detector spectrum is outside its calibrated energy band")
        if repetition_hz is not None and (not np.isfinite(repetition_hz) or repetition_hz < 0):
            raise ValueError("repetition_hz must be finite and nonnegative")
        absorption = np.interp(energy, self.energy_eV, self.efficiency)
        incident = float(np.trapezoid(photons * energy, energy) * _ELECTRON_VOLT_J)
        absorbed = float(np.trapezoid(photons * energy * absorption, energy) * _ELECTRON_VOLT_J)
        expected = absorbed * self.pulse_gain_v_per_j
        if not (np.isfinite(incident) and np.isfinite(absorbed) and np.isfinite(expected)):
            raise ValueError("calculated detector energy or voltage overflowed")
        saturated = self.saturation_v is not None and expected > self.saturation_v
        readout = min(expected, self.saturation_v) if self.saturation_v is not None else expected
        return OpticalPulseReading(
            incident_joule_per_pulse=incident,
            absorbed_joule_per_pulse=absorbed,
            expected_voltage_v=expected,
            reported_voltage_v=readout,
            noise_rms_voltage_v=self.noise_rms_v,
            saturated=saturated,
            response_sha256=self.response_sha256,
            average_incident_power_w=None if repetition_hz is None else incident * repetition_hz,
            average_absorbed_power_w=None if repetition_hz is None else absorbed * repetition_hz,
        )
