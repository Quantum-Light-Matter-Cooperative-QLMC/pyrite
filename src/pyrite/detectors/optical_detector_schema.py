"""Calibration schema for optical/IR detector classes (#617).

One record names a detector class, a provenance-bearing ``calibration_id``,
a measured spectral energy-absorption table and the class's scalar
calibration parameters (SI units encoded in each name). Supported classes
build the existing linear channel: a calibrated isolated-pulse gain ``G``
(:class:`CalibratedPulseEnergyDetector`) and a normalized impulse response
``h`` (:class:`DetectorTemporalResponse`), so ``V(t) = G sum_k U_k h(t - t_k)``.

Power-responsivity classes (bolometer, Golay cell) are calibrated in V/W
under sinusoidal power modulation at ``f_m``: the ratio of the voltage
fundamental to the absorbed-power fundamental is ``R(f_m) = G |H(2 pi f_m)|``,
hence

    G = R(f_m) / |H(2 pi f_m)|,

with ``f_m = 0`` the DC responsivity ``G H(0)``. Noise follows the NEP
definition ``V_rms = R(f_m) NEP sqrt(Delta f)`` with an explicitly declared
noise bandwidth; this assumes white output noise over that band, with NEP
quoted at ``f_m`` (conservative for thermal-noise-limited detectors).

Calibration reference plane (enter converted values, not raw datasheet numbers):

* ``responsivity_v_per_w`` and ``pulse_gain_v_per_j`` are per *absorbed*
  power/energy; the absorptance table is applied separately. A datasheet
  value per *incident* power at ``lambda_cal`` must be divided by
  ``A(lambda_cal)`` first, or absorption is counted twice.
* ``R(f_m)`` is a sinusoidal-fundamental ratio. A square-wave chopped
  calibration needs conversion: for a 50 % chop of on-power ``P_on`` read
  as lock-in RMS ``V_rms``, ``R = pi V_rms / (sqrt(2) P_on)``.
* Voltages refer to the shaped output: ``H`` includes
  ``shaping_time_constant_s``. A detector-head calibration combined with
  shaping biases ``G`` by ``sqrt(1 + (2 pi f_m tau_s)^2)``.

Classes whose forward operator does not exist yet (spectrometer, camera)
list their required calibration and refuse.

Validation: optical-instrument-observable
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .optical_energy import CalibratedPulseEnergyDetector
from .optical_temporal import DetectorTemporalResponse

__all__ = [
    "OPTICAL_DETECTOR_CLASSES",
    "UNSUPPORTED_DETECTOR_CLASSES",
    "OpticalChannel",
    "OpticalDetectorCalibration",
    "pulse_gain_from_power_responsivity",
]

_SHAPING = ("shaping_time_constant_s", "saturation_v")

# class -> (required, optional) scalar parameter names.
OPTICAL_DETECTOR_CLASSES: MappingProxyType[str, tuple[tuple[str, ...], tuple[str, ...]]] = (
    MappingProxyType(
        {
            "energy_channel": (
                ("pulse_gain_v_per_j", "time_constant_s"),
                ("noise_rms_v", *_SHAPING),
            ),
            "pyroelectric": (
                ("pulse_gain_v_per_j", "thermal_time_constant_s", "electrical_time_constant_s"),
                ("noise_rms_v", *_SHAPING),
            ),
            "bolometer": (
                (
                    "responsivity_v_per_w",
                    "thermal_time_constant_s",
                    "nep_w_per_sqrt_hz",
                    "noise_bandwidth_hz",
                    "operating_temperature_k",
                ),
                ("responsivity_modulation_hz", *_SHAPING),
            ),
            "golay": (
                (
                    "responsivity_v_per_w",
                    "responsivity_modulation_hz",
                    "time_constant_s",
                    "nep_w_per_sqrt_hz",
                    "noise_bandwidth_hz",
                ),
                _SHAPING,
            ),
        }
    )
)

# Classes with a declared calibration need but no forward operator.
UNSUPPORTED_DETECTOR_CLASSES: MappingProxyType[str, str] = MappingProxyType(
    {
        "spectrometer": (
            "polarization- and wavelength-dependent throughput, slit/angular coupling, "
            "line-spread function, per-channel detector response and wavelength calibration"
        ),
        "camera": (
            "spectral QE/responsivity, pixel geometry, imaging PSF and magnification, "
            "exposure, gain, dark current, read noise and full well"
        ),
    }
)

_NONNEGATIVE = frozenset({"noise_rms_v", "responsivity_modulation_hz"})


def pulse_gain_from_power_responsivity(
    responsivity_v_per_w: float,
    temporal: DetectorTemporalResponse,
    modulation_hz: float = 0.0,
) -> float:
    """Isolated-pulse gain ``G = R(f_m) / |H(2 pi f_m)|`` in V/J.

    ``R(f_m)`` is the sinusoidal-modulation responsivity (fundamental voltage
    over fundamental absorbed power); ``f_m = 0`` is DC. Refuses a modulation
    frequency where the channel transfer vanishes (DC of an AC-coupled
    channel).

    Validation: optical-instrument-observable
    """
    r = float(responsivity_v_per_w)
    f = float(modulation_hz)
    if not np.isfinite(r) or r <= 0:
        raise ValueError("responsivity_v_per_w must be positive and finite")
    if not np.isfinite(f) or f < 0:
        raise ValueError("modulation_hz must be nonnegative and finite")
    if f == 0 and temporal.is_ac_coupled:
        raise ValueError("an AC-coupled channel has no DC responsivity; give its modulation_hz")
    magnitude = float(abs(temporal.transfer(f)))
    scale = float(np.sum(np.abs(temporal.amplitudes) * temporal.time_constants_s))
    if not magnitude > 1e-12 * scale:
        raise ValueError("channel transfer vanishes at the calibration modulation frequency")
    return r / magnitude


@dataclass(frozen=True)
class OpticalChannel:
    """Calibrated pulse channel and its impulse response built from one record."""

    detector: CalibratedPulseEnergyDetector
    temporal: DetectorTemporalResponse
    detector_class: str
    schema_sha256: str


@dataclass(frozen=True, eq=False)
class OpticalDetectorCalibration:
    """Validated calibration record of one optical/IR detector class.

    ``energy_eV``/``absorptance`` are the measured spectral energy-absorption
    fraction (window, absorber and coatings). ``parameters`` holds exactly
    the class's required and allowed optional names (see
    ``OPTICAL_DETECTOR_CLASSES``); values are finite and positive (noise and
    modulation frequency may be zero). ``schema_sha256`` fingerprints the
    whole record.

    Validation: optical-instrument-observable
    """

    detector_class: str
    calibration_id: str
    energy_eV: Any
    absorptance: Any
    parameters: Any
    schema_sha256: str = field(init=False)

    def __post_init__(self):
        if self.detector_class in UNSUPPORTED_DETECTOR_CLASSES:
            raise NotImplementedError(
                f"no forward operator for {self.detector_class!r}; it requires "
                f"{UNSUPPORTED_DETECTOR_CLASSES[self.detector_class]}"
            )
        if self.detector_class not in OPTICAL_DETECTOR_CLASSES:
            raise ValueError(f"unknown optical detector class {self.detector_class!r}")
        if not self.calibration_id or not self.calibration_id.strip():
            raise ValueError("a provenance-bearing calibration_id is required")
        required, optional = OPTICAL_DETECTOR_CLASSES[self.detector_class]
        params = {str(k): float(v) for k, v in dict(self.parameters).items()}
        missing = sorted(set(required) - params.keys())
        unknown = sorted(params.keys() - set(required) - set(optional))
        if missing or unknown:
            raise ValueError(
                f"{self.detector_class} calibration: missing {missing}, unsupported {unknown}"
            )
        for name, value in params.items():
            bad = value < 0 if name in _NONNEGATIVE else not value > 0
            if not np.isfinite(value) or bad:
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if name in _NONNEGATIVE else 'positive'}"
                )
        if self.detector_class == "golay" and params["responsivity_modulation_hz"] <= 0:
            raise ValueError("a Golay cell responsivity is calibrated at a nonzero modulation")
        energy = np.array(self.energy_eV, dtype="<f8", copy=True)
        absorptance = np.array(self.absorptance, dtype="<f8", copy=True)
        energy.setflags(write=False)
        absorptance.setflags(write=False)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "absorptance", absorptance)
        object.__setattr__(self, "parameters", MappingProxyType(params))
        header = json.dumps(
            {
                "model": "optical-detector-calibration-v1",
                "detector_class": self.detector_class,
                "calibration_id": self.calibration_id,
                "parameters": params,
            },
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher = hashlib.sha256(header)
        hasher.update(energy.tobytes())
        hasher.update(absorptance.tobytes())
        object.__setattr__(self, "schema_sha256", hasher.hexdigest())
        self.build()  # validate the absorptance table and time constants eagerly

    def _temporal(self) -> DetectorTemporalResponse:
        p, cid = self.parameters, self.calibration_id
        if self.detector_class == "pyroelectric":
            h = DetectorTemporalResponse.pyroelectric(
                p["thermal_time_constant_s"], p["electrical_time_constant_s"], cid
            )
        else:
            tau = p.get("thermal_time_constant_s", p.get("time_constant_s"))
            h = DetectorTemporalResponse.single_pole(tau, cid)
        if "shaping_time_constant_s" in p:
            h = h.with_low_pass(p["shaping_time_constant_s"])
        return h

    def build(self) -> OpticalChannel:
        """Calibrated pulse detector and impulse response for this record."""
        p = self.parameters
        temporal = self._temporal()
        if "responsivity_v_per_w" in p:
            f_m = p.get("responsivity_modulation_hz", 0.0)
            gain = pulse_gain_from_power_responsivity(p["responsivity_v_per_w"], temporal, f_m)
            noise = p["responsivity_v_per_w"] * p["nep_w_per_sqrt_hz"]
            noise *= np.sqrt(p["noise_bandwidth_hz"])
        else:
            gain, noise = p["pulse_gain_v_per_j"], p.get("noise_rms_v", 0.0)
        detector = CalibratedPulseEnergyDetector(
            self.energy_eV,
            self.absorptance,
            pulse_gain_v_per_j=gain,
            calibration_id=f"{self.calibration_id} [schema {self.schema_sha256[:16]}]",
            noise_rms_v=float(noise),
            saturation_v=p.get("saturation_v"),
        )
        return OpticalChannel(detector, temporal, self.detector_class, self.schema_sha256)
