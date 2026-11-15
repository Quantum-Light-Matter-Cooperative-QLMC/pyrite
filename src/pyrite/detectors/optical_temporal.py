"""Temporal response, pulse trains and digitized counts of an optical channel (#617).

A calibrated pulse channel (:class:`CalibratedPulseEnergyDetector`) gives
the isolated-pulse peak voltage ``G * U`` for absorbed energy ``U`` per
pulse. Its linear time-invariant response to absorbed energies ``U_k`` at
times ``t_k`` is

    V(t) = G * sum_k U_k * h(t - t_k),

with ``h`` the *normalized* impulse response of the whole channel (detector,
coupling and amplifier): ``h(t) = 0`` for ``t < 0`` and ``max_t h = 1``, so the
calibrated pulse gain is unchanged. ``h`` is a sum of exponentials,

    h(t) = sum_j a_j exp(-t / tau_j),

closed under first-order low-pass shaping (``with_low_pass``) and AC
(high-pass) coupling (``with_ac_coupling``, ``h - h * lp``). A pyroelectric in
voltage mode is a thermal pole ``tau_T`` AC-coupled by the electrical
``tau_E = R C``,

    h(t) = [tau_T exp(-t/tau_E) - tau_E exp(-t/tau_T)] / (tau_T - tau_E),
    H(omega) proportional to i omega / [(1 + i omega tau_T)(1 + i omega tau_E)],

the standard pyroelectric voltage-responsivity shape (Porter, Ferroelectrics
33, 193 (1981)).

Derived quantities, for pulses of equal ``U`` at repetition rate ``f``
(period ``T = 1/f``):

* transfer ``H(omega) = sum_j a_j tau_j / (1 + i omega tau_j)`` in s;
* DC (average-power) responsivity ``G * H(0) = G * integral h`` in V/W. It is
  zero for an AC-coupled channel. The isolated-pulse gain is never reused as
  V/W;
* steady-state train ``V(s) = G U sum_j a_j exp(-s/tau_j) / (1 - exp(-T/tau_j))``
  for ``0 <= s < T`` after the latest pulse (geometric sum); an ``n``-pulse
  train keeps the factor ``1 - exp(-m T / tau_j)`` for the ``m`` pulses so far;
* its mean ``G U f H(0) = (G H(0)) * (U f)``, and the peak amplitude of its
  fundamental ``2 G U f |H(2 pi f)|`` (lock-in at ``f``; RMS is ``1/sqrt 2`` of
  it), from the Fourier series of a periodic impulse train;
* pile-up ratio: steady-state peak over the isolated-pulse peak.

Digitized counts ``N = N_0 + g V`` are clipped to ``[0, 2^bits - 1]``.
Their noise adds the analog noise and uniform quantization (``1/12`` count^2,
valid when the analog noise spans at least about one count). Shot noise,
nonlinearity, thermal saturation and dead time are not modelled.

Validation: optical-instrument-observable
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .._backend import array_namespace

__all__ = [
    "Digitizer",
    "DetectorTemporalResponse",
    "PulseTrainReading",
    "PulseTrainShape",
    "predict_pulse_train",
]

_DISTINCT = 1e-6


def _exponential_sum(t, amplitudes, taus):
    xp = array_namespace(t)
    t = xp.asarray(t, dtype=float)
    terms = xp.asarray(amplitudes)[:, None] * xp.exp(
        -xp.maximum(t.reshape(-1), 0.0)[None, :] / xp.asarray(taus)[:, None]
    )
    return xp.where(t.reshape(-1) >= 0, xp.sum(terms, axis=0), 0.0).reshape(t.shape)


def _refined_extremum(function, grid, sign):
    """Maximum (sign=+1) or minimum (sign=-1) of ``function`` on [grid[0], grid[-1]]."""
    values = sign * function(grid)
    k = int(np.argmax(values))
    best_t, best = float(grid[k]), float(values[k])
    lo, hi = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, grid.size - 1)])
    if hi > lo:
        res = minimize_scalar(
            lambda x: -sign * float(function(np.array([x]))[0]),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12 * max(hi, 1e-300)},
        )
        if -res.fun > best:
            best_t, best = float(res.x), float(-res.fun)
    return best_t, sign * best


@dataclass(frozen=True)
class PulseTrainShape:
    """Steady-state train per unit isolated-pulse peak (dimensionless).

    ``peak``/``minimum`` over one period; ``peak_delay_s`` after the latest
    pulse; ``mean = f H(0)``; ``fundamental_amplitude = 2 f |H(2 pi f)|``.
    """

    repetition_hz: float
    peak: float
    peak_delay_s: float
    minimum: float
    mean: float
    fundamental_amplitude: float


@dataclass(frozen=True, eq=False)
class DetectorTemporalResponse:
    """Normalized LTI impulse response ``h(t) = sum_j a_j exp(-t/tau_j)``.

    ``amplitudes`` are given in any common scale and rescaled so that the
    maximum over ``t >= 0`` is 1 (the calibrated isolated-pulse peak).
    Time constants in s must be positive and mutually distinct (relative
    1e-6). ``calibration_id`` identifies the measured response.

    Validation: optical-instrument-observable
    """

    amplitudes: np.ndarray
    time_constants_s: np.ndarray
    calibration_id: str
    peak_time_s: float = field(init=False)
    response_sha256: str = field(init=False)

    def __post_init__(self):
        a = np.array(self.amplitudes, dtype="<f8", copy=True).reshape(-1)
        tau = np.array(self.time_constants_s, dtype="<f8", copy=True).reshape(-1)
        if a.size < 1 or tau.shape != a.shape or not np.all(np.isfinite(a)):
            raise ValueError("amplitudes and time_constants_s must be finite, of one length")
        if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
            raise ValueError("time constants must be positive and finite")
        ordered = np.sort(tau)
        if np.any(np.diff(ordered) <= _DISTINCT * ordered[1:]):
            raise ValueError("time constants must be distinct")
        if not self.calibration_id or not self.calibration_id.strip():
            raise ValueError("a provenance-bearing calibration_id is required")
        grid = np.concatenate(([0.0], np.geomspace(1e-4 * tau.min(), 60 * tau.max(), 4097)))
        t_peak, peak = _refined_extremum(lambda t: _exponential_sum(t, a, tau), grid, 1)
        if not peak > 0:
            raise ValueError("impulse response has no positive peak")
        a = a / peak
        a.setflags(write=False)
        tau.setflags(write=False)
        object.__setattr__(self, "amplitudes", a)
        object.__setattr__(self, "time_constants_s", tau)
        object.__setattr__(self, "peak_time_s", t_peak)
        header = json.dumps(
            {"model": "optical-exponential-impulse-v1", "calibration_id": self.calibration_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher = hashlib.sha256(header)
        hasher.update(a.tobytes())
        hasher.update(tau.tobytes())
        object.__setattr__(self, "response_sha256", hasher.hexdigest())

    @classmethod
    def single_pole(cls, tau_s: float, calibration_id: str) -> DetectorTemporalResponse:
        """DC-coupled first-order response ``exp(-t/tau)`` (e.g. a bolometer)."""
        return cls(np.array([1.0]), np.array([tau_s]), calibration_id)

    @classmethod
    def pyroelectric(
        cls, thermal_s: float, electrical_s: float, calibration_id: str
    ) -> DetectorTemporalResponse:
        """Voltage-mode pyroelectric: thermal pole AC-coupled by ``R C``."""
        return cls.single_pole(thermal_s, calibration_id).with_ac_coupling(electrical_s)

    def _with_pole(self, tau_p: float, *, high_pass: bool) -> DetectorTemporalResponse:
        if not np.isfinite(tau_p) or tau_p <= 0:
            raise ValueError("coupling/shaping time constant must be positive and finite")
        a, tau = self.amplitudes, self.time_constants_s
        if np.any(np.abs(tau - tau_p) <= _DISTINCT * np.maximum(tau, tau_p)):
            raise ValueError("coupling/shaping time constant equals a response time constant")
        # h * (1/tau_p) exp(-t/tau_p): a_j tau_j/(tau_j - tau_p) on tau_j and
        # sum_j a_j tau_j/(tau_p - tau_j) on tau_p.
        lp_a = np.append(a * tau / (tau - tau_p), np.sum(a * tau / (tau_p - tau)))
        if high_pass:
            lp_a = np.append(a, 0.0) - lp_a
        return type(self)(lp_a, np.append(tau, tau_p), self.calibration_id)

    def with_low_pass(self, tau_s: float) -> DetectorTemporalResponse:
        """Convolve with ``exp(-t/tau)/tau`` (amplifier/shaping rise); renormalized."""
        return self._with_pole(tau_s, high_pass=False)

    def with_ac_coupling(self, tau_s: float) -> DetectorTemporalResponse:
        """Apply a first-order high-pass ``delta(t) - exp(-t/tau)/tau``; renormalized."""
        return self._with_pole(tau_s, high_pass=True)

    def impulse(self, t_s: Any) -> Any:
        """Normalized ``h(t)`` (dimensionless) on the input's NumPy/CuPy backend."""
        return _exponential_sum(t_s, self.amplitudes, self.time_constants_s)

    def transfer(self, frequency_hz: Any) -> np.ndarray:
        """``H(2 pi f) = integral h(t) exp(-2 pi i f t) dt`` in s."""
        w = 2 * np.pi * np.asarray(frequency_hz, dtype=float)[..., None]
        tau = self.time_constants_s
        return np.sum(self.amplitudes * tau / (1 + 1j * w * tau), axis=-1)

    @property
    def dc_integral_s(self) -> float:
        """``H(0) = integral h dt`` in s (zero for an AC-coupled channel)."""
        return float(np.sum(self.amplitudes * self.time_constants_s))

    @property
    def is_ac_coupled(self) -> bool:
        scale = float(np.sum(np.abs(self.amplitudes) * self.time_constants_s))
        return abs(self.dc_integral_s) <= 1e-9 * scale

    def train(self, t_s: Any, repetition_hz: float, n_pulses: int | None = None) -> Any:
        """Pulse-train response per unit isolated peak; pulses at ``k / f``, ``k >= 0``.

        ``n_pulses=None`` is the steady state (pulses since ``-inf``; ``t``
        taken modulo the period). Otherwise only ``k < n_pulses`` contribute.
        """
        if not np.isfinite(repetition_hz) or repetition_hz <= 0:
            raise ValueError("repetition_hz must be positive and finite")
        if n_pulses is not None and (isinstance(n_pulses, bool) or int(n_pulses) < 1):
            raise ValueError("n_pulses must be a positive integer or None")
        xp = array_namespace(t_s)
        t = xp.asarray(t_s, dtype=float)
        period = 1.0 / repetition_hz
        tau = xp.asarray(self.time_constants_s)[:, None]
        decay = -xp.expm1(-period / tau)
        flat = t.reshape(-1)[None, :]
        if n_pulses is None:
            since = flat - period * xp.floor(flat / period)
            gain = 1.0 / decay
        else:
            m = xp.clip(xp.floor(flat / period) + 1, 0, int(n_pulses))
            since = flat - (m - 1) * period
            gain = -xp.expm1(-m * period / tau) / decay
        terms = xp.asarray(self.amplitudes)[:, None] * xp.exp(-xp.maximum(since, 0) / tau)
        value = xp.sum(terms * gain, axis=0)
        if n_pulses is not None:
            value = xp.where(flat[0] >= 0, value, 0.0)
        return value.reshape(t.shape)

    def steady_state(self, repetition_hz: float) -> PulseTrainShape:
        """Peak, minimum, mean and fundamental of the steady-state train.

        Extrema are over the closed interval ``0 <= s <= T`` since the latest
        pulse: ``s = T`` is the left limit just before the next pulse.
        """
        period = 1.0 / float(repetition_hz)
        tau = self.time_constants_s
        weights = self.amplitudes / -np.expm1(-period / tau)
        grid = np.unique(
            np.concatenate(
                (
                    np.linspace(0.0, period, 4097),
                    np.geomspace(1e-9 * period, period, 2049),
                    [self.peak_time_s] if self.peak_time_s < period else [],
                )
            )
        )

        def wave(s):
            return np.sum(weights[:, None] * np.exp(-np.asarray(s)[None, :] / tau[:, None]), 0)

        t_peak, peak = _refined_extremum(wave, grid, 1)
        _, minimum = _refined_extremum(wave, grid, -1)
        return PulseTrainShape(
            repetition_hz=float(repetition_hz),
            peak=peak,
            peak_delay_s=t_peak,
            minimum=minimum,
            mean=float(repetition_hz) * self.dc_integral_s,
            fundamental_amplitude=float(2 * repetition_hz * abs(self.transfer(repetition_hz))),
        )


@dataclass(frozen=True)
class PulseTrainReading:
    """Steady-state voltages for equal absorbed pulses at one repetition rate.

    ``mean_v = dc_responsivity_v_per_w * average_absorbed_power_w``;
    ``fundamental_rms_v`` is what a lock-in at the repetition rate reads.
    ``saturated`` compares the steady-state peak with ``saturation_v``.
    """

    isolated_peak_v: float
    steady_peak_v: float
    steady_minimum_v: float
    mean_v: float
    fundamental_peak_v: float
    fundamental_rms_v: float
    pileup_ratio: float
    dc_responsivity_v_per_w: float
    average_absorbed_power_w: float
    saturated: bool
    response_sha256: str
    temporal_sha256: str


def predict_pulse_train(
    detector,
    absorbed_joule_per_pulse: float,
    temporal: DetectorTemporalResponse,
    repetition_hz: float,
) -> PulseTrainReading:
    """Steady-state train of a calibrated channel (absorbed J/pulse, Hz → V).

    Validation: optical-instrument-observable
    """
    energy = float(absorbed_joule_per_pulse)
    if not np.isfinite(energy) or energy < 0:
        raise ValueError("absorbed_joule_per_pulse must be finite and nonnegative")
    shape = temporal.steady_state(repetition_hz)
    gain = float(detector.pulse_gain_v_per_j)
    scale = gain * energy
    peak = scale * shape.peak
    return PulseTrainReading(
        isolated_peak_v=scale,
        steady_peak_v=peak,
        steady_minimum_v=scale * shape.minimum,
        mean_v=scale * shape.mean,
        fundamental_peak_v=scale * shape.fundamental_amplitude,
        fundamental_rms_v=scale * shape.fundamental_amplitude / np.sqrt(2),
        pileup_ratio=shape.peak,
        dc_responsivity_v_per_w=gain * temporal.dc_integral_s,
        average_absorbed_power_w=energy * float(repetition_hz),
        saturated=detector.saturation_v is not None and peak > detector.saturation_v,
        response_sha256=detector.response_sha256,
        temporal_sha256=temporal.response_sha256,
    )


@dataclass(frozen=True)
class Digitizer:
    """Linear ADC ``N = offset_counts + counts_per_v * V`` clipped to ``[0, 2^bits - 1]``.

    Validation: optical-instrument-observable
    """

    counts_per_v: float
    bits: int
    calibration_id: str
    offset_counts: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.counts_per_v) or self.counts_per_v == 0:
            raise ValueError("counts_per_v must be finite and nonzero")
        if isinstance(self.bits, bool) or not 1 <= int(self.bits) <= 32:
            raise ValueError("bits must be an integer in [1, 32]")
        if not np.isfinite(self.offset_counts):
            raise ValueError("offset_counts must be finite")
        if not self.calibration_id or not self.calibration_id.strip():
            raise ValueError("a provenance-bearing calibration_id is required")

    @property
    def full_scale_counts(self) -> int:
        return 2 ** int(self.bits) - 1

    def counts(self, voltage_v: Any) -> tuple[Any, Any]:
        """Expected counts (unrounded, clipped) and the clipped mask."""
        xp = array_namespace(voltage_v)
        raw = self.offset_counts + self.counts_per_v * xp.asarray(voltage_v, dtype=float)
        clipped = (raw < 0) | (raw > self.full_scale_counts)
        return xp.clip(raw, 0, self.full_scale_counts), clipped

    def noise_rms_counts(self, noise_rms_v: Any) -> Any:
        """Analog noise times |gain| plus uniform quantization, in quadrature."""
        xp = array_namespace(noise_rms_v)
        analog = abs(self.counts_per_v) * xp.asarray(noise_rms_v, dtype=float)
        return xp.sqrt(analog**2 + 1.0 / 12.0)
