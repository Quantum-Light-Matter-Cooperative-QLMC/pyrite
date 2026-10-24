"""Ideal polarized two-port Martin--Puplett forward interferogram (#617).

The input is a measured or independently predicted *spectral coherency* J(E)
with units photons/(eV pulse), in a specified orthonormal transverse Jones
basis. Intensity alone cannot determine an interferogram without a declared
input polarization. The ideal input wire grid projects onto its transmitting
axis; the two orthogonal optical paths acquire a relative phase and are
resolved at complementary, orthogonally polarized output ports. These ideal
Jones operators are an explicitly bounded **reference model**, not a fitted
instrument, roof-mirror ray trace, or inference of a measured bunch duration.

A moving roof mirror creates twice its displacement in optical path difference;
OPD jitter is distinguished from beam arrival-time jitter (common-mode phase).
All input optical spectra are already *collected* per pulse, not per sr.
No X-ray detector efficiency or uncalibrated optical transfer is assumed.

Validation: optical-instrument-observable
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._backend import _to_cpu, array_namespace
from .._optical_coherency import OpticalSpectralCoherency
from .optical_energy import CalibratedPulseEnergyDetector

__all__ = [
    "OpticalSpectralCoherency",
    "IdealMartinPuplett",
    "OpticalInterferogram",
    "polarized_interferogram",
    "load_optical_interferogram",
]

_HC_EV_M = 1.2398419843320026e-6
_C_M_S = 299792458.0
_EV_J = 1.602176634e-19


def _finite_fraction(x, name):
    if not np.isfinite(x) or not 0 <= x <= 1:
        raise ValueError(f"{name} must lie in [0,1] and be finite")


@dataclass(frozen=True)
class IdealMartinPuplett:
    """Loss-bounded orthogonal-arm interferometer in a stated Jones frame.

    Angles are referenced to the two ideal differential-delay arm axes.
    The input wire grid projects onto ``input_grid_angle_rad``; two ports
    project at ``output_grid_angle_rad`` and its orthogonal complement.
    ``arm_intensity_transmissions`` and ``port_efficiencies`` are passive
    *power* efficiencies, independent of energy in this first slice; all
    non-ideal wavelength dependence requires a future calibrated transfer.
    The optional Gaussian ``opd_jitter_rms_m`` is an RMS *optical path*
    fluctuation between shots and attenuates only the fringe term.
    ``phase_offset_rad`` adds a constant, energy-independent fringe phase
    (e.g. a residual retardance); a nonzero value makes the interferogram
    asymmetric about zero OPD. It is not a calibrated dispersion model. The
    interferogram is even in mirror displacement only for zero ``zero_opd_m``
    and ``phase_offset_rad``.

    Validation: optical-instrument-observable
    """

    instrument_id: str
    input_grid_angle_rad: float = np.pi / 4
    output_grid_angle_rad: float = np.pi / 4
    arm_intensity_transmissions: tuple[float, float] = (1.0, 1.0)
    port_efficiencies: tuple[float, float] = (1.0, 1.0)
    zero_opd_m: float = 0.0
    phase_offset_rad: float = 0.0
    opd_jitter_rms_m: float = 0.0

    def __post_init__(self):
        if not self.instrument_id or not self.instrument_id.strip():
            raise ValueError("instrument_id must identify the assumed interferometer configuration")
        for name in (
            "input_grid_angle_rad",
            "output_grid_angle_rad",
            "zero_opd_m",
            "phase_offset_rad",
            "opd_jitter_rms_m",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.opd_jitter_rms_m < 0:
            raise ValueError("opd_jitter_rms_m must be nonnegative")
        for name in ("arm_intensity_transmissions", "port_efficiencies"):
            values = getattr(self, name)
            if len(values) != 2:
                raise ValueError(f"{name} requires two entries")
            for value in values:
                _finite_fraction(value, name)
            object.__setattr__(self, name, tuple(float(v) for v in values))


@dataclass(frozen=True, eq=False)
class OpticalInterferogram:
    """Two-port calibrated energy/voltage against mirror travel.

    NumPy/CuPy arrays have shape (Ndelay, 2), except the spectrum (Ndelay,2,NE).
    A pulse is one physical electron bunch. Detector outputs are *isolated*
    peak voltages, not lock-in/interferometer AC responses or long-time DC V/W.
    :meth:`save` explicitly downloads device arrays for NPZ persistence.
    """

    mirror_displacement_m: Any
    optical_path_difference_m: Any
    optical_delay_fs: Any
    spectral_port_photons_per_eV_per_pulse: Any
    incident_joule_per_pulse_ports: Any
    absorbed_joule_per_pulse_ports: Any
    expected_voltage_v_ports: Any
    reported_voltage_v_ports: Any
    saturated_ports: Any
    noise_rms_voltage_v_ports: Any
    average_absorbed_power_w_ports: Any
    metadata: dict = field(default_factory=dict)

    def save(self, path) -> Path:
        """Save an NPZ (non-pickled arrays and JSON provenance)."""
        dest = Path(path)
        if dest.suffix != ".npz":
            dest = dest.with_suffix(dest.suffix + ".npz")
        names = (
            "mirror_displacement_m",
            "optical_path_difference_m",
            "optical_delay_fs",
            "spectral_port_photons_per_eV_per_pulse",
            "incident_joule_per_pulse_ports",
            "absorbed_joule_per_pulse_ports",
            "expected_voltage_v_ports",
            "reported_voltage_v_ports",
            "saturated_ports",
            "noise_rms_voltage_v_ports",
        )
        arrays: dict[str, Any] = {k: _to_cpu(getattr(self, k)) for k in names}
        if self.average_absorbed_power_w_ports is not None:
            arrays["average_absorbed_power_w_ports"] = _to_cpu(self.average_absorbed_power_w_ports)
        np.savez(dest, metadata=np.array(json.dumps(self.metadata, sort_keys=True)), **arrays)
        return dest


def load_optical_interferogram(path) -> OpticalInterferogram:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta.get("model") != "ideal-martin-puplett-v1":
            raise ValueError("not an ideal Martin--Puplett interferogram result")
        fields: dict[str, Any] = {k: np.array(data[k]) for k in data.files if k != "metadata"}
    fields.setdefault("average_absorbed_power_w_ports", None)
    return OpticalInterferogram(**fields, metadata=meta)


def _port_spectra(source, instrument, mirror_shift):
    """Two complementary analyzing axes following an ideal retarder."""
    xp = array_namespace(source.photons_per_eV_per_pulse, mirror_shift)
    e = xp.asarray(source.energy_eV)
    j = xp.asarray(source.photons_per_eV_per_pulse)
    a = instrument.input_grid_angle_rad
    b = instrument.output_grid_angle_rad
    u = xp.asarray([np.cos(a), np.sin(a)])
    # The input polarizer removes the component orthogonal to u: the passed
    # density is u^T J u, which keeps 2 cos(a) sin(a) Re J_xy, i.e. the real
    # part of the incident xy coherence, and nothing orthogonal to u.
    input_density = xp.einsum("a,eab,b->e", u, j, u).real
    amplitudes = (
        xp.asarray(
            [
                [np.cos(b) * np.cos(a), np.sin(b) * np.sin(a)],
                [-np.sin(b) * np.cos(a), np.cos(b) * np.sin(a)],
            ]
        )
        * xp.sqrt(xp.asarray(instrument.arm_intensity_transmissions))[None, :]
    )
    dc = xp.sum(amplitudes**2, axis=1)
    fringes = 2 * xp.prod(amplitudes, axis=1)
    opd = instrument.zero_opd_m + 2.0 * mirror_shift
    k = 2 * np.pi * e / _HC_EV_M
    fringe_reduction = xp.exp(-0.5 * (k * instrument.opd_jitter_rms_m) ** 2)
    phases = opd[:, None] * k[None, :] + instrument.phase_offset_rad
    cos = xp.cos(phases) * fringe_reduction[None, :]
    spectra = (dc[None, :, None] + fringes[None, :, None] * cos[:, None, :]) * input_density[
        None, None, :
    ]
    spectra *= xp.asarray(instrument.port_efficiencies)[None, :, None]
    # Roundoff cannot create a physical negative spectral density.
    if bool(xp.any(spectra < -1e-12 * xp.maximum(1.0, xp.max(xp.abs(spectra))))):
        raise ValueError("non-passive interferometer output")
    return xp.maximum(spectra, 0.0), opd


def _check_scan_arguments(
    source, mirror_displacement_m, instrument, detectors, repetition_hz, max_spectral_elements
) -> Any:
    if not isinstance(source, OpticalSpectralCoherency):
        raise TypeError("source must be an OpticalSpectralCoherency, not intensity alone")
    if not isinstance(instrument, IdealMartinPuplett):
        raise TypeError("instrument must be an IdealMartinPuplett")
    if len(detectors) != 2 or not all(
        isinstance(d, CalibratedPulseEnergyDetector) for d in detectors
    ):
        raise TypeError("supply two calibrated pulse energy detectors")
    xp = array_namespace(source.photons_per_eV_per_pulse, mirror_displacement_m)
    shifts = xp.asarray(mirror_displacement_m, dtype=float)
    if shifts.ndim != 1 or shifts.size == 0 or not bool(xp.all(xp.isfinite(shifts))):
        raise ValueError("mirror_displacement_m must be a finite nonempty 1-D scan")
    if repetition_hz is not None and (not np.isfinite(repetition_hz) or repetition_hz < 0):
        raise ValueError("repetition_hz must be finite and nonnegative")
    if (
        not isinstance(max_spectral_elements, int)
        or max_spectral_elements < 1
        or shifts.size * source.energy_eV.size * 2 > max_spectral_elements
    ):
        raise ValueError(
            "interferogram exceeds max_spectral_elements; use smaller delay/energy blocks"
        )
    return shifts


def polarized_interferogram(
    source: OpticalSpectralCoherency,
    mirror_displacement_m,
    *,
    instrument: IdealMartinPuplett,
    detectors: tuple[CalibratedPulseEnergyDetector, CalibratedPulseEnergyDetector],
    repetition_hz: float | None = None,
    max_spectral_elements: int = 4_000_000,
) -> OpticalInterferogram:
    """Predict two calibrated ports; displacement, not optical delay, is input.

    The scan parameter is differential **mirror mechanical displacement**;
    OPD = zero_opd_m + 2*displacement and dt = OPD/c. Array densities use
    photons/(eV pulse), so neither charge N nor repetition rate is multiplied
    into per-pulse energy again. Multiple pulses/exposure, detector recovery,
    beam-charge fluctuations and broadband instrument calibration are deferred.

    The source/scan arrays select NumPy or CuPy. Forward spectra and batched
    response reductions stay on that backend. Validation synchronizes scalar
    checks; provenance hashing downloads the small input (energy,2,2) once per
    scan. No delay spectra or output voltages are downloaded until persistence
    or an explicit caller transfer. Calibration data are staged from the host.

    Validation: optical-instrument-observable
    """
    shifts = _check_scan_arguments(
        source, mirror_displacement_m, instrument, detectors, repetition_hz, max_spectral_elements
    )
    spectra, opd = _port_spectra(source, instrument, shifts)
    xp = array_namespace(spectra)
    # Two batched calls, independent of scan length. No per-delay host readings.
    readings = [
        detector.predict_batch(source.energy_eV, spectra[:, p], repetition_hz=repetition_hz)
        for p, detector in enumerate(detectors)
    ]
    incident, absorbed, expected, reported, noise, saturated = (
        xp.stack([getattr(r, name) for r in readings], axis=1)
        for name in (
            "incident_joule_per_pulse",
            "absorbed_joule_per_pulse",
            "expected_voltage_v",
            "reported_voltage_v",
            "noise_rms_voltage_v",
            "saturated",
        )
    )
    avg_power = (
        None
        if repetition_hz is None
        else xp.stack([r.average_absorbed_power_w for r in readings], axis=1)
    )
    config = json.loads(json.dumps(asdict(instrument)))
    meta = {
        "model": "ideal-martin-puplett-v1",
        "assumptions": "ideal input wire grid, ideal orthogonal-arm retardance, two output linear polarizers; independent spectral bins",
        "source_provenance": source.provenance,
        "instrument": config,
        "detector_response_sha256": [d.response_sha256 for d in detectors],
        "source_units": "photons/(eV pulse) polarization coherency",
        "port_units": "collected photons/(eV pulse), J/pulse, isolated-pulse V",
        "mirror_displacement_to_opd": "OPD = zero_opd_m + 2*mirror_displacement_m",
        "opd_jitter": "Gaussian optical-path RMS; common electron arrival time cancels",
        "repetition_hz": repetition_hz,
        "validation": "optical-instrument-observable: unanchored; synthetic reference only",
    }
    # Provenance boundary: hash only the small input (E,2,2), never delay spectra.
    source_digest = hashlib.sha256(_to_cpu(source.energy_eV).tobytes())
    source_digest.update(_to_cpu(source.photons_per_eV_per_pulse).tobytes())
    meta["source_spectrum_sha256"] = source_digest.hexdigest()
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    meta["model_configuration_sha256"] = digest
    return OpticalInterferogram(
        mirror_displacement_m=shifts.copy(),
        optical_path_difference_m=opd,
        optical_delay_fs=opd / _C_M_S * 1e15,
        spectral_port_photons_per_eV_per_pulse=spectra,
        incident_joule_per_pulse_ports=incident,
        absorbed_joule_per_pulse_ports=absorbed,
        expected_voltage_v_ports=expected,
        reported_voltage_v_ports=reported,
        saturated_ports=saturated,
        noise_rms_voltage_v_ports=noise,
        average_absorbed_power_w_ports=avg_power,
        metadata=meta,
    )
