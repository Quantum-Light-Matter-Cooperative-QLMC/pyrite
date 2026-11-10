"""Delay-step, delay-range and spectral-node convergence of a two-port scan (#617).

A numerical diagnostic of ``polarized_interferogram``, not new physics. The
fringe part of each port, ``F_p(x) = [V_p(phi) - V_p(phi + pi)] / 2`` (the
constant level cancels exactly), gives the difference interferogram
``D(l) = F_0 - F_1 = integral W(E) cos(2 pi E l / hc) dE`` against optical
path difference ``l``. Its truncated, sampled cosine transform

    W_rec(E) = (2 / hc) * sum_j w_j D(l_j) cos(2 pi E l_j / hc)

(trapezoid weights ``w_j``) recovers the response-weighted fringe spectrum
``W`` [V/eV]. Three independent refinements are reported:

- spectral: scan voltages with ``n_energy`` versus ``2 n_energy - 1`` nested
  nodes (the fringe must be resolved in energy at the largest delay);
- delay step: ``W_rec`` with the mirror step halved (aliasing; the OPD step
  must stay below half the shortest in-band wavelength);
- delay range: ``W_rec`` with the mirror range doubled (truncation leakage and
  spectral resolution ``hc / (2 l_max)``).

Changes are relative: max-norm for voltages, L2 over the base energy nodes
for ``W_rec``. They bound refinement differences, not the distance to the
exact answer, and say nothing about source, optics or calibration accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .optical_energy import CalibratedPulseEnergyDetector
from .optical_interferometer import (
    _HC_EV_M,
    IdealMartinPuplett,
    OpticalSpectralCoherency,
    polarized_interferogram,
)

__all__ = ["InterferogramConvergence", "interferogram_convergence"]


@dataclass(frozen=True, eq=False)
class InterferogramConvergence:
    """Refinement changes of one symmetric two-port mirror scan."""

    spectral_change: float
    delay_step_change: float
    delay_range_change: float
    nyquist_ratio: float  # OPD step / (shortest in-band wavelength / 2); must be < 1
    spectral_resolution_eV: float  # hc / (2 * largest OPD)
    energy_eV: np.ndarray
    recovered_fringe_spectrum_v_per_eV: np.ndarray
    rtol: float
    converged: bool


def _fringe_difference(source, mirror, instrument, detectors, max_elements):
    """``F_0 - F_1`` [V] and OPD [m]; the constant port levels cancel exactly."""
    flipped = replace(instrument, phase_offset_rad=instrument.phase_offset_rad + np.pi)
    kwargs = dict(detectors=detectors, max_spectral_elements=max_elements)
    plus = polarized_interferogram(source, mirror, instrument=instrument, **kwargs)
    minus = polarized_interferogram(source, mirror, instrument=flipped, **kwargs)
    if np.any(plus.saturated_ports) or np.any(minus.saturated_ports):
        raise ValueError("saturated scan; the linear convergence diagnostic does not apply")
    fringe = 0.5 * (plus.expected_voltage_v_ports - minus.expected_voltage_v_ports)
    return fringe[:, 0] - fringe[:, 1], np.asarray(plus.optical_path_difference_m)


def _recovered_spectrum(energy, difference, opd):
    weights = np.gradient(opd)
    weights[[0, -1]] *= 0.5
    phase = 2 * np.pi * np.outer(energy, opd) / _HC_EV_M
    return (2 / _HC_EV_M) * (np.cos(phase) @ (weights * difference))


def _relative(change, reference):
    scale = float(np.linalg.norm(reference))
    return float(np.linalg.norm(change)) / scale if scale > 0 else float("inf")


def interferogram_convergence(
    spectrum: Callable[[np.ndarray], OpticalSpectralCoherency],
    band_eV: tuple[float, float],
    *,
    mirror_step_m: float,
    mirror_half_range_m: float,
    instrument: IdealMartinPuplett,
    detectors: tuple[CalibratedPulseEnergyDetector, CalibratedPulseEnergyDetector],
    n_energy: int = 257,
    rtol: float = 0.01,
    max_spectral_elements: int = 16_000_000,
) -> InterferogramConvergence:
    """Refine energy nodes, mirror step and mirror range of a symmetric scan.

    ``spectrum(energy_eV)`` returns the collected coherency on any increasing
    grid inside ``band_eV``; the source must vanish (or be negligible) at the
    band edges for the transform to describe it. The scan is
    ``x = -R ... R`` in steps of ``mirror_step_m`` (OPD ``2 x``), with
    ``zero_opd_m = 0`` and ``phase_offset_rad = 0`` (an even interferogram).
    Host NumPy; cost is four forward scans of at most
    ``2 * n_delay * (2 n_energy - 1)`` spectral elements each.
    """
    lo, hi = (float(v) for v in band_eV)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("band_eV must be finite with 0 < low < high")
    if isinstance(n_energy, bool) or not isinstance(n_energy, int) or n_energy < 3:
        raise ValueError("n_energy must be an integer >= 3")
    step, half = float(mirror_step_m), float(mirror_half_range_m)
    if not (np.isfinite(step) and np.isfinite(half) and 0 < step <= half / 2):
        raise ValueError("need 0 < mirror_step_m <= mirror_half_range_m / 2")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be positive and finite")
    if not isinstance(instrument, IdealMartinPuplett):
        raise TypeError("instrument must be an IdealMartinPuplett")
    if instrument.zero_opd_m != 0 or instrument.phase_offset_rad != 0:
        raise ValueError("the diagnostic needs an even scan: zero_opd_m = phase_offset_rad = 0")

    def scan(nodes, mirror_step, mirror_half):
        count = int(round(mirror_half / mirror_step))
        mirror = mirror_step * np.arange(-count, count + 1)
        source = spectrum(np.linspace(lo, hi, nodes))
        if not isinstance(source, OpticalSpectralCoherency):
            raise TypeError("spectrum must return an OpticalSpectralCoherency")
        return _fringe_difference(source, mirror, instrument, detectors, max_spectral_elements)

    energy = np.linspace(lo, hi, n_energy)
    base, opd = scan(n_energy, step, half)
    fine_nodes, _ = scan(2 * n_energy - 1, step, half)
    spectral_change = float(np.max(np.abs(fine_nodes - base))) / float(np.max(np.abs(fine_nodes)))
    recovered = _recovered_spectrum(energy, base, opd)
    half_step = _recovered_spectrum(energy, *scan(n_energy, step / 2, half))
    double_range = _recovered_spectrum(energy, *scan(n_energy, step, 2 * half))
    step_change = _relative(half_step - recovered, half_step)
    range_change = _relative(double_range - recovered, double_range)
    nyquist = 2 * step / (0.5 * _HC_EV_M / hi)
    return InterferogramConvergence(
        spectral_change=spectral_change,
        delay_step_change=step_change,
        delay_range_change=range_change,
        nyquist_ratio=float(nyquist),
        spectral_resolution_eV=float(_HC_EV_M / (2 * opd[-1])),
        energy_eV=energy,
        recovered_fringe_spectrum_v_per_eV=recovered,
        rtol=float(rtol),
        converged=bool(nyquist < 1 and max(spectral_change, step_change, range_change) <= rtol),
    )
