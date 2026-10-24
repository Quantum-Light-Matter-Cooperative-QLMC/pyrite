"""Paraxial complex-field handoff from #616 to the ideal #617 interferometer.

Only an actual coherently summed *field realization*, not a signed pair
intensity estimate, can yield its polarization coherency without additional
assumptions. Cartesian x/y Jones axes approximate physical fixed polarizer
axes only for a demonstrably paraxial collected field. This module therefore
refuses angular/evanescent leakage or a mismatch with exact Maxwell Poynting
flux at the selected plane. Angular, window and mesh convergence remain the
caller's responsibility. No instrument phase is inferred from intensity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..._backend import _to_cpu, array_namespace
from ..._optical_coherency import OpticalSpectralCoherency
from .optical_finite_collection import PlaneAcceptance, collect_screen_field
from .optical_finite_screen import (
    _E_CHARGE_C,
    _HBAR_EV_S,
    _Z0_OHM,
    ScreenSpectralField,
    _wavevectors,
    propagate_screen_field,
)

__all__ = ["coherency_from_screen_fields"]


def coherency_from_screen_fields(
    fields: Sequence[ScreenSpectralField],
    *,
    plane_z_m: float,
    aperture: PlaneAcceptance | None = None,
    provenance: str,
    max_angle_rad: float = 0.15,
    max_angular_leakage: float = 1e-3,
    max_flux_relative_error: float = 0.03,
) -> OpticalSpectralCoherency:
    """Integrate phase-resolved Ex/Ey spectral coherence over a physical plane.

    Input fields are single-realization complex two-sided *temporal* Fourier
    electric fields (V*s/m) already summed coherently across electrons. The
    returned Jones spectral density (photons/eV/pulse) is scaled by the
    paraxial ``(pi*hbar*Z0)^-1`` Poynting normalization, with no field-to-
    monochromatic-power conversion. The measured full-vector flux from
    :func:`collect_screen_field` must agree to within the declared tolerance.

    An input intensity-only ``FinitePlaneBunchBand`` is intentionally refused:
    its weighted pair estimate is a statistical expectation, not a physical
    complex field. An aperture keeps its existing spatial acceptance contract.
    """
    if (
        not isinstance(fields, Sequence)
        or len(fields) == 0
        or not all(isinstance(f, ScreenSpectralField) for f in fields)
    ):
        raise TypeError(
            "fields must be a nonempty sequence of complex ScreenSpectralField realizations"
        )
    if not np.isfinite(max_angle_rad) or not 0 < max_angle_rad < np.pi / 2:
        raise ValueError("max_angle_rad must be between 0 and pi/2")
    if not np.isfinite(max_angular_leakage) or not 0 <= max_angular_leakage < 1:
        raise ValueError("max_angular_leakage must be in [0,1)")
    if not np.isfinite(max_flux_relative_error) or not 0 < max_flux_relative_error < 1:
        raise ValueError("max_flux_relative_error must be in (0,1)")
    if not np.isfinite(plane_z_m):
        raise ValueError("plane_z_m must be finite")
    if aperture is not None and not isinstance(aperture, PlaneAcceptance):
        raise TypeError("aperture must be a PlaneAcceptance")
    energies = np.array([float(f.energy_eV) for f in fields])
    if energies.size < 2 or not np.all(np.diff(energies) > 0):
        raise ValueError("fields must be sorted on a strictly increasing energy grid of >=2 nodes")
    matrices = [
        _field_coherency(
            f,
            plane_z_m=plane_z_m,
            aperture=aperture,
            max_angle_rad=max_angle_rad,
            max_angular_leakage=max_angular_leakage,
            max_flux_relative_error=max_flux_relative_error,
        )
        for f in fields
    ]
    return OpticalSpectralCoherency(energies, np.asarray(matrices), provenance)


def _field_coherency(
    f: ScreenSpectralField,
    *,
    plane_z_m,
    aperture,
    max_angle_rad,
    max_angular_leakage,
    max_flux_relative_error,
) -> np.ndarray:
    """2x2 Jones coherency [photons/eV/pulse] of one energy node at the plane."""
    if plane_z_m <= f.z_m:
        raise ValueError("detector plane must be downstream of every field reference plane")
    plane = propagate_screen_field(f, plane_z_m - f.z_m)
    xp = array_namespace(plane.electric_xy_v_s_m)
    ft = xp.fft.fft2(plane.electric_xy_v_s_m, axes=(-2, -1))
    k0, kx, ky, _ = _wavevectors(plane)
    outside = (kx * kx + ky * ky) > (k0 * np.sin(max_angle_rad)) ** 2
    power_spectrum = xp.sum(xp.abs(ft) ** 2, axis=0)
    total = float(xp.sum(power_spectrum))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("screen field must have finite positive spectral power")
    leakage = float(xp.sum(power_spectrum[outside]) / total)
    if leakage > max_angular_leakage:
        raise ValueError(
            f"nonparaxial/evanescent field leakage {leakage:.3g} exceeds "
            f"{max_angular_leakage:g}; a fixed-xy Jones model is unsupported"
        )
    e = plane.electric_xy_v_s_m
    weight = xp.ones(e.shape[1:]) if aperture is None else xp.asarray(aperture._weights(plane))
    # Device-resident reduction to the 2x2 matrix; only it returns to the host.
    matrix = _to_cpu(xp.einsum("ayx,byx,yx->ab", e, e.conj(), weight))
    matrix *= plane.dx_m * plane.dy_m / (np.pi * _HBAR_EV_S * _Z0_OHM * f.energy_eV * _E_CHARGE_C)
    # The paraxial scalar-flux approximation is an explicit controlled
    # reduction of the vector Maxwell source, not a silent replacement.
    exact = collect_screen_field(f, plane_z_m=plane_z_m, aperture=aperture).photons_per_eV
    estimate = float(np.trace(matrix).real)
    if exact <= 0 or abs(estimate - exact) > max_flux_relative_error * exact:
        raise ValueError(
            "paraxial polarization density disagrees with full-vector collected flux; "
            "reduce collection angle or refine spatial sampling"
        )
    return matrix
