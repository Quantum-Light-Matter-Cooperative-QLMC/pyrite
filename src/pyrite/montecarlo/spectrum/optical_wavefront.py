"""Phase-preserving vacuum propagation from a *calibrated* optical field plane.

This is the wavefront propagation foundation of issue #616, not an optical
transition/diffraction-radiation source. ``electric_xy_v_m`` contains phasors
of the tangential electric field in V/m, with time convention ``exp(-i omega t)``.
The input is a finite transverse *reference plane*; it is not a far-field
photons/eV/sr spectrum and must not be inferred from one without phase and
normalization data. A monochromatic phasor has power units, not photon-density
units; spectral integration requires a separately defined Fourier convention.

The angular-spectrum step is the outgoing homogeneous-vacuum solution. FFT
boundary conditions periodically repeat the field; callers must establish
window/padding convergence independently, especially around sharp apertures.
Jones and Kirchhoff pupil masks are deliberately separate *thin-plane*
operators and are not a solver for finite surface currents or aperture DR.

Validation: optical-wavefront-propagation
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = [
    "OpticalWavefront",
    "propagate_vacuum",
    "apply_pupil",
    "forward_power_w",
]

# CODATA vacuum constants; hbar*c in eV*m and the vacuum wave impedance in ohms.
_HBAR_C_EV_M = 1.973269804593025e-7
_VACUUM_IMPEDANCE_OHM = 376.730313412


@dataclass(frozen=True, eq=False)
class OpticalWavefront:
    """One monochromatic outgoing electric-field plane in Cartesian vacuum.

    ``electric_xy_v_m`` has shape ``(2, ny, nx)`` and complex dtype: indices
    0/1 are lab-frame Ex and Ey, respectively. Cartesian x varies along the
    last axis and y along the penultimate axis; ``dx_m`` and ``dy_m`` are the
    strictly positive pixel pitches. The field is a complex electric *phasor*
    in V/m; positive z is downstream. Planes are equally spaced in vacuum and
    parallel to the x-y plane. Neither this type nor its propagation assigns
    an optical-spectral bandwidth or coherent bunch normalization.
    """

    energy_eV: float
    dx_m: float
    dy_m: float
    electric_xy_v_m: Any
    z_m: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.energy_eV) or self.energy_eV <= 0:
            raise ValueError("energy_eV must be positive and finite")
        if not np.isfinite(self.dx_m) or self.dx_m <= 0:
            raise ValueError("dx_m must be positive and finite")
        if not np.isfinite(self.dy_m) or self.dy_m <= 0:
            raise ValueError("dy_m must be positive and finite")
        if not np.isfinite(self.z_m):
            raise ValueError("z_m must be finite")
        field = self.electric_xy_v_m
        if getattr(field, "ndim", None) != 3 or field.shape[0] != 2:
            raise ValueError("electric_xy_v_m must have shape (2, ny, nx)")
        if min(field.shape[1:]) < 4:
            raise ValueError("the sampled reference plane needs >=4 pixels on each axis")
        if getattr(getattr(field, "dtype", None), "kind", None) != "c":
            raise TypeError("electric_xy_v_m must have a complex dtype (phasors)")


def _wavevectors(wavefront: OpticalWavefront, xp):
    ny, nx = wavefront.electric_xy_v_m.shape[1:]
    k0 = wavefront.energy_eV / _HBAR_C_EV_M
    kx = (2.0 * np.pi * xp.fft.fftfreq(nx, d=wavefront.dx_m))[None, :]
    ky = (2.0 * np.pi * xp.fft.fftfreq(ny, d=wavefront.dy_m))[:, None]
    kz = xp.sqrt((k0 * k0 - kx * kx - ky * ky).astype("complex128"))
    # Principal square root: Re(kz) >= 0, Im(kz) >= 0. This is an outgoing
    # wave (+z) for propagating components and decay for evanescent components.
    return k0, kx, ky, kz


def _check_finite(field, xp):
    if not bool(xp.all(xp.isfinite(field)).item()):
        raise ValueError("optical fields must be finite")


def propagate_vacuum(wavefront: OpticalWavefront, distance_m: float, *, xp=np):
    """Propagate a complex Ex/Ey plane forward in vacuum by angular spectrum.

    Uses ``exp(+i kz dz)`` (time convention ``exp(-i omega t)``), including
    exponential decay of evanescent components. Negative distance is refused:
    backward propagation would exponentially amplify unmeasured evanescence.
    Works on NumPy; use ``xp=cupy`` with device-resident input for a CUDA
    FFT. Does not perform padding/window selection on the caller's behalf.

    Validation: optical-wavefront-propagation
    """
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("distance_m must be nonnegative and finite")
    field = xp.asarray(wavefront.electric_xy_v_m)
    _check_finite(field, xp)
    if distance_m == 0:
        return replace(wavefront, electric_xy_v_m=field.copy())
    _, _, _, kz = _wavevectors(wavefront, xp)
    spectral = xp.fft.fft2(field, axes=(-2, -1))
    propagated = xp.fft.ifft2(spectral * xp.exp(1j * kz * distance_m), axes=(-2, -1))
    return replace(
        wavefront,
        electric_xy_v_m=propagated,
        z_m=wavefront.z_m + distance_m,
    )


def apply_pupil(
    wavefront: OpticalWavefront,
    *,
    transmission=None,
    phase_rad=None,
    jones=None,
    xp=np,
):
    """Apply a thin pupil in the existing reference plane (no propagation).

    ``transmission`` is a 2-D *amplitude* mask in [0,1], not intensity
    transmission; ``phase_rad`` is a real 2-D phase mask in radians. They
    share the field's (ny,nx) grid. ``jones`` is a constant 2x2 complex
    *transverse* Jones matrix; its largest singular value must be <=1.
    Jones passivity is defined in the paraxial transverse metric only;
    high-NA polarization conversion requires a full vector boundary model.
    This operation does not create DR from a physical metal aperture.
    """
    field = xp.asarray(wavefront.electric_xy_v_m)
    _check_finite(field, xp)
    shape = wavefront.electric_xy_v_m.shape[1:]
    out = field.copy()
    if transmission is not None:
        mask = xp.asarray(transmission)
        if mask.shape != shape or mask.dtype.kind == "c":
            raise ValueError("transmission must be a real (ny, nx) amplitude mask")
        if not bool(xp.all(xp.isfinite(mask) & (mask >= 0) & (mask <= 1)).item()):
            raise ValueError("amplitude transmission must be finite and between 0 and 1")
        out *= mask[None, :, :]
    if phase_rad is not None:
        phase = xp.asarray(phase_rad)
        if phase.shape != shape or phase.dtype.kind == "c":
            raise ValueError("phase_rad must be a real (ny, nx) phase")
        if not bool(xp.all(xp.isfinite(phase)).item()):
            raise ValueError("phase_rad must be finite")
        out *= xp.exp(1j * phase)[None, :, :]
    if jones is not None:
        matrix = np.asarray(jones, dtype=complex)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("jones must be a finite 2x2 complex matrix")
        if np.linalg.svd(matrix, compute_uv=False)[0] > 1.0 + 1e-12:
            raise ValueError("jones must be passive (largest singular value <= 1)")
        m = xp.asarray(matrix)
        out = xp.einsum("ab,bij->aij", m, out)
    return replace(wavefront, electric_xy_v_m=out)


def forward_power_w(wavefront: OpticalWavefront, *, xp=np) -> float:
    """Net positive-z *propagating* monochromatic power [W] in vacuum.

    Integrates the forward Poynting flux over one periodic FFT cell for the
    physical electric phasor [V/m]. For propagating plane waves the implied
    Ez is ``-(kx Ex + ky Ey)/kz``. Evanescent components convey no net normal
    power; grazing modes are omitted to avoid a singular Ex/Ey parametrization.
    This is not energy per pulse or photons/eV and cannot be integrated across
    energies until the field's spectral Fourier normalization is declared.

    Validation: optical-wavefront-propagation
    """
    field = xp.asarray(wavefront.electric_xy_v_m)
    _check_finite(field, xp)
    k0, kx, ky, kz = _wavevectors(wavefront, xp)
    propagating = (kx * kx + ky * ky) < k0 * k0 * (1 - 1e-12)
    kz_safe = xp.where(propagating, kz, 1.0)
    spectrum = xp.fft.fft2(field, axes=(-2, -1))
    ez = -(kx * spectrum[0] + ky * spectrum[1]) / kz_safe
    intensity = xp.abs(spectrum[0]) ** 2 + xp.abs(spectrum[1]) ** 2 + xp.abs(ez) ** 2
    flux = xp.where(propagating, (kz.real / k0) * intensity, 0.0)
    ny, nx = field.shape[1:]
    power = xp.sum(flux) * (
        wavefront.dx_m * wavefront.dy_m / (2.0 * _VACUUM_IMPEDANCE_OHM * nx * ny)
    )
    return float(power.item())
