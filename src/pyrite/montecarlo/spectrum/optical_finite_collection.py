"""Finite-distance, polarization-resolved optical TR/DR collection (#616/#617).

This opt-in CPU reference propagates *complex two-sided temporal Fourier* PEC
screen fields [V*s/m] to a detector plane and integrates the **normal
Poynting flux** over an explicitly bounded physical aperture. It does not use
far-field solid angle, the monochromatic V/m wavefront power formula, or an
incoherent-ray substitution. Source and plane have one common exp(-i omega t)
convention; no Fresnel factor is applied a second time.

The optional ideal Jones operator acts on outgoing *propagating* plane-wave
s/p components, not on local x/y intensities. It is uniform in direction,
passive in the vector-power metric, and discards evanescent modes. A realistic
angle-dependent mirror/polarizer must supply its own validated operator.

The periodic FFT cell, finite source, detector boundary and photon-energy grid
require independent convergence. Local normal Poynting flux can be signed in
near fields; negative collected energy is reported, never clipped into counts.
No measured-data anchor or CUDA parity is claimed.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .optical_finite_bunch import _form_factor, _validate_histories
from .optical_finite_screen import (
    _E_CHARGE_C,
    _HBAR_EV_S,
    _Z0_OHM,
    PECCircularScreen,
    ScreenSpectralField,
    _wavevectors,
    circular_pec_transition_field,
    propagate_screen_field,
)

if TYPE_CHECKING:
    from .optical_transition import OpticalBunchSource

__all__ = [
    "PlaneAcceptance",
    "FinitePlaneCollection",
    "FinitePlaneBunchBand",
    "collect_screen_field",
    "finite_screen_bunch_plane_band",
]


@dataclass(frozen=True)
class PlaneAcceptance:
    """Finite detector-plane aperture, centred in the local source x/y frame.

    Shape ``circle`` uses ``radius_m``; shape ``rectangle`` uses ``width_m`` and
    ``height_m``. Every detector boundary must lie strictly *inside* the FFT
    computational window, to expose periodic-wrap contamination separately.
    Rectangle pixel overlaps are exact; circles use an explicit 8x8 subpixel
    area approximation (converge pitch relative to aperture). Coordinates are
    physical metres, not angular cuts and not an optical transmission mask.
    """

    shape: Literal["circle", "rectangle"]
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    radius_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.center_x_m) or not np.isfinite(self.center_y_m):
            raise ValueError("detector-plane centre must be finite")
        if self.shape == "circle":
            if self.radius_m is None or not np.isfinite(self.radius_m) or self.radius_m <= 0:
                raise ValueError("circle requires positive finite radius_m")
            if self.width_m is not None or self.height_m is not None:
                raise ValueError("circle must not specify rectangle widths")
        elif self.shape == "rectangle":
            if self.radius_m is not None or self.width_m is None or self.height_m is None:
                raise ValueError("rectangle requires width_m and height_m only")
            if not all(np.isfinite(v) and v > 0 for v in (self.width_m, self.height_m)):
                raise ValueError("rectangle widths must be finite and positive")
        else:
            raise ValueError("detector shape must be 'circle' or 'rectangle'")

    def _weights(self, field: ScreenSpectralField) -> np.ndarray:
        ny, nx = field.electric_xy_v_s_m.shape[1:]
        dx, dy = field.dx_m, field.dy_m
        xx = (np.arange(nx) - (nx - 1) / 2) * dx
        yy = (np.arange(ny) - (ny - 1) / 2) * dy
        if self.shape == "circle":
            assert self.radius_m is not None  # enforced by __post_init__
            radius = self.radius_m
            half_x = half_y = radius
        else:
            assert self.width_m is not None and self.height_m is not None
            half_x, half_y = self.width_m / 2, self.height_m / 2
        if (
            abs(self.center_x_m) + half_x >= nx * dx / 2
            or abs(self.center_y_m) + half_y >= ny * dy / 2
        ):
            raise ValueError("detector aperture must fit strictly inside the FFT window")
        if self.shape == "rectangle":
            xover = (
                np.maximum(
                    0,
                    np.minimum(xx + dx / 2, self.center_x_m + half_x)
                    - np.maximum(xx - dx / 2, self.center_x_m - half_x),
                )
                / dx
            )
            yover = (
                np.maximum(
                    0,
                    np.minimum(yy + dy / 2, self.center_y_m + half_y)
                    - np.maximum(yy - dy / 2, self.center_y_m - half_y),
                )
                / dy
            )
            weights = yover[:, None] * xover[None, :]
        else:
            x = xx[None, :]
            y = yy[:, None]
            weights = np.zeros((ny, nx), float)
            # 8x8 subpixel area fractions prevent gross error for small circles.
            for row in range(8):
                for col in range(8):
                    px = x + ((col + 0.5) / 8 - 0.5) * dx - self.center_x_m
                    py = y + ((row + 0.5) / 8 - 0.5) * dy - self.center_y_m
                    weights += (px * px + py * py <= radius**2) / 64
        if not np.any(weights):
            raise ValueError("aperture captures no detector-grid area; refine the mesh")
        return weights


def _checked_jones(jones_sp):
    if jones_sp is None:
        return None
    m = np.asarray(jones_sp, complex)
    if m.shape != (2, 2) or not np.all(np.isfinite(m)):
        raise ValueError("jones_sp must be a finite 2x2 s/p Jones matrix")
    if np.linalg.svd(m, compute_uv=False)[0] > 1 + 1e-12:
        raise ValueError("jones_sp must be passive (largest singular value <= 1)")
    return m


def _polarized_spectrum(ft, field, matrix):
    """Power-orthonormal outgoing propagating s/p transfer (s first)."""
    if matrix is None:
        return ft
    k0, kx, ky, kz = _wavevectors(field)
    kt = np.sqrt(kx * kx + ky * ky)
    good = kt < k0 * (1 - 1e-12)
    nonaxial = good & (kt > 0)
    cx = np.divide(kx, kt, out=np.ones(np.broadcast_shapes(kx.shape, ky.shape)), where=nonaxial)
    cy = np.divide(ky, kt, out=np.zeros_like(cx), where=nonaxial)
    cos = np.where(good, kz.real / k0, 1.0)
    cos_safe = np.where(good, cos, 1.0)
    # E = Es*(-cy,cx,0) + Ep*(cos*cx,cos*cy,-sin(theta)).
    es = -cy * ft[0] + cx * ft[1]
    ep = (cx * ft[0] + cy * ft[1]) / cos_safe
    output_s = matrix[0, 0] * es + matrix[0, 1] * ep
    output_p = matrix[1, 0] * es + matrix[1, 1] * ep
    return np.stack(
        [
            np.where(good, -cy * output_s + cos * cx * output_p, 0.0),
            np.where(good, cx * output_s + cos * cy * output_p, 0.0),
        ]
    )


def _plane_electric_magnetic(field: ScreenSpectralField, jones_sp=None):
    """Reconstruct E_t and H_t on a common plane, including evanescence."""
    matrix = _checked_jones(jones_sp)
    k0, kx, ky, kz = _wavevectors(field)
    ft = np.fft.fft2(field.electric_xy_v_s_m, axes=(-2, -1))
    ft = _polarized_spectrum(ft, field, matrix)
    # kz=0 at the light circle is a singular angular-spectrum coordinate.
    # Refuse it rather than silently substituting a very large Ez/H field.
    if np.any((np.abs(kz) <= 1e-12 * k0) & (np.abs(ft[0]) + np.abs(ft[1]) > 0)):
        raise ValueError("grazing spectral mode is singular; change the FFT mesh")
    safe = np.where(np.abs(kz) > 1e-12 * k0, kz, 1.0)
    ez = -(kx * ft[0] + ky * ft[1]) / safe
    hx = (ky * ez - kz * ft[1]) / (_Z0_OHM * k0)
    hy = (kz * ft[0] - kx * ez) / (_Z0_OHM * k0)
    electric = np.fft.ifft2(ft, axes=(-2, -1))
    magnetic = np.fft.ifft2(np.stack([hx, hy]), axes=(-2, -1))
    return electric, magnetic


def _collected_energy(field, electric, magnetic, aperture):
    pz = np.real(electric[0] * magnetic[1].conj() - electric[1] * magnetic[0].conj())
    weights = (
        aperture._weights(field)
        if isinstance(aperture, PlaneAcceptance)
        else 1.0
        if aperture is None
        else aperture
    )
    return float(np.sum(pz * weights) * field.dx_m * field.dy_m / (np.pi * _HBAR_EV_S))


@dataclass(frozen=True)
class FinitePlaneCollection:
    """One-energy *signed* collected normal spectral energy/photons.

    Values are J/eV and photons/eV, not per-solid-angle and not W. The
    detector-plane acceptance has not yet applied any energy absorptance.
    Negative near-field flux or a noisy signed pair estimate must not be
    silently converted into positive detected photon counts.
    """

    energy_eV: float
    z_m: float
    spectral_energy_j_per_eV: float
    photons_per_eV: float
    polarization: str
    configuration: dict = field(default_factory=dict)


def _configuration(field, plane_z_m, aperture, jones_sp):
    matrix = _checked_jones(jones_sp)
    return {
        "model": "finite-pec-detector-plane-v1",
        "fourier_convention": "two-sided temporal V*s/m; exp(-i omega t)",
        "field_unit": "V*s/m",
        "collected_energy_unit": "J/eV",
        "plane_z_m": float(plane_z_m),
        "sampling": {
            "nx": int(field.electric_xy_v_s_m.shape[2]),
            "ny": int(field.electric_xy_v_s_m.shape[1]),
            "dx_m": float(field.dx_m),
            "dy_m": float(field.dy_m),
            "fft_boundary": "periodic; grid/padding convergence required",
        },
        "aperture": None if aperture is None else asdict(aperture),
        "jones_sp": None
        if matrix is None
        else [[[float(z.real), float(z.imag)] for z in row] for row in matrix],
        "polarization_basis": "s/p of each propagating angular-spectrum mode",
        "material_model": "separately specified ideal PEC; no Fresnel re-application",
    }


def collect_screen_field(
    field: ScreenSpectralField,
    *,
    plane_z_m: float,
    aperture: PlaneAcceptance | None = None,
    jones_sp=None,
) -> FinitePlaneCollection:
    """Integrate coherent single-charge/realization field at a finite plane.

    The optional Jones matrix acts in the outgoing (s,p) basis of each
    transverse Fourier component, on propagating modes only. It is an ideal
    passive analyzer before spatial collection; no lens/mirror geometry or
    detector QE is inferred. The coherent near field includes Ez and magnetic
    components reconstructed from source-free Maxwell equations.

    Validation: optical-wavefront-propagation
    """
    if not isinstance(field, ScreenSpectralField):
        raise TypeError("field must be a ScreenSpectralField in V*s/m")
    if not np.isfinite(plane_z_m) or plane_z_m <= field.z_m:
        raise ValueError("plane_z_m must be downstream of the field reference plane")
    if aperture is not None and not isinstance(aperture, PlaneAcceptance):
        raise TypeError("aperture must be a PlaneAcceptance")
    propagated = propagate_screen_field(field, plane_z_m - field.z_m)
    electric, magnetic = _plane_electric_magnetic(propagated, jones_sp)
    energy = _collected_energy(propagated, electric, magnetic, aperture)
    return FinitePlaneCollection(
        energy_eV=field.energy_eV,
        z_m=float(plane_z_m),
        spectral_energy_j_per_eV=energy,
        photons_per_eV=energy / (field.energy_eV * _E_CHARGE_C),
        polarization="total" if jones_sp is None else "propagating s/p Jones",
        configuration=_configuration(field, plane_z_m, aperture, jones_sp),
    )


@dataclass(frozen=True, eq=False)
class FinitePlaneBunchBand:
    """Finite-distance bunch spectrum; signed estimates remain signed."""

    energy_eV: np.ndarray
    collected_photons_per_eV_per_pulse: np.ndarray
    self_photons_per_eV_per_electron: np.ndarray
    pair_photons_per_eV_per_electron_pair: np.ndarray
    form_factor_squared: np.ndarray
    plane_z_m: float
    source_record: dict
    polarization: str
    configuration: dict = field(default_factory=dict)

    def detect(self, detector, *, repetition_hz=None):
        """Apply an explicitly supplied calibrated pulse detector (#617)."""
        if np.any(self.collected_photons_per_eV_per_pulse < 0):
            raise ValueError(
                "signed near-field flux or unresolved pair estimate cannot be detected as counts"
            )
        return detector.predict(
            self.energy_eV,
            self.collected_photons_per_eV_per_pulse,
            repetition_hz=repetition_hz,
        )


def _bunch_plane_one(
    energy, screens, source, weights, population, *, nx, ny, dx, dy, plane_z_m, aperture, jones_sp
):
    total_e = None
    total_h = None
    self_w = 0.0
    self_w2 = 0.0
    wsum = float(np.sum(weights))
    w2sum = float(np.sum(weights**2))
    region_weights = None
    for weight, screen in zip(weights, screens, strict=True):
        field = circular_pec_transition_field(energy, screen, nx=nx, ny=ny, dx_m=dx, dy_m=dy)
        field = propagate_screen_field(field, plane_z_m)
        e, h = _plane_electric_magnetic(field, jones_sp)
        if region_weights is None:
            region_weights = aperture._weights(field) if aperture is not None else 1.0
        self_flux = _collected_energy(field, e, h, region_weights)
        self_w += weight * self_flux
        self_w2 += weight * weight * self_flux
        if total_e is None:
            total_e = weight * e
            total_h = weight * h
        else:
            total_e += weight * e
            total_h += weight * h
    # Physical *bilinear* flux of the coherently summed E AND H, not |E|².
    total = _collected_energy(field, total_e, total_h, region_weights)
    denominator = wsum * wsum - w2sum
    pair = (total - self_w2) / denominator if denominator > 0 else 0.0
    selfflux = self_w / wsum
    factor = _form_factor(source, energy)
    bunchflux = (
        total
        if source.law == "realization"
        else (population * (selfflux + max(population - 1, 0) * factor * pair))
    )
    photon_joule = energy * _E_CHARGE_C
    return bunchflux / photon_joule, selfflux / photon_joule, pair / photon_joule, factor


def finite_screen_bunch_plane_band(
    energy_eV,
    screens: Sequence[PECCircularScreen],
    *,
    source: OpticalBunchSource,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    plane_z_m: float,
    aperture: PlaneAcceptance | None,
    jones_sp=None,
) -> FinitePlaneBunchBand:
    """Collect finite-screen CTR/CDR from a physical bunch at a finite plane.

    ``screens`` carry physical charge sign and sampled/realized phase offsets.
    Signed distinct-pair expectations use the established #615 source law.
    An actual realization is summed coherently BEFORE the local Poynting
    integration. Only the resulting photons/eV/pulse may enter the detector
    after an explicit positive-flux check. CPU NumPy reference only.
    """
    energies = np.asarray(energy_eV, dtype=float)
    if (
        energies.ndim != 1
        or energies.size < 2
        or not np.all(np.isfinite(energies))
        or np.any(energies <= 0)
        or np.any(np.diff(energies) <= 0)
    ):
        raise ValueError("energy_eV must be a strictly increasing positive finite 1-D grid")
    if not np.isfinite(plane_z_m) or plane_z_m <= 0:
        raise ValueError("plane_z_m must be positive and finite")
    if aperture is not None and not isinstance(aperture, PlaneAcceptance):
        raise TypeError("aperture must be a PlaneAcceptance")
    _checked_jones(jones_sp)
    histories, population, weights = _validate_histories(screens, source, float(energies[-1]))
    reference = circular_pec_transition_field(
        float(energies[0]), histories[0], nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m
    )
    if aperture is not None:
        aperture._weights(reference)  # refuse clipped/empty pupils before expensive moments
    values = np.array(
        [
            _bunch_plane_one(
                e,
                histories,
                source,
                weights,
                population,
                nx=nx,
                ny=ny,
                dx=dx_m,
                dy=dy_m,
                plane_z_m=plane_z_m,
                aperture=aperture,
                jones_sp=jones_sp,
            )
            for e in energies
        ]
    )
    result = FinitePlaneBunchBand(
        energy_eV=energies.copy(),
        collected_photons_per_eV_per_pulse=values[:, 0],
        self_photons_per_eV_per_electron=values[:, 1],
        pair_photons_per_eV_per_electron_pair=values[:, 2],
        form_factor_squared=values[:, 3],
        plane_z_m=float(plane_z_m),
        source_record=source.record(),
        polarization="total" if jones_sp is None else "propagating s/p Jones",
        configuration=_configuration(reference, plane_z_m, aperture, jones_sp),
    )
    if np.any(result.collected_photons_per_eV_per_pulse < 0):
        warnings.warn(
            "finite-distance spectral collection has signed negative bins; detector conversion will refuse them",
            RuntimeWarning,
            stacklevel=2,
        )
    return result
