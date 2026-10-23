"""Opt-in straight-charge PEC disc/annulus TR and aperture DR (#616).

A single straight electron drives a planar perfect conductor.  The outgoing
*frequency-domain* tangential electric field on the metal equals minus the
charge's incident transverse field (Shkvarunets & Fiorito, PRSTAB 11, 012801
(2008), Eq. 15).  The surface distribution is represented by the outgoing
Schelkunoff equivalent magnetic current, Eqs. 17 and 19–20.  A vector angular
spectrum propagates its outgoing vacuum field exactly for the sampled plane.

IMPORTANT: ``electric_xy_v_s_m`` is the full two-sided temporal Fourier
transform ``integral E(t) exp(+i omega t) dt`` in V*s/m, NOT a monochromatic
phasor in V/m. Do not pass it to OpticalWavefront/forward_power_w, or multiply
its photon-density output by a bandwidth twice. Source is one physical charge;
external coherent-bunch complex-field summation must keep the arrival phases.

This is a *bounded* planar, ideal-PEC, prescribed straight-current
reference, now admitting oblique incidence from the +z half-space. Finite-grid
truncation and pixel sampling must be converged; finite-thickness edge
currents, real materials, beam deflection, and coupled transport are not
modeled. The physical-optics source (-E_inc,t on the metal, zero off it) is
exact only for an unbounded PEC: screen/hole radii and edge distances must be
>> wavelength. The source is not yet experimentally anchored.

Validation: optical-finite-boundary-field
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import k0, k1

__all__ = [
    "PECCircularScreen",
    "ScreenSpectralField",
    "circular_pec_transition_field",
    "propagate_screen_field",
    "observe_screen_green",
    "spectral_energy_per_eV",
    "hemisphere_energy_per_eV",
    "far_field_photons_per_eV_sr",
    "ginzburg_frank_photons_per_eV_sr",
    "backward_formation_length_ang",
]

_HBAR_EV_S = 6.582119569e-16
_HBAR_C_EV_M = 1.973269804593025e-7
_E_CHARGE_C = 1.602176634e-19
_EPS0_F_M = 8.8541878128e-12
_C_M_S = 299792458.0
_Z0_OHM = 376.730313412


@dataclass(frozen=True)
class PECCircularScreen:
    """A metal disc or annulus centred at (0,0), in the z=0 plane.

    Metal occupies ``inner_radius_m <= rho <= outer_radius_m``.  Inner radius
    zero gives a foil struck by the electron (TR); positive inner radius with
    the beam passing through the opening gives aperture DR.  The finite FFT
    computational window must be larger than the *metal*, not only the hole.
    ``electron_x_m``/``electron_y_m`` are the beam intercept at z=0.
    ``(beta_x,beta_y,beta_z)`` is the signed velocity/c in screen coordinates:
    ``beta_z < 0`` approaches the plane from +z, while radiation is observed
    into +z. The full speed must be strictly less than c. The last two
    velocity components default to zero (the original normal-incidence case).
    Charge must be +/- one elementary charge, not a macro-particle charge,
    to prevent spurious N**2 self radiation.
    """

    outer_radius_m: float
    inner_radius_m: float = 0.0
    electron_x_m: float = 0.0
    electron_y_m: float = 0.0
    beta_z: float = -0.95
    charge_c: float = -_E_CHARGE_C
    arrival_time_s: float = 0.0
    beta_x: float = 0.0
    beta_y: float = 0.0

    @property
    def beta_xyz(self) -> np.ndarray:
        """Charge velocity/c, in the local PEC screen frame."""
        return np.array([self.beta_x, self.beta_y, self.beta_z], dtype=float)

    def __post_init__(self):
        values = (
            self.outer_radius_m,
            self.inner_radius_m,
            self.electron_x_m,
            self.electron_y_m,
            self.beta_z,
            self.beta_x,
            self.beta_y,
            self.charge_c,
            self.arrival_time_s,
        )
        if not all(np.isfinite(v) for v in values):
            raise ValueError("all screen and electron parameters must be finite")
        if self.outer_radius_m < 0 or not 0 <= self.inner_radius_m <= self.outer_radius_m:
            raise ValueError("screen radii require 0 <= inner_radius_m <= outer_radius_m")
        if self.beta_z >= 0 or np.linalg.norm(self.beta_xyz) >= 1:
            raise ValueError("PEC backward emission requires beta_z < 0 and |beta_xyz| < 1")
        if not np.isclose(abs(self.charge_c), _E_CHARGE_C, rtol=1e-12, atol=0):
            raise ValueError("charge_c must be one elementary charge, not a macro-particle charge")
        if (
            self.inner_radius_m > 0
            and np.hypot(self.electron_x_m, self.electron_y_m) >= self.inner_radius_m
        ):
            raise ValueError("for aperture DR, electron must pass strictly inside the hole")


@dataclass(frozen=True, eq=False)
class ScreenSpectralField:
    """Positive-frequency *temporal Fourier transform* of a field on a plane.

    Array ``electric_xy_v_s_m`` has (2, ny, nx) complex entries and units
    V*s/m. x/y grid centres use (i-(n-1)/2)*pitch. ``z_m`` is downstream (+z).
    The Fourier convention is Etilde(omega)=int E(t)e^{+i omega t}dt;
    E(t)=(1/2pi)int Etilde(omega)e^{-i omega t}domega.
    """

    energy_eV: float
    dx_m: float
    dy_m: float
    electric_xy_v_s_m: np.ndarray
    z_m: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.energy_eV) or self.energy_eV <= 0:
            raise ValueError("energy_eV must be finite and positive")
        if (
            not np.isfinite(self.dx_m)
            or self.dx_m <= 0
            or not np.isfinite(self.dy_m)
            or self.dy_m <= 0
        ):
            raise ValueError("pixel pitches must be finite and positive")
        if not np.isfinite(self.z_m) or self.z_m < 0:
            raise ValueError("z_m must be finite and nonnegative")
        f = np.asarray(self.electric_xy_v_s_m)
        if f.ndim != 3 or f.shape[0] != 2 or min(f.shape[1:]) < 16:
            raise ValueError("electric_xy_v_s_m must have shape (2,ny>=16,nx>=16)")
        if f.dtype.kind != "c" or not np.all(np.isfinite(f)):
            raise ValueError("electric_xy_v_s_m must be finite complex Fourier fields")
        # Prevent mutations after construction (inputs are copied).
        f = f.copy()
        f.setflags(write=False)
        object.__setattr__(self, "electric_xy_v_s_m", f)


def _wavevectors(field: ScreenSpectralField):
    ny, nx = field.electric_xy_v_s_m.shape[1:]
    k0 = field.energy_eV / _HBAR_C_EV_M
    kx = (2 * np.pi * np.fft.fftfreq(nx, field.dx_m))[None, :]
    ky = (2 * np.pi * np.fft.fftfreq(ny, field.dy_m))[:, None]
    kz = np.sqrt((k0 * k0 - kx * kx - ky * ky).astype(complex))
    return k0, kx, ky, kz


def circular_pec_transition_field(
    energy_eV: float,
    screen: PECCircularScreen,
    *,
    nx: int = 256,
    ny: int = 256,
    dx_m: float,
    dy_m: float,
) -> ScreenSpectralField:
    """Outgoing PEC tangential field driven by an oblique straight electron.

    Let ``u=(x-x0,y-y0,0)``, ``d=beta/|beta|``, ``v=c|beta|``,
    ``gamma=(1-|beta|²)^(-1/2)``, ``p=u-d*(d·u)`` and ``a=|p|/gamma``.
    Fourier transformation of the exact uniformly moving-charge electric
    field at the plane gives the **incident** vector field

    ``Einc_tilde = q*omega/(2*pi*eps0*gamma²*v²) * exp(i*omega*(t0+d·u/v))``
    ``       * [(p/a)*K1(omega*a/v) - i*d*K0(omega*a/v)]``.

    The outgoing PEC source has tangential field ``-Einc_tilde[:2]`` on the
    metal and zero outside. This preserves the longitudinal-field-induced
    *tangential* component and the transverse incidence-time phase, missing
    if ``beta_z`` is substituted for the full electron speed. For beta_x=y=0
    it reduces to the previous radial K1 source exactly. The 2-D outgoing
    propagation and induced-current approximation remain as in the normal
    incidence model; this is not a finite-thickness or stopping calculation.

    Off-metal incident virtual photons are *not* re-radiated through a hole.
    Even-sized centred FFT grids are required, and pixel sampling must resolve
    both the screen edge and the contracted virtual-photon transverse scale.

    Validation: optical-finite-boundary-field
    """
    if not isinstance(screen, PECCircularScreen):
        raise TypeError("screen must be a PECCircularScreen")
    if (
        not isinstance(nx, (int, np.integer))
        or not isinstance(ny, (int, np.integer))
        or nx < 16
        or ny < 16
        or nx % 2
        or ny % 2
    ):
        raise ValueError("nx and ny must be even integers >= 16")
    if not np.isfinite(energy_eV) or energy_eV <= 0:
        raise ValueError("energy_eV must be finite and positive")
    if not np.isfinite(dx_m) or dx_m <= 0 or not np.isfinite(dy_m) or dy_m <= 0:
        raise ValueError("dx_m and dy_m must be finite and positive")
    if 2 * screen.outer_radius_m >= min(nx * dx_m, ny * dy_m):
        raise ValueError("screen must fit strictly inside the transverse FFT window")
    x = (np.arange(nx) - (nx - 1) / 2) * dx_m
    y = (np.arange(ny) - (ny - 1) / 2) * dy_m
    xx, yy = np.meshgrid(x, y)
    # The screen mask uses origin, but the *charge field* uses its intercept.
    metal_r = np.hypot(xx, yy)
    metal = (metal_r >= screen.inner_radius_m) & (metal_r <= screen.outer_radius_m)
    rx, ry = xx - screen.electron_x_m, yy - screen.electron_y_m
    r = np.hypot(rx, ry)
    if np.any(metal & (r == 0)):
        raise ValueError("electron intercept exactly at a metal grid node: shift/refine mesh")
    result = np.zeros((2, ny, nx), complex)
    if np.any(metal):
        beta = screen.beta_xyz
        beta_abs = float(np.linalg.norm(beta))
        unit = beta / beta_abs
        speed = beta_abs * _C_M_S
        gamma = 1.0 / np.sqrt(1.0 - beta_abs**2)
        omega = energy_eV / _HBAR_EV_S
        # The virtual-photon field is Lorentz-contracted perpendicular to the
        # *3-D* trajectory, not in the screen's x-y plane. Its extra K0 term
        # couples the longitudinal field into Ex/Ey for oblique incidence.
        projection = unit[0] * rx[metal] + unit[1] * ry[metal]
        perp_x = rx[metal] - unit[0] * projection
        perp_y = ry[metal] - unit[1] * projection
        perp_z = -unit[2] * projection
        a = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2) / gamma
        if np.any(a <= 0):
            raise ValueError("electron trajectory intersects the sampled PEC node")
        argument = omega * a / speed
        phase = np.exp(1j * omega * (screen.arrival_time_s + projection / speed))
        coefficient = (
            -screen.charge_c * omega / (2.0 * np.pi * _EPS0_F_M * gamma**2 * speed**2)
        ) * phase
        k_one = k1(argument)
        k_zero = k0(argument)
        result[0, metal] = coefficient * (perp_x / a * k_one - 1j * unit[0] * k_zero)
        result[1, metal] = coefficient * (perp_y / a * k_one - 1j * unit[1] * k_zero)
        if not np.all(np.isfinite(result)):
            raise ValueError("PEC source exceeded finite spectral-field range; refine input")
    return ScreenSpectralField(energy_eV, dx_m, dy_m, result)


def backward_formation_length_ang(
    energy_eV: float, screen: PECCircularScreen, *, theta_max_rad: float
) -> float:
    """Conservative 1-radian backward-TR kinematic phase length [Å].

    On a straight path the source/observed-wave phase accumulates at rate
    ``k0*(1-beta·n)`` per ``beta*c*dt``. Across an outgoing +z collection
    cone of polar angle <= theta_max, the slowest rate is
    ``k0*[1 - (|beta_perp|*sin(theta_max) + beta_z*cos(theta_max))]``.
    The returned distance along the incident trajectory is ``|beta|/rate``.
    It is a *diagnostic scale*, not a rigorous finite-history error bound or
    a physical straight-through PEC foil model. For a bounded trajectory,
    a caller can require multiple such scales before treating the charge as
    asymptotically straight; no default correction is applied.
    """
    if not np.isfinite(energy_eV) or energy_eV <= 0:
        raise ValueError("energy_eV must be finite and positive")
    if not np.isfinite(theta_max_rad) or not 0 <= theta_max_rad < np.pi / 2:
        raise ValueError("theta_max_rad must lie in [0, pi/2)")
    transverse = np.hypot(screen.beta_x, screen.beta_y)
    slowest = 1 - (transverse * np.sin(theta_max_rad) + screen.beta_z * np.cos(theta_max_rad))
    length_m = np.linalg.norm(screen.beta_xyz) * _HBAR_C_EV_M / (energy_eV * slowest)
    return float(length_m * 1e10)


def propagate_screen_field(field: ScreenSpectralField, distance_m: float) -> ScreenSpectralField:
    """Outgoing *vector* angular-spectrum propagation, preserving V*s/m units.

    The x and y components obey the source-free Helmholtz equation, with
    ``Ez=-(kx Ex + ky Ey)/kz`` for every propagating component. Evanescent
    components decay forward. There is no distance-to-far-field substitution.
    Periodic FFT wraparound requires separate padding/window convergence.

    Validation: optical-wavefront-propagation
    """
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("distance_m must be finite and nonnegative")
    if distance_m == 0:
        return replace(field, electric_xy_v_s_m=field.electric_xy_v_s_m.copy())
    _, _, _, kz = _wavevectors(field)
    spectrum = np.fft.fft2(field.electric_xy_v_s_m, axes=(-2, -1))
    electric = np.fft.ifft2(spectrum * np.exp(1j * kz * distance_m), axes=(-2, -1))
    return replace(field, electric_xy_v_s_m=electric, z_m=field.z_m + distance_m)


def observe_screen_green(
    field: ScreenSpectralField, observation_xyz_m, *, max_interactions: int = 250_000
) -> np.ndarray:
    """Direct free-space vector Green evaluation, Eq. (19)--(20) of [#616].

    Return the complex *temporal-Fourier* vector (Ex,Ey,Ez) [V*s/m] at
    arbitrary observation points with positive z **relative to the field's
    reference plane**. Source is the Schelkunoff magnetic-equivalent sheet
    ``2 (zhat cross E_s)``; using the retarded vacuum Green function
    ``G(R)=exp(+ik0 R)/(4*pi*R)``,

    ``E(r) = 2 int_surface [grad_r G(r-r') cross (zhat cross E_s(r'))] dA'``.

    This is equivalent to the *continuous*, nonperiodic outgoing angular
    spectrum, and includes exact 1/R and 1/R² terms. A finite FFT plane
    periodically repeats the source, whereas this quadrature does not. At
    very small distances, the pixel-centre quadrature requires refinement;
    the caller must converge both source sampling and observation distances.
    O(number of metal pixels x observation points), CPU reference only.

    Validation: optical-finite-boundary-field
    """
    points = np.asarray(observation_xyz_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("observation_xyz_m must be a finite (n,3) array")
    if np.any(points[:, 2] <= field.z_m):
        raise ValueError("all observation z positions must lie downstream of the source plane")
    if not isinstance(max_interactions, int) or max_interactions < 1:
        raise ValueError("max_interactions must be a positive integer")
    source = field.electric_xy_v_s_m
    keep = np.any(source != 0, axis=0)
    result = np.zeros((points.shape[0], 3), complex)
    if not np.any(keep) or points.shape[0] == 0:
        return result
    ny, nx = source.shape[1:]
    x = (np.arange(nx) - (nx - 1) / 2) * field.dx_m
    y = (np.arange(ny) - (ny - 1) / 2) * field.dy_m
    xx, yy = np.meshgrid(x, y)
    xs, ys = xx[keep], yy[keep]
    ex, ey = source[0, keep], source[1, keep]
    k0 = field.energy_eV / _HBAR_C_EV_M
    # Chunk both source and observation work to bound peak temporaries.
    for i, point in enumerate(points):
        for start in range(0, xs.size, max_interactions):
            stop = start + max_interactions
            rx = point[0] - xs[start:stop]
            ry = point[1] - ys[start:stop]
            rz = point[2] - field.z_m
            rr = np.sqrt(rx * rx + ry * ry + rz * rz)
            kernel = np.exp(1j * k0 * rr) * (1j * k0 * rr - 1) / (4 * np.pi * rr**3)
            # curl( G * 2(z cross E_s) ) evaluated with the exact dG/dR.
            result[i, 0] += np.sum(-2 * rz * kernel * ex[start:stop])
            result[i, 1] += np.sum(-2 * rz * kernel * ey[start:stop])
            result[i, 2] += np.sum(2 * kernel * (rx * ex[start:stop] + ry * ey[start:stop]))
    result *= field.dx_m * field.dy_m
    return result


def _propagating_spectral_flux(field: ScreenSpectralField):
    k0, kx, ky, kz = _wavevectors(field)
    propagated = (kx * kx + ky * ky) < k0 * k0 * (1 - 1e-12)
    fft = np.fft.fft2(field.electric_xy_v_s_m, axes=(-2, -1))
    kz_safe = np.where(propagated, kz, 1.0)
    ez = -(kx * fft[0] + ky * fft[1]) / kz_safe
    norm = abs(fft[0]) ** 2 + abs(fft[1]) ** 2 + abs(ez) ** 2
    flux = np.where(propagated, (kz.real / k0) * norm, 0.0)
    return k0, kx, ky, kz, propagated, fft, ez, flux


def spectral_energy_per_eV(field: ScreenSpectralField) -> float:
    """Positive-z energy density dU/dE [J/eV], per physical electron.

    With a *two-sided* Fourier transform, positive frequencies have
    dU/domega = (pi*Z0)^-1 times the outgoing normal flux sum over
    propagating modes of (kz/k0)|Etilde(k)|^2 (paraxially int(dA)|Etilde|^2),
    not the time-averaged V/m phasor expression. Includes the longitudinal
    field and obliquity factors;
    evanescent components do not transport net normal energy.

    This is a Riemann sum on the periodic FFT k-grid. The k-space flux
    density carries an integrable 1/kz singularity on the light circle (the
    |Ez|² term), so the sum converges slowly and non-monotonically with the
    FFT window (errors of several percent are typical even when pitch is
    converged). For a compact source use :func:`hemisphere_energy_per_eV`.
    """
    *_, flux = _propagating_spectral_flux(field)
    ny, nx = field.electric_xy_v_s_m.shape[1:]
    return float(np.sum(flux) * field.dx_m * field.dy_m / (nx * ny * np.pi * _Z0_OHM * _HBAR_EV_S))


def hemisphere_energy_per_eV(
    field: ScreenSpectralField, *, n_u: int = 64, n_phi: int = 128, chunk: int = 512
) -> float:
    """Outgoing dU/dE [J/eV] by window-independent angular quadrature.

    The sampled surface field is Fourier transformed *directly* (no periodic
    images) at Gauss--Legendre nodes in ``u = cos(theta)`` and uniform nodes in
    phi. With ``d²k = k0² u du dphi`` the light-circle 1/kz singularity of the
    k-space flux density cancels, so the integrand is smooth and the rule
    converges rapidly. Same normalization as :func:`spectral_energy_per_eV`.
    The field must be compact inside the window (zero on its border), as the
    PEC source is when its metal lies inside the grid; propagated fields are
    refused. Cost scales as nonzero pixels x ``n_u * n_phi``.

    Validation: optical-finite-boundary-field
    """
    e = field.electric_xy_v_s_m
    if min(n_u, n_phi, chunk) < 1:
        raise ValueError("quadrature sizes must be positive")
    border = np.concatenate([e[:, 0, :], e[:, -1, :], e[:, :, 0], e[:, :, -1]], axis=1)
    if np.any(border != 0):
        raise ValueError("field is not compact inside the window; quadrature needs a source")
    ny, nx = e.shape[1:]
    k0 = field.energy_eV / _HBAR_C_EV_M
    x = (np.arange(nx) - (nx - 1) / 2) * field.dx_m
    y = (np.arange(ny) - (ny - 1) / 2) * field.dy_m
    nonzero = np.any(e != 0, axis=0)
    rows, cols = np.nonzero(nonzero)
    xs, ys = x[cols], y[rows]
    ex, ey = e[0][nonzero] * field.dx_m * field.dy_m, e[1][nonzero] * field.dx_m * field.dy_m
    t, w = np.polynomial.legendre.leggauss(n_u)
    u, wu = (t + 1) / 2, w / 2
    phi = np.arange(n_phi) * (2 * np.pi / n_phi)
    uu, pp = (a.ravel() for a in np.meshgrid(u, phi, indexing="ij"))
    weights = np.repeat(wu, n_phi) * (2 * np.pi / n_phi)
    kx = k0 * np.sqrt(1 - uu * uu) * np.cos(pp)
    ky = k0 * np.sqrt(1 - uu * uu) * np.sin(pp)
    kz = k0 * uu
    total = 0.0
    for lo in range(0, uu.size, chunk):
        sl = slice(lo, lo + chunk)
        phase = np.exp(-1j * (np.outer(kx[sl], xs) + np.outer(ky[sl], ys)))
        fx, fy = phase @ ex, phase @ ey
        fz = -(kx[sl] * fx + ky[sl] * fy) / kz[sl]
        norm = np.abs(fx) ** 2 + np.abs(fy) ** 2 + np.abs(fz) ** 2
        # (kz/k0)|F|²/(4 pi²) per d²k, with d²k = k0² u du dphi.
        total += float(np.sum(weights[sl] * kz[sl] * k0 * uu[sl] * norm))
    return total / (4 * np.pi**2 * np.pi * _Z0_OHM * _HBAR_EV_S)


def far_field_photons_per_eV_sr(field: ScreenSpectralField):
    """Grid-sampled far-zone d²N/(dE dOmega), [photons/eV/sr].

    Returns (theta_rad, phi_rad, yield_density), each (ny,nx), on *only* the
    outgoing propagating FFT angular spectrum. Evanescent / near-grazing modes
    are assigned exactly zero yield. This is the angular distribution of the
    already finite-screen source; it is not a detector map or solid-angle
    integral, and the discrete angular mesh must be converged independently.
    """
    k0, kx, ky, kz, good, fft, ez, _ = _propagating_spectral_flux(field)
    ft_power = (abs(fft[0]) ** 2 + abs(fft[1]) ** 2 + abs(ez) ** 2) * (field.dx_m * field.dy_m) ** 2
    cosine = np.where(good, kz.real / k0, 0.0)
    # Jacobian d(kx ky) = k0² cos(theta) dOmega. Field amplitude at infinity
    # carries a second cos(theta), giving k0² cos²(theta) |FT|².
    du_domega_sr = k0**2 * cosine**2 * ft_power / (4 * np.pi**3 * _Z0_OHM)
    density = np.where(good, du_domega_sr / (_HBAR_EV_S * field.energy_eV * _E_CHARGE_C), 0.0)
    theta = np.where(good, np.arccos(np.clip(cosine, -1, 1)), np.nan)
    phi = np.broadcast_to(np.arctan2(ky, kx), cosine.shape).copy()
    return theta, phi, density


def ginzburg_frank_photons_per_eV_sr(energy_eV, theta_rad, beta: float):
    """Ideal infinite-PEC Ginzburg–Frank single-charge backward TR [ph/eV/sr].

    This is an **independent analytical limit**, not the finite-screen solver.
    ``d²U/(domega dOmega) = e² beta² sin²(theta) /
        (4 pi³ eps0 c (1-beta² cos²(theta))²)`` in SI.
    """
    e = np.asarray(energy_eV, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    if (
        np.any(~np.isfinite(e) | (e <= 0))
        or np.any(~np.isfinite(theta))
        or not np.isfinite(beta)
        or not 0 < abs(beta) < 1
    ):
        raise ValueError("energy/theta/beta must be finite, E>0 and 0<|beta|<1")
    sin = np.sin(theta)
    dU_domega = (
        _E_CHARGE_C**2
        * beta**2
        * sin**2
        / (4 * np.pi**3 * _EPS0_F_M * _C_M_S * (1 - beta**2 * np.cos(theta) ** 2) ** 2)
    )
    return dU_domega / (_HBAR_EV_S * e * _E_CHARGE_C)
