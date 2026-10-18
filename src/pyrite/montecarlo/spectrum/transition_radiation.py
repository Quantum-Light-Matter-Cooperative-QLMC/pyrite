"""Exact classical transition radiation of a straight charge through a planar stack.

Screening tool for the transition-radiation (TR) omission study (#102): TR is
not a production emission component (that is #103). This module computes the
radiation a point charge on a straight, uniform-velocity trajectory emits into
the vacuum above a planar stack of homogeneous, isotropic layers, and compares
it with a production source density.

Method (Ginzburg--Frank boundary-value problem; Ginzburg & Tsytovich,
*Transition Radiation and Transition Scattering*, 1990, Ch. 2; Jackson,
*Classical Electrodynamics* 3rd ed., Sec. 13.7). Gaussian units with
``hbar = c = 1``, energies in eV. For ``rho = e delta(r - v t)`` the charge's
own field in an unbounded medium ``eps`` has, at transverse wavevector
``k_perp`` and frequency ``omega``, the single longitudinal wavenumber
``q = (omega - k_perp . v_perp) / v_z`` and

    phi = 4 pi e / (|v_z| eps (k^2 - eps omega^2)),
    E_p = i phi (eps omega v - k),   H_p = i eps phi (k x v),   k = (k_perp, q).

In every layer the total field is that particular field plus free plane waves
``K = (k_perp, +-lambda)``, ``lambda = sqrt(eps omega^2 - k_perp^2)``
(``Im lambda >= 0``); the semi-infinite media carry only outgoing waves.
Continuity of tangential ``E`` and ``H`` at every interface fixes all free
amplitudes in one ``4 N_interface`` linear solve. Each free wave is referenced
to the interface it decays away from, so absorbing layers never overflow.
Stationary phase turns the vacuum amplitude ``a(k_perp)`` at
``k_perp = omega n_perp`` into the far field, giving

    d^2N / dE dOmega = omega cos^2(theta) |a|^2 / (16 pi^4)   [photons/eV/sr],

summed over both polarizations. Interference between interfaces, formation
zones, absorption (complex ``eps``), backward emission, oblique incidence and
the Cherenkov regime of a layer are all contained in the same solve.

Assumptions: classical radiation (photon energy far below the electron kinetic
energy, no recoil); a straight, constant-velocity trajectory through the whole
stack (no scattering or energy loss across the formation zones); infinite,
sharp, flat interfaces; local isotropic ``eps(omega)``, here
:func:`medium_dielectric`, which is the independent-atom Chantler forward
scattering factor and therefore crude near and below the plasma energy. No
diffracted transition radiation: the layers have no lattice periodicity.

Limits (checked in ``tests/montecarlo/test_transition_radiation.py``): a matched
medium or a vanishing film radiates nothing; a single interface at normal
incidence reproduces the closed Ginzburg--Frank form
:func:`ginzburg_frank_interface` in both directions; a conductor
``|eps| -> infinity`` at any incidence reproduces the charge-plus-image
sudden-stop amplitude; an opaque film reduces to its exit interface; a
transparent film averages to twice one interface and peaks at four times; the
relativistic small-angle X-ray limit is the Jackson Eq. 13.91 form used by
:mod:`pyrite.montecarlo.channeling.transition_radiation`.

Validation: tr-detected-omission-bound
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...materials.atomic import henke_dispersion
from ...materials.attenuation import _resolve_composition
from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG, HC_EV_ANG, M_E_EV, R_E_ANG, Z_TABLE

__all__ = [
    "TransitionRadiationOmission",
    "beta_from_kinetic_eV",
    "ginzburg_frank_interface",
    "interface_transition_radiation_amplitude",
    "medium_dielectric",
    "omission_tolerance",
    "planar_transition_radiation",
    "transition_radiation_omission",
]


def beta_from_kinetic_eV(kinetic_energy_eV: float) -> float:
    """Electron speed ``v/c`` for kinetic energy ``T`` in eV."""
    gamma = 1.0 + float(kinetic_energy_eV) / M_E_EV
    return float(np.sqrt(1.0 - 1.0 / gamma**2))


def medium_dielectric(material, photon_energy_eV) -> np.ndarray:
    """Complex relative permittivity of a catalog medium, ``exp(-i omega t)`` convention.

    ``eps = 1 - (r_e lambda^2 / pi) sum_i n_i (Z_i + f'_i - i f''_i)``, the exact
    ``k^2 = eps omega^2`` relation behind :func:`pyrite.materials.crystal.chi_0`
    (``eps = 1 + conj(chi_0)``: that module uses ``exp(+i omega t)``), extended to
    any medium the attenuation tables resolve. ``Im eps >= 0`` is absorption.
    Independent-atom forward scattering: reliable well above ``hbar omega_p``
    and away from edges, crude near and below the plasma energy. Energies
    outside the Chantler table give NaN.

    Validation: tr-detected-omission-bound
    """
    energy = np.asarray(photon_energy_eV, dtype=float)
    wavelength = HC_EV_ANG / energy
    total = np.zeros(energy.shape, dtype=complex)
    for element, density in _resolve_composition(material):
        f_prime, f_double_prime = henke_dispersion(element, energy)
        total = total + density * (Z_TABLE[element] + f_prime - 1j * f_double_prime)
    return 1.0 - R_E_ANG * wavelength**2 / np.pi * total


def ginzburg_frank_interface(eps, beta, theta_rad, photon_energy_eV) -> np.ndarray:
    """Closed-form single-interface TR at normal incidence [photons/eV/sr].

    Charge leaving a medium ``eps`` into vacuum; ``theta`` from the velocity:

        d^2N/dE dOmega = (alpha beta^2 sin^2 cos^2 / (pi^2 E))
            |(eps - 1)(1 - beta^2 - beta s) /
             ((1 - beta^2 cos^2)(1 - beta s)(eps cos + s))|^2,
        s = sqrt(eps - sin^2 theta).

    A charge entering the medium radiates backward into the vacuum by the same
    expression with ``beta -> -beta`` and ``theta`` from the reversed velocity
    (Ginzburg & Tsytovich 1990, Ch. 2; DeRaad, Tsai & Erber, Phys. Rev. D 18,
    2152 (1978) give its angle integral; Jackson 3rd ed. Eq. 13.91 is its
    ``gamma >> 1``, small-angle X-ray limit). A perfect conductor gives
    ``alpha beta^2 sin^2 / (pi^2 E (1 - beta^2 cos^2)^2)`` for either sign.

    Validation: tr-detected-omission-bound
    """
    eps = np.asarray(eps, dtype=complex)
    theta = np.asarray(theta_rad, dtype=float)
    beta = np.asarray(beta, dtype=float)
    sin2 = np.sin(theta) ** 2
    cos = np.cos(theta)
    root = np.sqrt(eps - sin2)
    amplitude = (eps - 1.0) * (1.0 - beta**2 - beta * root)
    denominator = (1.0 - beta**2 * cos**2) * (1.0 - beta * root) * (eps * cos + root)
    return (
        ALPHA_FS
        * beta**2
        * sin2
        * cos**2
        / (np.pi**2 * np.asarray(photon_energy_eV, dtype=float))
        * np.abs(amplitude / denominator) ** 2
    )


def _outgoing_root(value, xp=np):
    root = xp.sqrt(value)
    flip = (root.imag < 0) | ((root.imag == 0) & (root.real < 0))
    return xp.where(flip, -root, root)


def _cross(a, b, xp=np):
    return xp.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def _checked_kinematics(energy, n, v):
    if np.any(energy <= 0.0):
        raise ValueError("photon_energy_eV must be positive")
    if np.any(n[..., 2] <= 0.0):
        raise ValueError("direction must point into the vacuum above the stack (n_z > 0)")
    speed = np.linalg.norm(v, axis=-1)
    if np.any((speed >= 1.0) | (v[..., 2] == 0.0)):
        raise ValueError("velocity must satisfy |v| < 1 and v_z != 0")


def planar_transition_radiation(
    photon_energy_eV,
    direction,
    velocity,
    layers: Sequence[tuple[complex | np.ndarray, float]],
    substrate_eps=1.0,
    *,
    xp=np,
    return_amplitude: bool = False,
):
    """Transition radiation into the vacuum above a planar stack [photons/eV/sr].

    Parameters
    ----------
    photon_energy_eV
        Photon energies ``E > 0`` in eV.
    direction
        Unit emission directions ``(..., 3)`` with ``n_z > 0`` (into the vacuum
        above the stack, whose normal is ``+z``).
    velocity
        Electron velocity ``v/c`` ``(..., 3)``, ``0 < |v| < 1``, ``v_z != 0``.
        ``v_z > 0`` leaves the stack into the vacuum (forward TR); ``v_z < 0``
        enters it from the vacuum (backward TR).
    layers
        ``(eps, thickness_ang)`` from the substrate side up to the vacuum. Each
        ``eps`` is a scalar or broadcasts against the energies. Empty for a
        single interface.
    substrate_eps
        Permittivity of the semi-infinite medium below the stack; ``1`` for a
        free-standing film.
    xp
        Array module for the solve (NumPy, or CuPy for the CUDA backend). The
        arithmetic is complex128 on every backend; inputs are validated on the
        host.

    return_amplitude
        If true, return ``(yield, amplitude)``. The default returns only yield.

    Returns
    -------
    xp.ndarray or tuple[xp.ndarray, xp.ndarray]
        Spectral-angular yield per crossing electron, broadcast over the
        energy, direction and velocity shapes, summed over polarization.
        With ``return_amplitude=True``, the second array is the outgoing complex
        electric-field vector ``a`` with shape ``(*yield.shape, 3)`` in the
        stack's Cartesian ``(x, y, z)`` basis. Both arrays stay on ``xp``.
        This is the existing transverse Fourier field amplitude [1/eV in the
        Gaussian, ``hbar = c = 1`` convention above], not a photon amplitude:
        ``yield = E * n_z**2 * sum(abs(a)**2) / (16 * pi**4)``.

        The free vacuum wave uses ``exp(-i omega t)`` and is referenced to the
        top interface ``z_top = sum(thickness_ang)``: its spatial factor is
        ``exp[i(k_perp . r_perp + lambda_vac (z - z_top))]`` (lengths converted
        to 1/eV). The driving charge passes ``r = 0`` at ``t = 0`` at the
        substrate-side interface. No extra position/time translation phase or
        far-field stationary-phase prefactor is applied to the returned vector.
        Inter-electron or cross-component coherent summation is not implemented.

    Validation: tr-detected-omission-bound
    """
    energy = np.asarray(photon_energy_eV, dtype=float)
    n = np.asarray(direction, dtype=float)
    v = np.asarray(velocity, dtype=float)
    shape = np.broadcast_shapes(energy.shape, n.shape[:-1], v.shape[:-1])
    _checked_kinematics(energy, n, v)
    omega = xp.asarray(np.broadcast_to(energy, shape).ravel())
    n = xp.asarray(np.broadcast_to(n, (*shape, 3)).reshape(-1, 3))
    v = xp.asarray(np.broadcast_to(v, (*shape, 3)).reshape(-1, 3))
    media = _stack_media(substrate_eps, layers, shape, xp)
    z_if = _interface_heights(layers)
    k_perp = omega[:, None] * n[:, :2]
    q = (omega - xp.sum(k_perp * v[:, :2], axis=-1)) / v[:, 2]
    matrix, top, lam = _free_system(omega, k_perp, media, z_if, xp)
    particular = [_particular_field(eps, omega, k_perp, q, v, xp) for eps in media]
    rhs = xp.zeros(matrix.shape[:2], dtype=complex)
    for i, z0 in enumerate(z_if):
        row = 4 * i
        phase = xp.exp(1j * q * z0)[:, None]
        for j, side in ((i + 1, +1.0), (i, -1.0)):
            e_vec, h_vec = particular[j]
            rhs[:, row : row + 2] -= side * e_vec[:, :2] * phase
            rhs[:, row + 2 : row + 4] -= side * h_vec[:, :2] * phase

    solution = xp.linalg.solve(matrix, rhs[..., None])[..., 0]
    e_t = solution[:, top : top + 2]
    e_z = -xp.sum(k_perp * e_t, axis=-1) / lam[-1]
    power = xp.sum(xp.abs(e_t) ** 2, axis=-1) + xp.abs(e_z) ** 2
    yield_ = omega * n[:, 2] ** 2 * power / (16.0 * np.pi**4)
    yield_ = yield_.reshape(shape)
    if return_amplitude:
        amplitude = xp.concatenate([e_t, e_z[:, None]], axis=-1)
        return yield_, amplitude.reshape((*shape, 3))
    return yield_


def _stack_media(substrate_eps, layers, shape, xp):
    """Flattened permittivities, substrate first and the vacuum above last."""

    def _medium(eps):
        host = eps.get() if hasattr(eps, "get") else eps  # CuPy -> host for broadcasting
        return xp.asarray(np.broadcast_to(np.asarray(host, dtype=complex), shape).ravel())

    media = [_medium(substrate_eps)] + [_medium(e) for e, _ in layers]
    media.append(xp.ones(int(np.prod(shape, dtype=int)), dtype=complex))
    return media


def _interface_heights(layers):
    """Interface heights in 1/eV (hbar = c = 1), the substrate interface at 0."""
    return np.concatenate([[0.0], np.cumsum([float(d) for _, d in layers])]) / HBARC_EV_ANG


def _free_system(omega, k_perp, media, z_if, xp):
    """Tangential E/H continuity matrix of the free waves, its top column, and ``lambda``."""
    batch, n_media = omega.size, len(media)
    k_perp2 = xp.sum(k_perp**2, axis=-1)
    columns: dict[tuple[int, int], int] = {}
    for j in range(n_media):
        for wave in (+1, -1):
            if (wave == +1 and j > 0) or (wave == -1 and j < n_media - 1):
                columns[(j, wave)] = 2 * len(columns)
    size = 2 * len(columns)
    matrix = xp.zeros((batch, size, size), dtype=complex)

    lam = [_outgoing_root(eps * omega**2 - k_perp2, xp) for eps in media]
    unit = np.eye(2)
    free = {}
    for (j, wave), _ in columns.items():
        kz = wave * lam[j]
        wavevector = xp.concatenate([k_perp, kz[:, None]], axis=-1)
        fields = []
        for comp in range(2):
            e_t = xp.broadcast_to(xp.asarray(unit[comp], dtype=complex), (batch, 2))
            e_z = -xp.sum(k_perp * e_t, axis=-1) / kz
            e_vec = xp.concatenate([e_t, e_z[:, None]], axis=-1)
            fields.append((e_vec, _cross(wavevector, e_vec, xp) / omega[:, None]))
        free[(j, wave)] = fields

    for i, z0 in enumerate(z_if):
        row = 4 * i
        for j, side in ((i + 1, +1.0), (i, -1.0)):
            for wave in (+1, -1):
                if (j, wave) not in columns:
                    continue
                z_ref = z_if[j - 1] if wave == +1 else z_if[j]
                phase = xp.exp(1j * wave * lam[j] * (z0 - z_ref))
                for comp, (e_vec, h_vec) in enumerate(free[(j, wave)]):
                    col = columns[(j, wave)] + comp
                    matrix[:, row : row + 2, col] += side * e_vec[:, :2] * phase[:, None]
                    matrix[:, row + 2 : row + 4, col] += side * h_vec[:, :2] * phase[:, None]
    return matrix, columns[(n_media - 1, +1)], lam


def _particular_field(eps, omega, k_perp, q, v, xp):
    """Charge ``+sqrt(alpha)`` own field ``(E_p, H_p)`` at its own crossing, phase 1.

    Broadcasts over leading axes: ``eps``, ``omega`` and ``q`` share a shape
    ``s``; ``k_perp`` and ``v`` are ``(*s, 2)`` and ``(*s, 3)``.
    """
    wavevector = xp.concatenate(
        [xp.broadcast_to(k_perp, (*q.shape, 2)), q[..., None]], axis=-1
    ).astype(complex)
    v_complex = xp.broadcast_to(v, (*q.shape, 3)).astype(complex)
    denom = xp.sum(wavevector**2, axis=-1) - eps * omega**2
    phi = 4.0 * np.pi * np.sqrt(ALPHA_FS) / (xp.abs(v_complex[..., 2].real) * eps * denom)
    e_vec = 1j * phi[..., None] * (eps[..., None] * omega[..., None] * v_complex - wavevector)
    h_vec = 1j * (eps * phi)[..., None] * _cross(wavevector, v_complex, xp)
    return e_vec, h_vec


def interface_transition_radiation_amplitude(
    photon_energy_eV,
    direction,
    layers: Sequence[tuple[complex | np.ndarray, float]],
    substrate_eps,
    velocity,
    time_ang,
    position_ang,
    interface,
    *,
    xp=np,
):
    """Vacuum amplitude of each interface crossing of a planar stack [1/eV].

    The solve of :func:`planar_transition_radiation` is linear in its source:
    the right-hand side at interface ``i`` is the jump of the charge's own
    field between the media on either side, ``-(P_{i+1} - P_i)``, with phase
    ``exp(i q z_i)``. For a charge through the origin at ``t = 0`` that phase is
    ``exp[i(omega t_i - k_perp . r_perp,i)]`` at its crossing ``(r_i, t_i)``, so
    the vacuum amplitude is a sum over crossings of

        a_c = G_i [-(P_{i+1}(v_c) - P_i(v_c))] exp[i(omega t_c - k_perp . r_perp,c)],

    ``G_i`` the top-wave response of the stack to a tangential jump at
    interface ``i``, with every layer's propagation, absorption and multiple
    reflection, and ``P(v_c)`` the own field of an unbounded straight charge at
    the crossing's velocity. A straight trajectory's crossings sum exactly to
    the ``return_amplitude`` vector of :func:`planar_transition_radiation`.
    For a scattered trajectory each crossing is treated as a locally straight
    passage at its own velocity; radiation from deflections between crossings
    is bremsstrahlung, not this term.

    Parameters
    ----------
    photon_energy_eV, direction
        Photon energies and unit emission directions (``n_z > 0``), broadcast
        to a shape ``s``.
    layers, substrate_eps
        The stack, as for :func:`planar_transition_radiation` (substrate side
        first; the vacuum above is the emission side).
    velocity, time_ang, position_ang, interface
        ``K`` crossings in the same frame: velocity ``v/c`` ``(K, 3)`` with
        ``0 < |v| < 1`` and ``v_z != 0``, time ``(K,)`` and position ``(K, 3)`` in
        Å (``c = 1``; only ``x, y`` enter), interface index ``(K,)`` in
        ``0..len(layers)``.

    Returns
    -------
    xp.ndarray
        Complex Cartesian vectors ``(*s, K, 3)`` in the ``return_amplitude``
        convention (charge ``+e``; scale by the charge sign): far-field origin
        ``(0, 0, z_top)``, ``exp(-i omega t)``, and
        ``d^2N/dE dOmega = E n_z^2 |sum_c a_c|^2 / (16 pi^4)``.

    Validation: tr-crossing-superposition
    """
    energy = np.asarray(photon_energy_eV, dtype=float)
    n = np.asarray(direction, dtype=float)
    shape = np.broadcast_shapes(energy.shape, n.shape[:-1])
    v = np.atleast_2d(np.asarray(velocity, dtype=float))
    face = np.asarray(interface, dtype=np.int64).reshape(-1)
    _checked_kinematics(energy, n, v)
    if np.any((face < 0) | (face > len(layers))):
        raise ValueError("interface must index 0..len(layers)")
    omega = xp.asarray(np.broadcast_to(energy, shape).ravel())
    n = xp.asarray(np.broadcast_to(n, (*shape, 3)).reshape(-1, 3))
    media = _stack_media(substrate_eps, layers, shape, xp)
    k_perp = omega[:, None] * n[:, :2]
    matrix, top, lam = _free_system(omega, k_perp, media, _interface_heights(layers), xp)
    # Top-wave rows of the inverse: solve the transposed system once.
    select = xp.zeros((omega.size, matrix.shape[1], 2), dtype=complex)
    select[:, top, 0] = select[:, top + 1, 1] = 1.0
    response = xp.linalg.solve(xp.swapaxes(matrix, 1, 2), select)  # (B, size, 2)
    response = response.reshape(omega.size, -1, 4, 2)  # rows 4i..4i+3: interface i

    v_dev = xp.asarray(v)[None]
    q = (omega[:, None] - xp.sum(k_perp[:, None] * v_dev[..., :2], axis=-1)) / v_dev[..., 2]
    eps = xp.stack(media)  # (n_media, B)
    face_dev = xp.asarray(face)
    jump = []
    for offset, sign in ((1, -1.0), (0, 1.0)):
        e_vec, h_vec = _particular_field(
            eps[face_dev + offset].T, omega[:, None], k_perp[:, None], q, v_dev, xp
        )
        jump.append(sign * xp.concatenate([e_vec[..., :2], h_vec[..., :2]], axis=-1))
    r_perp = xp.asarray(np.atleast_2d(np.asarray(position_ang, dtype=float))[:, :2])
    phase = xp.exp(
        1j
        * (
            omega[:, None] * xp.asarray(np.asarray(time_ang, dtype=float).reshape(-1))
            - xp.sum(k_perp[:, None] * r_perp[None], axis=-1)
        )
        / HBARC_EV_ANG
    )
    source = (jump[0] + jump[1]) * phase[..., None]  # (B, K, 4)
    e_t = xp.einsum("bkrc,bkr->bkc", response[:, face_dev], source)
    e_z = -xp.sum(k_perp[:, None] * e_t, axis=-1) / lam[-1][:, None]
    return xp.concatenate([e_t, e_z[..., None]], axis=-1).reshape((*shape, face.size, 3))


def omission_tolerance(target_rse: float) -> float:
    """Largest acceptable relative omission bias for a precision target.

    An omitted, deterministic contribution biases an observable by ``b``; the
    total error ``sqrt(rse^2 + b^2)`` stays within 5% of the statistical error
    when ``b <= rse / 3`` (``sqrt(1 + 1/9) = 1.054``). For a fixed-count run pass
    the observable's realized relative standard error.

    Validation: tr-detected-omission-bound
    """
    rse = float(target_rse)
    if not np.isfinite(rse) or rse <= 0.0:
        raise ValueError("target_rse must be finite and positive")
    return rse / 3.0


@dataclass(frozen=True)
class TransitionRadiationOmission:
    """Pointwise TR-to-source bound and its verdict against a tolerance.

    Attributes
    ----------
    photon_energy_eV
        Energies at which the source density was compared [eV].
    transition_radiation
        Upper bound on TR per incident electron [photons/eV/sr].
    ratio
        ``transition_radiation / source_density``.
    bias_bound
        ``max(ratio)`` over the band: for any detector response, filter, and
        acceptance that act as one positive linear map on both densities, the
        relative change of every non-negative observable drawn from this band
        is at most this value.
    tolerance
        Tolerance from :func:`omission_tolerance`.
    omit
        ``bias_bound <= tolerance``.
    """

    photon_energy_eV: np.ndarray
    transition_radiation: np.ndarray
    ratio: np.ndarray
    bias_bound: float
    tolerance: float
    omit: bool


def transition_radiation_omission(
    photon_energy_eV,
    source_density,
    transition_radiation,
    *,
    target_rse: float,
    band_eV: tuple[float, float] | None = None,
) -> TransitionRadiationOmission:
    """Bound the detected-observable change from omitting transition radiation.

    ``source_density`` is the production density the detector already scores
    (photons/eV/sr per incident electron); taking only the continuum keeps the
    bound conservative. ``transition_radiation`` is an upper bound on the TR
    density in the same units at the same energies, already maximized over the
    acceptance directions and multiplied by the number of boundary crossings
    per incident electron. If ``TR(E) <= c S(E)`` at every energy that can
    reach an observable, any positive linear response ``R`` (filters, quantum
    efficiency, resolution, redistribution, solid-angle integration) gives
    ``R[TR] <= c R[S]`` pointwise, so the relative change of any non-negative
    window, sideband, or total is at most ``c``. ``band_eV`` limits the
    maximum to the source energies the observable can draw on.

    A net line after sideband subtraction is not non-negative in this sense:
    with a continuum-only ``source_density`` its bias is at most ``c B / L``
    (otherwise only ``c (1 + B / L)``) for window background ``B`` and net line
    ``L``. A coherent signal is not covered: the TR amplitude can interfere
    with it (see the physics page).

    Validation: tr-detected-omission-bound
    """
    energy = np.asarray(photon_energy_eV, dtype=float)
    source = np.asarray(source_density, dtype=float)
    tr = np.broadcast_to(np.asarray(transition_radiation, dtype=float), energy.shape)
    keep = np.ones(energy.shape, dtype=bool)
    if band_eV is not None:
        keep &= (energy >= band_eV[0]) & (energy <= band_eV[1])
    if not np.any(keep):
        raise ValueError("no energies in the band")
    # A non-positive or undefined source with any TR left is an unbounded bias.
    ratio = np.full(energy.shape, np.inf)
    positive = np.isfinite(source) & (source > 0.0)
    ratio[positive] = tr[positive] / source[positive]
    ratio[~positive & (tr == 0.0)] = 0.0
    bound = float(np.max(ratio[keep]))
    tolerance = omission_tolerance(target_rse)
    return TransitionRadiationOmission(
        photon_energy_eV=energy,
        transition_radiation=np.asarray(tr, dtype=float),
        ratio=ratio,
        bias_bound=bound,
        tolerance=tolerance,
        omit=bound <= tolerance,
    )
