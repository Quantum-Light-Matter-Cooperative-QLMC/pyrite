r"""Off-axis polarization source of the carrier-fibred absorbing axial model (#992).

Research CPU reference extending :func:`.axial_fiber_regression.fiber_on_axis_source`
to a photon direction $\mathbf n=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$
about the axis $z$, with the polarization basis $\mathbf e_1=\hat\theta$,
$\mathbf e_2=\hat\phi$.

**Amplitude (acceleration form).** The first-order amplitude
$\mathbf e_\lambda\cdot\langle f|(\mathbf v/c)e^{-i\mathbf q\cdot\mathbf r_\perp}|i\rangle$
($\mathbf q=(E\sin\theta/\hbar c)\hat n_\perp$, $v_z=\beta c$) is, to transverse
dipole order with $\mathbf r_{fi}=i\hbar c\,\mathbf J_{fi}/\Delta_{fi}$,
$A_1=\hat n_\perp\cdot\mathbf J(\cos\theta-\beta E\sin^2\theta/\Delta)$ (velocity
form). Its $E$-dependent part differs from its value at the line resonance
$E(1-\beta\cos\theta)=\Delta$ by a total depth derivative of the transition's
phase, i.e. by entrance/exit boundary terms (plus, under dissipation, an
$O(\sin^2\theta\,(E(1-\beta\cos\theta)-\Delta)/\Delta)$ in-line correction).
Those boundary terms are cancelled by the vacuum side, which is not modelled
(crystal transition radiation is negligible at keV), and left in they grow as
$E$ off resonance. This module therefore uses the resonance-evaluated
(Liénard--Wiechert acceleration) form

$$
A_1=\hat n_\perp\cdot\mathbf J\,\frac{\cos\theta-\beta}{1-\beta\cos\theta},\qquad
A_2=\hat\phi\cdot\mathbf J ,
$$

the field of :func:`.dipole.dipole_far_field`. Neglected: $O(qa)$ terms ($a$
the bound-state size), which vanish at first order for parity-changing lines
and carry off-axis quadrupole/magnetic lines forbidden on axis. Hence, for an
ensemble with $C_4$ symmetry about the axis, the intensity is azimuthally
uniform at this order.

**Two-time source.** The same velocity currents as on axis are regressed
(``cr-axial-fiber-regression``), $G_{ab}(z_i,z_j)$. With
$v_i=w_iA_i(E)^{1/\cos\theta}e^{iE(1-\beta\cos\theta)z_i/\hbar v}$ (trapezoid
weights $w_i$; attenuation along $(L-z)/\cos\theta$),

$$
S_{\lambda\lambda'}(E,\mathbf n)=\frac{\alpha E}{4\pi^2(\hbar v)^2}
\sum_{ij}v_iv_j^*\sum_{ab}m_{\lambda a}m_{\lambda'b}G_{ab}(z_i,z_j)
\quad[\text{photons/(eV sr incident electron)}],
$$

$S_{\lambda\lambda'}=\langle A_\lambda A_{\lambda'}^*\rangle$. On axis its trace
is :func:`.axial_fiber_regression.signed_on_axis_spectrum`. The phase depends
on $\theta$ only. The trapezoid depth sum is periodic in the phase rate with
period $2\pi/\Delta z$ and so aliases line replicas; the phase step
$E(1-\beta\cos\theta)\Delta z/\hbar v$ is therefore capped
(``max_phase_step``).

Scope: as ``cr-axial-fiber-regression`` (absorbing retained subspace only);
transverse dipole order; no photon refraction; no longitudinal recoil.

Units: eV, Å, rad; spectra photons/(eV sr incident electron).

Validation: cr-axial-fiber-angular
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG

from .axial_fiber_regression import (
    _admit,
    _amplitude,
    _primaries,
)
from .axial_fiber_unraveling import FiberEvent
from .bloch import longitudinal_kinematics
from .quantum_regression import RegressionCorrelation


def angular_coefficients(theta_rad, phi_rad, beta: float) -> np.ndarray:
    r"""Real coefficients ``m[t, p, lambda, a]`` of $A_\lambda=\sum_a m_{\lambda a}J_a$.

    ``a`` runs over ``(J_x, J_y)``, ``lambda`` over ``(e_theta, e_phi)``:
    $m_\theta=\hat n_\perp(\cos\theta-\beta)/(1-\beta\cos\theta)$,
    $m_\phi=\hat\phi$. Shape ``(n_theta, n_phi, 2, 2)``.

    Validation: cr-axial-fiber-angular
    """
    t = np.asarray(theta_rad, dtype=float)[:, None]
    p = np.asarray(phi_rad, dtype=float)[None, :]
    t, p = np.broadcast_arrays(t, p)
    c, s = np.cos(p), np.sin(p)
    # (cos t - beta)/(1 - beta cos t) with 1 - beta cos t cancellation-free
    one_minus = (1.0 - beta) + 2.0 * beta * np.sin(t / 2) ** 2
    aberration = (np.cos(t) - beta) / one_minus
    first = np.stack([aberration * c, aberration * s], -1)
    second = np.stack([-s, c], -1)
    return np.stack([first, second], -2)


def hermitian_complete(lower: np.ndarray) -> np.ndarray:
    """Full ``G[i, j, c, d]`` from its ``i >= j`` part via ``G[j,i,c,d] = conj(G[i,j,d,c])``.

    Validation: cr-axial-fiber-angular
    """
    full = np.array(lower, dtype=complex)
    upper = np.triu_indices(full.shape[0], 1)
    full[upper] = np.swapaxes(full, 0, 1)[upper].conj().swapaxes(-1, -2)
    return full


def _check_directions(theta_rad, phi_rad):
    theta = np.atleast_1d(np.asarray(theta_rad, dtype=float))
    phi = np.atleast_1d(np.asarray(phi_rad, dtype=float))
    if theta.ndim != 1 or not np.all(np.isfinite(theta)) or np.any(theta < 0):
        raise ValueError("theta_rad must be finite, nonnegative and one-dimensional")
    if np.any(theta >= np.pi / 2):
        raise ValueError("theta_rad must be below pi/2 (forward hemisphere)")
    if phi.ndim != 1 or not np.all(np.isfinite(phi)):
        raise ValueError("phi_rad must be finite and one-dimensional")
    return theta, phi


def _phase_rate(energies, angle, beta, hbar_v):
    """``E (1 - beta cos theta) / hbar v``, cancellation-free.

    Validation: cr-axial-fiber-angular
    """
    return energies * ((1.0 - beta) + 2.0 * beta * np.sin(angle / 2) ** 2) / hbar_v


def _check_phase_step(z, energies, theta, beta, hbar_v, max_phase_step):
    """Refuse a depth step whose largest Doppler phase step exceeds the cap.

    Validation: cr-axial-fiber-angular
    """
    step = float(_phase_rate(energies.max(), theta.max(), beta, hbar_v) * np.diff(z).max())
    if not step <= max_phase_step:
        raise ValueError(
            f"depth step undersamples the Doppler phase ({step:.3g} rad > max_phase_step)"
        )


def linear_depth_weights(depths_ang, rate_inv_ang) -> np.ndarray:
    r"""Exact weights ``c[e, i] = int hat_i(z) e^{i omega_e z} dz`` of linear interpolation.

    ``hat_i`` is the piecewise-linear hat function of node ``z_i``; on an
    interval ``[z_k, z_k + h]`` with ``u = omega h`` the left and right nodes
    receive ``h e^{i omega z_k}`` times ``int_0^1 (1-t) e^{iut} dt`` and
    ``int_0^1 t e^{iut} dt``. As ``omega -> 0`` these are the trapezoid weights
    ``w_i e^{i omega z_i}``; unlike them they never alias a fast external phase
    (#1148). Small ``|u|`` uses the Taylor series.

    Validation: cr-axial-directional-observation
    """
    z = np.asarray(depths_ang, dtype=float)
    omega = np.asarray(rate_inv_ang, dtype=float)[:, None]
    h = np.diff(z)[None, :]
    u = omega * h
    small = np.abs(u) < 1e-3
    safe = np.where(small, 1.0, u)
    e = np.exp(1j * safe)
    right = np.where(small, 0.5 + 1j * u / 3 - u**2 / 8, e / (1j * safe) + (e - 1.0) / safe**2)
    left = np.where(small, 0.5 + 1j * u / 6 - u**2 / 24, 1j / safe - (e - 1.0) / safe**2)
    base = h * np.exp(1j * omega * z[None, :-1])
    out = np.zeros((omega.shape[0], z.size), dtype=complex)
    out[:, :-1] += base * left
    out[:, 1:] += base * right
    return out


def polarization_spectrum(
    values: np.ndarray,
    depths_ang,
    photon_energy_eV,
    kinetic_energy_eV: float,
    theta_rad,
    phi_rad,
    *,
    amplitude: np.ndarray | None = None,
    max_phase_step: float = 1.0,
    quadrature: str = "trapezoid",
) -> np.ndarray:
    r"""Polarization matrices ``S[t, p, lambda, lambda', e]`` from a full two-current ``G``.

    ``values`` is the Hermitian-complete ``(depth, depth, 2, 2)`` correlation
    (:func:`hermitian_complete`). ``amplitude`` (energy, depth) is the
    on-axis amplitude attenuation $e^{-\mu(L-z)/2}$; it is raised to
    $1/\cos\theta$ for the oblique exit path. The largest phase step
    $E(1-\beta\cos\theta)\Delta z/\hbar v$ must not exceed
    ``max_phase_step`` (rad), or line replicas alias into the grid.
    ``quadrature="linear"`` instead integrates the external phase exactly
    against the linearly interpolated correlation
    (:func:`linear_depth_weights`), so no step cap applies (#1148).
    Signed, unclipped estimator.

    Validation: cr-axial-fiber-angular, cr-axial-directional-observation
    """
    z = np.asarray(depths_ang, dtype=float)
    energies = np.asarray(photon_energy_eV, dtype=float)
    theta, phi = _check_directions(theta_rad, phi_rad)
    g = np.asarray(values, dtype=complex)
    if g.shape != (z.size, z.size, 2, 2):
        raise ValueError("values must have shape (depth, depth, 2, 2)")
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    hbar_v = HBARC_EV_ANG * beta
    spacing = np.diff(z)
    if quadrature not in ("trapezoid", "linear"):
        raise ValueError("quadrature must be 'trapezoid' or 'linear'")
    if quadrature == "trapezoid":
        _check_phase_step(z, energies, theta, beta, hbar_v, max_phase_step)
    weights = np.r_[spacing[0], spacing[:-1] + spacing[1:], spacing[-1]] / 2
    coeff = angular_coefficients(theta, phi, beta)
    out = np.zeros((theta.size, phi.size, 2, 2, energies.size), dtype=complex)
    flat = g.transpose(2, 3, 0, 1).reshape(4, z.size, z.size)
    for t, angle in enumerate(theta):
        rate = _phase_rate(energies, angle, beta, hbar_v)
        if quadrature == "linear":
            v = linear_depth_weights(z, rate)
        else:
            v = weights[None, :] * np.exp(1j * rate[:, None] * z[None, :])
        if amplitude is not None:
            v = v * np.asarray(amplitude) ** (1.0 / np.cos(angle))
        moments = (np.matmul(v[None], flat) * v.conj()[None]).sum(-1).T.reshape(-1, 2, 2)
        out[t] = np.einsum("plc,ecd,pmd->plme", coeff[t], moments, coeff[t])
    return out * (ALPHA_FS * energies / (4 * np.pi**2 * hbar_v**2))


def stokes(polarization: np.ndarray) -> np.ndarray:
    """Stokes ``(I, Q, U, V)`` on axis ``-3`` from ``S[..., lambda, lambda', e]``.

    ``Q, U`` refer to ``(e_theta, e_phi)``; ``V = 2 Im S[1, 0]`` follows
    :func:`.axial_emission.axis_stokes` (on axis at ``phi = 0`` the basis is
    the crystal ``x, y``).

    Validation: cr-axial-fiber-angular
    """
    s = np.asarray(polarization)
    return np.stack(
        [
            (s[..., 0, 0, :] + s[..., 1, 1, :]).real,
            (s[..., 0, 0, :] - s[..., 1, 1, :]).real,
            2.0 * s[..., 1, 0, :].real,
            2.0 * s[..., 1, 0, :].imag,
        ],
        axis=-2,
    )


def integrate_solid_angle(values: np.ndarray, theta_rad, phi_rad) -> np.ndarray:
    r"""$\int d\Omega$ over a cone of ``values[t, p, ...]``.

    Trapezoid in $\theta$ with weight $\sin\theta$ from ``theta_rad[0]`` to
    ``theta_rad[-1]``; ``phi_rad`` must be uniform and cover one period
    (``phi[k] = phi[0] + 2 pi k / n``), averaged and times $2\pi$ (exact for
    the degree-2 trigonometric dependence of a single direction's
    coefficients when ``n >= 5``).

    Validation: cr-axial-fiber-angular
    """
    theta, phi = _check_directions(theta_rad, phi_rad)
    step = 2 * np.pi / phi.size
    if not np.allclose(phi - phi[0], step * np.arange(phi.size), rtol=0, atol=1e-12):
        raise ValueError("phi_rad must be a uniform grid over one period")
    azimuth = 2 * np.pi * np.asarray(values).mean(axis=1)
    weight = np.sin(theta).reshape((-1,) + (1,) * (azimuth.ndim - 1))
    return np.trapezoid(azimuth * weight, theta, axis=0)


@dataclass(frozen=True)
class FiberAngularSource:
    """Signed ensemble estimate of the off-axis polarization source.

    ``polarization[t, p, lambda, lambda', e]`` is the mean $S$
    (photons/(eV sr incident electron)); ``stokes`` its ``(I, Q, U, V)``
    (axis 2) and ``stokes_sem`` their standard errors from the spread of
    contiguous incident batches. ``correlation.values`` is the
    Hermitian-complete two-current mean. ``events`` match
    :func:`.unravel_fiber_outgoing` for the same inputs.
    """

    photon_energy_eV: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray
    polarization: np.ndarray
    stokes: np.ndarray
    stokes_sem: np.ndarray
    correlation: RegressionCorrelation
    batches: int
    events: list[FiberEvent]


def fiber_angular_source(
    channels,
    entry_orders: np.ndarray,
    *,
    kinetic_energy_eV: float,
    hbar_v_eV_ang: float,
    depths_ang,
    photon_energy_eV,
    theta_rad,
    phi_rad,
    incident_ids,
    seed: int = 0,
    carrier_inv_ang: tuple[float, float] = (0.0, 0.0),
    boundary_eV: float | None = None,
    absorption_inv_ang=None,
    exit_path_ang=None,
    window=None,
    batches: int = 8,
    max_phase_step: float = 1.0,
    search_step_ang: float = 100.0,
    depth_tolerance_ang: float = 1e-6,
    max_working_bytes: int = 256 * 1024**2,
) -> FiberAngularSource:
    r"""Estimate the off-axis polarization source of the fibred absorbing model.

    Same histories, draws and terminal events as
    :func:`.axial_fiber_regression.fiber_on_axis_source`; the regressed
    currents are the same velocity currents. ``incident_ids`` is split into
    ``batches`` contiguous groups (at least two, each nonempty); because $S$
    is linear in $G$, the mean is evaluated once and the standard error of
    the Stokes parameters follows from the size-weighted batch spread.
    ``exit_path_ang`` is the on-axis path to the exit face.

    Validation: cr-axial-fiber-angular
    """
    z, energies = _admit(
        depths_ang,
        photon_energy_eV,
        kinetic_energy_eV,
        hbar_v_eV_ang,
        max_working_bytes,
    )
    theta, phi = _check_directions(theta_rad, phi_rad)
    ids = list(incident_ids)
    if isinstance(batches, bool) or not isinstance(batches, int) or not 2 <= batches <= len(ids):
        raise ValueError("batches must be an integer between 2 and the number of incidents")
    amplitude = _amplitude(absorption_inv_ang, exit_path_ang, energies, z)
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    _check_phase_step(z, energies, theta, beta, HBARC_EV_ANG * beta, max_phase_step)
    groups = np.array_split(np.asarray(ids), batches)
    sums, sizes, events = [], [], []
    alive = np.zeros(z.size)
    for group in groups:
        total = np.zeros((z.size, z.size, 2, 2), dtype=complex)
        for event, estimate, live in _primaries(
            channels,
            entry_orders,
            hbar_v_eV_ang,
            gamma,
            z,
            group.tolist(),
            seed,
            carrier_inv_ang,
            boundary_eV,
            window,
            search_step_ang,
            depth_tolerance_ang,
        ):
            total += estimate
            alive += live
            events.append(event)
        sums.append(total)
        sizes.append(group.size)
    count = len(events)
    sizes = np.asarray(sizes, dtype=float)
    batch_stokes = np.stack(
        [
            stokes(
                polarization_spectrum(
                    hermitian_complete(s / n),
                    z,
                    energies,
                    kinetic_energy_eV,
                    theta,
                    phi,
                    amplitude=amplitude,
                    max_phase_step=max_phase_step,
                )
            )
            for s, n in zip(sums, sizes, strict=True)
        ]
    )
    mean_g = hermitian_complete(sum(sums) / count)
    pol = polarization_spectrum(
        mean_g,
        z,
        energies,
        kinetic_energy_eV,
        theta,
        phi,
        amplitude=amplitude,
        max_phase_step=max_phase_step,
    )
    mean_stokes = stokes(pol)
    w = sizes.reshape((-1,) + (1,) * mean_stokes.ndim)
    spread = np.sum(w * (batch_stokes - mean_stokes) ** 2, axis=0) / (batches - 1)
    return FiberAngularSource(
        energies.copy(),
        theta.copy(),
        phi.copy(),
        pol,
        mean_stokes,
        np.sqrt(spread / count),
        RegressionCorrelation(z.copy(), mean_g, alive / count, 0),
        batches,
        events,
    )
