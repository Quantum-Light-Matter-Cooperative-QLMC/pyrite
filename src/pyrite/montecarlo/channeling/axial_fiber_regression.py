r"""On-axis two-time photon source of the carrier-fibred absorbing axial model (#992).

Research CPU reference. The electronic (kick) and continuum TDS jumps of
:mod:`.axial_fiber_unraveling` move the Bloch carrier, so the regressed
operator $\rho J_b^\dagger$ leaves the one-sector space of
:func:`.quantum_regression.regression_correlation`. Here every primary
trajectory carries *companion* vectors $w=J_b u$ inserted at each recorded
depth and propagated by exactly the operators its own history applies
(no-jump propagator, drawn jump, retained projection), scaled by the
primary's norm factors. Because the regression summand
$\langle Kw|J_a|Ku\rangle$ is linear in the record operator applied to the
primary, sampling records with the primary's own law is unbiased, with no
auxiliary trajectories or draws. Derivation:
``docs/validation/channeling/cr-axial-fiber-regression.md``.

Scope: the absorbing no-return model and its retained subspace only. Once an
electron leaves the retained subspace its later emission, and the
interference with its earlier emission, are not represented here. On-axis
observation only; lowering currents are velocity/c in the retained
eigenbasis of each fibre. Grid, window, cut and ensemble convergence and
any measured-data agreement are caller evidence, not implied.

Units: eV, Å; spectra photons/(eV sr incident electron).

Validation: cr-axial-fiber-regression
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV

from .axial_fiber_unraveling import FiberChannels, FiberEvent, _Engine, _trajectory
from .bloch import longitudinal_kinematics
from .quantum_regression import RegressionCorrelation


def fiber_lowering_currents(
    vectors: np.ndarray,
    energies_eV: np.ndarray,
    momenta_inv_ang: np.ndarray,
    gamma: float,
    *,
    boundary_eV: float | None = None,
    bohr_tolerance_eV: float = 1e-10,
) -> np.ndarray:
    r"""Velocity/c lowering currents ``(2, n, n)`` in one fibre's retained eigenbasis.

    ``vectors`` (K, n) are orthonormal retained eigenvectors on the window
    orders, whose absolute transverse wavevectors ``kappa + n g`` are
    ``momenta_inv_ang`` (K, 2). $J_a=\hbar c\,V^\dagger\mathrm{diag}(k_a)V/(\gamma
    mc^2)$ keeps elements ``[f, i]`` with $E_i-E_f$ above ``bohr_tolerance_eV``
    (positive-frequency part). ``boundary_eV`` keeps only final states strictly
    below it (the axial block of ``cr-axial-source-partition``); a boundary
    inside a degenerate multiplet is refused. The Hamiltonian kinetic term
    must be $\hbar^2k^2/(2\gamma m)$ for these to be its velocities.

    Validation: cr-axial-fiber-regression
    """
    v = np.asarray(vectors, dtype=complex)
    e = np.asarray(energies_eV, dtype=float)
    k = np.asarray(momenta_inv_ang, dtype=float)
    if v.ndim != 2 or e.shape != (v.shape[1],) or k.shape != (v.shape[0], 2):
        raise ValueError("vectors, energies and momenta have inconsistent shapes")
    scale = HBARC_EV_ANG / (gamma * M_E_EV)
    currents = np.stack([v.conj().T @ (k[:, a, None] * v) * scale for a in range(2)])
    mask = e[None, :] - e[:, None] > bohr_tolerance_eV
    if boundary_eV is not None:
        bound = e < boundary_eV
        if bound.any() and (~bound).any():
            if e[~bound].min() - e[bound].max() <= bohr_tolerance_eV:
                raise ValueError("boundary_eV splits a degenerate multiplet")
        mask &= bound[:, None]
    return currents * mask[None]


class _Companions:
    """Companion regression of one primary history (``carry`` of ``_trajectory``).

    ``y`` holds companion eigen-coordinates of the no-jump generator in the
    frame where the primary's unnormalized state is ``right @ (e^{-i z lam /
    hbar v} a)``; the physical companion is that vector divided by the same
    norm as the primary. ``values[i, j, a, b]`` accumulates
    ``<w_jb(z_i)|J_a|u(z_i)>`` for ``i >= j``. With ``flux``, ``flux_sum[i]``
    accumulates the local-emission factors ``F_c`` of the normalized state
    at each recorded depth (#1148).

    Validation: cr-axial-fiber-regression, cr-axial-fiber-angular
    Validation: cr-zone-incoherent-flux-weight
    """

    def __init__(self, engine, depths, gamma, boundary_eV, flux=None):
        self.engine, self.depths, self.gamma = engine, depths, gamma
        self.boundary = boundary_eV
        # Optional local-emission factors F_c of each recorded state (#1148).
        self.flux = flux
        self.flux_sum = None if flux is None else np.zeros((depths.size, len(flux.keys)))
        nx, ny = (np.arange(-m, m + 1) for m in engine.m)
        self.orders = np.stack(np.meshgrid(nx, ny, indexing="ij"), -1).reshape(-1, 2)
        size = depths.size
        self.values = np.zeros((size, size, 2, 2), dtype=complex)
        self.alive = np.zeros(size)
        self.y = None
        self.labels = np.zeros((0, 2), dtype=int)  # (insertion depth, component)

    def _currents(self, fiber, kappa):
        momenta = np.asarray(kappa, dtype=float) + self.orders * self.engine.g
        return fiber_lowering_currents(
            fiber.vectors, fiber.energies, momenta, self.gamma, boundary_eV=self.boundary
        )

    def _record(self, i, u, norm, right, inverse, currents):
        ju = np.einsum("anm,m->an", currents, u)
        if self.y is not None:
            # <w|J_a u> with w = right @ y / norm, without forming w.
            projected = right.conj().T @ ju.T
            self.values[i, self.labels[:, 0], :, self.labels[:, 1]] += (
                self.y.conj().T @ projected / norm
            )
        self.values[i, i] += (ju.conj() @ ju.T).T
        live = np.flatnonzero(np.linalg.norm(ju, axis=1) > 0)
        if live.size:
            fresh = inverse @ (norm * ju[live].T)
            self.y = fresh if self.y is None else np.hstack([self.y, fresh])
            self.labels = np.vstack([self.labels, np.stack([np.full(live.size, i), live], -1)])
        self.alive[i] += 1.0

    def segment(self, fiber, c, kappa, start, stop, final):
        lam, right, inverse = fiber.eigen
        rate = -1j * lam / fiber.hbar_v
        a = inverse @ c
        currents = self._currents(fiber, kappa)
        here = start
        for i, z in enumerate(self.depths):
            if not (start <= z < stop or (final and z == stop)):
                continue
            if self.y is not None:
                self.y = self.y * np.exp(rate * (z - here))[:, None]
            here = z
            u = right @ (np.exp(rate * (z - start)) * a)
            norm = float(np.linalg.norm(u))
            self._record(i, u / norm, norm, right, inverse, currents)
            if self.flux is not None and self.flux_sum is not None:
                self.flux_sum[i] += self.flux.factors(fiber.vectors @ (u / norm))
        if not final and self.y is not None:
            self.y = self.y * np.exp(rate * (stop - here))[:, None]

    def jump(self, fiber, norm, operator, new_fiber, c_new):
        if self.y is None:
            return
        chi = fiber.vectors @ (fiber.eigen[1] @ self.y) / norm
        box = operator(chi)
        moved = new_fiber.vectors.conj().T @ box[self.engine.window_in_box]
        self.y = new_fiber.eigen[2] @ (moved / np.linalg.norm(c_new))


@dataclass(frozen=True)
class FiberOnAxisSource:
    """Signed ensemble estimate of the on-axis photon source and its correlation.

    ``spectrum[E]`` is photons/(eV sr incident electron), summed over the two
    transverse polarizations; ``sem`` is its standard error over independent
    primaries. Per-primary estimates are signed and are never clipped.
    ``correlation`` is the ensemble mean (Hermitian completed), with
    ``trace`` the retained weight per depth; ``correlation_sem`` the
    elementwise standard error of its real and imaginary parts (``i >= j``).
    ``events`` are the primary terminal events, identical to
    :func:`.unravel_fiber_outgoing` for the same inputs.
    ``flux_keys``/``flux_factors`` ``(n_z, n_keys)`` are the ensemble
    local-emission factors ``F_c(z)`` of the retained density when requested
    (#1148, :mod:`.axial_flux_weights`); ``F = 1`` where nothing survives.
    """

    photon_energy_eV: np.ndarray
    spectrum: np.ndarray
    sem: np.ndarray
    correlation: RegressionCorrelation
    correlation_sem: tuple[np.ndarray, np.ndarray]
    events: list[FiberEvent]
    flux_keys: tuple[str, ...] = ()
    flux_factors: np.ndarray | None = None


def signed_on_axis_spectrum(
    values: np.ndarray,
    depths_ang: np.ndarray,
    photon_energy_eV: np.ndarray,
    kinetic_energy_eV: float,
    *,
    amplitude: np.ndarray | None = None,
) -> np.ndarray:
    r"""On-axis photons/(eV sr incident electron) from a lower-triangle correlation.

    Same law as :func:`.axial_qrt_source.on_axis_photon_spectrum`:
    $\alpha E\sum_a\sum_{ij}v_{ia}G_{ij,aa}v_{ja}^*/(4\pi^2(\hbar v)^2)$ with
    $v_{ia}=w_iA_i(E)e^{iE(1-\beta)z_i/\hbar v}$ and trapezoid weights, using
    only ``values[i, j]`` for ``i >= j`` and Hermitian symmetry for the rest.
    Returns a signed result (no positivity check), as needed for an unbiased
    Monte Carlo estimator. ``amplitude`` (E, depth) is the amplitude-level
    attenuation; default one.

    Validation: cr-axial-fiber-regression
    """
    z = np.asarray(depths_ang, dtype=float)
    energies = np.asarray(photon_energy_eV, dtype=float)
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    hbar_v = HBARC_EV_ANG * beta
    phase = energies * (1 / (gamma**2 * (1 + beta))) / hbar_v
    spacing = np.diff(z)
    weights = np.r_[spacing[0], spacing[:-1] + spacing[1:], spacing[-1]] / 2
    v = weights[None, :] * np.exp(1j * phase[:, None] * z[None, :])
    if amplitude is not None:
        v = v * amplitude
    total = np.zeros(energies.size)
    for a in range(2):
        lower = np.tril(values[:, :, a, a])
        lower[np.diag_indices_from(lower)] *= 0.5
        total += 2 * np.einsum("ei,ij,ej->e", v, lower, v.conj()).real
    return ALPHA_FS * energies * total / (4 * np.pi**2 * hbar_v**2)


def _amplitude(absorption_inv_ang, exit_path_ang, energies, depths):
    if (absorption_inv_ang is None) != (exit_path_ang is None):
        raise ValueError("absorption_inv_ang and exit_path_ang must be supplied together")
    if absorption_inv_ang is None:
        return None
    mu = np.asarray(absorption_inv_ang, dtype=float)
    path = np.asarray(exit_path_ang, dtype=float)
    if mu.shape != energies.shape or path.shape != depths.shape:
        raise ValueError("absorption and exit path must match the energy and depth grids")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in (mu, path)):
        raise ValueError("absorption and exit path must be finite and nonnegative")
    return np.exp(-0.5 * mu[:, None] * path[None, :])


def _admit(depths_ang, photon_energy_eV, kinetic_energy_eV, hbar_v_eV_ang, max_working_bytes):
    z = np.asarray(depths_ang, dtype=float)
    e = np.asarray(photon_energy_eV, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("depths_ang needs at least two finite nonnegative depths")
    if np.any(np.diff(z) <= 0):
        raise ValueError("depths_ang must be strictly increasing")
    if e.ndim != 1 or not e.size or not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("photon_energy_eV must be a nonempty positive one-dimensional grid")
    if not np.isfinite(kinetic_energy_eV) or kinetic_energy_eV <= 0:
        raise ValueError("kinetic_energy_eV must be finite and positive")
    _, beta = longitudinal_kinematics(kinetic_energy_eV)
    if not np.isclose(hbar_v_eV_ang, HBARC_EV_ANG * beta, rtol=1e-9, atol=0):
        raise ValueError("hbar_v_eV_ang must equal hbar c beta of kinetic_energy_eV")
    if isinstance(max_working_bytes, bool) or not isinstance(max_working_bytes, Integral):
        raise ValueError("max_working_bytes must be a positive integer")
    estimate = 16 * (6 * 4 * z.size**2 + 4 * e.size * z.size)
    if max_working_bytes < 1 or estimate > max_working_bytes:
        raise ValueError(f"fiber regression exceeds working-memory budget ({estimate} bytes)")
    return z, e


def _primaries(
    channels,
    entry_orders,
    hbar_v_eV_ang,
    gamma,
    z,
    incident_ids,
    seed,
    carrier_inv_ang,
    boundary_eV,
    window,
    search_step_ang,
    depth_tolerance_ang,
    flux=None,
    flux_total=None,
):
    """Yield ``(event, W * values, W * alive)`` for each primary history.

    With ``flux`` (:class:`.axial_flux_weights.AxialFluxKernels`) each history
    also adds ``W * sum_alive F_c`` per depth into ``flux_total`` in place
    (#1148); no draw changes.

    Validation: cr-axial-fiber-regression, cr-axial-fiber-angular
    """
    engine = _Engine(channels, hbar_v_eV_ang)
    entry = np.asarray(entry_orders, dtype=complex)
    if entry.shape != (engine.k,) or not np.all(np.isfinite(entry)):
        raise ValueError("entry_orders must be finite on the order window")
    if window is None:
        window = np.zeros(engine.box_shape, dtype=complex)
        window[0, 0] = 1.0
    window = np.asarray(window, dtype=complex)
    if window.shape != engine.box_shape or not np.isclose(np.vdot(window, window).real, 1.0):
        raise ValueError("window must be a unit-norm array on the outgoing box shape")
    kappa = (float(carrier_inv_ang[0]), float(carrier_inv_ang[1]))
    for incident in incident_ids:
        carry = _Companions(engine, z, gamma, boundary_eV, flux)
        event = _trajectory(
            engine,
            entry,
            kappa,
            window,
            float(z[-1]),
            seed,
            int(incident),
            search_step_ang,
            depth_tolerance_ang,
            carry=carry,
        )
        if flux is not None:
            flux_total += event.weight * carry.flux_sum
        yield event, event.weight * carry.values, event.weight * carry.alive


def fiber_on_axis_source(
    channels: FiberChannels,
    entry_orders: np.ndarray,
    *,
    kinetic_energy_eV: float,
    hbar_v_eV_ang: float,
    depths_ang,
    photon_energy_eV,
    incident_ids,
    seed: int = 0,
    carrier_inv_ang: tuple[float, float] = (0.0, 0.0),
    boundary_eV: float | None = None,
    absorption_inv_ang=None,
    exit_path_ang=None,
    window=None,
    search_step_ang: float = 100.0,
    depth_tolerance_ang: float = 1e-6,
    max_working_bytes: int = 256 * 1024**2,
    flux_kernels=None,
) -> FiberOnAxisSource:
    r"""Estimate the on-axis two-time source of the fibred absorbing model.

    For recorded depths $z_j\le z_i$,
    $G_{ab}(z_i,z_j)=\mathrm{Tr}[J_a\Lambda_{z_i-z_j}(\rho(z_j)J_b^\dagger)]$,
    with $\Lambda$ the absorbing fibred map of :func:`.unravel_fiber_outgoing`
    restricted to the retained subspaces and $J$ the lowering currents of
    :func:`fiber_lowering_currents` in the *current* fibre of each record.
    Each primary history (``fiber_stream(seed, id)``, unchanged draws) gives
    $\hat G=W\langle\hat w_{jb}(z_i)|J_a|\hat u(z_i)\rangle$ with $W$ the
    incident weight, $\hat u$ the normalized primary and $\hat w_{jb}$ the
    companion inserted as $J_b\hat u(z_j)$; its mean over primaries is
    unbiased because the summand vanishes on every record of zero primary
    probability. The spectrum is :func:`signed_on_axis_spectrum` of each
    primary's estimate, averaged; ``sem`` follows from the per-primary spread.

    ``depths_ang`` must end at the slab path; ``window`` is the outgoing Weyl
    window (default a unit delta), which affects no jump draw. Emission after
    an electron leaves the retained subspace is excluded by construction.
    ``flux_kernels`` (:class:`.axial_flux_weights.AxialFluxKernels` on the
    channels' order window) adds the ensemble factors ``flux_factors``;
    draws, events and spectra are unchanged.

    Validation: cr-axial-fiber-regression, cr-zone-incoherent-flux-weight
    """
    z, energies = _admit(
        depths_ang, photon_energy_eV, kinetic_energy_eV, hbar_v_eV_ang, max_working_bytes
    )
    amplitude = _amplitude(absorption_inv_ang, exit_path_ang, energies, z)
    gamma, _ = longitudinal_kinematics(kinetic_energy_eV)
    flux_total = None if flux_kernels is None else np.zeros((z.size, len(flux_kernels.keys)))
    primaries = _primaries(
        channels,
        entry_orders,
        hbar_v_eV_ang,
        gamma,
        z,
        incident_ids,
        seed,
        carrier_inv_ang,
        boundary_eV,
        window,
        search_step_ang,
        depth_tolerance_ang,
        flux_kernels,
        flux_total,
    )
    shape = (z.size, z.size, 2, 2)
    totals = [np.zeros(shape, dtype=complex), np.zeros(shape), np.zeros(shape)]
    spectra, alive, events = [], np.zeros(z.size), []
    for event, estimate, live in primaries:
        totals[0] += estimate
        totals[1] += estimate.real**2
        totals[2] += estimate.imag**2
        alive += live
        spectra.append(
            signed_on_axis_spectrum(estimate, z, energies, kinetic_energy_eV, amplitude=amplitude)
        )
        events.append(event)
    count = len(events)
    if count < 2:
        raise ValueError("at least two primaries are needed for an error estimate")
    spectra = np.asarray(spectra)
    mean = totals[0] / count
    sems = tuple(
        np.sqrt(np.maximum(total / count - part**2, 0.0) / (count - 1))
        for total, part in ((totals[1], mean.real), (totals[2], mean.imag))
    )
    full = mean.copy()
    upper = np.triu_indices(z.size, 1)
    full[upper] = np.swapaxes(mean, 0, 1)[upper].conj().swapaxes(-1, -2)
    correlation = RegressionCorrelation(z.copy(), full, alive / count, 0)
    flux_keys: tuple[str, ...] = ()
    flux_factors = None
    if flux_kernels is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            factors = flux_total / alive[:, None]
        flux_keys = tuple(flux_kernels.keys)
        flux_factors = np.where(alive[:, None] > 0.0, factors, 1.0)
    return FiberOnAxisSource(
        energies.copy(),
        spectra.mean(axis=0),
        spectra.std(axis=0, ddof=1) / np.sqrt(count),
        correlation,
        sems,
        events,
        flux_keys,
        flux_factors,
    )


def fiber_thickness_spectra(
    channels: FiberChannels,
    entry_orders: np.ndarray,
    *,
    kinetic_energy_eV: float,
    hbar_v_eV_ang: float,
    depths_ang,
    thickness_indices,
    photon_energy_eV,
    incident_ids,
    seed: int = 0,
    carrier_inv_ang: tuple[float, float] = (0.0, 0.0),
    boundary_eV: float | None = None,
    absorption_inv_ang=None,
    window=None,
    search_step_ang: float = 100.0,
    depth_tolerance_ang: float = 1e-6,
    max_working_bytes: int = 256 * 1024**2,
) -> tuple[np.ndarray, list[FiberEvent]]:
    r"""Per-primary on-axis spectra for slabs ending at several recorded depths.

    For each primary history of :func:`fiber_on_axis_source` and each index
    $k$ in ``thickness_indices``, the estimate restricted to depths
    $z_0..z_k$ is the estimator of a slab of thickness $z_k$ on that depth
    grid: records up to $z_k$ never depend on later depths, and the
    trapezoid weights and exit path $z_k-z$ (attenuation ``absorption_inv_ang``
    per photon energy) are those of the truncated grid. Thicknesses are
    therefore paired exactly within each primary. Returns an array
    ``(primary, thickness, energy)`` in photons/(eV sr incident electron) and
    the primaries' terminal events (as :func:`fiber_on_axis_source`).

    Validation: cr-axial-fiber-regression
    """
    z, energies = _admit(
        depths_ang, photon_energy_eV, kinetic_energy_eV, hbar_v_eV_ang, max_working_bytes
    )
    stops = [int(k) for k in thickness_indices]
    if not stops or any(not 1 <= k < z.size for k in stops):
        raise ValueError("thickness_indices must select recorded depths after the first")
    amplitudes = [
        None
        if absorption_inv_ang is None
        else _amplitude(absorption_inv_ang, z[k] - z[: k + 1], energies, z[: k + 1])
        for k in stops
    ]
    gamma, _ = longitudinal_kinematics(kinetic_energy_eV)
    primaries = _primaries(
        channels,
        entry_orders,
        hbar_v_eV_ang,
        gamma,
        z,
        incident_ids,
        seed,
        carrier_inv_ang,
        boundary_eV,
        window,
        search_step_ang,
        depth_tolerance_ang,
    )
    out, events = [], []
    for event, estimate, _ in primaries:
        events.append(event)
        out.append(
            [
                signed_on_axis_spectrum(
                    estimate[: k + 1, : k + 1],
                    z[: k + 1],
                    energies,
                    kinetic_energy_eV,
                    amplitude=amplitude,
                )
                for k, amplitude in zip(stops, amplitudes, strict=True)
            ]
        )
    return np.asarray(out).reshape(len(out), len(stops), energies.size), events
