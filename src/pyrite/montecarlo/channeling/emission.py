"""Channeling-radiation emission from the transverse density matrix (ADR-0018, P3).

**Operator.** A photon ``(k, epsilon)`` is emitted by the first-order
interaction ``(e/c) v . A``. In the transverse Bloch basis the transition
operator is ``A_fi = <f| (v . epsilon* / c) exp(-i k . r) |i>`` with
``v = (p_x / (gamma m), 0, beta c)``. The golden rule with the longitudinal
energy--momentum balance ``hbar omega (1 - beta cos theta) = Delta E_perp``
gives, per unit path,

``dN / dz dOmega = alpha (omega / c) |A_fi|^2 / (2 pi beta (1 - beta cos theta))``.

For ``q = k_x -> 0`` (state size ``<= 1 A`` against a rest-frame wavelength of
``>= 10^2 A``) ``A_fi -> -i x_fi (omega / c)[(1 - beta cos theta) eps_x* + beta n_x eps_z*]``,
which is the boosted electric dipole: the rest-frame rate
``A' = (4/3) alpha omega'^3 |x_fi|^2 / c^2`` at ``omega' = gamma Delta E_perp / hbar``,
the rest-frame pattern ``(3 / 8 pi) sin^2 Theta'`` about the plane normal
``x``, aberration ``dOmega'/dOmega = 1 / (gamma^2 (1 - beta cos theta)^2)`` and
the lab path rate ``A' / (gamma beta c)``.

**Estimator.** Lines are scored as the expectation
``<d_c^dagger d_c>_rho(z)`` per cluster ``c`` of transitions that share a
final state and are degenerate within their widths, ``d_c = sum a_fi
|f><i|`` with ``|a_fi|^2`` the dipole path rate; cross terms are kept.
``rho`` is block-diagonal in Bloch ``k``. Scored transitions end on bound
states: bound -> bound lines and free -> bound radiative capture from the
above-barrier states the solver tracks (maintainer decision 2026-10-15,
reversing v1's bound -> bound-only rule for capture). Free -> free is
coherent bremsstrahlung (PXR/CBS kernel) and is never scored. Per-trajectory
MCWF field amplitudes are never squared.

**Line shape.** The radiating coherences ``<f_k|rho|i_k>`` of one transition
in all Bloch blocks obey a linear secular equation: phase ``Delta_k /
hbar v``, loss ``(Gamma_i + Gamma_f)/2`` and coherence transfer between
blocks by the scattering kernel. Its eigenmodes (quantum regression theorem)
give complex-weighted Lorentzians, scaled by the Doppler factor and convolved
with the Gaussian of the beam energy spread. The detector aperture and the
in-plane beam divergence add the Doppler--geometric width by quadrature over
directions. Photons are attenuated on the way out by
:func:`~pyrite.materials.attenuation._stack_tau`.

Lengths in angstroms, energies in eV, angles in radians. CPU only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eig
from scipy.special import voigt_profile, wofz

from ...materials.attenuation import _resolve_composition, _stack_tau
from ...materials.channeling.transverse import HBAR2_2M_EV_ANG2, _hamiltonian
from ...materials.crystal import HBARC_EV_ANG, M_E_EV
from .density import DensityMatrixSolver
from .dipole import _beta, _denominator, boosted_dipole_pattern, dipole_rate_per_ang, dipole_stokes

DEFAULT_CAPTURE_THRESHOLD = 0.03


# ---------------------------------------------------------------------------
# Line catalog and expected-value estimator on rho(z)
# ---------------------------------------------------------------------------

KIND_LINE = 0
KIND_CAPTURE = 1


def _block_dipoles(solver: DensityMatrixSolver, block: int, states: np.ndarray) -> np.ndarray:
    """``x_fi`` [angstrom] between eigenstates ``states`` of one block."""
    basis = solver.basis
    vectors = basis.vectors[block][:, states]
    waves = solver.kappa[block] + solver.orders * solver.profile.g_inv_ang
    momentum = vectors.conj().T @ (waves[:, None] * vectors)
    energies = basis.energies[block][states]
    gap = energies[:, None] - energies[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dipole = -1j * 2.0 * HBAR2_2M_EV_ANG2 / basis.gamma * momentum / gap
    dipole[~np.isfinite(dipole)] = 0.0
    np.fill_diagonal(dipole, 0.0)
    return dipole


def _complex_profile(detuning_eV, hwhm_eV, sigma_eV):
    """``(i/pi)/(x + i h)`` convolved with a unit Gaussian of RMS ``sigma`` [1/eV].

    Its real part is the Voigt profile; its imaginary part is the matching
    dispersion profile. Uses the Faddeeva function; ``sigma = 0`` is the
    complex Lorentzian.
    """
    x = np.asarray(detuning_eV, dtype=float)
    h = np.asarray(hwhm_eV, dtype=float)
    s = np.asarray(sigma_eV, dtype=float)
    x, h, s = np.broadcast_arrays(x, h, s)
    out = np.empty(x.shape, dtype=complex)
    gauss = s > 0.0
    root = np.sqrt(2.0) * np.where(gauss, s, 1.0)
    out[gauss] = (wofz((x + 1j * h) / root) / (root * np.sqrt(np.pi)))[gauss]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[~gauss] = (1j / np.pi / (x + 1j * h))[~gauss]
    return out


@dataclass(frozen=True)
class LineGroup:
    """One transition ``(upper, lower)`` across the Bloch blocks that carry it.

    The radiating coherences ``sigma^(k) = <f_k|rho|i_k>`` of the group obey
    ``d sigma / dz = K sigma`` with
    ``K_kk = i Delta_k / (hbar v) - (Gamma_i + Gamma_f)_k / 2 + J_kk`` and
    ``K_k'k = <f_k'|J(|f_k><i_k|)|i_k'>``: the scattering kernel transfers
    coherence between blocks (a momentum kick acting on ket and bra alike).
    ``K = V diag(mu) V^-1`` gives the line as a sum of complex Lorentzian
    modes (quantum regression theorem).

    Attributes
    ----------
    lines
        Catalog indices of the group's lines, one per block.
    amplitude
        Representative transition amplitude ``a`` of each line (eigenbasis gauge).
    eigenvalues, right, left
        ``mu`` [1/angstrom], ``V`` and ``V^-1`` of ``K``.
    hbar_v_eV_ang
        ``hbar v`` [eV angstrom].
    """

    lines: np.ndarray
    amplitude: np.ndarray
    eigenvalues: np.ndarray
    right: np.ndarray
    left: np.ndarray
    hbar_v_eV_ang: float

    def modes(self, yields: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mode centres, half widths [eV, transverse] and complex weights.

        ``yields`` are the (absorbed) photon yields of the group's lines; the
        source coherence of line ``k`` is ``yield_k / conj(a_k)`` and the
        weights ``w_m = (a^dagger V)_m (V^-1 s)_m`` sum to the total yield.
        The transverse spectrum is ``Re sum_m conj(w_m) (i/pi)/(x_m + i h_m)``,
        ``x_m = E - centre_m``.

        Validation: cr-line-shape
        """
        source = np.asarray(yields, dtype=float) / np.conj(self.amplitude)
        weight = (np.conj(self.amplitude) @ self.right) * (self.left @ source)
        return (
            self.hbar_v_eV_ang * self.eigenvalues.imag,
            -self.hbar_v_eV_ang * self.eigenvalues.real,
            weight,
        )


@dataclass(frozen=True)
class EmissionLines:
    """Dipole lines ending on bound states, from a density-matrix solver's basis.

    One entry per cluster of transitions in one Bloch block. ``kind`` is
    :data:`KIND_LINE` (bound -> bound) or :data:`KIND_CAPTURE` (tracked free
    -> bound). Members are the ``(lower, upper, amplitude)`` triples of the
    cluster; ``|amplitude|^2`` is the dipole path rate [1/angstrom] and its
    phase follows the eigenvector gauge of the solver's basis (rates use
    ``rho`` in the same gauge).

    Attributes
    ----------
    gamma, beta
        Longitudinal kinematics of the basis.
    block, kind
        Bloch block and kind of each line.
    upper, lower
        Band indices of each line's strongest member.
    delta_eV
        Rate-weighted transverse transition energy [eV].
    hwhm_eV
        All-block secular half width ``hbar v kappa`` [eV] (exact for flat
        bands; the spectrum uses the group modes).
    max_member_hwhm_eV
        Largest member pair half width of each line [eV]; it bounds every
        clustering link, so :func:`capture_bound` clusters at least as coarsely.
    gamma_exponent
        ``d ln Delta E_perp / d ln gamma`` at fixed band indices.
    strength_per_ang
        Sum of member path rates ``|a_fi|^2`` [1/angstrom].
    members
        Per line, ``(lower, upper, amplitude)`` arrays.
    group
        Index into ``groups`` of each line.
    groups
        Coherence-transfer groups (:class:`LineGroup`).
    """

    gamma: float
    beta: float
    block: np.ndarray
    kind: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    delta_eV: np.ndarray
    hwhm_eV: np.ndarray
    max_member_hwhm_eV: np.ndarray
    gamma_exponent: np.ndarray
    strength_per_ang: np.ndarray
    members: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    group: np.ndarray
    groups: tuple[LineGroup, ...]

    @property
    def size(self) -> int:
        """Number of lines."""
        return int(self.block.size)

    def rates(self, rho: np.ndarray) -> np.ndarray:
        """Expected photon yield per unit path of each line [1/angstrom].

        ``<d_c^dagger d_c> = sum_f sum_{i, i'} a_fi' * a_fi rho_ii'`` over the
        members of each cluster, on a density matrix in the solver's
        eigenbasis, shape ``(n_cells, N, N)``. Cross terms between initial
        states of one cluster are kept. It is linear in ``rho``, so it is an
        unbiased estimator on any ensemble of states.

        Validation: cr-yield-assembly, cr-capture-yield
        """
        out = np.zeros(self.size)
        for line, (lower, upper, amplitude) in enumerate(self.members):
            sub = rho[self.block[line]][np.ix_(upper, upper)]
            same = lower[:, None] == lower[None, :]
            out[line] = np.sum(same * amplitude[:, None] * np.conj(amplitude[None, :]) * sub).real
        return out

    def forward_energy_eV(self) -> np.ndarray:
        """Forward (``theta = 0``) lab photon energy of each line [eV]."""
        return self.delta_eV / _denominator(self.gamma, 0.0)

    def spectrum(
        self,
        photon_energy_eV,
        yields: np.ndarray,
        theta_e_rad: float,
        sigma_gamma_rel: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Line and capture spectra per unit angular density [photons/eV].

        Each group contributes ``Re sum_m conj(w_m) F(E - E_m)`` with lab
        centres ``E_m = centre_m / D``, half widths ``h_m / D``
        (``D = 1 - beta cos theta_e``) and the complex Voigt ``F`` of RMS
        ``sigma = |E_m| s sigma_gamma_rel`` (``s = d ln E_c / d ln gamma``, see
        :func:`channeling_spectrum`). Multiply by the angular density
        :func:`boosted_dipole_pattern` for photons/eV/sr.

        Validation: cr-line-shape, cr-yield-assembly
        """
        energies = np.asarray(photon_energy_eV, dtype=float)
        denom = float(_denominator(self.gamma, theta_e_rad))
        kinematic = np.cos(theta_e_rad) / (self.gamma**2 * self.beta * denom)
        out = [np.zeros(energies.shape), np.zeros(energies.shape)]
        for group in self.groups:
            members = group.lines
            centre, hwhm, weight = group.modes(np.asarray(yields)[members])
            centre, hwhm = centre / denom, hwhm / denom
            strength = self.strength_per_ang[members]
            exponent = float(np.dot(strength, self.gamma_exponent[members]) / strength.sum())
            sigma = np.abs(centre * (kinematic + exponent)) * float(sigma_gamma_rel)
            if np.any((hwhm <= 0.0) & (sigma <= 0.0)):
                raise ValueError(
                    "line has zero width: supply a scattering kernel or a beam energy spread"
                )
            profile = _complex_profile(energies[..., None] - centre, hwhm, sigma)
            out[int(self.kind[members[0]])] += (np.conj(weight) * profile).real.sum(axis=-1)
        return out[KIND_LINE], out[KIND_CAPTURE]


def _gauge(amplitude: Any) -> complex:
    return complex(amplitude) if amplitude != 0 else 1.0 + 0j


def emission_lines(
    solver: DensityMatrixSolver,
    *,
    include_capture: bool = True,
    cluster_tol_eV: float | None = None,
    gamma_step: float = 1e-4,
    relative_cut: float = 1e-12,
) -> EmissionLines:
    """Build the dipole line catalog of a density-matrix solver.

    In each Bloch block, every bound final state ``f`` (``E < barrier``) and
    every higher initial state ``i`` contributes
    ``a_fi = sqrt(dN/dz) x_fi / |x_fi|`` with :func:`dipole_rate_per_ang`.
    Initial states are bound (lines) and, with ``include_capture``, the
    above-barrier states the solver tracks (``barrier <= E < barrier +
    margin``: radiative capture). ``x_fi`` comes from the commutator in the
    solver's eigenbasis, so parity selection emerges; transitions below
    ``relative_cut`` of the block's strongest of the same kind are dropped.
    Free -> free (coherent bremsstrahlung) is never scored.

    **Widths.** For each transition ``(upper, lower)`` the coherences of all
    blocks form a :class:`LineGroup` with generator ``K`` (scattering
    transfers coherence between blocks). The reported half width is the
    all-block secular value
    ``kappa_k = (Gamma_i + Gamma_f)_k / 2 - Re sum_k' a_k'* K^J_k'k a_k / |a_k|^2``,
    exact when the band energies are flat across blocks; the spectrum uses
    the modes of ``K``, which also covers dispersive lines.

    **Clusters.** Transitions of one block into the same final state merge
    when their initial energies differ by at most ``cluster_tol_eV``
    (default: half the sum of their half widths, i.e. unresolved), and their
    cross terms are kept in :meth:`EmissionLines.rates`. Bound levels of one
    block are nondegenerate, so bound lines stay single; above-barrier
    states come in near-degenerate pairs, whose cross terms vanish by parity
    only at ``k = 0`` and ``g/2``.

    Zero bound states give an empty catalog, hence exactly zero emission.

    Validation: cr-dipole-rate, cr-line-shape, cr-capture-yield
    """
    basis = solver.basis
    gamma = float(basis.gamma)
    beta = float(_beta(gamma))
    hbar_v = HBARC_EV_ANG * beta
    size = solver.orders.size
    bound = basis.energies < solver.barrier_eV
    initial_ok = bound | (basis.subspace & ~bound) if include_capture else bound
    shifted = [
        np.array(
            [
                np.linalg.eigvalsh(
                    _hamiltonian(solver.profile, gamma * factor, kappa, solver.orders)
                )
                for kappa in solver.kappa
            ]
        )
        for factor in (1.0 + gamma_step, 1.0 - gamma_step)
    ]
    losses: dict[tuple[int, int], float] = {}

    def loss(block: int, state: int) -> float:
        if (block, state) not in losses:
            rho = np.zeros((solver.n_cells, size, size), dtype=complex)
            rho[block, state, state] = 1.0
            losses[(block, state)] = solver.loss_rate(rho)
        return losses[(block, state)]

    # 1. transitions per block
    pairs: list[dict] = []
    for block in range(solver.n_cells):
        finals = np.flatnonzero(bound[block])
        states = np.flatnonzero(initial_ok[block])
        if finals.size == 0 or states.size < 2:
            continue
        dipoles = _block_dipoles(solver, block, states)
        energies = basis.energies[block][states]
        found = []
        for a, upper in enumerate(states):
            for b, lower in enumerate(states):
                if not bound[block, lower] or energies[a] <= energies[b] + 1e-12:
                    continue
                x = dipoles[b, a]
                if abs(x) == 0.0:
                    continue
                delta = float(energies[a] - energies[b])
                rate = float(dipole_rate_per_ang(delta, abs(x), gamma))
                up = shifted[0][block, upper] - shifted[0][block, lower]
                down = shifted[1][block, upper] - shifted[1][block, lower]
                found.append(
                    {
                        "block": block,
                        "kind": KIND_LINE if bound[block, upper] else KIND_CAPTURE,
                        "upper": int(upper),
                        "lower": int(lower),
                        "delta": delta,
                        "amp": np.sqrt(rate) * x / abs(x),
                        "rate": rate,
                        "exponent": float(
                            (np.log(up) - np.log(down))
                            / (np.log1p(gamma_step) - np.log1p(-gamma_step))
                        ),
                    }
                )
        for kind in (KIND_LINE, KIND_CAPTURE):
            subset = [item for item in found if item["kind"] == kind]
            if subset:
                floor = relative_cut * max(item["rate"] for item in subset)
                pairs.extend(item for item in subset if item["rate"] >= floor)

    # 2. coherence-transfer generator per (upper, lower) across blocks
    by_pair: dict[tuple[int, int], list[int]] = {}
    for index, item in enumerate(pairs):
        by_pair.setdefault((item["upper"], item["lower"]), []).append(index)
    generators: dict[tuple[int, int], np.ndarray] = {}
    for key, indices in by_pair.items():
        upper, lower = key
        matrix = np.zeros((len(indices), len(indices)), dtype=complex)
        blocks = [pairs[j]["block"] for j in indices]
        for col, j in enumerate(indices):
            rho = np.zeros((solver.n_cells, size, size), dtype=complex)
            rho[blocks[col], lower, upper] = 1.0
            inside, _, _ = solver.jump(rho)
            matrix[:, col] = inside[blocks, lower, upper]
            matrix[col, col] += 1j * pairs[j]["delta"] / hbar_v - 0.5 * (
                loss(blocks[col], upper) + loss(blocks[col], lower)
            )
        generators[key] = matrix
        amps = np.array([_gauge(pairs[j]["amp"]) for j in indices])
        for col, j in enumerate(indices):
            gain = np.conj(amps) @ matrix[:, col] * amps[col] / abs(amps[col]) ** 2
            pairs[j]["hwhm"] = max(-hbar_v * float(gain.real), 0.0)

    # 3. clusters within a block (shared final, unresolved initial energies)
    order = sorted(
        range(len(pairs)),
        key=lambda j: (pairs[j]["block"], pairs[j]["lower"], pairs[j]["delta"]),
    )
    clusters: list[list[int]] = []
    for j in order:
        if clusters:
            last = pairs[clusters[-1][-1]]
            item = pairs[j]
            if last["block"] == item["block"] and last["lower"] == item["lower"]:
                tol = (
                    0.5 * (last["hwhm"] + item["hwhm"])
                    if cluster_tol_eV is None
                    else float(cluster_tol_eV)
                )
                if item["delta"] - last["delta"] <= tol:
                    clusters[-1].append(j)
                    continue
        clusters.append([j])
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        items = [pairs[j] for j in cluster]
        weight = np.array([item["rate"] for item in items])
        strongest = items[int(np.argmax(weight))]

        def mean(key, items=items, weight=weight):
            return float(np.dot(weight, [item[key] for item in items]) / weight.sum())

        rows.append(
            {
                "block": strongest["block"],
                "kind": strongest["kind"],
                "upper": strongest["upper"],
                "lower": strongest["lower"],
                "amp": strongest["amp"],
                "delta": mean("delta"),
                "hwhm": mean("hwhm"),
                "max_hwhm": max(item["hwhm"] for item in items),
                "exponent": mean("exponent"),
                "strength": float(weight.sum()),
                "members": (
                    np.array([item["lower"] for item in items]),
                    np.array([item["upper"] for item in items]),
                    np.array([item["amp"] for item in items], dtype=complex),
                ),
            }
        )

    # 4. line groups from the representatives
    groups: list[LineGroup] = []
    group_of = np.zeros(len(rows), dtype=int)
    by_rep: dict[tuple[int, int], list[int]] = {}
    for line, row in enumerate(rows):
        by_rep.setdefault((row["upper"], row["lower"]), []).append(line)
    for key, lines in by_rep.items():
        pair_blocks = [pairs[j]["block"] for j in by_pair[key]]
        position = [pair_blocks.index(rows[line]["block"]) for line in lines]
        matrix = generators[key][np.ix_(position, position)]
        mu, right = eig(matrix)
        groups.append(
            LineGroup(
                lines=np.array(lines),
                amplitude=np.array([_gauge(rows[line]["amp"]) for line in lines]),
                eigenvalues=mu,
                right=right,
                left=np.linalg.inv(right),
                hbar_v_eV_ang=hbar_v,
            )
        )
        group_of[lines] = len(groups) - 1

    def column(key, dtype=float):
        return np.array([row[key] for row in rows], dtype=dtype)

    return EmissionLines(
        gamma=gamma,
        beta=beta,
        block=column("block", int),
        kind=column("kind", int),
        upper=column("upper", int),
        lower=column("lower", int),
        delta_eV=column("delta"),
        hwhm_eV=column("hwhm"),
        max_member_hwhm_eV=column("max_hwhm"),
        gamma_exponent=column("exponent"),
        strength_per_ang=column("strength"),
        members=tuple(row["members"] for row in rows),
        group=group_of,
        groups=tuple(groups),
    )


# ---------------------------------------------------------------------------
# Line shape
# ---------------------------------------------------------------------------


def line_shape(photon_energy_eV, centre_eV, lorentz_hwhm_eV, gaussian_sigma_eV):
    """Return the unit-area Voigt line profile [1/eV].

    Lorentzian of half width ``lorentz_hwhm_eV`` (coherence decay) convolved
    with a Gaussian of RMS ``gaussian_sigma_eV`` (beam energy spread). It is
    the real part of the mode profile used by :meth:`EmissionLines.spectrum`
    for a single real-weight mode. The Doppler--geometric width is added by
    the caller's direction quadrature. Both widths zero is a delta function
    and is refused.

    Validation: cr-line-shape
    """
    lorentz = np.asarray(lorentz_hwhm_eV, dtype=float)
    sigma = np.asarray(gaussian_sigma_eV, dtype=float)
    if np.any((lorentz <= 0.0) & (sigma <= 0.0)):
        raise ValueError("line has zero width: supply a scattering kernel or a beam energy spread")
    detuning = np.asarray(photon_energy_eV, dtype=float) - np.asarray(centre_eV, dtype=float)
    return voigt_profile(detuning, sigma, lorentz)


# ---------------------------------------------------------------------------
# Depth history and spectrum assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _History:
    z_ang: np.ndarray
    rates: np.ndarray  # (n_z, n_lines) [1/angstrom]
    bound: np.ndarray  # (n_z,)
    interval_weight: np.ndarray | None = None

    def integrate(self, values: np.ndarray) -> np.ndarray:
        """Path integral per incident electron; values have depth as axis zero.

        Validation: cr-persisted-channel
        """
        if self.interval_weight is None:
            return np.trapezoid(values, self.z_ang, axis=0)
        weight = np.diff(self.z_ang) * self.interval_weight
        weight = weight.reshape((-1,) + (1,) * (values.ndim - 1))
        return np.sum(0.5 * (values[:-1] + values[1:]) * weight, axis=0)

    def censored(self, path_limits_ang: np.ndarray | Sequence[float]) -> _History:
        """Interpolate at face/cutoff endpoints and weight surviving entrance paths.

        Each incident primary has one geometric/cutoff limit, including zero
        for a miss. Quantum dechanneling is already in ``rates`` and must not
        enter these weights a second time. Piecewise linear rates are the
        same trapezoid approximation as the uncensored history.

        Validation: cr-persisted-channel
        """
        limits = np.asarray(path_limits_ang, dtype=float)
        if limits.ndim != 1 or not limits.size or np.any(~np.isfinite(limits)):
            raise ValueError("path_limits_ang must be a nonempty finite one-dimensional array")
        limits = np.sort(np.clip(limits, 0.0, self.z_ang[-1]))
        z = np.unique(np.concatenate((self.z_ang, limits)))
        rates = np.empty((z.size, self.rates.shape[1]))
        for i in range(self.rates.shape[1]):
            rates[:, i] = np.interp(z, self.z_ang, self.rates[:, i])
        weight = (limits.size - np.searchsorted(limits, z[1:], side="left")) / limits.size
        return _History(z, rates, np.interp(z, self.z_ang, self.bound), weight)


def _validated_entry_initial(solver: DensityMatrixSolver, initial) -> tuple[np.ndarray, float]:
    """Validate a preprojected entrance density before CR depth integration.

    This is the exact density/eigenbasis cutoff partition returned by the
    established planar projector. Initially free/outside populations belong
    to the entry sink, not an independent dechanneling weight.
    """
    if not isinstance(initial, tuple) or len(initial) != 2:
        raise TypeError("entry_initial must be (subspace_density, entry_population)")
    rho = np.array(initial[0], dtype=complex, copy=True)
    n_cells, states = solver.basis.energies.shape
    if rho.shape != (n_cells, states, states):
        raise ValueError("entry_initial density shape does not match solver's Bloch basis")
    if not np.all(np.isfinite(rho)):
        raise ValueError("entry_initial density must be finite")
    if not np.allclose(rho, np.swapaxes(rho.conj(), 1, 2), atol=1e-10, rtol=0.0):
        raise ValueError("entry_initial density must be Hermitian")
    if np.linalg.eigvalsh(rho).min() < -2e-8:
        raise ValueError("entry_initial density must be positive semidefinite")
    kept = solver.basis.subspace
    if np.any(np.abs(rho * ~(kept[:, :, None] & kept[:, None, :])) > 1e-8):
        raise ValueError("entry_initial has weight outside the retained subspace")
    entry = float(initial[1])
    trace = float(np.einsum("rii->", rho).real)
    if (
        not np.isfinite(entry)
        or entry < -1e-12
        or not np.isclose(trace + entry, 1.0, atol=2e-3, rtol=0.0)
    ):
        raise ValueError("entry_initial population must be normalized and nonnegative")
    return rho, max(entry, 0.0)


def _depth_history(
    solver: DensityMatrixSolver,
    lines: EmissionLines,
    thickness_ang: float,
    dz_ang: float,
    sample_ang: float,
    incidence_rad: float,
    divergence_rad: float,
    entry_state=None,
    impact: str = "averaged",
    entry_initial=None,
) -> _History:
    if thickness_ang <= 0.0 or dz_ang <= 0.0 or sample_ang <= 0.0:
        raise ValueError("thickness_ang, dz_ang and sample_ang must be positive")
    inner = max(1, int(round(sample_ang / dz_ang)))
    segments = max(1, int(np.ceil(thickness_ang / (inner * dz_ang))))
    seg = thickness_ang / segments
    step = seg / inner
    if entry_initial is not None:
        rho, entry = _validated_entry_initial(solver, entry_initial)
    elif entry_state is None:
        rho, free, outside = solver.entry_state(incidence_rad, divergence_rad)
        entry = float(free.sum()) + outside
    else:
        from .states import planar_entry_density

        if divergence_rad:
            raise ValueError("entry_state carries its own momentum spread; leave divergence_rad 0")
        rho, free, outside = solver.entry_from_plane(
            *planar_entry_density(solver, entry_state, incidence_rad=incidence_rad, impact=impact)
        )
        entry = float(free.sum()) + outside
    z, rates, bound = [0.0], [lines.rates(rho)], [solver.bound_population(rho)]
    for index in range(1, segments + 1):
        result = solver.propagate(seg, step, initial=(rho, entry))
        rho, entry = result.final_density, result.total_flux[-1]
        z.append(index * seg)
        rates.append(lines.rates(rho))
        bound.append(float(result.bound_population[-1]))
    return _History(
        z_ang=np.asarray(z),
        rates=np.asarray(rates).reshape(len(z), lines.size),
        bound=np.asarray(bound),
    )


@dataclass(frozen=True)
class ChannelingSpectrum:
    """Diagnostic channeling-radiation spectrum at the crystal exit.

    Attributes
    ----------
    photon_energy_eV
        Photon energy grid [eV].
    theta_rad, phi_rad
        Detector directions (polar angle to the beam axis, azimuth from the
        plane normal), broadcast shape ``S``.
    spectrum
        Total ``d^2N / dE dOmega`` [photons/eV/sr/electron], bound -> bound
        lines plus capture, shape ``S + (n_E,)``.
    capture_spectrum
        The free -> bound capture component of ``spectrum``.
    stokes
        Intensity-weighted normalized Stokes ``(Q, U)`` of ``spectrum``,
        shape ``S + (n_E, 2)``; NaN where the spectrum is zero.
    lines
        Line catalog (``lines.kind`` separates lines from capture).
    z_ang, line_rates
        Sampled depths [angstrom] and the estimator per catalog entry
        [1/angstrom], shape ``(n_z, n_lines)``.
    bound_population
        Channeled (bound) population at each sampled depth.
    line_yield
        Depth-integrated photons per electron per catalog entry over
        ``4 pi``, before absorption.
    """

    photon_energy_eV: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray
    spectrum: np.ndarray
    capture_spectrum: np.ndarray
    stokes: np.ndarray
    lines: EmissionLines
    z_ang: np.ndarray
    line_rates: np.ndarray
    bound_population: np.ndarray
    line_yield: np.ndarray

    @property
    def total_line_yield(self) -> float:
        """Bound -> bound photons per electron (``4 pi``, unabsorbed)."""
        return float(self.line_yield[self.lines.kind == KIND_LINE].sum())

    @property
    def total_capture_yield(self) -> float:
        """Free -> bound capture photons per electron (``4 pi``, unabsorbed)."""
        return float(self.line_yield[self.lines.kind == KIND_CAPTURE].sum())

    def quantify(self):
        """Return a copied :class:`QuantifiedChannelingSpectrum` quantity view."""
        from .quantified_emission import QuantifiedChannelingSpectrum

        return QuantifiedChannelingSpectrum.from_spectrum(self)


def _direction_nodes(aperture_rad: float, n_aperture: int):
    """Offsets ``(r, chi, weight)`` over a cone of half angle ``aperture_rad``."""
    if aperture_rad <= 0.0:
        return np.zeros(1), np.zeros(1), np.ones(1)
    x, w = np.polynomial.legendre.leggauss(n_aperture)
    radius = 0.5 * aperture_rad * (x + 1.0)
    radial = 0.5 * aperture_rad * w * np.sin(radius)
    n_chi = 2 * n_aperture
    chi = 2.0 * np.pi * (np.arange(n_chi) + 0.5) / n_chi
    r_grid, chi_grid = np.meshgrid(radius, chi, indexing="ij")
    weight = np.repeat(radial[:, None], n_chi, axis=1) * (2.0 * np.pi / n_chi)
    return r_grid.ravel(), chi_grid.ravel(), (weight / weight.sum()).ravel()


def channeling_spectrum(
    solver: DensityMatrixSolver,
    thickness_ang: float,
    photon_energy_eV,
    theta_rad,
    phi_rad=0.0,
    *,
    incidence_rad: float = 0.0,
    divergence_rad: float = 0.0,
    divergence_y_rad: float = 0.0,
    energy_spread_eV: float = 0.0,
    aperture_rad: float = 0.0,
    dz_ang: float = 50.0,
    sample_ang: float = 500.0,
    absorption: bool = True,
    layers: Sequence | None = None,
    include_capture: bool = True,
    cluster_tol_eV: float | None = None,
    n_aperture: int = 6,
    n_divergence: int = 9,
    surface_normal: np.ndarray | Sequence[float] | None = None,
    path_limits_ang: np.ndarray | Sequence[float] | None = None,
    entry_state=None,
    entry_impact: str = "averaged",
    entry_initial: tuple[np.ndarray, float] | None = None,
) -> ChannelingSpectrum:
    """Return the channeling-radiation spectrum per electron.

    ``d^2N/dE dOmega = sum_groups Re sum_m conj(w_m(z-integrated)) P(n_e)
    F_m(E)``, with ``w_m`` built from the absorbed line yields
    ``Y_c = int_0^t dz <d_c^dagger d_c>_rho(z) exp(-tau(z, E_c, n))``,

    averaged over the detector cone (``aperture_rad``) and the in-plane
    (``y``) beam divergence. Lines end on bound states: bound -> bound, plus
    radiative capture from the tracked above-barrier states when
    ``include_capture`` (reported separately as ``capture_spectrum``).
    ``rho(z)`` is propagated by ``solver`` from the divergence-averaged entry
    state at constant energy, sampled every ``sample_ang`` and integrated by
    the trapezoid rule; ``P`` is :func:`boosted_dipole_pattern` in the
    electron frame (photon angle ``theta_e`` to the electron velocity) and
    ``F_m`` the complex Voigt mode profile of :meth:`EmissionLines.spectrum`:
    centres ``/ (1 - beta cos theta_e)``, half widths likewise, Gaussian
    ``sigma = |E_m| s sigma_gamma / gamma`` with
    ``s = cos theta_e / (gamma^2 beta D) + d ln Delta E_perp / d ln gamma``
    (``-> 2 + d ln Delta E / d ln gamma`` forward; the walkthrough's
    ``2 delta gamma / gamma`` is its kinematic part). The face normal is the
    mean beam axis ``z`` by default; ``surface_normal`` allows a tilted slab.
    Photons leave along the lab direction ``n``, attenuated by ``_stack_tau`` at each line centre over
    ``layers`` (default: one slab of ``solver.profile.crystal`` over
    ``[0, thickness]``).

    Parameters
    ----------
    solver
        Density-matrix solver (profile, entry energy, kernel and margin). The
        margin sets which free states are tracked and can capture.
    thickness_ang
        Mean electron path length [angstrom]; the slab-normal thickness is
        this length times the normalized ``surface_normal`` z component.
    photon_energy_eV
        Photon energy grid [eV].
    theta_rad, phi_rad
        Detector direction(s), broadcast together.
    incidence_rad, divergence_rad
        Mean incidence and RMS divergence normal to the plane (entry state).
    divergence_y_rad
        RMS beam divergence along the plane (electron direction spread).
    energy_spread_eV
        RMS beam kinetic-energy spread [eV].
    aperture_rad
        Half angle of a circular detector cone [rad]; 0 is a point detector.
    dz_ang, sample_ang
        Solver step and estimator sampling interval [angstrom].
    absorption
        Apply exit attenuation.
    layers
        ``_stack_tau`` layers ``[(z_top, z_bot, composition), ...]``.
    include_capture
        Score free -> bound capture from the tracked free states.
    cluster_tol_eV
        Override the cluster tolerance of :func:`emission_lines`.
    n_aperture, n_divergence
        Quadrature orders of the aperture and in-plane divergence averages;
        raise them when the Doppler spread exceeds the Lorentzian width.
    surface_normal
        Slab inward normal in the beam/plane frame (x normal to the planes,
        z along the mean beam). Default ``(0, 0, 1)``. Layer depths use this
        normal, rather than the electron path coordinate.
    path_limits_ang
        Optional face/cutoff path lengths of all incident primaries, including
        zeros for misses. Weight the ensemble source by geometric survival;
        dechanneling must not be included here because it is already in rho.
    entry_state
        Optional :mod:`.states` entry state (``x`` along the plane normal)
        replacing the plane-wave/``divergence_rad`` entry; it is projected by
        :func:`~.states.planar_entry_density` and tilted by ``incidence_rad``.
        A 2D state's in-plane spread is not inferred; pass
        ``divergence_y_rad``.
    entry_impact
        ``"averaged"`` (uniform entry point) or ``"fixed"`` (the state's own
        position relative to the planes); see :func:`~.states.planar_entry_density`.
    entry_initial
        Preprojected ``(rho_S, initially_unretained_population)`` from the
        planar source projector. Reuses the same initial density as diagnostic
        propagation without projecting twice. Mutually exclusive with
        ``entry_state``, nonzero ``incidence_rad``/``divergence_rad`` and
        nondefault ``entry_impact``; these are already encoded in ``rho_S``.

    Returns
    -------
    ChannelingSpectrum
        Spectrum in photons/eV/sr/electron with line diagnostics.

    Assumptions: dipole lines ending on bound states; free -> free is
    coherent bremsstrahlung and is excluded; constant electron energy in the
    zone (no ``gamma(z)`` drift, ``Delta T / T << 1``); recoil
    ``O(hbar omega / E)`` and the radiative natural width neglected;
    dechanneled (sink) flux emits nothing, so capture depends on the
    collapse margin. Limits: zero bound states give exactly zero; zero
    tracked free states give zero capture; with ``absorption=False`` the
    energy and solid-angle integral of a point detector equals the yields.

    Validation: cr-yield-assembly, cr-line-shape, cr-capture-yield, cr-persisted-channel
    """
    if entry_initial is not None and (
        entry_state is not None
        or incidence_rad != 0.0
        or divergence_rad != 0.0
        or entry_impact != "averaged"
    ):
        raise ValueError("entry_initial conflicts with entry_state or incidence/divergence/impact")
    energies = np.asarray(photon_energy_eV, dtype=float)
    theta, phi = np.broadcast_arrays(
        np.asarray(theta_rad, dtype=float), np.asarray(phi_rad, dtype=float)
    )
    shape = theta.shape
    normal = np.asarray((0.0, 0.0, 1.0) if surface_normal is None else surface_normal, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0:
        raise ValueError("surface_normal must be a finite nonzero three-vector")
    normal = normal / np.linalg.norm(normal)
    if normal[2] <= 0:
        raise ValueError("surface_normal must point into the slab along the mean beam")
    lines = emission_lines(solver, include_capture=include_capture, cluster_tol_eV=cluster_tol_eV)
    history = _depth_history(
        solver,
        lines,
        thickness_ang,
        dz_ang,
        sample_ang,
        incidence_rad,
        divergence_rad,
        entry_state,
        entry_impact,
        entry_initial,
    )
    if path_limits_ang is not None:
        history = history.censored(path_limits_ang)
    line_yield = history.integrate(history.rates)
    spectrum = np.zeros(shape + (energies.size,))
    capture = np.zeros_like(spectrum)
    q_sum = np.zeros_like(spectrum)
    u_sum = np.zeros_like(spectrum)
    if lines.size and np.any(line_yield > 0.0):
        if absorption and layers is None:
            layers = [
                (
                    0.0,
                    float(thickness_ang) * normal[2],
                    _resolve_composition(solver.profile.crystal),
                )
            ]
        gamma = lines.gamma
        sigma_gamma_rel = float(energy_spread_eV) / M_E_EV / gamma
        radius, chi, ap_weight = _direction_nodes(float(aperture_rad), n_aperture)
        if divergence_y_rad > 0.0:
            nodes, psi_weight = np.polynomial.hermite_e.hermegauss(n_divergence)
            psi = divergence_y_rad * nodes
            psi_weight = psi_weight / psi_weight.sum()
        else:
            psi, psi_weight = np.zeros(1), np.ones(1)
        flat = [a.reshape(-1, energies.size) for a in (spectrum, capture, q_sum, u_sum)]
        for d, (t0, p0) in enumerate(zip(theta.ravel(), phi.ravel(), strict=True)):
            n0 = np.array([np.sin(t0) * np.cos(p0), np.sin(t0) * np.sin(p0), np.cos(t0)])
            t1 = np.array([np.cos(t0) * np.cos(p0), np.cos(t0) * np.sin(p0), -np.sin(t0)])
            t2 = np.array([-np.sin(p0), np.cos(p0), 0.0])
            for r, c, wa in zip(radius, chi, ap_weight, strict=True):
                n = np.cos(r) * n0 + np.sin(r) * (np.cos(c) * t1 + np.sin(c) * t2)
                normal_cosine = float(n @ normal)
                if absorption and abs(normal_cosine) < 1e-12:
                    continue
                for s, wp in zip(psi, psi_weight, strict=True):
                    ye = n[1] * np.cos(s) - n[2] * np.sin(s)
                    ze = n[1] * np.sin(s) + n[2] * np.cos(s)
                    theta_e = float(np.arctan2(np.hypot(n[0], ye), ze))
                    phi_e = float(np.arctan2(ye, n[0]))
                    if absorption:
                        centre = lines.delta_eV / float(_denominator(gamma, theta_e))
                        e_grid = np.broadcast_to(centre[:, None], (lines.size, history.z_ang.size))
                        tau = _stack_tau(
                            layers, history.z_ang[None, :] * normal[2], normal_cosine, e_grid
                        )
                        attenuated = np.exp(-np.asarray(tau, dtype=float))
                        yields = history.integrate(history.rates * attenuated.T)
                    else:
                        yields = line_yield
                    weight = wa * wp * float(boosted_dipole_pattern(theta_e, phi_e, gamma))
                    bound_part, capture_part = lines.spectrum(
                        energies, yields, theta_e, sigma_gamma_rel
                    )
                    total = weight * (bound_part + capture_part)
                    stokes = dipole_stokes(theta_e, phi_e, gamma)
                    flat[0][d] += total
                    flat[1][d] += weight * capture_part
                    flat[2][d] += total * stokes[0]
                    flat[3][d] += total * stokes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        stokes_out = np.stack([q_sum / spectrum, u_sum / spectrum], axis=-1)
    return ChannelingSpectrum(
        photon_energy_eV=energies,
        theta_rad=theta,
        phi_rad=phi,
        spectrum=spectrum,
        capture_spectrum=capture,
        stokes=stokes_out,
        lines=lines,
        z_ang=history.z_ang,
        line_rates=history.rates,
        bound_population=history.bound,
        line_yield=line_yield,
    )


# ---------------------------------------------------------------------------
# Free -> bound capture bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureBound:
    """Free -> bound radiative-capture diagnostic against the line yield.

    Attributes
    ----------
    estimate
        Scored capture yield of the tracked free states, the
        ``cr-capture-yield`` estimator (cross terms kept)
        [photons/electron, ``4 pi``].
    upper_bound
        ``lambda_max(R) int dz (1 - P_bound(z))``: every unbound electron,
        dechanneled or not, captures at most at the largest eigenvalue of the
        clustered capture-rate operator ``R = sum_c d_c^dagger d_c`` over every
        above-barrier basis state [photons/electron].
    line_yield
        Bound -> bound yield of the same solve [photons/electron].
    max_rate_per_ang
        ``lambda_max(R)`` [1/angstrom].
    edge_eV
        Forward capture edge ``(barrier - E_0) / (1 - beta)`` per block [eV].
    threshold
        Fraction of the line yield above which capture must be scored.
    """

    estimate: float
    upper_bound: float
    line_yield: float
    max_rate_per_ang: float
    edge_eV: np.ndarray
    threshold: float

    @property
    def estimate_ratio(self) -> float:
        """``estimate / line_yield``."""
        return self.estimate / self.line_yield if self.line_yield > 0.0 else np.inf

    @property
    def bound_ratio(self) -> float:
        """``upper_bound / line_yield``."""
        return self.upper_bound / self.line_yield if self.line_yield > 0.0 else np.inf

    @property
    def promote(self) -> bool:
        """True when the estimate exceeds ``threshold`` of the line yield."""
        return self.estimate_ratio > self.threshold


def _capture_rate_bound(solver: DensityMatrixSolver, gamma: float, tol_eV: float) -> float:
    """Largest eigenvalue of the clustered capture-rate operator [1/angstrom]."""
    basis = solver.basis
    best = 0.0
    for block in range(solver.n_cells):
        energies = basis.energies[block]
        bound = np.flatnonzero(energies < solver.barrier_eV)
        free = np.flatnonzero(energies >= solver.barrier_eV)
        if bound.size == 0 or free.size == 0:
            continue
        states = np.concatenate([bound, free])
        dipoles = _block_dipoles(solver, block, states)
        delta = energies[free][None, :] - energies[bound][:, None]
        x = dipoles[: bound.size, bound.size :]
        amp = np.sqrt(dipole_rate_per_ang(delta, np.abs(x), gamma)) * np.exp(1j * np.angle(x))
        # free states sorted by energy; clusters are runs with gaps <= tol
        label = np.concatenate([[0], np.cumsum(np.diff(energies[free]) > tol_eV)])
        same = label[:, None] == label[None, :]
        rate = np.zeros((free.size, free.size), dtype=complex)
        for row in amp:
            rate += same * np.conj(row)[:, None] * row[None, :]
        best = max(best, float(np.linalg.eigvalsh(rate).max()))
    return best


def capture_bound(
    solver: DensityMatrixSolver,
    thickness_ang: float,
    *,
    incidence_rad: float = 0.0,
    divergence_rad: float = 0.0,
    dz_ang: float = 50.0,
    sample_ang: float = 500.0,
    threshold: float = DEFAULT_CAPTURE_THRESHOLD,
    cluster_tol_eV: float | None = None,
    entry_state=None,
    entry_impact: str = "averaged",
) -> CaptureBound:
    """Bound the free -> bound radiative-capture yield against the line yield.

    Capture of an above-barrier state ``i`` into bound ``f`` is the same
    dipole emission as a line, at ``hbar omega >= (barrier - E_f)/(1 - beta
    cos theta)``. The *estimate* is the scored capture yield of
    :func:`emission_lines` on the tracked free states
    (``barrier <= E < barrier + margin``), cross terms between near-degenerate
    free pairs kept. The diagonal form would hold by parity only at the
    symmetric blocks ``k = 0, g/2``; on finer Bloch grids that protection is
    lost. The *upper bound* charges every unbound electron, including
    dechanneled flux, at ``lambda_max`` of the clustered capture-rate
    operator over all above-barrier basis states (clusters: runs of free
    levels within ``cluster_tol_eV``, default the largest tracked capture
    member half width, which every scored clustering link stays within).

    Validation: cr-capture-bound
    """
    lines = emission_lines(solver, include_capture=True, cluster_tol_eV=cluster_tol_eV)
    history = _depth_history(
        solver,
        lines,
        thickness_ang,
        dz_ang,
        sample_ang,
        incidence_rad,
        divergence_rad,
        entry_state,
        entry_impact,
    )
    yields = np.trapezoid(history.rates, history.z_ang, axis=0)
    is_capture = lines.kind == KIND_CAPTURE
    if cluster_tol_eV is None:
        # Each scored link joins pairs within half their summed half widths, at
        # most the largest pair half width: the bound's runs are then coarser.
        tol = float(lines.max_member_hwhm_eV[is_capture].max()) if is_capture.any() else 0.0
    else:
        tol = float(cluster_tol_eV)
    max_rate = _capture_rate_bound(solver, lines.gamma, tol)
    unbound = float(np.trapezoid(1.0 - history.bound, history.z_ang))
    ground = solver.basis.energies[:, 0]
    edge = (solver.barrier_eV - ground) / _denominator(lines.gamma, 0.0)
    return CaptureBound(
        estimate=float(yields[is_capture].sum()),
        upper_bound=max_rate * unbound,
        line_yield=float(yields[~is_capture].sum()),
        max_rate_per_ang=max_rate,
        edge_eV=np.asarray(edge),
        threshold=float(threshold),
    )
