"""Monte Carlo wave-function (quantum-jump) unravelling of channeling evolution.

Each electron carries a pure transverse state in one Bloch block and follows
the same operators as :class:`~.density.DensityMatrixSolver`:

- **No-jump evolution.** ``psi`` evolves unnormalized under the non-Hermitian
  ``H_eff = H_perp - (i hbar v / 2) C(x, x)`` restricted to the subspace ``S``.
  Its norm decays at the full loss rate ``<psi|C(x, x)|psi>``.
- **Jump times.** Waiting-time rule (Dalibard, Castin & Mølmer 1992;
  Carmichael 1993): draw ``r ~ U(0, 1)`` and jump when ``||psi||^2`` falls
  below ``r``. The step in which the norm crosses ``r`` places the jump at its
  midpoint, between the two half-step propagators the density solver uses.
  At most one jump is taken per step: a second crossing is deferred to the
  next step's midpoint, a relative jump-rate bias of about
  ``0.46 (rate dz)^2``.
- **Jump operators.** The kept kernel block of residue ``r`` is
  ``K_r = sum_s lambda_s u_s u_s^dagger``, so
  ``C(x, x') = sum_{r,s} L_rs(x) L_rs(x')*`` with
  ``L_rs = sqrt(lambda_s) sum_h u_s[h] e^{i(kappa_r + h g) x}``. ``L_rs`` moves a
  state from block ``b`` to ``b + r`` (mod ``n_cells``). A jump picks channel
  ``(r, s)`` with probability ``||P L_rs psi||^2 / <psi|C(x, x)|psi>``.
  The remainder, momenta outside the plane-wave or kept band, is an overflow
  jump that dechannels.
- **Collapse.** After a kept jump the electron stays channeled with probability
  ``||P_S phi||^2 / ||phi||^2`` and continues as ``P_S phi``. Otherwise it is
  dechanneled into an above-cutoff state ``n`` drawn with weight
  ``|<n|phi>|^2``, which records its ``E_perp``. Entry uses the same rule on the
  sampled entry plane wave.

The ensemble mean of ``|psi_hat><psi_hat|`` over surviving electrons reproduces
the density solver's ``rho_S``. A single trajectory is an unravelling, not a
physical history: only ensemble observables are physical.

**RNG.** Electron ``i`` owns a Philox stream keyed on ``(seed, i)`` under its own
salt. Its draws are consumed in a fixed order: entry momentum, entry collapse,
then per jump a channel, a collapse and the next threshold, and at each
re-projection a collapse. Results therefore do not depend on batch
composition or size.

**Energy loss.** As in the density solver, a constant stopping power lowers
``gamma``. When it drifts by more than a tolerance, every electron's state is
re-projected onto the new eigenstates (the kernel rescaled by
``(beta_0/beta)^2``). It stays channeled with probability
``||P_S psi'||^2 / ||psi'||^2`` with its norm, and so its jump threshold,
unchanged; otherwise it is counted as reprojection flux. In expectation this is
the density solver's ``P_S rho P_S`` re-projection.

Lengths in angstroms, energies in eV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.channeling.transverse import longitudinal_kinematics
from ...materials.crystal import HBARC_EV_ANG
from ..transport.kinematics import (
    _SM64_GOLDEN,
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_ONE,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
)
from .density import DensityMatrixSolver

_MCWF_STREAM_SALT = np.uint64(0x1F83D9AB5BE0CD19)
# Kernel eigenvalues below this fraction of the block maximum are dropped as
# numerical zeros of a positive semidefinite matrix.
_CHANNEL_RTOL = 1e-12


def _mix(x):
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


def mcwf_stream_key(seed, electron):
    """Philox key of electron ``electron`` under ``seed``; SplitMix64-mixed.

    Validation: cr-mcwf-unravelling
    """
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ _MCWF_STREAM_SALT
        index = np.asarray(electron, dtype=np.uint64)
        return _mix(x + _SM64_GOLDEN * (index + _SM64_ONE))


def _rng(seed: int, electron: int) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(key=int(mcwf_stream_key(seed, electron))))


@dataclass(frozen=True)
class MCWFResult:
    """Ensemble statistics of an MCWF channeling run.

    Attributes
    ----------
    z_ang
        Recorded depths [angstrom].
    band_populations, band_populations_sem
        Ensemble mean and standard error of the ``n``-th lowest state's
        population, summed over Bloch blocks; shape ``(n_z, n_bands)``.
    bound_population, bound_population_sem
        Mean and standard error of the population below the barrier.
    trace
        Fraction of electrons still channeled (in ``S``).
    free_flux, overflow_flux, reprojection_flux, entry_flux
        Cumulative fraction of electrons dechanneled by each channel.
    kinetic_energy_eV
        Electron kinetic energy at each recorded depth [eV].
    dechanneled_energy_eV
        ``E_perp`` of every free or entry dechanneling event [eV]; ``nan`` for
        overflow.
    dechanneled_depth_ang
        Depth of each dechanneling event [angstrom], aligned with the energies.
    dechanneled_electron
        Electron index (``first_electron`` offset included) of each event.
    final_density
        Ensemble mean of ``|psi_hat><psi_hat|`` over surviving electrons at the
        final depth, normalized per electron launched; shape ``(n_cells, N, N)``.
    jumps
        Total number of jumps over the ensemble.
    """

    z_ang: np.ndarray
    band_populations: np.ndarray
    band_populations_sem: np.ndarray
    bound_population: np.ndarray
    bound_population_sem: np.ndarray
    trace: np.ndarray
    free_flux: np.ndarray
    overflow_flux: np.ndarray
    reprojection_flux: np.ndarray
    kinetic_energy_eV: np.ndarray
    entry_flux: float
    dechanneled_energy_eV: np.ndarray
    dechanneled_depth_ang: np.ndarray
    dechanneled_electron: np.ndarray
    final_density: np.ndarray
    jumps: int


class MCWFSolver:
    """Quantum-jump stepper sharing a :class:`~.density.DensityMatrixSolver`'s operators.

    Validation: cr-mcwf-unravelling
    """

    def __init__(self, solver: DensityMatrixSolver) -> None:
        self.solver = solver
        self.n_cells = solver.n_cells
        self.max_order = solver.max_order
        self.size = solver.orders.size
        self.channels = []  # per residue: (sqrt(lambda), u) with u (N, n_channels)
        for block in solver.kernel.momentum_kernel:
            values, vectors = np.linalg.eigh(block)
            keep = values > _CHANNEL_RTOL * max(values.max(), 0.0)
            self.channels.append((np.sqrt(values[keep]), vectors[:, keep]))

    # -- single-electron operations -----------------------------------------

    def _loss(self, block: int, coefficients: np.ndarray) -> float:
        """``<psi|C(x, x)|psi>`` of a normalized eigenbasis state in ``block``."""
        plane = self.solver.basis.vectors[block] @ coefficients
        return float(np.real(np.vdot(plane, self.solver._loss_toeplitz @ plane)))

    def _jump_candidates(self, block: int, coefficients: np.ndarray):
        """All kept jump results ``P L_rs psi`` as ``(target block, amplitudes)`` per residue."""
        plane = self.solver.basis.vectors[block] @ coefficients
        size, top = self.size, self.max_order
        out = []
        for residue, (root, vectors) in enumerate(self.channels):
            if root.size == 0:
                out.append((0, np.zeros((size, 0), dtype=complex)))
                continue
            target = block + residue
            carry = target // self.n_cells
            # (L psi)[o] = sum_h u[h] psi[o - h - carry], orders o, h in -H..H.
            lag = np.arange(size)[:, None] - np.arange(size)[None, :] - carry + top
            valid = (lag >= 0) & (lag < size)
            toeplitz = np.where(valid, plane[np.clip(lag, 0, size - 1)], 0.0)
            out.append((target % self.n_cells, (toeplitz @ vectors) * root))
        return out

    def _collapse(self, block: int, plane: np.ndarray, rng: np.random.Generator):
        """Apply the collapse rule to a plane-wave state; return the survivor or ``E_perp``."""
        basis = self.solver.basis
        eigen = basis.vectors[block].conj().T @ plane
        weight = np.abs(eigen) ** 2
        keep = basis.subspace[block]
        total = weight.sum()
        stay = weight[keep].sum()
        if rng.random() * total < stay:
            return np.where(keep, eigen, 0.0) / np.sqrt(stay), None
        free = np.where(keep, 0.0, weight)
        state = int(np.searchsorted(np.cumsum(free), rng.random() * free.sum(), "right"))
        return None, float(basis.energies[block, min(state, free.size - 1)])

    def _entry_mixture(self, plane: np.ndarray, outside: float):
        """Pure plane-wave-basis components ``(block, vector)`` and weights of an entry density."""
        blocks, vectors, weights = [], [], []
        for r in range(self.n_cells):
            values, modes = np.linalg.eigh(plane[r])
            for value, mode in zip(values, modes.T, strict=True):
                if value > 1e-14:
                    blocks.append(r)
                    vectors.append(mode)
                    weights.append(value)
        weights.append(max(outside, 0.0))
        weights = np.asarray(weights)
        return blocks, vectors, weights / weights.sum()

    def _entry_from_mixture(self, rng, blocks, vectors, weights):
        """Sample one pure entry component (or out-of-band flux) and collapse it onto ``S``."""
        pick = min(
            int(np.searchsorted(np.cumsum(weights), rng.random(), "right")), weights.size - 1
        )
        if pick == len(blocks):
            rng.random()  # keep the draw order fixed
            return 0, None, np.nan
        state, energy = self._collapse(blocks[pick], vectors[pick], rng)
        return blocks[pick], state, energy

    def _entry(self, rng, lattice, weights):
        """Sample an entry plane wave and collapse it onto ``S``."""
        pick = int(np.searchsorted(np.cumsum(weights), rng.random(), "right"))
        momentum = int(lattice[min(pick, lattice.size - 1)])
        block = momentum % self.n_cells
        order = (momentum - block) // self.n_cells
        if abs(order) > self.max_order:
            rng.random()  # keep the draw order fixed
            return block, None, np.nan
        plane = np.zeros(self.size, dtype=complex)
        plane[order + self.max_order] = 1.0
        state, energy = self._collapse(block, plane, rng)
        return block, state, energy

    # -- ensemble -----------------------------------------------------------------

    def run(
        self,
        n_electrons: int,
        thickness_ang: float,
        dz_ang: float,
        *,
        seed: int,
        incidence_rad: float = 0.0,
        divergence_rad: float = 0.0,
        record_every: int = 1,
        first_electron: int = 0,
        stopping_eV_per_ang: float = 0.0,
        reprojection_tolerance: float = 1e-4,
        entry_state=None,
        entry_impact: str = "averaged",
    ) -> MCWFResult:
        """Run ``n_electrons`` trajectories ``first_electron, ...`` to ``thickness_ang``.

        ``entry_state`` (a :mod:`.states` state, ``x`` along the plane normal)
        replaces the plane-wave/``divergence_rad`` entry: each electron draws one
        pure component of its impact-averaged density
        (:func:`~.states.planar_entry_density`, ``impact=entry_impact``), tilted by
        ``incidence_rad``.

        Validation: cr-mcwf-unravelling, cr-entry-state-projection
        """
        if n_electrons < 1 or thickness_ang <= 0.0 or dz_ang <= 0.0:
            raise ValueError("n_electrons, thickness_ang and dz_ang must be positive")
        solver = self.solver
        # A shared density solver may have left its basis at a slowed-down gamma.
        if solver.basis.gamma != solver.gamma0 or solver._scale != 1.0:
            solver.basis, solver._scale = solver._basis(solver.gamma0), 1.0
        steps = max(1, int(round(thickness_ang / dz_ang)))
        dz = thickness_ang / steps
        half = solver._propagators(dz, solver.hbar_v0)
        if entry_state is None:
            lattice, weights = solver.entry_lattice(incidence_rad, divergence_rad)

            def draw_entry(rng):
                return self._entry(rng, lattice, weights)
        else:
            from .states import planar_entry_density

            if divergence_rad:
                raise ValueError(
                    "entry_state carries its own momentum spread; leave divergence_rad 0"
                )
            mixture = self._entry_mixture(
                *planar_entry_density(
                    solver, entry_state, incidence_rad=incidence_rad, impact=entry_impact
                )
            )

            def draw_entry(rng):
                return self._entry_from_mixture(rng, *mixture)

        streams = [_rng(seed, first_electron + i) for i in range(n_electrons)]
        blocks = np.zeros(n_electrons, dtype=int)
        states = np.zeros((n_electrons, self.size), dtype=complex)
        alive = np.zeros(n_electrons, dtype=bool)
        thresholds = np.zeros(n_electrons)
        energies, depths, owners = [], [], []
        counts = {"free": 0, "overflow": 0, "reprojection": 0}
        energy_eV, gamma_basis = solver.energy_eV, solver.gamma0
        entry = 0
        for i, rng in enumerate(streams):
            blocks[i], state, energy = draw_entry(rng)
            if state is None:
                entry += 1
                energies.append(energy)
                depths.append(0.0)
                owners.append(first_electron + i)
                continue
            states[i], alive[i] = state, True
            thresholds[i] = rng.random()
        keys = ("z", "E", "bands", "bands2", "bound", "bound2", "alive", "free", "over", "repro")
        history = {k: [] for k in keys}

        def record(z: float) -> None:
            norms = np.sum(np.abs(states) ** 2, axis=1)
            pops = np.abs(states) ** 2 / np.where(norms > 0.0, norms, 1.0)[:, None]
            pops[~alive] = 0.0
            bound = solver.basis.energies < solver.barrier_eV
            b = np.einsum("tn,tn->t", pops, bound[blocks])
            history["z"].append(z)
            history["E"].append(energy_eV)
            history["bands"].append(pops.mean(axis=0))
            history["bands2"].append(pops.std(axis=0) / np.sqrt(n_electrons))
            history["bound"].append(b.mean())
            history["bound2"].append(b.std() / np.sqrt(n_electrons))
            history["alive"].append(alive.mean())
            history["free"].append(counts["free"] / n_electrons)
            history["over"].append(counts["overflow"] / n_electrons)
            history["repro"].append(counts["reprojection"] / n_electrons)

        jumps = 0
        record(0.0)
        for step in range(1, steps + 1):
            live = np.flatnonzero(alive)
            middle = np.einsum("tij,tj->ti", half[blocks[live]], states[live])
            end = np.einsum("tij,tj->ti", half[blocks[live]], middle)
            norm_end = np.sum(np.abs(end) ** 2, axis=1)
            states[live] = end
            for k in np.flatnonzero(norm_end < thresholds[live]):
                i = live[k]
                rng = streams[i]
                jumps += 1
                psi = middle[k] / np.linalg.norm(middle[k])
                block, survivor, energy = self._jump(blocks[i], psi, rng)
                if survivor is None:
                    alive[i] = False
                    states[i] = 0.0
                    counts["overflow" if np.isnan(energy) else "free"] += 1
                    energies.append(energy)
                    depths.append((step - 0.5) * dz)
                    owners.append(first_electron + i)
                    continue
                blocks[i] = block
                states[i] = half[block] @ survivor
                thresholds[i] = rng.random()
            if stopping_eV_per_ang:
                energy_eV = solver.energy_eV - stopping_eV_per_ang * step * dz
                gamma, beta = longitudinal_kinematics(energy_eV)
                if abs(gamma / gamma_basis - 1.0) > reprojection_tolerance:
                    old = solver.basis.vectors
                    solver.basis = solver._basis(gamma)
                    solver._scale = (solver.beta0 / beta) ** 2
                    half = solver._propagators(dz, HBARC_EV_ANG * beta)
                    gamma_basis = gamma
                    for i in np.flatnonzero(alive):
                        state, lost = self._reproject(
                            blocks[i], old[blocks[i]], states[i], streams[i]
                        )
                        if state is None:
                            alive[i] = False
                            states[i] = 0.0
                            counts["reprojection"] += 1
                            energies.append(lost)
                            depths.append(step * dz)
                            owners.append(first_electron + i)
                        else:
                            states[i] = state
            if step % record_every == 0 or step == steps:
                record(step * dz)
        final = np.zeros((self.n_cells, self.size, self.size), dtype=complex)
        for i in np.flatnonzero(alive):
            psi = states[i] / np.linalg.norm(states[i])
            final[blocks[i]] += np.outer(psi, psi.conj())
        return MCWFResult(
            z_ang=np.asarray(history["z"]),
            band_populations=np.asarray(history["bands"]),
            band_populations_sem=np.asarray(history["bands2"]),
            bound_population=np.asarray(history["bound"]),
            bound_population_sem=np.asarray(history["bound2"]),
            trace=np.asarray(history["alive"]),
            free_flux=np.asarray(history["free"]),
            overflow_flux=np.asarray(history["over"]),
            reprojection_flux=np.asarray(history["repro"]),
            kinetic_energy_eV=np.asarray(history["E"]),
            entry_flux=entry / n_electrons,
            dechanneled_energy_eV=np.asarray(energies, dtype=float),
            dechanneled_depth_ang=np.asarray(depths, dtype=float),
            dechanneled_electron=np.asarray(owners, dtype=int),
            final_density=final / n_electrons,
            jumps=jumps,
        )

    def _reproject(self, block, old_vectors, state, rng):
        """Move one electron onto the current basis; ``(state | None, E_perp | nan)``.

        The plane-wave content is kept; the electron survives into ``S`` with
        probability ``||P_S psi'||^2 / ||psi'||^2`` and keeps its norm.
        """
        plane = old_vectors @ state
        basis = self.solver.basis
        eigen = basis.vectors[block].conj().T @ plane
        weight = np.abs(eigen) ** 2
        keep = basis.subspace[block]
        total, stay = weight.sum(), weight[keep].sum()
        if rng.random() * total < stay:
            return np.where(keep, eigen, 0.0) * np.sqrt(total / stay), np.nan
        free = np.where(keep, 0.0, weight)
        index = int(np.searchsorted(np.cumsum(free), rng.random() * free.sum(), "right"))
        return None, float(basis.energies[block, min(index, free.size - 1)])

    def _jump(self, block: int, psi: np.ndarray, rng: np.random.Generator):
        """One jump from normalized ``psi``: ``(block, survivor | None, E_perp | nan)``."""
        candidates = self._jump_candidates(block, psi)
        probabilities = np.concatenate([np.sum(np.abs(a) ** 2, axis=0) for _, a in candidates])
        loss = self._loss(block, psi)
        overflow = max(loss - probabilities.sum(), 0.0)
        pick = rng.random() * (probabilities.sum() + overflow)
        cumulative = np.cumsum(probabilities)
        index = int(np.searchsorted(cumulative, pick, "right"))
        if index >= probabilities.size:
            rng.random()  # keep the draw order fixed
            return block, None, np.nan
        for target, amplitudes in candidates:
            if index < amplitudes.shape[1]:
                plane = amplitudes[:, index]
                return (target, *self._collapse(target, plane, rng))
            index -= amplitudes.shape[1]
        raise AssertionError("jump channel index out of range")
