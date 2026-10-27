"""Quantum-jump (MCWF) unravelling of the axial tier on a real-space grid (#508).

Each electron's transverse state ``psi = exp(i k_t . r) phi`` evolves under
the non-Hermitian split-step of :mod:`.realspace` with the dissipator's full
loss rate ``W = (hbar v / 2) C(r, r)``. A waiting-time draw ``xi ~ U(0, 1)``
fixes the jump depth (``||phi||^2 = xi``); the mechanism is chosen by its
share of ``int |psi|^2 C(r, r)``:

- **TDS in grid:** a source string ``(s, R)`` is drawn with weight
  ``lambda_s int |psi|^2 Var_u phi(r)``, then a displacement ``u`` from
  ``p(u) ||L_u psi||^2`` by rejection, and ``phi <- L_u phi``;
- **TDS sink:** the transfer leaves the grid band; the electron leaves the
  channeled zone (recorded, removed from the ensemble);
- **electronic:** a transfer node ``q`` is drawn with weight
  ``c_q <rho_q>``, a uniform azimuth fixes ``q``, ``phi <- sqrt(rho_q) phi`` and
  ``k_t <- k_t + q``.

The ensemble average of normalized surviving states, with removed electrons
counted as zero, is the Lindblad density matrix (Dalibard, Castin & Mølmer,
PRL 68, 580 (1992)); the jump depth is resolved to one step. Streams are the
per-electron Philox keys of :mod:`.mcwf`.

Lengths in angstroms, energies in eV.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.fft

from ..._units import reject_quantity
from ...materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics
from .axial_dissipator import AxialDissipator
from .mcwf import _rng
from .realspace import RealSpaceGrid
from .states import EntryState, MixedState, momentum_inv_ang

#: Rejection trials allowed per TDS displacement draw.
MAX_REJECTION_TRIALS = 100_000
#: Displacement proposals evaluated together per rejection round.
_PROPOSALS = 64
#: Annulus edges of the displacement envelope [standard deviations]; |u| > 8 sigma
#: (probability e^-32) is not drawn.
_SHELL_EDGES = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
)
#: Electrons advanced together (stacked FFTs).
BATCH = 128

Observable = Callable[[np.ndarray, tuple[float, float]], np.ndarray]
#: ``hook(electron, phi, tilt)`` after entry and every kept jump: a replacement
#: state, or ``None`` to absorb the electron.
JumpHook = Callable[[int, np.ndarray, tuple[float, float]], "np.ndarray | None"]


def density_observable(phi: np.ndarray, tilt: tuple[float, float]) -> np.ndarray:
    """Default observable: ``|psi|^2`` per sample of a normalized state."""
    return np.abs(phi) ** 2


@dataclass(frozen=True)
class AxialMCWFResult:
    """Ensemble averages of an axial MCWF run.

    Attributes
    ----------
    depths_ang
        Observation depths [angstrom].
    observables
        Ensemble mean of the observable at each depth (removed electrons
        contribute zero), shape ``(n_depths, ...)``.
    surviving
        Fraction of electrons not removed by a sink jump or a ``jump_hook``
        absorption, per depth.
    sink_depths_ang
        Depth of every sink jump [angstrom].
    jumps
        Total jumps per mechanism: ``tds``, ``sink``, ``electronic``, plus
        ``absorbed`` (hook absorptions) when a ``jump_hook`` is given.
    n_electrons
        Trajectories run.
    jump_history
        Cumulative raw ensemble jump counts at each recorded depth, by
        mechanism. Divide by ``n_electrons`` for counts per incident electron.
    """

    depths_ang: np.ndarray
    observables: np.ndarray
    surviving: np.ndarray
    sink_depths_ang: np.ndarray
    jumps: dict[str, int]
    n_electrons: int
    jump_history: dict[str, np.ndarray] | None = None


class AxialMCWF:
    """Per-electron quantum-jump stepper for the axial tier.

    Parameters
    ----------
    dissipator
        Grid kernels and jump data (:func:`.axial_dissipator.axial_dissipator`).
    kinetic_energy_eV
        Electron kinetic energy [eV].
    potential_eV
        Real ``U`` on the dissipator's grid [eV] (``None``: free space).
    backend
        ``cpu`` (default) or explicit ``cuda``. CUDA uses complex128 and
        independent counter-addressed per-electron Philox streams; CPU draw
        order is preserved. CUDA imports are lazy.

    Validation: cr-axial-realspace-dissipator
    """

    def __init__(
        self,
        dissipator: AxialDissipator,
        kinetic_energy_eV: float,
        potential_eV: np.ndarray | None = None,
        *,
        backend: str = "cpu",
    ) -> None:
        if backend not in ("cpu", "cuda"):
            raise ValueError("axial MCWF backend must be 'cpu' or 'cuda'")
        self.backend = backend
        self.dissipator = dissipator
        self.grid: RealSpaceGrid = dissipator.grid
        self.kinetic_energy_eV = float(kinetic_energy_eV)
        self.gamma, _ = longitudinal_kinematics(self.kinetic_energy_eV)
        self.hbar_v = dissipator.hbar_v_eV_ang
        self.momentum = momentum_inv_ang(self.kinetic_energy_eV)
        u = np.zeros(self.grid.shape) if potential_eV is None else np.asarray(potential_eV, float)
        if u.shape != self.grid.shape:
            raise ValueError(f"potential must have the grid shape {self.grid.shape}")
        self._complex_potential = u - 1j * dissipator.absorptive_eV
        self._rates = {
            "tds": dissipator.tds_grid_rate,
            "sink": dissipator.tds_sink_rate,
            "electronic": dissipator.electronic_rate,
        }

    # -- stepping -----------------------------------------------------------

    def _half_kinetic(self, tilt, step):
        kx = self.grid.kx[:, None] + tilt[0]
        ky = self.grid.ky[None, :] + tilt[1]
        energy = HBAR2_2M_EV_ANG2 / self.gamma * (kx**2 + ky**2)
        return np.exp(-0.5j * step * energy / self.hbar_v)

    # -- jumps --------------------------------------------------------------

    def _jump(self, state, tilt, rng):
        """Draw one jump: ``(kind, normalized state or None for a sink, tilt)``."""
        density = np.abs(state) ** 2
        shares = [self._expect(density, rate) for rate in self._rates.values()]
        kind = list(self._rates)[_choose(rng, shares)]
        if kind == "sink":
            return kind, None, tilt
        if kind == "tds":
            state = self._tds_jump(state, density, rng)
        else:
            state, tilt = self._electronic_jump(state, density, tilt, rng)
        return kind, state / np.sqrt(np.sum(np.abs(state) ** 2) * self.grid.cell_area_ang2), tilt

    def _expect(self, density: np.ndarray, rate: np.ndarray) -> float:
        return float(np.sum(density * rate) * self.grid.cell_area_ang2)

    def _tds_jump(self, phi, density, rng):
        d = self.dissipator
        weights = np.einsum("sij,ij->s", d.tds_source_rates, density)
        source = _choose(rng, weights)
        src = d.tds_sources[source]
        sigma = np.sqrt(src.displacement_ang2)
        edges = sigma * _SHELL_EDGES
        coefficients = d.tds_weight_spectrum(source, density)
        envelope = np.maximum(d.tds_weight_envelope(coefficients, edges), 0.0)
        # Rayleigh shell probabilities of |u| ~ 2D Gaussian.
        shell_p = -np.diff(np.exp(-0.5 * _SHELL_EDGES**2))
        proposal = shell_p * envelope
        for _ in range(MAX_REJECTION_TRIALS // _PROPOSALS):
            shells = np.searchsorted(
                np.cumsum(proposal), rng.random(_PROPOSALS) * proposal.sum(), "right"
            )
            shells = np.minimum(shells, proposal.size - 1)
            lo = np.exp(-0.5 * _SHELL_EDGES[shells] ** 2)
            hi = np.exp(-0.5 * _SHELL_EDGES[shells + 1] ** 2)
            radius = sigma * np.sqrt(-2.0 * np.log(hi + (lo - hi) * rng.random(_PROPOSALS)))
            angle = 2.0 * np.pi * rng.random(_PROPOSALS)
            offsets = np.stack([radius * np.cos(angle), radius * np.sin(angle)], 1)
            uniforms = rng.random(_PROPOSALS)
            weights = d.tds_weight(coefficients, offsets)
            if np.any(weights > envelope[shells] * (1.0 + 1e-12)):
                raise RuntimeError("TDS displacement weight exceeded its rejection envelope")
            accepted = np.flatnonzero(uniforms * envelope[shells] < weights)
            if accepted.size:
                return d.tds_phase(source, offsets[accepted[0]]) * phi
        raise RuntimeError("TDS displacement rejection sampling did not converge")

    def _electronic_jump(self, phi, density, tilt, rng):
        d = self.dissipator
        weights = d.electronic_weights * np.maximum(d.electronic_expectations(density), 0.0)
        node = _choose(rng, weights)
        q = float(d.electronic_q_inv_ang[node])
        azimuth = 2.0 * np.pi * rng.random()
        kicked = (tilt[0] + q * np.cos(azimuth), tilt[1] + q * np.sin(azimuth))
        return np.sqrt(d.smoothed_density(q)) * phi, kicked

    # -- ensemble -----------------------------------------------------------

    def run(
        self,
        entry: EntryState,
        depths_ang: Sequence[float],
        n_electrons: int,
        *,
        seed: int = 0,
        step_ang: float = 5.0,
        tilt_rad: tuple[float, float] = (0.0, 0.0),
        observable: Observable = density_observable,
        jump_hook: JumpHook | None = None,
        batch_size: int | None = None,
        memory_budget_bytes: int | None = None,
    ) -> AxialMCWFResult:
        """Run ``n_electrons`` trajectories from ``entry`` and average ``observable``.

        ``depths_ang`` must be ascending multiples of ``step_ang`` (rounded to
        the nearest step). ``tilt_rad`` is the mean carrier tilt.
        ``jump_hook`` (research diagnostics) sees each normalized state at entry
        and after every TDS or electronic jump; it returns the state to
        continue, or ``None`` to absorb the electron (counted as ``absorbed``).
        It must draw from its own random streams; ``None`` leaves every
        trajectory unchanged.

        ``batch_size`` caps the number of simultaneously advanced electrons;
        CUDA also admits batches against the existing resource policy and
        optional ``memory_budget_bytes``. CUDA callbacks must advertise
        ``cuda_compatible = True`` and consume/return CuPy arrays. Observable
        ``recording_bytes`` and ``scratch_bytes`` declarations reserve custom
        reductions before device allocation. Results contain host arrays.
        Depths, step and tilt are numeric-only; Pint quantities raise ``TypeError``.
        """
        reject_quantity(depths_ang, field="AxialMCWF.run.depths_ang", unit="angstrom")
        reject_quantity(step_ang, field="AxialMCWF.run.step_ang", unit="angstrom")
        reject_quantity(tilt_rad, field="AxialMCWF.run.tilt_rad", unit="rad")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if self.backend == "cuda":
            from ._axial_cuda import run_cuda

            return run_cuda(
                self,
                entry,
                depths_ang,
                n_electrons,
                seed=seed,
                step_ang=step_ang,
                tilt_rad=tilt_rad,
                observable=observable,
                jump_hook=jump_hook,
                batch_size=batch_size,
                memory_budget_bytes=memory_budget_bytes,
            )
        if memory_budget_bytes is not None:
            raise ValueError("memory_budget_bytes is only supported for CUDA axial MCWF")
        return self._run_cpu(
            entry,
            depths_ang,
            n_electrons,
            seed=seed,
            step_ang=step_ang,
            tilt_rad=tilt_rad,
            observable=observable,
            jump_hook=jump_hook,
            batch_size=batch_size or BATCH,
        )

    def _run_cpu(
        self,
        entry,
        depths_ang,
        n_electrons,
        *,
        seed,
        step_ang,
        tilt_rad,
        observable,
        jump_hook,
        batch_size,
    ) -> AxialMCWFResult:
        """Historical NumPy trajectory loop; preserves CPU RNG draws/event order."""
        steps = _depth_steps(depths_ang, n_electrons, step_ang)
        components = _sample_components(self.grid, entry)
        mix = np.array([w for w, _ in components])
        base_tilt = (self.momentum * tilt_rad[0], self.momentum * tilt_rad[1])
        potential_phase = np.exp(-1j * step_ang * self._complex_potential / self.hbar_v)
        sums: list[np.ndarray | None] = [None] * steps.size
        alive = np.zeros(steps.size)
        sinks: list[float] = []
        jumps = {"tds": 0, "sink": 0, "electronic": 0}
        history = {kind: np.zeros(steps.size, dtype=np.int64) for kind in jumps}
        absorbed = 0
        area = self.grid.cell_area_ang2
        cells = self.grid.shape[0] * self.grid.shape[1]

        batch = batch_size
        for start in range(0, n_electrons, batch):
            rngs = [_rng(seed, e) for e in range(start, min(start + batch, n_electrons))]
            size = len(rngs)
            phi = np.empty((size, *self.grid.shape), dtype=complex)
            for b, rng in enumerate(rngs):
                phi[b] = components[_choose(rng, mix)][1]
            phi /= np.sqrt(np.sum(np.abs(phi) ** 2, axis=(1, 2)) * area)[:, None, None]
            tilts = [base_tilt] * size
            live = _enter(jump_hook, start, phi, base_tilt, area)
            absorbed += int(size - live.sum())
            half = np.broadcast_to(self._half_kinetic(base_tilt, step_ang), phi.shape).copy()
            full = half**2
            thresholds = np.array([rng.random() for rng in rngs])
            # Stored spectrum lags the physical one by a half kinetic step:
            # physical = half * lagged, so consecutive half steps fuse into one.
            lagged = _fft2(phi) * half.conj()
            step = 0
            local_jumps = dict.fromkeys(jumps, 0)
            for index, target in enumerate(steps):
                while step < target and live.any():
                    lagged *= full
                    lagged = _fft2(potential_phase * _ifft2(lagged))
                    step += 1
                    norms = np.sum(np.abs(lagged) ** 2, axis=(1, 2)) * area / cells
                    for b in np.flatnonzero(live & (norms <= thresholds)):
                        state = _ifft2(lagged[b] * half[b]) / np.sqrt(norms[b])
                        kind, state, tilts[b] = self._jump(state, tilts[b], rngs[b])
                        jumps[kind] += 1
                        local_jumps[kind] += 1
                        if kind == "sink":
                            sinks.append(step * step_ang)
                        elif kind == "electronic":
                            half[b] = self._half_kinetic(tilts[b], step_ang)
                            full[b] = half[b] ** 2
                        kept = _apply_hook(jump_hook, start + b, state, tilts[b], area)
                        # A sink removes before the hook; count only hook removals.
                        absorbed += int(kept is None) - int(kind == "sink")
                        state = kept
                        if state is None:
                            live[b] = False
                            lagged[b] = 0.0
                            continue
                        lagged[b] = _fft2(state) * half[b].conj()
                        thresholds[b] = rngs[b].random()
                sums[index] = _observe(observable, lagged, half, live, tilts, area, sums[index])
                alive[index] += live.sum()
                _record_jumps(history, index, local_jumps)

        if jump_hook is not None:
            jumps["absorbed"] = absorbed
        template = next((s for s in sums if s is not None), None)
        if template is None:
            template = np.asarray(observable(components[0][1], base_tilt)) * 0.0
        observables = np.stack(
            [(s if s is not None else template * 0.0) / n_electrons for s in sums]
        )
        return AxialMCWFResult(
            depths_ang=steps * step_ang,
            observables=observables,
            surviving=alive / n_electrons,
            sink_depths_ang=np.asarray(sinks),
            jumps=jumps,
            n_electrons=int(n_electrons),
            jump_history=history,
        )


def _depth_steps(depths, n_electrons, step):
    if n_electrons < 1 or not np.isfinite(step) or step <= 0:
        raise ValueError("n_electrons must be >= 1 and step_ang positive")
    depths = np.asarray(depths, dtype=float)
    if depths.ndim != 1 or not depths.size or not np.all(np.isfinite(depths)):
        raise ValueError("depths must be nonempty finite one-dimensional samples")
    steps = np.rint(depths / step).astype(int)
    if np.any(np.diff(steps) < 0) or np.any(steps < 0):
        raise ValueError("depths must be ascending and non-negative")
    return steps


def _sample_components(grid, entry):
    components = entry.components if isinstance(entry, MixedState) else [(1.0, entry)]
    return [(float(weight), grid.sample(state)) for weight, state in components]


def _record_jumps(history, index, jumps):
    for kind, counts in history.items():
        counts[index] += jumps[kind]


def _enter(hook, start, phi, tilt, area):
    """Apply ``hook`` to each entry state in place; return the live mask."""
    live = np.ones(len(phi), dtype=bool)
    for b in range(len(phi)):
        entered = _apply_hook(hook, start + b, phi[b], tilt, area)
        live[b] = entered is not None
        phi[b] = 0.0 if entered is None else entered
    return live


def _observe(observable, lagged, half, live, tilts, area, total):
    """Add ``observable`` over the live, normalized physical states to ``total``.

    ``total`` is ``None`` until a live state is observed.
    """
    states = _ifft2(lagged[live] * half[live])
    for b, state in zip(np.flatnonzero(live), states, strict=True):
        state = state / np.sqrt(np.sum(np.abs(state) ** 2) * area)
        value = np.asarray(observable(state, tilts[b]))
        total = value if total is None else total + value
    return total


def _apply_hook(hook, electron, state, tilt, area):
    """Return the normalized state a :data:`JumpHook` keeps, or ``None``.

    Without a hook, or for a removed (``None``) state, ``state`` is returned
    untouched (no renormalization).
    """
    if hook is None or state is None:
        return state
    out = hook(electron, state, tilt)
    if out is None:
        return None
    out = np.asarray(out, dtype=complex)
    norm = np.sum(np.abs(out) ** 2) * area
    if out.shape != state.shape or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("jump_hook must return a finite nonzero state of the grid shape")
    return out / np.sqrt(norm)


def _fft2(values: np.ndarray) -> np.ndarray:
    return scipy.fft.fft2(values, axes=(-2, -1), workers=-1)


def _ifft2(values: np.ndarray) -> np.ndarray:
    return scipy.fft.ifft2(values, axes=(-2, -1), workers=-1)


def _choose(rng: np.random.Generator, weights) -> int:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if not total > 0.0:
        raise ValueError("no positive weight to sample")
    return min(
        int(np.searchsorted(np.cumsum(weights), rng.random() * total, "right")), weights.size - 1
    )


__all__ = ["AxialMCWF", "AxialMCWFResult", "density_observable"]
