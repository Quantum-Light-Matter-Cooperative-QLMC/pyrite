"""CPU research reference for compressed axial no-jump evolution (#623).

This does not define physical escape or enable standard axial runs. Between
jumps it evaluates exp[-i P(H-iW)P dz/(hbar v)] on the retained space, or the
full represented space when no cut is supplied. Production jump operators and
per-incident RNG streams are reused; jump times remain resolved to one step.

Source: Castin, Dalibard & Molmer, arXiv:0805.4002, Eqs. (2), (4), (8),
compressed onto an orthogonal spectral subspace of H. Units: eV and angstrom.
Validation: cr-axial-zone-handoff-accounting (unverified).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

import numpy as np
from scipy.linalg import expm

from .axial_mcwf import (
    AxialMCWF,
    AxialMCWFResult,
    JumpHook,
    Observable,
    _apply_hook,
    _choose,
    _depth_steps,
    _sample_components,
    density_observable,
)
from .mcwf import _rng
from .states import EntryState, MixedState

Index = tuple[np.ndarray, np.ndarray]
Spectrum = Sequence[tuple[Index, np.ndarray, np.ndarray]]
Coefficients = tuple[np.ndarray, ...]


class SpectralProjector(Protocol):
    """Cell-periodic H and positive absorptive W in matching Fourier blocks."""

    @property
    def n(self) -> tuple[int, int]: ...

    @property
    def loss_blocks(self) -> Sequence[np.ndarray]: ...

    def spectrum(self, tilt: tuple[float, float]) -> Spectrum: ...

    def gap_cut(
        self, tilt: tuple[float, float], target_eV: float, window_eV: float
    ) -> tuple[float, float]: ...


@dataclass(frozen=True)
class _Block:
    index: Index
    vectors: np.ndarray
    propagator: np.ndarray


class SpectralNoJump:
    """Matrix-exponential semigroup on full or retained Fourier blocks.

    Castin et al. Eq. (4) gives H_eff=H-iW. In a spectral subspace of H,
    H_PP is diagonal and W_PP=V_P^dagger W V_P; exponentiate their compressed
    combination, rather than projecting a full-space exponential. Constant W
    attenuates amplitude by exp[-W dz/(hbar v)]; W=0 is unitary on the subspace.
    A full/empty cut recovers full propagation/the zero subspace.

    The provider must include every Fourier block and supply a cell-periodic
    positive W [eV]. The cache holds at most ``cache_size`` tilt entries; it
    does not bound the provider's own spectrum cache or caller-held arrays.
    Energies are eV, lengths are angstroms, and hbar_v is eV angstrom.
    Validation: cr-axial-zone-handoff-accounting (unverified).
    """

    def __init__(
        self,
        projector: SpectralProjector,
        cell_area_ang2: float,
        hbar_v: float,
        step_ang: float,
        *,
        cut_eV: float | None = None,
        window_eV: float = 0.0,
        cache_size: int = 8,
    ) -> None:
        if any(not np.isfinite(v) or v <= 0 for v in (cell_area_ang2, hbar_v, step_ang)):
            raise ValueError("cell area, hbar_v and step_ang must be finite and positive")
        if (cut_eV is not None and not np.isfinite(cut_eV)) or (
            not np.isfinite(window_eV) or window_eV < 0
        ):
            raise ValueError("cut must be finite and matching window finite and nonnegative")
        if isinstance(cache_size, bool) or not isinstance(cache_size, Integral) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if not projector.loss_blocks:
            raise ValueError("spectral propagation requires absorptive loss blocks")
        self.projector = projector
        self.hbar_v, self.step_ang = hbar_v, step_ang
        self.cut_eV, self.window_eV = cut_eV, window_eV
        self.cache_size = cache_size
        self._factor = float(np.sqrt(cell_area_ang2 / np.prod(projector.n)))
        self._cache: OrderedDict[tuple[float, float], tuple[_Block, ...]] = OrderedDict()

    def _blocks(self, tilt: tuple[float, float]) -> tuple[_Block, ...]:
        """Compress the Eq. (4) generator before exponentiating [eV / eV Å].

        Full-space cuts recover the finite-grid no-jump semigroup.
        Validation: cr-axial-zone-handoff-accounting (unverified).
        """
        key = (float(tilt[0]), float(tilt[1]))
        if not np.all(np.isfinite(key)):
            raise ValueError("carrier tilt must be finite")
        if key not in self._cache:
            cut = (
                np.inf
                if self.cut_eV is None
                else self.projector.gap_cut(tilt, self.cut_eV, self.window_eV)[0]
            )
            blocks = []
            for (index, energies, vectors), loss in zip(
                self.projector.spectrum(tilt), self.projector.loss_blocks, strict=True
            ):
                keep = energies < cut
                v = vectors[:, keep]
                h_eff = np.diag(energies[keep]) - 1j * (v.conj().T @ loss @ v)
                u = expm(-1j * self.step_ang / self.hbar_v * h_eff) if keep.any() else h_eff
                blocks.append(_Block(index, v, u))
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = tuple(blocks)
        self._cache.move_to_end(key)
        return self._cache[key]

    def encode(self, phi: np.ndarray, tilt: tuple[float, float]) -> Coefficients:
        """Project physical phi [Å^-1] into orthonormal coefficients.

        Parseval gives ||c||²=integral |P phi|² dA; a full cut preserves norm.
        Validation: cr-axial-zone-handoff-accounting (unverified).
        """
        if phi.shape != self.projector.n or not np.all(np.isfinite(phi)):
            raise ValueError("state must be finite and match the projector grid")
        a = np.fft.fft2(phi) * self._factor
        return tuple(b.vectors.conj().T @ a[b.index] for b in self._blocks(tilt))

    def decode(self, coefficients: Coefficients, tilt: tuple[float, float]) -> np.ndarray:
        """Reconstruct physical phi [Å^-1]; zero coefficients give zero state.

        Validation: cr-axial-zone-handoff-accounting (unverified).
        """
        a = np.zeros(self.projector.n, dtype=complex)
        for b, c in zip(self._blocks(tilt), coefficients, strict=True):
            a[b.index] = b.vectors @ c
        return np.fft.ifft2(a) / self._factor

    def advance(self, coefficients: Coefficients, tilt: tuple[float, float]) -> Coefficients:
        """Advance one step without jumps or extra projections; W=0 is unitary.

        Source: compressed Castin et al. Eq. (4), eV/Å conventions above.
        Validation: cr-axial-zone-handoff-accounting (unverified).
        """
        return tuple(
            b.propagator @ c for b, c in zip(self._blocks(tilt), coefficients, strict=True)
        )

    @staticmethod
    def norm2(coefficients: Coefficients) -> float:
        """Dimensionless no-jump norm squared (Parseval); zero state gives zero.

        Validation: cr-axial-zone-handoff-accounting (unverified).
        """
        return sum(float(np.vdot(c, c).real) for c in coefficients)


def _prepare(stepper: SpectralNoJump, state: np.ndarray, tilt: tuple[float, float]) -> Coefficients:
    coefficients = stepper.encode(state, tilt)
    norm = stepper.norm2(coefficients)
    if not np.isclose(norm, 1.0, atol=1e-9, rtol=0):
        raise ValueError("entry/jump hook must supply a normalized state in the retained space")
    return tuple(c / np.sqrt(norm) for c in coefficients)


def _trajectory(
    solver, stepper, components, mix, steps, seed, electron, tilt, observable, hook
) -> AxialMCWFResult:
    """One incident trajectory using native draws and step-resolved jumps.

    Castin et al. Eqs. (4), (8): no-jump norm is the waiting-time survival law.
    W=0 triggers no jump; sink/hook removals contribute zero at later depths.
    Validation: cr-axial-zone-handoff-accounting (unverified).
    """
    rng = _rng(seed, electron)
    area = solver.grid.cell_area_ang2
    state = components[_choose(rng, mix)][1].copy()
    state /= np.sqrt(np.sum(np.abs(state) ** 2) * area)
    template = np.asarray(observable(state, tilt))
    values = np.zeros((len(steps), *template.shape), dtype=np.result_type(template, float))
    surviving = np.zeros(len(steps))
    jumps = dict.fromkeys(("tds", "sink", "electronic"), 0)
    history = {k: np.zeros(len(steps), dtype=np.int64) for k in jumps}
    if hook is not None:
        jumps["absorbed"] = 0
    state = _apply_hook(hook, electron, state, tilt, area)
    if state is None:
        jumps["absorbed"] += 1
    coefficients = None if state is None else _prepare(stepper, state, tilt)
    threshold, step, sinks = rng.random(), 0, []
    for index, target in enumerate(steps):
        while step < target and coefficients is not None:
            coefficients = stepper.advance(coefficients, tilt)
            step += 1
            norm = stepper.norm2(coefficients)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("no-jump state vanished numerically; reduce step_ang")
            if norm <= threshold:
                state = stepper.decode(coefficients, tilt) / np.sqrt(norm)
                kind, state, tilt = solver._jump(state, tilt, rng)
                jumps[kind] += 1
                if kind == "sink":
                    sinks.append(step * stepper.step_ang)
                state = _apply_hook(hook, electron, state, tilt, area)
                if state is None and kind != "sink":
                    jumps["absorbed"] += 1
                coefficients = None if state is None else _prepare(stepper, state, tilt)
                if coefficients is not None:
                    threshold = rng.random()
        if coefficients is not None:
            state = stepper.decode(coefficients, tilt) / np.sqrt(stepper.norm2(coefficients))
            values[index] = observable(state, tilt)
            surviving[index] = 1
        for k in history:
            history[k][index] = jumps[k]
    return AxialMCWFResult(
        steps * stepper.step_ang, values, surviving, np.asarray(sinks), jumps, 1, history
    )


def run_spectral_mcwf(
    solver: AxialMCWF,
    projector: SpectralProjector,
    entry: EntryState | MixedState,
    depths_ang: Sequence[float],
    n_electrons: int,
    *,
    seed: int = 0,
    step_ang: float = 5.0,
    cut_eV: float | None = None,
    window_eV: float = 0.0,
    tilt_rad: tuple[float, float] = (0.0, 0.0),
    observable: Observable = density_observable,
    jump_hook: JumpHook | None = None,
) -> AxialMCWFResult:
    """Research CPU MCWF with exact finite-grid no-jump matrix propagation.

    Castin et al. Eqs. (2), (4), (8) with generator compressed as in
    SpectralNoJump. Native jump draws and per-incident keys are reused; crossing
    a waiting-time threshold triggers at most one jump per step, as in the
    production CPU solver. Step convergence is still required for jump times.

    A finite cut requires entry and post-jump states already in that retained
    space; a hook may supply them and absorb outcomes outside it. Such entry
    conditioning must be incident-weighted by the caller. No outgoing state
    is interpreted as physical escape. Full cut plus W=0 gives unitary motion.
    Removed electrons contribute zero, never survivor-renormalized output.
    This streams one trajectory at a time; no individual histories are saved.
    Validation: cr-axial-zone-handoff-accounting (unverified).
    """
    if solver.backend != "cpu":
        raise ValueError("spectral MCWF research reference supports only CPU")
    steps = _depth_steps(depths_ang, n_electrons, step_ang)
    stepper = SpectralNoJump(
        projector,
        solver.grid.cell_area_ang2,
        solver.hbar_v,
        step_ang,
        cut_eV=cut_eV,
        window_eV=window_eV,
    )
    components = _sample_components(solver.grid, entry)
    mix = np.array([w for w, _ in components])
    tilt = (solver.momentum * tilt_rad[0], solver.momentum * tilt_rad[1])
    total = None
    sinks: list[float] = []
    for electron in range(n_electrons):
        result = _trajectory(
            solver, stepper, components, mix, steps, seed, electron, tilt, observable, jump_hook
        )
        sinks.extend(result.sink_depths_ang.tolist())
        if total is None:
            total = result
        else:
            total.observables[:] += result.observables
            total.surviving[:] += result.surviving
            for k in total.jumps:
                total.jumps[k] += result.jumps[k]
            assert total.jump_history is not None and result.jump_history is not None
            for k in total.jump_history:
                total.jump_history[k][:] += result.jump_history[k]
    assert total is not None
    return AxialMCWFResult(
        total.depths_ang,
        total.observables / n_electrons,
        total.surviving / n_electrons,
        np.asarray(sinks),
        total.jumps,
        n_electrons,
        total.jump_history,
    )
