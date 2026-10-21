"""Memory-admitted CUDA axial MCWF propagation and recorded reductions.

Wavefunctions and jump data remain on device; Python receives scalar control
and indices. Callbacks opt in with ``cuda_compatible = True`` and consume and
return CuPy arrays. No implicit host callback adapter is provided.
Validation: cr-axial-realspace-propagation
Validation: cr-axial-realspace-dissipator
"""

from __future__ import annotations

import numpy as np

from ..._backend import BackendResourceError, select_backend
from .._resources import admitted_chunk, resolve_resource_policy
from .axial_mcwf import (
    BATCH,
    AxialMCWFResult,
    _depth_steps,
    _record_jumps,
    _sample_components,
    density_observable,
)


def _callback_check(observable, hook):
    for name, callback in (("observable", observable), ("jump_hook", hook)):
        if callback is not None and callback is not density_observable:
            if not getattr(callback, "cuda_compatible", False):
                raise ValueError(
                    f"CUDA {name} must declare cuda_compatible=True and use CuPy arrays"
                )


def _hook(cp, hook, electron, state, tilt, area):
    if hook is None or state is None:
        return state
    out = hook(electron, state, tilt)
    if out is None:
        return None
    if not isinstance(out, cp.ndarray):
        raise TypeError("CUDA jump_hook must return a CuPy array or None")
    norm = cp.sum(cp.abs(out) ** 2) * area
    if out.shape != state.shape or not bool(cp.isfinite(norm) & (norm > 0)):
        raise ValueError("jump_hook must return a finite nonzero state of the grid shape")
    return out / cp.sqrt(norm)


def _observe(cp, observable, lagged, half, live, tilts, area, total):
    if not bool(live.any()):
        return total
    states = cp.fft.ifft2(lagged[live] * half[live], axes=(-2, -1))
    states /= cp.sqrt(cp.sum(cp.abs(states) ** 2, axis=(1, 2)) * area)[:, None, None]
    if observable is density_observable:
        value = cp.sum(cp.abs(states) ** 2, axis=0)
        return value if total is None else total + value
    for b, state in zip(cp.asnumpy(cp.flatnonzero(live)), states, strict=True):
        value = observable(state, tilts[b])
        if not isinstance(value, cp.ndarray):
            raise TypeError("CUDA observable must return a CuPy array")
        total = value.copy() if total is None else total + value
    return total


def _admit(solver, backend, batch_size, budget, depths, observable):
    cp = backend.xp
    policy = resolve_resource_policy(backend)
    available = int(cp.cuda.runtime.memGetInfo()[0])
    limit = min(available // 2, policy.device_budget_bytes or available // 2)
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("memory_budget_bytes must be a positive integer")
        limit = min(limit, budget)
    cells = int(np.prod(solver.grid.shape))
    d = solver.dissipator
    static = sum(v.nbytes for v in vars(d).values() if isinstance(v, np.ndarray))
    static += sum(v.nbytes for v in d.phase_spectra.values())
    # Fixed device spectra/scratch: fine-grid TDS coefficients, rejection GEMMs,
    # source spectra, potential and recorded density. FFT workspace receives a
    # factor of two over the live complex128 arrays. Admission fails before upload.
    fixed = static + cells * (16 * 48 + 8 * depths) + 131 * 131 * 8 * 4
    fixed += len(d.tds_sources) * (
        cells * 16 * 12 + 15 * 131**2 + 131 * sum(solver.grid.shape) * 32
    )
    fixed += int(getattr(observable, "recording_bytes", 0))
    fixed += int(getattr(observable, "scratch_bytes", 0))
    if limit <= fixed:
        raise BackendResourceError(
            "axial CUDA memory budget cannot admit immutable data and scratch"
        )
    return admitted_chunk(
        requested_chunk=batch_size,
        bins=cells,
        itemsize=16,
        budget_bytes=limit - fixed,
        intermediates=12,
        minimum_chunk=1,
    )


def run_cuda(
    solver,
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
    memory_budget_bytes,
):
    """Execute CUDA MCWF; returned result owns NumPy arrays only."""
    _callback_check(observable, jump_hook)
    steps = _depth_steps(depths_ang, n_electrons, step_ang)
    backend = select_backend("cuda")
    cp = backend.xp
    size_limit = _admit(
        solver, backend, batch_size or BATCH, memory_budget_bytes, steps.size, observable
    )
    from ._axial_cuda_dissipator import DeviceDissipator
    from ._axial_cuda_jumps import DeviceJumps

    device = DeviceJumps(solver, DeviceDissipator(solver.dissipator))
    components = _sample_components(solver.grid, entry)
    weights = cp.asarray([float(w) for w, _ in components])
    sampled = cp.asarray(np.stack([s for _, s in components]))
    phase = cp.exp(-1j * step_ang * cp.asarray(solver._complex_potential) / solver.hbar_v)
    base_tilt = (solver.momentum * tilt_rad[0], solver.momentum * tilt_rad[1])
    sums = [None] * len(steps)
    alive = np.zeros(len(steps))
    sinks = []
    jumps = {"tds": 0, "sink": 0, "electronic": 0}
    history = {kind: np.zeros(steps.size, dtype=np.int64) for kind in jumps}
    absorbed = 0
    for start in range(0, n_electrons, size_limit):
        batch = _Batch(
            device,
            sampled,
            weights,
            start,
            min(size_limit, n_electrons - start),
            seed,
            step_ang,
            base_tilt,
            jump_hook,
        )
        absorbed += batch.absorbed
        for index, target in enumerate(steps):
            batch.advance(target, step_ang, phase, jumps, sinks)
            _record_jumps(history, index, batch.jumps)
            sums[index] = _observe(
                cp,
                observable,
                batch.lagged,
                batch.half,
                batch.live,
                batch.tilts,
                solver.grid.cell_area_ang2,
                sums[index],
            )
            alive[index] += int(batch.live.sum())
        absorbed += batch.absorbed_after_entry
    if jump_hook is not None:
        jumps["absorbed"] = absorbed
    template = next((s for s in sums if s is not None), None)
    if template is None:
        template = (
            cp.abs(sampled[0]) ** 2
            if observable is density_observable
            else observable(sampled[0], base_tilt)
        ) * 0
    values = cp.stack([(s if s is not None else template * 0) / n_electrons for s in sums])
    return AxialMCWFResult(
        steps * step_ang,
        cp.asnumpy(values),
        alive / n_electrons,
        np.asarray(sinks),
        jumps,
        int(n_electrons),
        history,
    )


class _Batch:
    def __init__(self, device, sampled, weights, start, size, seed, step, tilt, hook):
        from ._axial_cuda_jumps import _choose, electron_rng

        self.device = device
        self.cp = cp = __import__("cupy")
        self.start, self.step, self.hook = start, 0, hook
        self.jumps = {"tds": 0, "sink": 0, "electronic": 0}
        self.rngs = [electron_rng(seed, e) for e in range(start, start + size)]
        phi = cp.stack([sampled[_choose(rng, weights)] for rng in self.rngs])
        self.area = device.grid.cell_area_ang2
        phi /= cp.sqrt(cp.sum(cp.abs(phi) ** 2, axis=(1, 2)) * self.area)[:, None, None]
        self.tilts = [tilt] * size
        self.live = cp.ones(size, dtype=bool)
        for b in range(size):
            kept = _hook(cp, hook, start + b, phi[b], tilt, self.area)
            self.live[b] = kept is not None
            phi[b] = 0 if kept is None else kept
        self.absorbed = size - int(self.live.sum())
        self.absorbed_after_entry = 0
        self.half = cp.broadcast_to(device._half_kinetic(tilt, step), phi.shape).copy()
        self.full = self.half**2
        self.thresholds = cp.stack([rng.random() for rng in self.rngs])
        self.lagged = cp.fft.fft2(phi, axes=(-2, -1)) * self.half.conj()

    def advance(self, target, step_ang, phase, jumps, sinks):
        cp = self.cp
        while self.step < target and bool(self.live.any()):
            self.lagged *= self.full
            self.lagged = cp.fft.fft2(
                phase * cp.fft.ifft2(self.lagged, axes=(-2, -1)), axes=(-2, -1)
            )
            self.step += 1
            norms = cp.sum(cp.abs(self.lagged) ** 2, axis=(1, 2)) * self.area / self.lagged[0].size
            for b in cp.asnumpy(cp.flatnonzero(self.live & (norms <= self.thresholds))):
                self.jump(int(b), norms[b], step_ang, jumps, sinks)

    def jump(self, b, norm, step_ang, jumps, sinks):
        cp = self.cp
        state = cp.fft.ifft2(self.lagged[b] * self.half[b]) / cp.sqrt(norm)
        kind, state, self.tilts[b] = self.device._jump(state, self.tilts[b], self.rngs[b])
        jumps[kind] += 1
        self.jumps[kind] += 1
        if kind == "sink":
            sinks.append(self.step * step_ang)
        elif kind == "electronic":
            self.half[b] = self.device._half_kinetic(self.tilts[b], step_ang)
            self.full[b] = self.half[b] ** 2
        kept = _hook(cp, self.hook, self.start + b, state, self.tilts[b], self.area)
        self.absorbed_after_entry += int(kept is None) - int(kind == "sink")
        if kept is None:
            self.live[b] = False
            self.lagged[b] = 0
        else:
            self.lagged[b] = cp.fft.fft2(kept) * self.half[b].conj()
            self.thresholds[b] = self.rngs[b].random()
