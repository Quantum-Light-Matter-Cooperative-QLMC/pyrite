"""CUDA execution of energy-resolved axial MCWF trajectories (issue #627).

Electrons are admitted in global-ID chunks, as on CPU. Inside a chunk they are
grouped by their fixed entrance energy, so every device batch shares one
energy-specific Hamiltonian and dissipator; batches are sized by the existing
axial CUDA admission rule after reserving the other resident energy solvers.
Each trajectory draws from its own Philox4x32-10 stream keyed by its *global*
electron ID, so batch membership, grouping and admission never move a draw.
Energy and entrance draws stay on the host NumPy streams and therefore match
the CPU reference exactly; trajectory draws use a different generator, so
CPU/CUDA agreement is statistical (energy-resolved MCWF design note, CUDA
envelope section).

Observables are linear in the normalized state's density matrix and are summed
on device per energy group; radiation is integrated from the group's summed
level history, which equals the per-electron sum because the trapezoid, line
polarization and dipole yield (proportional to line strength) are linear.
Propagation and jumps are the ledgered kernels of :mod:`._axial_cuda`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._backend import BackendResourceError, select_cupy_backend
from ._axial_cuda import (
    _admit,
    _Batch,
    device_jumps,
    device_static_bytes,
    initial_states,
    potential_phase,
    record_batch,
)
from .axial_energy_mcwf import require_cuda_callback
from .axial_mcwf import AxialMCWF, _depth_steps, _sample_components


@dataclass(frozen=True)
class GroupSums:
    """Raw sums of one energy group inside one admission chunk.

    ``data`` is ``(depth, column)`` in the CPU packed layout (density, pixel
    coherence, custom, level history), summed over the group's electrons;
    ``surviving`` counts live electrons per depth.
    """

    energy_eV: float
    electron_ids: tuple[int, ...]
    data: np.ndarray
    surviving: np.ndarray
    depths_ang: np.ndarray
    jumps: dict[str, int]
    custom: Any
    level_observer: Any
    level_size: int


class PackedObservable:
    """Device reduction of the CPU packed per-electron observable layout."""

    cuda_compatible = True

    def __init__(self, cp, pairs, dx, dy, custom, level_observer, columns, depths):
        self.cp = cp
        self.custom = custom
        self.level_observer = level_observer
        self.columns = int(columns)
        self.pairs = None
        if pairs:
            index = np.asarray(pairs, dtype=np.int64)
            self.pairs = (cp.asarray(index[:, 0]), cp.asarray(index[:, 1]))
            self.dx, self.dy = cp.asarray(dx, dtype=float), cp.asarray(dy, dtype=float)
        self.recording_bytes = int(depths) * self.columns * 16
        self.scratch_bytes = 4 * self.columns * 16

    def sum_batch(self, states, tilts):
        cp = self.cp
        count = states.shape[0]
        parts = [cp.sum(cp.abs(states) ** 2, axis=0).ravel().astype(complex)]
        if self.pairs is not None:
            flat = states.reshape(count, -1)
            a, b = self.pairs
            carrier = cp.exp(1j * (tilts[:, 0:1] * self.dx + tilts[:, 1:2] * self.dy))
            parts.append(cp.sum(flat[:, a] * flat[:, b].conj() * carrier, axis=0))
        for observer in (self.custom, self.level_observer):
            if observer is None:
                continue
            if hasattr(observer, "sum_batch"):
                value = observer.sum_batch(states, tilts)
            else:
                value = sum(
                    observer(state, tilt) for state, tilt in zip(states, tilts, strict=True)
                )
            parts.append(cp.asarray(value, dtype=complex).ravel())
        return cp.concatenate(parts)


class _DeviceSolvers:
    """Energy-keyed resident device dissipators, evicted least-recently-used first."""

    def __init__(self, cp, max_cached: int, step_ang: float):
        self.cp = cp
        self.max_cached = max_cached
        self.step_ang = step_ang
        self._cached: OrderedDict[float, tuple[Any, Any, int]] = OrderedDict()

    def get(self, energy: float, solver) -> tuple[Any, Any, int]:
        """``(device jumps, phase, bytes reserved by the other resident energies)``."""
        cached = self._cached
        if energy in cached:
            cached.move_to_end(energy)
        else:
            while len(cached) >= self.max_cached:
                cached.popitem(last=False)
            cached[energy] = (
                device_jumps(solver),
                potential_phase(self.cp, solver, self.step_ang),
                device_static_bytes(solver),
            )
        device, phase, _ = cached[energy]
        reserved = sum(size for key, (_, _, size) in cached.items() if key != energy)
        return device, phase, reserved

    def evict_others(self, energy: float) -> None:
        """Release every resident energy except ``energy``."""
        for key in [key for key in self._cached if key != energy]:
            del self._cached[key]
        self.cp.get_default_memory_pool().free_all_blocks()


def chunk_ends(start: int, n_electrons: int, chunk_size: int, checkpoint_every: int) -> list[int]:
    """Admission chunk ends in global-ID order, split at every checkpoint boundary."""
    ends = []
    while start < n_electrons:
        end = min(start + chunk_size, n_electrons)
        if checkpoint_every:
            end = min(end, (start // checkpoint_every + 1) * checkpoint_every)
        ends.append(end)
        start = end
    return ends


def cuda_energy_groups(
    *,
    energies: np.ndarray,
    start_id: int,
    options: Any,
    solvers: Any,
    entry_for: Callable[[int, float], tuple[Any, Sequence[float]]],
    custom_for: Callable[[float, int], Any],
    pairs: Sequence[tuple[int, int]],
    dx: np.ndarray,
    dy: np.ndarray,
    npix: int,
    depths_ang: Sequence[float],
    step_ang: float,
    trajectory_seed: int,
) -> Iterator[tuple[int, list[GroupSums]]]:
    """Yield ``(chunk end ID, energy groups)`` for each admission chunk."""
    backend = select_cupy_backend()
    cp = backend.xp
    steps = _depth_steps(depths_ang, 1, step_ang)
    resident = _DeviceSolvers(cp, options.max_cached_solvers, step_ang)
    start = start_id
    for end in chunk_ends(
        start_id, options.n_electrons, options.chunk_size, options.checkpoint_every
    ):
        groups: OrderedDict[float, list[tuple[int, Any, Sequence[float]]]] = OrderedDict()
        for eid in range(start, end):
            energy = float(energies[eid])
            entry, tilt_rad = entry_for(eid, energy)
            groups.setdefault(energy, []).append((eid, entry, tilt_rad))
        yield (
            end,
            [
                _run_group(
                    cp,
                    backend,
                    energy,
                    members,
                    resident,
                    solvers,
                    custom_for,
                    pairs,
                    dx,
                    dy,
                    npix,
                    steps,
                    step_ang,
                    trajectory_seed,
                    options.memory_budget_bytes,
                )
                for energy, members in groups.items()
            ],
        )
        start = end


def _run_group(
    cp,
    backend,
    energy,
    members,
    resident,
    solvers,
    custom_for,
    pairs,
    dx,
    dy,
    npix,
    steps,
    step_ang,
    trajectory_seed,
    budget,
):
    from ._axial_cuda_rng import ElectronRNG

    solver, level_observer, level_size = solvers.get(energy)
    if not isinstance(solver, AxialMCWF):
        raise TypeError("CUDA energy-resolved MCWF requires the built-in AxialMCWF solver")
    require_cuda_callback(level_observer, "radiation level observer")
    custom = custom_for(energy, level_size)
    require_cuda_callback(custom, "observable_factory output")
    custom_size = 0
    if custom is not None:
        probe = cp.ones(solver.grid.shape, complex) / np.sqrt(solver.grid.area_ang2)
        custom_size = int(cp.asarray(custom(probe, cp.zeros(2))).size)
    columns = npix + len(pairs) + custom_size + level_size
    observable = PackedObservable(cp, pairs, dx, dy, custom, level_observer, columns, steps.size)
    device, phase, reserved = resident.get(energy, solver)
    sampled = {}
    for _, entry, _ in members:
        if id(entry) not in sampled:
            components = _sample_components(solver.grid, entry)
            sampled[id(entry)] = (
                cp.asarray(np.stack([s for _, s in components])),
                cp.asarray([float(w) for w, _ in components]),
            )
    components = max(s.shape[0] for s, _ in sampled.values())
    try:
        size_limit = _admit(
            solver,
            backend,
            len(members),
            budget,
            steps.size,
            observable,
            components,
            reserved_bytes=reserved,
        )
    except BackendResourceError:
        if not reserved:
            raise
        # Other cached energies crowd out this one: release them and retry.
        resident.evict_others(energy)
        size_limit = _admit(
            solver, backend, len(members), budget, steps.size, observable, components
        )
    sums: list[Any] = [None] * steps.size
    alive = np.zeros(steps.size)
    jumps = {"tds": 0, "sink": 0, "electronic": 0}
    history = {kind: np.zeros(steps.size, dtype=np.int64) for kind in jumps}
    for first in range(0, len(members), size_limit):
        batch_members = members[first : first + size_limit]
        ids = [eid for eid, _, _ in batch_members]
        rngs = [ElectronRNG(trajectory_seed, eid) for eid in ids]
        if len(sampled) == 1:
            phi = initial_states(*next(iter(sampled.values())), rngs)
        else:
            phi = cp.concatenate(
                [
                    initial_states(*sampled[id(entry)], [rng])
                    for (_, entry, _), rng in zip(batch_members, rngs, strict=True)
                ]
            )
        tilts = [
            (solver.momentum * float(t[0]), solver.momentum * float(t[1]))
            for _, _, t in batch_members
        ]
        batch = _Batch(device, phi, rngs, ids, step_ang, tilts, None)
        record_batch(cp, batch, steps, step_ang, phase, observable, sums, alive, history, jumps, [])
    zero = cp.zeros(columns, dtype=complex)
    data = cp.asnumpy(cp.stack([zero if s is None else s for s in sums]))
    return GroupSums(
        energy,
        tuple(eid for eid, _, _ in members),
        data,
        alive,
        steps * step_ang,
        jumps,
        custom,
        level_observer,
        level_size,
    )


__all__ = ["GroupSums", "PackedObservable", "chunk_ends", "cuda_energy_groups"]
