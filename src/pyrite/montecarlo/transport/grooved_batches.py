"""Batch adapter for the exact per-electron grooved CPU core (#972).

The generic per-electron batch driver (:func:`.batching._drive_per_electron_batches`)
owns segment capacity replay, compaction and exit counts. Grooved histories also
emit vacuum legs between relief facets. This adapter gives the grooved core the
generic driver's call signature. It allocates the per-batch vacuum slots,
replays the batch when one history overflows them, and keeps the compacted legs
of the last pass for each batch start.

Counter-addressed streams make a replay exact. Restoring the batch's mutable
entry state (position, direction, energy, clock) is enough, since the core resets
its per-history counters and straggling accumulator itself. Vacuum rows are
electron-major and leg-minor, a function of the electron index only, so they
are identical across batch and block partitions.
"""

from __future__ import annotations

import numpy as np

from .batching import _capacity_for

#: First vacuum slot budget per history; grows by replay when exceeded.
INITIAL_VACUUM_CAPACITY = 8

_EMPTY_VACUUM = (
    np.empty((0, 3), dtype=float),
    np.empty((0, 3), dtype=float),
    np.empty(0, dtype=float),
    np.empty(0, dtype=float),
    np.empty(0, dtype=np.int64),
)


def _alloc_vacuum(n):
    return (
        np.empty((n, 3), dtype=float),
        np.empty((n, 3), dtype=float),
        np.empty(n, dtype=float),
        np.empty(n, dtype=float),
        np.empty(n, dtype=np.int64),
    )


class GroovedVacuumCore:
    """Callable with the generic per-electron core signature, plus vacuum legs.

    ``groove_geometry`` is ``(spacing_ang, depth_ang, sin_tilt, cos_tilt)``.
    Call :meth:`vacuum` after the driver finishes to get the joined
    ``(start, end, E_keV, t_ang, electron_id)`` arrays.

    Validation: blazed-groove-geometry
    """

    def __init__(self, core, groove_geometry, config, capacity=None):
        self._core = core
        self._geometry = tuple(groove_geometry)
        self._config = config
        self._cap = max(1, int(INITIAL_VACUUM_CAPACITY if capacity is None else capacity))
        self._batches: dict[int, tuple] = {}

    def __call__(
        self, run, control, geometry, materials, mott, state, segments, pe_out, straggling
    ):
        e, m = int(run[0]), int(run[1])
        _alive, clock, pos, dirs, E_keV, _E_cut = state
        sl = slice(e, e + m)
        snap = (pos[sl].copy(), dirs[sl].copy(), E_keV[sl].copy(), clock[sl].copy())
        seg_count, exit_code = pe_out
        while True:
            cap = self._cap
            bufs = _alloc_vacuum(m * cap)
            vac_count = np.zeros(m, dtype=np.int64)
            self._core(
                run,
                control,
                geometry,
                (*self._geometry, cap, *bufs),
                materials,
                mott,
                state,
                segments,
                (seg_count, exit_code, vac_count),
                straggling,
            )
            needed = int(vac_count.max()) if m else 0
            if needed <= cap:
                break
            self._cap = _capacity_for(needed, self._config)
            pos[sl], dirs[sl], E_keV[sl], clock[sl] = snap
        keep = (np.arange(cap)[None, :] < vac_count[:, None]).ravel()
        # A segment-capacity replay of the same batch start overwrites this.
        self._batches[e] = tuple(buf[keep] for buf in bufs)

    def vacuum(self):
        """Joined vacuum legs of every accepted batch, in electron order."""
        if not self._batches:
            return _EMPTY_VACUUM
        parts = [self._batches[e] for e in sorted(self._batches)]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(len(_EMPTY_VACUUM)))
