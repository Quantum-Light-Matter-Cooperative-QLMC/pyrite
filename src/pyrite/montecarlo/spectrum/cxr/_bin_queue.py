"""Per-target bin-mean line queues for the per-hkl scorer (#1192).

The per-hkl route visits one (reflection, orientation) row at a time. When
the device Fourier reduction applies, each row's lines are queued per output
target and reduced together, so the cost scales with lines rather than rows.
"""

from __future__ import annotations

from typing import Any


class BinLineQueues:
    """Lazily created ``_CxrBatch`` queues keyed by output target."""

    def __init__(self, E_grid: Any, bin_edges: Any, bin_inv_width: Any) -> None:
        self._grid = (E_grid, bin_edges, bin_inv_width)
        self._queues: dict[int, Any] = {}

    def add(self, targets, E_r, a_width, weights_mask) -> None:
        """Queue ``(E_r, a_width, w)[mask]`` for every ``(w, target)`` pair."""
        from ._batched import _CxrBatch  # _batched imports the per-hkl module

        for w, tgt in targets:
            queue = self._queues.get(id(tgt))
            if queue is None:
                queue = self._queues[id(tgt)] = _CxrBatch(tgt, *self._grid)
            queue.append(E_r[weights_mask], a_width[weights_mask], w[weights_mask])

    def flush(self) -> None:
        """Reduce every queued line into its target."""
        for queue in self._queues.values():
            queue.flush()
        self._queues.clear()
