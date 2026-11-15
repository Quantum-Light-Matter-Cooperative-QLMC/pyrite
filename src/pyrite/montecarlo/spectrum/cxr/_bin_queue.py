"""Per-target bin-mean line queues for the per-hkl scorer (#1192).

The per-hkl route visits one (reflection, orientation) row at a time. When
the device Fourier reduction applies, each row's lines are queued per output
target and reduced together, so the cost scales with lines rather than rows.
"""

from __future__ import annotations

from typing import Any

from ._bin_quadrature import resolve_bin_reduction
from ._cpu_reduce import numpy_backend


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


def open_bin_line_queues(st) -> None:
    """Attach queues to ``st`` when its bin-mean lines use the device Fourier reduction."""
    edges = st.bin_edges
    if edges is not None and not numpy_backend() and resolve_bin_reduction(edges)[0] == "fourier":
        st.bin_line_queues = BinLineQueues(st.E_grid, edges, st.bin_inv_width)


def flush_bin_line_queues(st) -> None:
    """Reduce the lines ``st`` queued during this call, if any, and detach the queues."""
    if st.bin_line_queues is not None:
        st.bin_line_queues.flush()
    st.bin_line_queues = None
