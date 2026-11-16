"""Per-target bin-mean line queues for the per-hkl scorer (#1192).

The per-hkl route visits one (reflection, orientation) row at a time. When
a Fourier reduction applies (the device route, or on the NumPy backend the
host route of ``_fourier_host``), each row's lines are queued per output
target and reduced together, so the cost scales with lines rather than rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._bin_quadrature import resolve_bin_reduction, resolve_host_bin_reduction
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


class HostBinLineQueues:
    """Per-target host line queues reduced by the host Fourier route (#1230).

    Each target collects ``(E_r, a_width, w)`` until ``flush_lines`` lines are
    queued in total, then every target is reduced on the uniform ``axis``.

    Validation: cxr-line-fourier-reduction
    """

    def __init__(self, axis, flush_lines: int) -> None:
        self._axis = axis
        self._flush_lines = int(flush_lines)
        self._queued = 0
        self._queues: dict[int, tuple[Any, list[tuple[Any, Any, Any]]]] = {}

    def add(self, targets, E_r, a_width, weights_mask) -> None:
        """Queue ``(E_r, a_width, w)[mask]`` for every ``(w, target)`` pair."""
        for w, tgt in targets:
            _, lines = self._queues.setdefault(id(tgt), (tgt, []))
            lines.append((E_r[weights_mask], a_width[weights_mask], w[weights_mask]))
            self._queued += int(lines[-1][0].size)
        if self._queued >= self._flush_lines:
            self.flush()

    def flush(self) -> None:
        """Reduce every queued line into its target.

        Validation: cxr-line-fourier-reduction
        """
        from ._fourier_host import host_fourier_bin_mean_reduce

        for tgt, lines in self._queues.values():
            if not lines:
                continue
            E_r, a_width, w = (np.concatenate(part) for part in zip(*lines, strict=True))
            lines.clear()
            tgt += host_fourier_bin_mean_reduce(E_r, a_width, w, self._axis).astype(tgt.dtype)
        self._queued = 0


def open_bin_line_queues(st) -> None:
    """Attach queues to ``st`` when its bin-mean lines use a Fourier reduction.

    The device route queues into ``_CxrBatch`` flushes; the NumPy backend
    queues into :class:`HostBinLineQueues`. Other host backends keep the dense
    evaluator.
    """
    edges = st.bin_edges
    if edges is None:
        return
    if numpy_backend():
        host = resolve_host_bin_reduction(edges)
        if host is not None:
            st.bin_line_queues = HostBinLineQueues(*host)
        return
    if resolve_bin_reduction(edges)[0] == "fourier":
        st.bin_line_queues = BinLineQueues(st.E_grid, edges, st.bin_inv_width)


def open_host_bin_line_queues(st) -> None:
    """Attach host Fourier queues for the batched route on the NumPy backend.

    The device batched route reduces through its own ``_CxrBatch``; bin-mean
    is incoherent-only, so a coherent request carries no ``bin_edges``.
    """
    if st.bin_edges is not None and numpy_backend():
        open_bin_line_queues(st)


def flush_bin_line_queues(st) -> None:
    """Reduce the lines ``st`` queued during this call, if any, and detach the queues."""
    if st.bin_line_queues is not None:
        st.bin_line_queues.flush()
    st.bin_line_queues = None
