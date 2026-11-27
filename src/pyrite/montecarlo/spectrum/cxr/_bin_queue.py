"""Per-target bin-mean line queues for the per-hkl scorer (#1192).

The per-hkl route visits one (reflection, orientation) row at a time. When
a Fourier reduction applies (the device route, or on the NumPy backend the
host route of ``_fourier_host``), each row's lines are queued per output
target and reduced together, so the cost scales with lines rather than rows.
On the NumPy backend a non-uniform axis queues into
:class:`HostSegmentedLineQueues` (#1390): uniform runs take the shared-mode
Fourier route and the other bins the numba near/far evaluator, never the
dense ``(line, bin)`` matrix.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

from ._bin_quadrature import (
    host_near_far_applies,
    resolve_bin_reduction,
    resolve_host_bin_reduction,
)
from ._cpu_reduce import numpy_backend

#: Lines queued per target before a host near/far flush (three FP64 arrays each).
HOST_NEAR_FAR_FLUSH_LINES = 4_000_000


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
        for tgt, lines in self._queues.values():
            if not lines:
                continue
            E_r, a_width, w = (np.concatenate(part) for part in zip(*lines, strict=True))
            lines.clear()
            tgt += self._reduce(E_r, a_width, w).astype(tgt.dtype)
        self._queued = 0

    def _reduce(self, E_r, a_width, w):
        from ._fourier_host import host_fourier_bin_mean_reduce

        return host_fourier_bin_mean_reduce(E_r, a_width, w, self._axis)


class HostSegmentedLineQueues(HostBinLineQueues):
    """Host line queues for non-uniform bin axes (#1390).

    Each uniform run of at least ``HOST_SEGMENT_MIN_BINS`` bins takes the host
    Fourier route on its sub-axis; the remaining bins take the numba near/far
    evaluator (no ``(line, bin)`` matrix, the far-field envelope beyond
    ``BIN_MEAN_EXACT_WIDTHS``). Sums are independent of thread count.

    Validation: cxr-line-fourier-reduction, sinc-bin-far-envelope
    """

    def __init__(self, edges, inv_width, flush_lines: int) -> None:
        from ._fourier_host import uniform_runs

        super().__init__(None, flush_lines)
        self._edges = np.asarray(edges, dtype=np.float64)
        self._inv_width = np.asarray(inv_width, dtype=np.float64)
        self._runs = uniform_runs(self._edges)

    def _reduce(self, E_r, a_width, w):
        from ._fourier_host import host_segmented_bin_mean_reduce

        return host_segmented_bin_mean_reduce(
            E_r, a_width, w, self._edges, self._inv_width, self._runs
        )


class SharedLineSink:
    """Host bin-mean lines of many incoherent calls, reduced together (#1509).

    A full-fiber case scores every roll node and reflection batch as its own
    spectrum call. Reducing each call's lines separately pays the host Fourier
    route's per-flush cost (the mode grid the narrowest queued line sets) once
    per call. Inside :func:`shared_bin_lines`, every NumPy-backend call whose
    bin-mean lines would take a host queue instead queues them here, with its
    line weights scaled by :attr:`weight` ``/ Ne`` (the call's quadrature
    weight and its per-electron normalization), so the reduced sum equals the
    sum of the calls' own reductions up to the Fourier route's declared image
    residual. Calls that do not queue on the host (device, dense override,
    components) are unaffected and return their full spectrum.

    Validation: cxr-line-fourier-reduction
    """

    def __init__(self) -> None:
        self.weight = 1.0
        self._key: Any = None
        self._queues: HostBinLineQueues | None = None
        self.target: np.ndarray | None = None

    def bind(self, st, queues: HostBinLineQueues):
        """``st``'s queue view writing into the shared target."""
        key = (type(queues), np.asarray(st.bin_edges).tobytes())
        if self._queues is None:
            self._key, self._queues = key, queues
            self.target = np.zeros(np.asarray(st.bin_edges).size - 1, dtype=np.float64)
        elif key != self._key:
            raise ValueError("shared bin-mean lines need one energy axis")
        return _SinkView(self._queues, self.target, self.weight / float(st.Ne))

    def result(self, shape) -> np.ndarray:
        """Reduce every queued line; zeros if no call queued any."""
        if self._queues is None:
            return np.zeros(shape, dtype=np.float64)
        self._queues.flush()
        assert self.target is not None
        return self.target.reshape(shape)


class _SinkView:
    """One call's handle on a :class:`SharedLineSink`; the sink owns the flush."""

    def __init__(self, queues, target, scale) -> None:
        self._queues, self._target, self._scale = queues, target, scale

    def add(self, targets, E_r, a_width, weights_mask) -> None:
        if len(targets) != 1:
            raise ValueError("shared bin-mean lines score one target per call")
        weights = targets[0][0][weights_mask] * self._scale
        self._queues.add(
            [(weights, self._target)], E_r[weights_mask], a_width[weights_mask], slice(None)
        )

    def flush(self) -> None:
        """Deferred to :meth:`SharedLineSink.result`."""


_SHARED_SINK: ContextVar[SharedLineSink | None] = ContextVar(
    "pyrite_shared_bin_lines", default=None
)


@contextmanager
def shared_bin_lines() -> Generator[SharedLineSink]:
    """Route host bin-mean lines of the enclosed spectrum calls to one sink (#1509)."""
    sink = SharedLineSink()
    token = _SHARED_SINK.set(sink)
    try:
        yield sink
    finally:
        _SHARED_SINK.reset(token)


def _host_queues(st, queues):
    """``queues``, or a shared-sink view of them for a single-target incoherent call."""
    sink = _SHARED_SINK.get()
    if sink is None or st.request.components or st.request.coherent:
        return queues
    return sink.bind(st, queues)


def open_bin_line_queues(st) -> None:
    """Attach queues to ``st`` when its bin-mean lines use a Fourier reduction.

    The device route queues into ``_CxrBatch`` flushes; the NumPy backend
    queues into :class:`HostBinLineQueues` on uniform axes and into
    :class:`HostSegmentedLineQueues` otherwise (#1390), unless a
    :func:`bin_reduction_override` selects the dense evaluator. Inside
    :func:`shared_bin_lines`, host queues of single-target incoherent calls are
    views of the shared sink. Other host
    backends keep the dense evaluator.
    """
    edges = st.bin_edges
    if edges is None:
        return
    if numpy_backend():
        host = resolve_host_bin_reduction(edges)
        if host is not None:
            st.bin_line_queues = _host_queues(st, HostBinLineQueues(*host))
        elif host_near_far_applies():
            st.bin_line_queues = _host_queues(
                st, HostSegmentedLineQueues(edges, st.bin_inv_width, HOST_NEAR_FAR_FLUSH_LINES)
            )
        return
    if resolve_bin_reduction(edges)[0] in ("fourier", "segmented"):
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
