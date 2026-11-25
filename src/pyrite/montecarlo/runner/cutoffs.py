"""Per-history termination energies of a transport population (#1452).

A case's cutoffs are piecewise constant in the electron index: at most two
runs (shared histories, then the line-only or continuum-only tail). The dense
``(max(Ne, Ne_brem),)`` array is only needed where a kernel reads it, and an
adaptive case sizes ``Ne`` by ``max_histories`` (~1e8), so
:class:`PiecewiseCutoffs` keeps the runs and materializes just the requested
block slice. Every value a slice returns equals the dense array's.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PiecewiseCutoffs", "case_cutoffs", "cutoff_floor", "electron_cutoffs"]


class PiecewiseCutoffs:
    """Constant-run cutoffs ``[keV]`` by electron index, sliceable without a dense array.

    ``runs`` is ``((stop, value), ...)`` with strictly increasing ``stop``;
    run ``i`` covers ``[stop_{i-1}, stop_i)``. Only the operations the transport
    callers use are provided: ``size``, slicing, prefix minimum, truncation,
    a tail rewrite, and ``np.asarray`` (dense, for trajectory capture).
    """

    __slots__ = ("_runs",)

    def __init__(self, runs):
        merged: list[tuple[int, float]] = []
        start = 0
        for stop, value in runs:
            stop, value = int(stop), float(value)
            if stop <= start:
                continue
            if merged and merged[-1][1] == value:
                merged[-1] = (stop, value)
            else:
                merged.append((stop, value))
            start = stop
        self._runs = tuple(merged)

    @property
    def size(self) -> int:
        return self._runs[-1][0] if self._runs else 0

    def __len__(self) -> int:
        return self.size

    @property
    def runs(self) -> tuple[tuple[int, float], ...]:
        return self._runs

    def __getitem__(self, index) -> np.ndarray:
        if not isinstance(index, slice) or index.step not in (None, 1):
            raise TypeError("PiecewiseCutoffs supports contiguous slices only")
        start, stop, _ = index.indices(self.size)
        out = np.empty(max(stop - start, 0), dtype=np.float64)
        lo = 0
        for run_stop, value in self._runs:
            a, b = max(lo, start), min(run_stop, stop)
            if a < b:
                out[a - start : b - start] = value
            lo = run_stop
        return out

    def __array__(self, dtype=None, copy=None):
        dense = self[:]
        return dense if dtype is None else dense.astype(dtype, copy=False)

    def prefix_min(self, n: int) -> float:
        """``min`` over electrons ``[0, n)``; ``n`` must be positive."""
        n = min(int(n), self.size)
        if n <= 0:
            raise ValueError("prefix_min needs at least one electron")
        lo, best = 0, np.inf
        for run_stop, value in self._runs:
            if lo >= n:
                break
            best = min(best, value)
            lo = run_stop
        return float(best)


def case_cutoffs(case) -> PiecewiseCutoffs:
    """Termination energies (keV) of ``case``'s transport population, as runs.

    Line-only histories stop at ``E_cut_lines_keV`` (default 5), continuum-only
    ones at ``E_cut_brem_keV`` (default 1), and shared histories at the lower.
    """
    Ne, Ne_brem = int(case["Ne"]), int(case["Ne_brem"])
    E_cut_lines = case.get("E_cut_lines_keV", 5.0)
    E_cut_brem = case.get("E_cut_brem_keV", 1.0)
    tail = E_cut_lines if Ne > Ne_brem else E_cut_brem
    shared = min(E_cut_lines, E_cut_brem)
    return PiecewiseCutoffs(((min(Ne, Ne_brem), shared), (max(Ne, Ne_brem), tail)))


def electron_cutoffs(case, *, piecewise=False) -> np.ndarray | PiecewiseCutoffs:
    """Per-history termination energies (keV) of ``case``'s transport population.

    Line-only histories stop at ``E_cut_lines_keV`` (default 5), continuum-only
    ones at ``E_cut_brem_keV`` (default 1), and shared histories at the lower.
    ``piecewise`` returns the equal :class:`.cutoffs.PiecewiseCutoffs` runs
    instead of the dense array (adaptive cases; #1452).
    """
    if piecewise:
        return case_cutoffs(case)
    Ne, Ne_brem = case["Ne"], case["Ne_brem"]
    electron_ids = np.arange(max(Ne, Ne_brem))
    cxr_mask = electron_ids < Ne
    brem_mask = electron_ids < Ne_brem
    E_cut_lines = case.get("E_cut_lines_keV", 5.0)
    E_cut_brem = case.get("E_cut_brem_keV", 1.0)
    cutoffs = np.empty(electron_ids.size, dtype=np.float64)
    cutoffs[cxr_mask & ~brem_mask] = E_cut_lines
    cutoffs[brem_mask & ~cxr_mask] = E_cut_brem
    cutoffs[cxr_mask & brem_mask] = min(E_cut_lines, E_cut_brem)
    return cutoffs


def cutoff_floor(cutoffs, n: int) -> float:
    """Smallest cutoff of electrons ``[0, n)`` of a dense or piecewise ``cutoffs``."""
    if isinstance(cutoffs, PiecewiseCutoffs):
        return cutoffs.prefix_min(n)
    return float(np.min(cutoffs[:n]))
