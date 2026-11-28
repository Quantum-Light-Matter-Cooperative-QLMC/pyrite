"""Fixed-size exact stop-lattice evidence for measured fiber grids (#1527).

The old resolver retains triples (resonance, coefficient, first-zero width).
Here each production chunk contributes its mass and the same upper-tail sum
at every possible 100 eV stop (including the non-lattice ceiling). Only the
candidate vector and scalar extrema/counts survive the chunk. No line list
or lines-by-stops array is materialized. GPU reductions reuse the existing
fused stop-search product, with stop tiles and reduction rows.

Validation: line-grid-resonance-bandwidth
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._backend import REAL, _to_cpu
from ._device_sizing import _ROW_LINES, _kernels, device_resident
from ._population_tail import block_tail_sums

STOP_STEP_EV = 100.0
_STOP_TILE = 16


def stop_lattice(ceiling: float) -> np.ndarray:
    """The old search's positive lattice points plus the exact ceiling."""
    return np.append(np.arange(0.0, ceiling, STOP_STEP_EV), float(ceiling))


@dataclass(frozen=True)
class TailSummary:
    """Immutable-by-convention partial sums, safe for audit retry snapshots."""

    total: Any
    n_lines: Any
    min_energy: Any
    max_energy: Any
    sums: Any
    raw_lines: int


def merge_tail(left: TailSummary | None, right: TailSummary) -> TailSummary:
    """Combine two chunks without mutating either (including their arrays)."""
    if left is None:
        return right
    # Device scalar comparisons must not synchronize per radiation chunk.
    if device_resident(left.sums):
        import cupy as cp

        minimum, maximum = cp.minimum, cp.maximum
    else:
        minimum, maximum = np.minimum, np.maximum
    return TailSummary(
        left.total + right.total,
        left.n_lines + right.n_lines,
        minimum(left.min_energy, right.min_energy),
        maximum(left.max_energy, right.max_energy),
        left.sums + right.sums,
        left.raw_lines + right.raw_lines,
    )


def _device_stop_sums(cp, energy, weight, width, stops):
    """Reduce broadcast views in tiles; only row partials are allocated."""
    _, kernel = _kernels(cp)
    arrays = (energy, weight, width)
    split = energy.size // _ROW_LINES * _ROW_LINES
    result = []
    for offset in range(0, stops.size, _STOP_TILE):
        tile = cp.asarray(stops[offset : offset + _STOP_TILE], dtype=energy.dtype)
        sums = cp.zeros(tile.size, dtype=cp.float64)
        if split:
            rows = [a[:split].reshape(1, -1, _ROW_LINES) for a in arrays]
            sums = kernel(*rows, tile[:, None, None], axis=2).sum(axis=1)
        if split < energy.size:
            rest = [a[None, split:] for a in arrays]
            sums = sums + kernel(*rest, tile[:, None], axis=1)
        result.append(sums)
    return cp.concatenate(result)


def summarize_lines(energy, width, weight, stops, roll_weight=1.0) -> TailSummary:
    """The old collected triple's arithmetic, before adding its tail sums.

    Cast each field to REAL as collect_lines does. Apply the roll weight to
    the coefficient *before* its product with width, matching the old fiber
    collector's FP32 rounding and its finite-positive mass filter.
    """
    if device_resident(energy):
        import cupy as cp

        module = cp
        energy, width, weight = (cp.asarray(a, dtype=REAL) for a in (energy, width, weight))
    else:
        module = np
        energy, width, weight = (
            np.asarray(_to_cpu(a), dtype=REAL) for a in (energy, width, weight)
        )
    if energy.size == 0:
        return TailSummary(0.0, 0, np.inf, -np.inf, module.zeros(stops.size), 0)
    weight = weight * roll_weight
    mass = weight * width
    keep = module.isfinite(mass) & (mass > 0.0)
    total = module.where(keep, mass, 0.0).sum(dtype=module.float64)
    count = keep.sum(dtype=module.int64)
    low = module.where(keep, energy, module.inf).min()
    high = module.where(keep, energy, -module.inf).max()
    if module is np:
        blocks = [(energy, weight, width)]
        sums = np.array([block_tail_sums(blocks, float(stop))[0] for stop in stops])
    else:
        sums = _device_stop_sums(module, energy, weight, width, stops)
    return TailSummary(total, count, low, high, sums, int(energy.size))


class LatticeTail:
    """Host view of the bounded evidence, implementing _lattice_stop's contract."""

    def __init__(self, stops, summary: TailSummary | None):
        self.stops = stops
        self.total = float(_to_cpu(summary.total)) if summary is not None else 0.0
        self.n_lines = int(_to_cpu(summary.n_lines)) if summary is not None else 0
        self.min_energy = float(_to_cpu(summary.min_energy)) if summary is not None else np.inf
        self.max_energy = float(_to_cpu(summary.max_energy)) if summary is not None else -np.inf
        self.sums = (
            np.asarray(_to_cpu(summary.sums), dtype=float)
            if summary is not None
            else np.zeros(stops.size)
        )
        self.raw_lines = summary.raw_lines if summary is not None else 0

    def numerator(self, stop: float) -> float:
        """Exact candidate sum; the old search may also probe below all lines."""
        if stop <= self.min_energy:
            return self.total
        index = int(np.searchsorted(self.stops, stop))
        if index == self.stops.size or self.stops[index] != stop:
            raise ValueError(f"stop {stop:g} eV is outside the accumulated lattice")
        return float(self.sums[index])

    @property
    def population_bytes(self) -> int:
        """Conservative host admission for a replay requiring individual lines.

        Reserve space for two copies of the collected triples (roll folding,
        packing/refinement). Additional scratch is admitted by its own owner.
        """
        return self.raw_lines * 6 * np.dtype(REAL).itemsize
