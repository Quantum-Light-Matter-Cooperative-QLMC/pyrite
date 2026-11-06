"""Exact-mass roulette over primary histories for incoherent line scoring (#879).

Each primary's line mass ``m_i = sum w pi / a_w`` comes from a pass with no grid
reduction. That pass also fills the truncation audit, so line-yield statistics
and guards stay analog. The pilot's masses fix one threshold, then every later
primary is scored surely or with probability ``floor`` and weight ``1/floor``,
from a uniform keyed only on the case seed and primary ID. A primary with no
surviving line has zero contribution everywhere and is skipped. The strata reuse
the production scorers unchanged, so CPU, CUDA and every route share one estimator.

Validation: line-scoring-roulette
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..._backend import _to_cpu
from ..._line_roulette import LineRoulette
from ..spectrum.cxr._kernels import _SEG_ARRAYS
from .electron_blocks import _electron_ids, _take

_DRAW_SALT = np.uint64(0x4C494E45524F554C)  # "LINEROUL": apart from transport streams
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
#: Audit sums a screen pass adds to the caller's truncation audit.
_AUDIT_SUMS = ("line_mass", "mass_above", "mass_below", "electron_mass")


def roulette_uniforms(seed: int, primaries) -> np.ndarray:
    """Uniforms in ``[0, 1)`` keyed on (seed, primary), for any block or order.

    Validation: line-scoring-roulette
    """
    ids = np.asarray(primaries, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        state = np.uint64(seed & 0xFFFFFFFFFFFFFFFF) ^ _DRAW_SALT
        x = state + _GOLDEN * (ids + np.uint64(1))
        x = (x ^ (x >> np.uint64(30))) * _MIX1
        x = (x ^ (x >> np.uint64(27))) * _MIX2
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * 2.0**-53


def _rows_of(segments: Mapping[str, Any], rows: np.ndarray) -> dict[str, Any]:
    """Whole-primary rows; original IDs and scalar normalization stay unchanged."""
    view = dict(segments)
    for field in _SEG_ARRAYS:
        if field in view:
            view[field] = _take(view[field], rows)
    return view


class LineScreen:
    """One line population's masses, threshold and draws, across all its blocks.

    Blocks must hold whole primaries in ascending ID order, as transport and
    replay produce them. Rescoring a block after an allocation retry overwrites
    that block's per-primary state with the same values.

    Validation: line-scoring-roulette
    """

    def __init__(self, case: Mapping[str, Any], n_electrons: int):
        self.policy = LineRoulette.from_dict(case["line_roulette"])
        self.seed = int(case["seed"])
        self.n = int(n_electrons)
        self.mass = np.zeros(self.n)
        self.probability = np.ones(self.n)
        self.kept = np.zeros(self.n, dtype=bool)
        self.rows = np.zeros(self.n, dtype=np.int64)
        self.threshold: float | None = None
        self._grid = None
        self._tables: dict = {}

    def _screen(self, batch, audit_band) -> dict[str, Any]:
        from .. import runner

        if self._grid is None:
            # The batched table cache keys on this array's identity.
            self._grid = np.array(audit_band, dtype=float)
        audit = {
            "start_eV": audit_band[0],
            "stop_eV": audit_band[1],
            "n_electrons": self.n,
            "mass_only": True,
        }
        with np.errstate(over="ignore", divide="ignore"):
            runner._cxr_for_segments(
                batch.segments,
                self._grid,
                batch.case,
                batch.n_hat,
                batch.abs_layers,
                batch.groove,
                coherent=False,
                Ne=self.n,
                table_cache=self._tables,
                truncation_audit=audit,
            )
        return audit

    def _resolve_threshold(self) -> None:
        pilot = self.mass[: self.policy.pilot_electrons]
        self.threshold = float(np.quantile(pilot, 1.0 - self.policy.keep_fraction))

    def _select(self, primaries: np.ndarray) -> None:
        """Set each primary's probability and draw, given its exact mass."""
        policy = self.policy
        later = primaries[primaries >= policy.pilot_electrons]
        if later.size and self.threshold is None:
            self._resolve_threshold()
        self.probability[primaries] = 1.0
        self.kept[primaries] = self.mass[primaries] > 0.0
        if not later.size:
            return
        low = later[(self.mass[later] > 0.0) & (self.mass[later] < self.threshold)]
        self.probability[low] = policy.floor
        self.kept[low] = roulette_uniforms(self.seed, low) < policy.floor

    def score(self, batch) -> np.ndarray:
        """The reweighted line spectrum of one block of whole primaries."""
        from .. import runner

        grid = np.asarray(_to_cpu(batch.energy_eV), dtype=float)
        caller = batch.truncation_audit
        band = (
            (caller["start_eV"], caller["stop_eV"])
            if caller is not None
            else (float(grid[0]), float(grid[-1]))
        )
        audit = self._screen(batch, band)
        ids = np.asarray(_electron_ids(batch.segments))
        in_population = ids < self.n
        primaries, counts = np.unique(ids[in_population], return_counts=True)
        masses = audit.get("electron_mass")
        if masses is not None:
            self.mass[primaries] = np.asarray(_to_cpu(masses), dtype=float)[primaries]
        else:
            self.mass[primaries] = 0.0
        self.rows[primaries] = counts
        self._select(primaries)
        if caller is not None:
            for key in _AUDIT_SUMS:
                if key in audit and (key != "electron_mass" or "n_electrons" in caller):
                    caller[key] = caller.get(key, 0.0) + audit[key]
        spectrum = None
        for weight, members in (
            (1.0, self.kept & (self.probability == 1.0)),
            (1.0 / self.policy.floor, self.kept & (self.probability < 1.0)),
        ):
            rows = np.flatnonzero(in_population & members[np.minimum(ids, self.n - 1)])
            if rows.size == 0:
                continue
            part = runner._cxr_for_segments(
                _rows_of(batch.segments, rows),
                batch.energy_eV,
                batch.case,
                batch.n_hat,
                batch.abs_layers,
                batch.groove,
                coherent=False,
                Ne=self.n,
                table_cache=batch.table_cache,
            )
            part = part if weight == 1.0 else part * weight
            spectrum = part if spectrum is None else spectrum + part
        return np.zeros(grid.shape) if spectrum is None else spectrum

    def record(self) -> dict[str, Any]:
        """Provenance and the estimator's exact, conditional yield variance."""
        culled = self.probability < 1.0
        estimate = float(self.mass[self.kept] @ (1.0 / self.probability[self.kept]))
        return {
            **self.policy.to_dict(),
            "threshold": self.threshold,
            "primaries": self.n,
            "primaries_scored": int(self.kept.sum()),
            "primaries_at_floor": int(culled.sum()),
            "primaries_zero_mass": int(np.count_nonzero(self.mass == 0.0)),
            "rows": int(self.rows.sum()),
            "rows_scored": int(self.rows[self.kept].sum()),
            "line_mass_exact": float(self.mass.sum()),
            "line_mass_estimate": estimate,
            "added_mass_variance": float(
                (1.0 / self.probability[culled] - 1.0) @ self.mass[culled] ** 2
            ),
        }


def line_screen_for(case: Mapping[str, Any], n_electrons: int) -> LineScreen | None:
    """The case's screen, else ``None``: analog scoring takes the unchanged path."""
    if case.get("line_roulette") is None:
        return None
    return LineScreen(case, n_electrons)


__all__ = ["LineScreen", "line_screen_for", "roulette_uniforms"]
