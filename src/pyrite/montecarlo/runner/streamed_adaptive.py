"""Streamed adaptive cases scored from the monitored pass (#1280, #1397).

An adaptive case streams through its monitored adaptive transport
(:func:`.block_transport.transport_monitored_blocks`). The stopping monitor
reads each chunk; a chunk containing a stop or brem freeze is replayed on its
prefix ``[start, cut)``; an energy-range replay restarts at electron 0. The
chunks the driver keeps are exactly the realized population, so a sink fed
those chunks sees what the fixed case at the realized ``(Ne, Ne_brem)``
transports. Three routes (:func:`adaptive_route`):

- ``single`` (explicit grid): the monitored pass scores its kept chunks. The
  realized counts are unknown until the stop, so blocks are scored at a
  provisional power-of-two count ``L >= max_histories``: every estimator's
  electron filter passes every kept row, and its ``sum / L`` is exact. At the
  stop the totals are renormalized by ``L / Ne`` and ``L / Ne_brem``
  (:meth:`.streamed_scoring.BlockSpectrumScorer.result`). A chunk past the
  brem freeze scores lines only. One transport.
- ``folded`` (``line_grid_policy``): the monitored pass feeds
  :class:`.streamed_line_grid.BlockGridEvidence`, re-blocked to the fixed
  route's integer blocks so the evidence, and the grid record, are the
  separate grid pass's bit for bit. At the stop a speed-cache hit discards
  it; a miss resolves the grid from it. The fixed case then streams on that
  grid. Two transports, cached or not.
- ``measure`` (cascades, line roulette, or :data:`MONITORED_PASS` off): the
  #1280 measure-only pass, then the fixed streamed route with its own grid
  pass: two or three transports.

Every sink commits a kept chunk once: a fresh sink state per pass
(``begin_pass``), and a scoring allocator OOM rolls the sink back and
rescores a host copy of the chunk (:func:`commit_once`).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

#: Score in, or collect grid evidence from, the monitored pass (#1397).
#: ``False`` keeps the #1280 measure-only route: a same-revision A/B arm.
MONITORED_PASS = True


def provisional_count(limit: int) -> int:
    """The smallest power of two at or above ``limit`` (and at least 1)."""
    return 1 << max(int(limit) - 1, 0).bit_length()


def adaptive_route(case: Mapping[str, Any]) -> str:
    """``"single"``, ``"folded"`` or ``"measure"`` for an adaptive ``case`` (module doc)."""
    if not MONITORED_PASS or case.get("secondary_threshold_eV") is not None:
        return "measure"  # cascades keep their #1240 route
    if case.get("line_roulette") is not None:
        return "measure"  # the roulette sizes per-primary state by the population
    return "single" if case.get("line_grid_policy") is None else "folded"


def commit_once(add, block, owner=None) -> None:
    """``add(block)`` once; a device allocator OOM rolls ``owner`` back and adds host rows.

    ``owner`` (optional) has ``snapshot()``/``restore(state)``. Grid evidence
    commits only after a whole block succeeds and needs none.
    """
    from ..._backend import is_device_array
    from .. import runner
    from .oom import _drop_traceback_locals
    from .pool import _admit_host_segments
    from .recovery import _host_block

    state = None if owner is None else owner.snapshot()
    try:
        add(block)
        return
    except Exception as error:
        if not runner._is_gpu_oom(error) or not is_device_array(block.get("L_ang")):
            raise
        _drop_traceback_locals(error)
        error.__traceback__ = None
    if owner is not None:
        owner.restore(state)
    runner.BACKEND.release_memory()
    _admit_host_segments()
    host = _host_block(block)
    del block
    runner.BACKEND.release_memory()
    add(host)


def _fixed_template(case, limit):
    """The requested case as a fixed case at ``limit`` electrons of both populations."""
    fixed = {key: value for key, value in case.items() if key != "adaptive_precision"}
    fixed.update(Ne=int(limit), Ne_brem=int(limit))
    return fixed


def _max_histories(case) -> int:
    from .adaptive import Precision

    return int(Precision.from_dict(case["adaptive_precision"]).max_histories)


class MonitoredScoringSink:
    """Score the monitored pass's kept chunks at a provisional count (``single``)."""

    def __init__(self, case, limit):
        from .artifacts import _case_geometry

        self.count = provisional_count(limit)
        self.case = _fixed_template(case, self.count)
        E_grid, E_brem, _, n_hat, groove = _case_geometry(self.case)
        self.inputs = dict(E_grid=E_grid, E_brem=E_brem, n_hat=n_hat, groove=groove)
        self.scorer: Any = None
        self.passes = 0

    def begin_pass(self, monitor) -> None:
        from .streamed_selection import block_scorer

        def brem_stop():
            return getattr(monitor, "brem_count", None)

        inputs = dict(
            self.inputs,
            Ne_lines=self.count,
            Ne_brem=self.count,
            provisional=True,
            brem_stop=brem_stop,
        )
        self.scorer = block_scorer(self.case, inputs)
        self.passes += 1

    def __call__(self, block) -> None:
        commit_once(self.scorer.add, block, self.scorer)

    def result(self, realized, counts, line_coverage) -> dict[str, Any]:
        from .streamed_selection import PrefixSelectingScorer

        population = (int(realized["Ne"]), int(realized["Ne_brem"]))
        kwargs = dict(
            Ne=int(counts["Ne"]),
            n_backscattered=int(counts["n_backscattered"]),
            n_missed=int(counts["n_missed"]),
            line_coverage=line_coverage,
            population=population,
        )
        if isinstance(self.scorer, PrefixSelectingScorer):
            kwargs["realized_case"] = realized
        return self.scorer.result(**kwargs)


class MonitoredGridEvidence:
    """Grid evidence from the monitored pass's kept chunks (``folded``).

    With an integer ``block_electrons`` the chunks are re-blocked to the fixed
    route's blocks ``[kB, (k+1)B)`` before :meth:`BlockGridEvidence.add`, so
    the evidence (line populations in block order, hence the
    ``resonance-population`` tail sums) is the separate grid pass's bit for
    bit. A block that spans chunks waits on the host for its remainder. Under
    a :class:`.stream_sizing.MemoryBlockSizer` the evidence follows the
    chunks; the sinc samples still concatenate exactly, and the tail sums may
    differ in the last ulp, as the grid pass already does from the joined one.
    """

    def __init__(self, case, block_electrons, spill, limit):
        from .artifacts import _case_geometry
        from .cxr_grid import _auditable_policy
        from .line_grid_spill import GridSpillPolicy

        self.case = _fixed_template(case, provisional_count(limit))
        _, _, _, self.n_hat, self.groove = _case_geometry(self.case)
        self.payload = _auditable_policy(self.case, self.case["line_grid_policy"])
        self.span = block_electrons if isinstance(block_electrons, int) else None
        self.spill = GridSpillPolicy() if spill is None else spill
        self.evidence: Any = None
        self._pending: list = []
        self._seen = 0
        self.passes = 0

    def begin_pass(self, monitor) -> None:
        from .streamed_line_grid import BlockGridEvidence

        del monitor  # evidence reads line rows only; lines run past a brem freeze
        self.close()
        self.evidence = BlockGridEvidence(
            self.case,
            self.payload,
            self.n_hat,
            self.case["Ne"],
            self.case.get("abs_layers"),
            self.groove,
            spill=self.spill,
            project=False,
        )
        self._pending, self._seen = [], 0
        self.passes += 1

    def __call__(self, block) -> None:
        start = self._seen
        stop = start + int(block["Ne"])
        self._seen = stop
        if self.span is None:
            commit_once(self.evidence.add, block)
            return
        cut = start
        boundary = (start // self.span + 1) * self.span
        while boundary <= stop:
            piece = _electron_slice(block, start, cut, boundary)
            commit_once(self.evidence.add, _join([*self._pending, piece]))
            self._pending, cut = [], boundary
            boundary += self.span
        if cut < stop:
            from .recovery import _host_block

            # The rest of this block arrives with the next chunk.
            self._pending.append(_host_block(_electron_slice(block, start, cut, stop)))

    def finish(self) -> None:
        """Add the last, short block ``[kB, Ne)`` of the realized population."""
        if self._pending:
            commit_once(self.evidence.add, _join(self._pending))
            self._pending = []

    def resolve(self, realized):
        """``(E_grid, record, grid_pass)`` at the stop: the speed cache, else the evidence."""
        from .cxr_grid import _auditable_policy, _resolve_policy_grid, cached_policy_cxr_grid

        self.finish()
        grid_pass = {
            "transported": False,
            "monitored_evidence": True,
            "evidence_bytes": self.evidence.nbytes,
            "spilled": self.evidence.spilled,
            "spill_peak_bytes": self.evidence.spill_peak_bytes,
        }
        hit = cached_policy_cxr_grid(realized)
        if hit is not None:
            return *hit, {**grid_pass, "cache": "hit"}
        grid, record = _resolve_policy_grid(
            _auditable_policy(realized, realized["line_grid_policy"]),
            realized,
            None,
            (self.n_hat,),
            realized["Ne"],
            realized.get("abs_layers"),
            self.groove,
            observation=False,
            evidence=self.evidence,
        )
        return grid, record, {**grid_pass, "cache": "miss"}

    def close(self) -> None:
        if self.evidence is not None:
            self.evidence.close()
        self._pending = []


def _electron_slice(block, start, first, stop):
    """Rows of the electrons ``[first, stop)`` of a kept chunk starting at ``start``."""
    from ..spectrum.cxr._kernels import _SEG_ARRAYS
    from .electron_blocks import _electron_ids, _take

    if (first, stop) == (start, start + int(block["Ne"])):
        return block
    ids = _electron_ids(block)
    rows = slice(*(int(v) for v in np.searchsorted(ids, (first, stop), side="left")))
    piece = dict(block)
    for key in _SEG_ARRAYS:
        if key in piece:
            piece[key] = _take(piece[key], rows)
    piece["Ne"] = stop - first
    return piece


def _join(pieces):
    """One block from consecutive electron slices (host rows when they mix)."""
    from ..spectrum.cxr._kernels import _SEG_ARRAYS
    from .block_transport import _concatenate

    if len(pieces) == 1:
        return pieces[0]
    block = dict(pieces[0])
    for key in _SEG_ARRAYS:
        if key in block:
            block[key] = _concatenate([piece[key] for piece in pieces])
    block["Ne"] = sum(int(piece["Ne"]) for piece in pieces)
    return block


class _MeasureOnly:
    """The #1280 measure-only sink: keeps nothing, counts the monitored passes."""

    def __init__(self):
        self.passes = 0

    def begin_pass(self, monitor) -> None:
        self.passes += 1

    def __call__(self, block) -> None:
        del block


def _monitored_pass(case, transport_core, sink):
    """Run the monitored adaptive transport into ``sink``; the realized fixed case and record.

    Validation: adaptive-sample-size-stopping
    """
    from . import _transport_case

    tp = _transport_case(
        case,
        transport_core=transport_core,
        keep_segments_on_device=True,
        block_consumer=sink,
    )
    sampling = dict(tp["adaptive_sampling"])
    fixed = {key: value for key, value in case.items() if key != "adaptive_precision"}
    fixed.update(Ne=int(tp["Ne_lines"]), Ne_brem=int(tp["Ne_brem"]))
    return fixed, sampling, tp


def run_adaptive_streamed(case, block_electrons, *, transport_core="auto", spill=None):
    """Stream an adaptive ``case`` on its :func:`adaptive_route` (module doc).

    Returns :func:`.streamed_scoring.run_case_streamed`'s output, with the
    monitor's ``adaptive_sampling`` and ``_adaptive_route`` naming the route
    and its transport count.

    Validation: adaptive-sample-size-stopping
    """
    from .recovery import RECOVERY_KEYS
    from .streamed_scoring import score_fixed_streamed

    route = adaptive_route(case)
    if route == "single":
        from ..spectrum.cxr._coverage import collect_line_data_coverage
        from .oom import _ensure_pool_limit

        _ensure_pool_limit()
        sink = MonitoredScoringSink(case, _max_histories(case))
        with collect_line_data_coverage() as line_coverage:
            fixed, sampling, tp = _monitored_pass(case, transport_core, sink)
        out = sink.result(fixed, tp["segs"], line_coverage)
        out.update({key: tp[key] for key in RECOVERY_KEYS if key in tp})
        transports = sink.passes
    elif route == "folded":
        evidence = MonitoredGridEvidence(case, block_electrons, spill, _max_histories(case))
        try:
            fixed, sampling, tp = _monitored_pass(case, transport_core, evidence)
            grid = evidence.resolve(fixed)
        finally:
            evidence.close()
        out = score_fixed_streamed(
            fixed, block_electrons, transport_core=sampling["transport_core"], grid=grid
        )
        transports = evidence.passes + 1
    else:
        sink = _MeasureOnly()
        fixed, sampling, tp = _monitored_pass(case, transport_core, sink)
        out = score_fixed_streamed(
            fixed, block_electrons, transport_core=sampling["transport_core"], spill=spill
        )
        grid_pass = out.get("_streamed_line_grid_pass") or {}
        transports = sink.passes + 1 + bool(grid_pass.get("transported"))
    out["adaptive_sampling"] = sampling
    out["_adaptive_route"] = {
        "route": "streamed",
        "scoring": route,
        # Full-population transports; each energy-range replay is a pass.
        "transports": transports,
    }
    return out
