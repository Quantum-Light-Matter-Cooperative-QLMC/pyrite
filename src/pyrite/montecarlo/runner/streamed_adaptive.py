"""Streamed adaptive cases scored from the monitored pass (#1280, #1397).

An adaptive case streams through its monitored adaptive transport
(:func:`.block_transport.transport_monitored_blocks`). The stopping monitor
reads each chunk; a chunk containing a stop or brem freeze is replayed on its
prefix ``[start, cut)``; an energy-range replay restarts at electron 0. The
chunks the driver keeps are exactly the realized population, so a sink fed
those chunks sees what the fixed case at the realized ``(Ne, Ne_brem)``
transports. Two routes (:func:`adaptive_route`):

- ``single`` (explicit grid): the monitored pass scores its kept chunks. The
  realized counts are unknown until the stop, so blocks are scored at a
  provisional power-of-two count ``L >= max_histories``: every estimator's
  electron filter passes every kept row, and its ``sum / L`` is exact. At the
  stop the totals are renormalized by ``L / Ne`` and ``L / Ne_brem``
  (:meth:`.streamed_scoring.BlockSpectrumScorer.result`). A chunk past the
  brem freeze scores lines only. One transport.
- ``measure`` (``line_grid_policy``, cascades, line roulette, or
  :data:`MONITORED_PASS` off): the #1280 measure-only pass, then the fixed
  streamed route with its own grid pass: two or three transports. Folding
  the policy grid's evidence into the monitored pass was measured slower
  than its separate grid pass on an RTX 5080 (#1397, jobs 15587-15598).

Every sink commits a kept chunk once: a fresh sink state per pass
(``begin_pass``), and a scoring allocator OOM rolls the sink back and
rescores a host copy of the chunk (:func:`commit_once`).
"""

from collections.abc import Mapping
from contextlib import nullcontext
from numbers import Integral
from typing import Any

#: Score explicit-grid cases in the monitored pass (#1397).
#: ``False`` keeps the #1280 measure-only route: a same-revision A/B arm.
MONITORED_PASS = True


def provisional_count(limit: int) -> int:
    """The smallest power of two at or above ``limit`` (and at least 1)."""
    return 1 << max(int(limit) - 1, 0).bit_length()


def adaptive_route(case: Mapping[str, Any]) -> str:
    """``"single"`` or ``"measure"`` for an adaptive ``case`` (module doc)."""
    if not MONITORED_PASS or case.get("secondary_threshold_eV") is not None:
        return "measure"  # cascades keep their #1240 route
    if case.get("line_roulette") is not None:
        return "measure"  # the roulette sizes per-primary state by the population
    if case.get("line_grid_policy") is not None:
        return "measure"  # the grid needs the whole population (#1001)
    return "single"


def commit_once(add, block, owner=None) -> None:
    """``add(block)`` once; a device allocator OOM rolls ``owner`` back and adds host rows.

    ``owner`` (optional) has ``snapshot()``/``restore(state)``.
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
    """Score the monitored pass's kept chunks at a provisional count (``single``).

    Adaptive chunks are small (:class:`.stream_sizing.AdaptiveChunkPlan`), and
    scoring each one alone cost more than the transport the single pass saves
    (#1397, RTX 5080 job 15605). With an integer ``block_electrons`` the kept
    chunks are therefore joined (:func:`.block_transport.merge_electron_blocks`,
    the retained route's join) into blocks of at least that many electrons
    before :meth:`add`. A joined block never straddles the brem freeze: the
    buffer is scored first when the next chunk lies on the other side. A
    memory-sized run (:class:`.stream_sizing.MemoryBlockSizer`) has no fixed
    span and scores each chunk alone.

    ``coverage`` (the run's line-data-coverage list, when set) is cleared at
    each pass: an energy-range replay discards the earlier pass's scorer, so
    its records must not count twice.
    """

    def __init__(self, case, limit, block_electrons=None):
        from .artifacts import _case_geometry

        self.count = provisional_count(limit)
        self.case = _fixed_template(case, self.count)
        E_grid, E_brem, _, n_hat, groove = _case_geometry(self.case)
        self.inputs = dict(E_grid=E_grid, E_brem=E_brem, n_hat=n_hat, groove=groove)
        self.span = int(block_electrons) if isinstance(block_electrons, Integral) else None
        self.coverage: list | None = None
        self.scorer: Any = None
        self.passes = 0
        self._monitor: Any = None
        self._buffer: list = []
        self._buffered = self._seen = 0
        self._buffer_lines_only = False

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
        self._monitor = monitor
        self._buffer, self._buffered, self._seen = [], 0, 0
        if self.coverage is not None:
            self.coverage.clear()

    def __call__(self, block) -> None:
        if self.span is None:
            commit_once(self.scorer.add, block, self.scorer)
            return
        start = self._seen
        self._seen += int(block["Ne"])
        frozen = getattr(self._monitor, "brem_count", None)
        lines_only = frozen is not None and start >= frozen
        if self._buffer and lines_only != self._buffer_lines_only:
            self.finish()
        self._buffer.append(block)
        self._buffered += int(block["Ne"])
        self._buffer_lines_only = lines_only
        if self._buffered >= self.span:
            self.finish()

    def finish(self) -> None:
        """Score the buffered chunks as one block (the pass's last, short one at the stop)."""
        from .block_transport import merge_electron_blocks

        if not self._buffer:
            return
        block = merge_electron_blocks(self._buffer)
        self._buffer, self._buffered = [], 0
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
        sink = MonitoredScoringSink(case, _max_histories(case), block_electrons)
        sized = not isinstance(block_electrons, Integral)
        # Same scoped allocator cap as the fixed streamed route.
        with block_electrons.pool_limit() if sized else nullcontext():
            with collect_line_data_coverage() as line_coverage:
                sink.coverage = line_coverage
                fixed, sampling, tp = _monitored_pass(case, transport_core, sink)
                sink.finish()
        out = sink.result(fixed, tp["segs"], line_coverage)
        out.update({key: tp[key] for key in RECOVERY_KEYS if key in tp})
        transports = sink.passes
    else:
        sink = _MeasureOnly()
        fixed, sampling, tp = _monitored_pass(case, transport_core, sink)
        out = score_fixed_streamed(
            fixed, block_electrons, transport_core=sampling["transport_core"], spill=spill
        )
        grid_pass = out.get("_streamed_line_grid_pass") or {}
        transports = (
            sink.passes
            + 1
            + bool(grid_pass.get("transported"))
            # A refused default axis reruns its passes on the extension (#1243).
            + grid_pass.get("fallback_transports", 0)
        )
    out["adaptive_sampling"] = sampling
    out["_adaptive_route"] = {
        "route": "streamed",
        "scoring": route,
        # Full-population transports; each energy-range replay is a pass.
        "transports": transports,
    }
    return out
