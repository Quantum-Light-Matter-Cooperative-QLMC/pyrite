"""Population-resolved line grids for live streamed scoring (#1001).

A ``line_grid_policy`` grid is resolved from the whole line population, but
live streaming scores each block as it finishes. :func:`resolve_streamed_line_grid`
therefore runs two transport passes over the same electron blocks. Primary
streams are addressed by ``(seed, electron, draw)``, so both passes see the
same histories. The first pass keeps only what policy resolution reads, as
compact per-row arrays (:class:`BlockGridEvidence`), and resolves the grid
with the joined path's own code; the second pass scores against it.

A speed-cache hit needs no trajectories, so it skips the first pass.

Equality with the joined grid: the sinc samples of consecutive whole-electron
blocks concatenate to the joined samples, so the sinc spacing is exact, and
``resonance-local`` seeds are integer coverage counts. The measured
``resonance-population`` stop sums line masses chunk by chunk, so its float
sums may differ in the last ulp from the joined path's; the stop is rounded up
to a whole 100 eV step, so the grid differs only if a tail ratio lands within
rounding of the truncation limit.

A ``sinc-oversampled`` axis over its point budget refines node-locally around
every production line (#1257). Full-fiber grids defer individual line collection
until that refinement is actually needed (#1527); other routes keep lines on
the first pass as before.

A production line population can be far larger than the transport rows it
comes from (full-fiber CXR holds thousands of lines per row), while transport
blocks are sized by their rows. Each block's lines are therefore collected over
consecutive electron-aligned slices of about :data:`LINE_SLICE_BYTES`, and each
slice is recorded, and spilled, before the next is collected (#1527).

Windowed policies read whole-population seed providers and are refused.

Under CuPy each block's sinc samples come from the device route the joined
sizing uses (:func:`..spectrum._device_sizing.device_sinc_samples`), so only
the two sample arrays cross to the host and the streamed samples equal the
joined ones bit for bit there too (#1481).
"""

import sys
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._cxr_grid_policy import OVERSAMPLED_RESOLUTION_POLICY, RESONANCE_BANDWIDTH_POLICY

if TYPE_CHECKING:
    from .line_grid_spill import LineFile, SpillStore

#: Line evidence one collection call of the grid pass aims to produce.
LINE_SLICE_BYTES = 2**30
#: Transport rows of the first line slice, which measures the bytes per row.
LINE_SLICE_FIRST_ROWS = 1024
#: Most a line slice's row count may grow over the one before.
LINE_SLICE_GROWTH = 8

#: Line-grid policy features live streaming cannot resolve from block evidence.
WINDOWED_POLICY_REFUSAL = "windowed line_grid_policy is not supported by live streamed scoring"


def policy_streaming_refusal(case) -> str | None:
    """Why the case's line-grid policy cannot be streamed, or ``None``."""
    payload = case.get("line_grid_policy")
    if payload is not None and payload.get("windows") is not None:
        return WINDOWED_POLICY_REFUSAL
    return None


def _block_sinc_samples(segs, n_hat, electron_limit):
    """Host ``(width, weight)`` of one block, by the route the joined sizing takes."""
    from ..._backend import _to_cpu
    from ..spectrum._device_sizing import device_sinc_samples, sizing_module
    from ..spectrum.diagnostics import sinc_feature_samples

    module = sizing_module(segs["L_ang"])
    if module is not None:
        resident = device_sinc_samples(segs, n_hat, electron_limit=electron_limit, module=module)
        if resident is not None:
            return tuple(np.asarray(_to_cpu(a)) for a in resident)
    return sinc_feature_samples(segs, n_hat, electron_limit=electron_limit)


class _NeedsLinePopulation(Exception):
    """Local refinement needs a replay with lines, optionally with a byte estimate."""

    def __init__(self, required_bytes: int | None):
        super().__init__("local line-grid refinement requires individual production lines")
        self.required_bytes = required_bytes


class BlockGridEvidence:
    """Grid-resolution evidence accumulated over whole-electron blocks.

    Serves the evidence :class:`.cxr_grid.SegmentGridEvidence` reads from
    a joined population, for the case's single direction ``n_hat``: the sinc
    ``(width, weight)`` samples of every valid line row, and either production
    line triples or fixed-size tail sums. Full-fiber measured grids keep only
    tail sums until local refinement actually needs individual lines (#1527).
    Full-fiber kinematic grids, including bandwidth extensions, defer line
    collection too: a uniform axis reads only the sinc samples. Other routes
    retain their original evidence. Whenever lines are needed they are
    collected a slice of a block at a time (:meth:`_line_slices`).

    Under ``spill`` (:class:`.line_grid_spill.GridSpillPolicy`), once the
    evidence held, or projected over the run from the blocks so far, exceeds
    the threshold, it moves to a scratch directory and later blocks are written
    there as they arrive (#1141); :meth:`close` removes it. A projected spill
    the scratch disk cannot hold raises
    :class:`.line_grid_spill.LineGridSpillError` at that block.
    """

    def __init__(
        self,
        case,
        payload,
        n_hat,
        Ne,
        abs_layers=None,
        groove=None,
        spill=None,
        *,
        keep_lines=False,
    ):
        from .line_grid_spill import GridSpillPolicy

        self.case, self.n_hat, self.Ne = case, np.asarray(n_hat, dtype=float), int(Ne)
        self.abs_layers, self.groove = abs_layers, groove
        bandwidth = payload["bandwidth"]
        self.band = (float(bandwidth["start_eV"]), float(bandwidth["stop_eV"]))
        self.measured = bandwidth["policy"] == RESONANCE_BANDWIDTH_POLICY
        self.defers_lines = case.get("cxr_fiber") is not None and not keep_lines
        self.summarizes_tail = self.measured and self.defers_lines
        self._tail_summary = None
        self._tail_stops = None
        self._tail_replay = None
        self._resolved_tail = None
        self._pilot_rows = 0
        self._pilot_certified: bool | None = None
        self.tail_replays = 0
        # Mixed transport scores no lines; its local axis is the backbone alone.
        oversampled = payload["resolution"]["policy"] == OVERSAMPLED_RESOLUTION_POLICY
        self.keeps_lines = not self.defers_lines and (
            self.measured or (oversampled and case.get("transport_mode") != "mixed")
        )
        self.policy = GridSpillPolicy(threshold_bytes=None) if spill is None else spill
        self._electrons = max(int(Ne), int(case.get("Ne_brem") or 0), 1)
        self._width: list[np.ndarray] | None = []
        self._weight: list[np.ndarray] | None = []
        self._populations: list = []
        self._dtypes = None
        # Transported electrons, sinc bytes, line bytes, electrons those lines cover.
        self._seen = [0, 0, 0, 0]
        self._slice_rows, self._row_bytes = LINE_SLICE_FIRST_ROWS, 0.0
        self._store: SpillStore | None = None
        self._lines: LineFile | None = None
        self._runs: list = []
        self.spill_peak_bytes = 0

    def add(self, segs) -> None:
        """Record one block; a failure leaves none of the block recorded."""
        from ..spectrum._lattice_tail import merge_tail
        from .cxr_grid import _direction_tail

        width, weight = _block_sinc_samples(segs, self.n_hat, self.Ne)
        if self._width is None or self._weight is None:
            raise RuntimeError("streamed sinc evidence was already reduced")
        if self._dtypes is None:
            self._dtypes = (width.dtype, weight.dtype)
        # The joined samples take the first block's dtypes (_drain).
        width = width.astype(self._dtypes[0], copy=False)
        weight = weight.astype(self._dtypes[1], copy=False)
        mark = self._mark()
        summary, stops = self._tail_summary, self._tail_stops
        pilot_rows = self._pilot_rows
        try:
            self._record_samples(width, weight, int(segs.get("Ne", 0)))
            if self.summarizes_tail:
                if self._tail_replay is not None and stops is None:
                    if int(segs["L_ang"].shape[0]):
                        from ..._cxr_grid_policy import BANDWIDTH_PROXY_SAFETY
                        from ._pilot_tail import pilot_block

                        limit = float(self.case["line_grid_policy"]["bandwidth"]["truncation_limit"])

                        def collect(part, candidates):
                            return _direction_tail(
                                self.case, part, self.n_hat, self.Ne, self.abs_layers,
                                self.groove, *self.band, stops=candidates,
                            )

                        stops, summary, pilot_rows = pilot_block(
                            segs, self.band[1], limit / BANDWIDTH_PROXY_SAFETY, collect
                        )
                        print(
                            f"line-grid #1527: pilot rows={pilot_rows} "
                            f"certification candidates={len(stops)}",
                            file=sys.stderr, flush=True,
                        )
                else:
                    options = {} if stops is None else {"stops": stops}
                    part = _direction_tail(
                        self.case, segs, self.n_hat, self.Ne, self.abs_layers, self.groove,
                        *self.band, **options,
                    )
                    if part is not None:
                        summary = merge_tail(summary, part)
            elif self.keeps_lines and int(segs["L_ang"].shape[0]):
                for populations, electrons in self._line_slices(segs):
                    self._record_lines(populations, electrons)
        except BaseException:
            self._rollback(mark)
            raise
        self._tail_summary, self._tail_stops, self._pilot_rows = summary, stops, pilot_rows

    def _line_slices(self, segs):
        """``(populations, electrons)`` of consecutive electron-aligned block slices.

        Each slice's row count comes from the most line bytes per row seen so
        far, aiming at :data:`LINE_SLICE_BYTES`. ``electrons`` is the slice's
        share of the block's electrons by rows, for the spill projection; the
        shares sum to the block's count.
        """
        from .cxr_grid import _direction_populations
        from .electron_blocks import iter_sized_electron_blocks

        total, electrons = int(segs["L_ang"].shape[0]), int(segs.get("Ne", 0))
        done = covered = 0
        for part in iter_sized_electron_blocks(segs, lambda: self._slice_rows):
            populations = _direction_populations(
                self.case,
                part,
                self.n_hat,
                self.Ne,
                self.abs_layers,
                self.groove,
                *self.band,
                None,
            )
            rows = int(part["L_ang"].shape[0])
            self._size_next_slice(rows, _population_bytes(populations))
            done += rows
            share = -(-electrons * done // total) - covered
            covered += share
            yield populations, share

    def _size_next_slice(self, rows, nbytes) -> None:
        if not nbytes:
            return  # rows past the line electrons say nothing about the next slice
        self._row_bytes = max(self._row_bytes, nbytes / rows)
        fit = int(LINE_SLICE_BYTES / self._row_bytes)
        self._slice_rows = max(1, min(fit, LINE_SLICE_GROWTH * self._slice_rows))

    def _record_samples(self, width, weight, electrons) -> None:
        seen = list(self._seen)
        seen[0] += electrons
        seen[1] += width.nbytes + weight.nbytes
        self._admit(seen)
        if self._store is not None:
            self._runs.append(_write_run(self._store, width, weight))
            self._track_peak()
        else:
            assert self._width is not None and self._weight is not None
            self._width.append(width)
            self._weight.append(weight)
        self._seen = seen

    def _record_lines(self, populations, electrons) -> None:
        seen = list(self._seen)
        seen[2] += _population_bytes(populations)
        seen[3] += electrons
        self._admit(seen)
        if self._lines is not None:
            self._populations.extend(self._lines.append(populations))
            self._track_peak()
        else:
            self._populations.extend(populations)
        self._seen = seen

    def _admit(self, seen) -> None:
        """Before recording up to ``seen``: spill, or check the scratch disk."""
        if self._store is not None:
            self._check_disk(self._store, seen)
        elif self._should_spill(seen):
            self._start_spill(seen)

    def _mark(self):
        """``(seen, sample blocks, populations)`` that :meth:`_rollback` returns to."""
        blocks = self._runs if self._store is not None else self._width or ()
        return list(self._seen), len(blocks), len(self._populations)

    def _rollback(self, mark) -> None:
        """Drop everything recorded since ``mark``, held or spilled.

        A spill that started since then stays: it holds the earlier blocks.
        """
        self._seen, blocks, populations = mark
        del self._populations[populations:]
        if self._store is None:
            assert self._width is not None and self._weight is not None
            del self._width[blocks:], self._weight[blocks:]
            return
        for run in self._runs[blocks:]:
            self._store.unlink(run.width)
            self._store.unlink(run.weight)
        del self._runs[blocks:]
        assert self._lines is not None
        self._lines.rewind(self._populations)

    def _projected(self, seen) -> tuple[float, float]:
        """Run-total ``(sinc, line)`` bytes at the per-electron rate seen so far."""
        return (
            seen[1] * (self._electrons / max(seen[0], 1)),
            seen[2] * (self._electrons / max(seen[3], 1)),
        )

    def _should_spill(self, seen) -> bool:
        threshold = self.policy.threshold_bytes
        if threshold is None:
            return False
        return max(sum(self._projected(seen)), seen[1] + seen[2]) > threshold

    def _check_disk(self, store, seen) -> None:
        from .line_grid_spill import LineGridSpillError, projected_disk_bytes

        need = projected_disk_bytes(*self._projected(seen)) - store.bytes
        free = store.free_bytes()
        if need > free:
            raise LineGridSpillError(
                f"streamed line-grid evidence needs about {need / 1024**3:.1f} GiB more "
                f"scratch disk in {store.root}, but only {free / 1024**3:.1f} GiB is free; "
                "free space or set KernelTuning.stream_spill_dir to a larger file system"
            )

    def _start_spill(self, seen) -> None:
        """Move the held blocks to a new store; on failure keep them in memory."""
        from .line_grid_spill import LineFile, SpillStore

        store = SpillStore(self.policy.parent())
        self._store = store
        try:
            self._check_disk(store, seen)
            runs, populations = self._runs, []
            for width, weight in zip(self._width or (), self._weight or (), strict=True):
                runs.append(_write_run(store, width, weight))
            self._lines = LineFile(store)
            populations = self._lines.append(self._populations)
        except BaseException:
            store.close()
            self._store, self._lines, self._runs = None, None, []
            raise
        self._width, self._weight, self._populations = [], [], populations
        self._track_peak()

    def _track_peak(self) -> None:
        if self._store is not None:
            self.spill_peak_bytes = max(self.spill_peak_bytes, self._store.peak_bytes)

    @property
    def spilled(self) -> bool:
        """Whether the evidence lives on disk."""
        return self._store is not None

    @property
    def nbytes(self) -> int:
        """Evidence bytes collected, held in memory or spilled."""
        tail_bytes = 0 if self._tail_summary is None else self._tail_summary.sums.nbytes
        return int(self._seen[1] + self._seen[2] + tail_bytes)

    @property
    def line_evidence_mode(self) -> str:
        """Distinguish a bounded tail pass from collection and spacing alone."""
        if self.summarizes_tail:
            return "tail-lattice"
        return "line-population" if self.keeps_lines else "spacing-only"

    def line_evidence_record(self) -> dict[str, Any]:
        """Actual retained line bytes, not the projected raw population size."""
        summary = self._tail_summary
        record: dict[str, Any] = {
            "line_evidence_mode": self.line_evidence_mode,
            "line_population_bytes": self._seen[2],
            "tail_summary_bytes": 0 if summary is None else int(summary.sums.nbytes),
            "summarized_lines": 0 if summary is None else summary.raw_lines,
        }
        if self._pilot_rows and self._tail_stops is not None:
            record.update(
                pilot_rows=self._pilot_rows,
                pilot_candidates=len(self._tail_stops),
                pilot_certified=self._pilot_certified,
                tail_replays=self.tail_replays,
            )
        return record

    def close(self) -> None:
        """Remove any spilled evidence."""
        if self._lines is not None:
            self._lines.close()
        if self._store is not None:
            self._track_peak()
            self._store.close()

    def _check_direction(self, n_hat) -> None:
        if not np.array_equal(np.asarray(n_hat, dtype=float), self.n_hat):
            raise ValueError("streamed line-grid evidence was collected along another direction")

    def sinc_spacing(self, n_hat, aliased_weight_limit):
        """:func:`.diagnostics.sinc_feature_spacing` of the accumulated rows."""
        from ..spectrum.diagnostics import sinc_spacing_from_samples
        from ..spectrum.spacing_selection import sinc_spacing_by_selection

        self._check_direction(n_hat)
        if self._width is None or self._weight is None:
            raise RuntimeError("streamed sinc evidence was already reduced")
        if self._store is None and self._sort_refused():
            self._start_spill(self._seen)
        width, weight = self._width, self._weight
        self._width = self._weight = None
        if self._store is not None:
            return self._spilled_sinc_spacing(self._store, aliased_weight_limit)
        selected = sinc_spacing_by_selection(_paired(width, weight), aliased_weight_limit)
        if selected is not None:
            return selected
        samples = [_drain(width), _drain(weight)]
        del width, weight
        return sinc_spacing_from_samples(
            samples, aliased_weight_limit=aliased_weight_limit, select=False
        )

    def _sort_refused(self) -> bool:
        """Whether host admission refuses the in-memory spacing sort (#1453).

        The sort allocates :func:`.diagnostics.sinc_spacing_sort_bytes` beyond
        the held samples, whose bytes headroom already excludes. A refusal
        moves the held evidence to disk instead; a policy that never spills
        (``threshold_bytes=None``) has nowhere to go and keeps the sort.
        """
        from ..spectrum.diagnostics import sinc_spacing_sort_bytes
        from ..transport import host_memory

        if self.policy.threshold_bytes is None or self._dtypes is None:
            return False
        width, weight = (np.dtype(dtype).itemsize for dtype in self._dtypes)
        rows = self._seen[1] // (width + weight)
        try:
            host_memory.admit_host_allocation(
                sinc_spacing_sort_bytes(rows, width, weight), "streamed line-grid spacing sort"
            )
        except host_memory.HostMemoryResourceError:
            return True
        return False

    def _spilled_sinc_spacing(self, store, aliased_weight_limit):
        from ..spectrum.spacing_selection import sinc_spacing_by_selection
        from .line_grid_spill import external_stable_sort, iter_run_chunks, sorted_sinc_spacing

        runs = self._runs
        selected = sinc_spacing_by_selection(lambda: iter_run_chunks(runs), aliased_weight_limit)
        if selected is not None:
            self._runs = []
            return selected
        self._runs = []
        ordered = external_stable_sort(store, runs, self._dtypes or (np.dtype(float),) * 2)
        self._track_peak()
        try:
            return sorted_sinc_spacing(ordered, aliased_weight_limit=aliased_weight_limit)
        finally:
            store.unlink(ordered.width)
            store.unlink(ordered.weight)

    def tail(self, n_hat, start, ceiling, profile):
        """Certify the full-population neighborhood, or replay exact search."""
        from ..spectrum._lattice_tail import LatticeTail, stop_lattice

        self._check_direction(n_hat)
        if not self.summarizes_tail:
            return None
        if (float(start), float(ceiling)) != self.band:
            raise ValueError("streamed tail evidence was collected over another band")
        if self._tail_replay is None or self._tail_stops is None:
            return LatticeTail(stop_lattice(ceiling), self._tail_summary)
        if self._resolved_tail is None:
            from ..._cxr_grid_policy import BANDWIDTH_PROXY_SAFETY
            from ._pilot_tail import resolve_pilot_tail

            limit = float(self.case["line_grid_policy"]["bandwidth"]["truncation_limit"])
            self._resolved_tail, self._pilot_certified = resolve_pilot_tail(
                ceiling, limit / BANDWIDTH_PROXY_SAFETY, self._tail_stops,
                self._tail_summary, self._tail_replay,
            )
            print(
                f"line-grid #1527: pilot certified={self._pilot_certified} "
                f"tail_replays={self.tail_replays}",
                file=sys.stderr, flush=True,
            )
        return self._resolved_tail

    def populations(self, n_hat, start, ceiling, profile):
        """Every accumulated production line over the policy band."""
        del profile  # stage profiling covers the joined resolver only
        self._check_direction(n_hat)
        if (float(start), float(ceiling)) != self.band:
            raise ValueError("streamed line-grid evidence holds no lines for this band")
        if self.defers_lines:
            tail = self.tail(n_hat, start, ceiling, None)
            # A kinematic grid has no tail pass: the replay uses the existing
            # bounded slices and first-slice disk projection instead.
            raise _NeedsLinePopulation(None if tail is None else tail.population_bytes)
        if not self.keeps_lines:
            raise ValueError("streamed line-grid evidence holds no lines for this band")
        if self._lines is not None:
            return self._lines.populations(self._populations)
        return list(self._populations)


def _population_bytes(populations) -> int:
    return sum(
        array.nbytes
        for population in populations
        for array in (population.energy_eV, population.weight, population.width_eV)
    )


def _write_run(store, width, weight):
    from .line_grid_spill import write_run

    return write_run(store, width, weight)


def _paired(width: list[np.ndarray], weight: list[np.ndarray]):
    """A re-iterable source of the held ``(width, weight)`` blocks, in order."""
    return lambda: zip(width, weight, strict=True)


def _drain(blocks: list[np.ndarray]) -> np.ndarray:
    """Join ``blocks`` in order into one array, releasing each as it is copied.

    Equal to ``np.concatenate(blocks)``; the peak is the result plus one block
    rather than twice the total (#1001).
    """
    size = sum(block.size for block in blocks)
    joined = np.empty(size, dtype=blocks[0].dtype if blocks else float)
    offset = 0
    blocks.reverse()
    while blocks:
        block = blocks.pop()
        joined[offset : offset + block.size] = block
        offset += block.size
    return joined


def resolve_streamed_line_grid(
    case, block_electrons, *, transport_core, n_hat, groove, spill=None, _keep_lines=False
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """The case's policy line grid, from the speed cache or a transport pass.

    Returns ``(E_grid, record, grid_pass)``; ``record`` equals the joined
    :func:`.cxr_grid.resolve_cxr_grid` record and ``grid_pass`` describes the
    first pass (``transported``, ``evidence_bytes``, ``spilled``,
    ``spill_peak_bytes``). ``spill`` is a
    :class:`.line_grid_spill.GridSpillPolicy` (default: spill above
    :func:`.line_grid_spill.default_spill_threshold_bytes`, a share of host
    memory, under the output cache); spilled files are removed however the
    pass ends.
    """
    from . import _TIMING, _announce_transport_pass, _transport_case
    from .cxr_grid import _auditable_policy, _resolve_policy_grid, cached_policy_cxr_grid
    from .line_grid_spill import GridSpillPolicy

    hit = cached_policy_cxr_grid(case)
    if hit is not None:
        return *hit, {
            "transported": False,
            "evidence_bytes": 0,
            "spilled": False,
            "spill_peak_bytes": 0,
        }
    payload = _auditable_policy(case, case["line_grid_policy"])
    abs_layers = case.get("abs_layers")
    spill = GridSpillPolicy() if spill is None else spill
    evidence = BlockGridEvidence(
        case, payload, n_hat, case["Ne"], abs_layers, groove, spill=spill, keep_lines=_keep_lines
    )
    profile = case.get("cxr_fiber") is not None and (
        _TIMING or case.get("_profile_line_grid_stages", False)
    )
    if evidence.summarizes_tail:
        from ..spectrum._lattice_tail import merge_tail
        from .cxr_grid import _direction_tail

        def replay(stops):
            # No sinc samples, line collection or adaptive stopping on this pass.
            summary = None
            evidence.tail_replays += 1
            started = perf_counter()
            print(
                f"line-grid #1527: refinement candidates={len(stops)}",
                file=sys.stderr, flush=True,
            )

            def consume(segs):
                nonlocal summary
                part = _direction_tail(
                    case, segs, n_hat, case["Ne"], abs_layers, groove,
                    *evidence.band, stops=stops,
                )
                if part is not None:
                    summary = merge_tail(summary, part)

            _announce_transport_pass("grid")
            _transport_case(
                case, transport_core=transport_core, block_electrons=block_electrons,
                block_consumer=consume, keep_segments_on_device=True,
            )
            print(
                f"line-grid #1527: refinement wall_s={perf_counter() - started:.3f}",
                file=sys.stderr, flush=True,
            )
            return summary

        evidence._tail_replay = replay
        print("line-grid #1527: pilot-guided certification", file=sys.stderr, flush=True)
    began = perf_counter()
    if profile:
        print(
            f"line-grid #1527: mode={evidence.line_evidence_mode} "
            f"Ne={case['Ne']} band={evidence.band} eV",
            file=sys.stderr,
            flush=True,
        )
    retry = None
    resolved = None
    _announce_transport_pass("grid")
    try:
        _transport_case(
            case,
            transport_core=transport_core,
            block_electrons=block_electrons,
            block_consumer=evidence.add,
            keep_segments_on_device=True,
        )
        resolved = _resolve_policy_grid(
            payload,
            case,
            None,
            (n_hat,),
            case["Ne"],
            abs_layers,
            groove,
            observation=False,
            evidence=evidence,
        )
    except _NeedsLinePopulation as exc:
        # Do not retain the exception (its traceback owns the transport/evidence).
        retry = (exc.required_bytes,)
    finally:
        evidence._tail_replay = None
        evidence.close()
    if profile:
        print(
            f"line-grid #1527: {evidence.line_evidence_record()} "
            f"evidence_bytes={evidence.nbytes} spill_peak_bytes={evidence.spill_peak_bytes} "
            f"wall_s={perf_counter() - began:.3f} replay={retry is not None}",
            file=sys.stderr,
            flush=True,
        )
    if retry is not None:
        from ..transport.host_memory import admit_host_allocation

        retry_bytes = retry[0]
        transports = 1 + evidence.tail_replays
        del evidence
        if retry_bytes is not None:
            admit_host_allocation(retry_bytes, "streamed fiber line-grid refinement")
        # Without a measured tail count the kinematic replay is still bounded
        # by the workaround's electron-aligned slices and disk projection.
        grid, record, replay = resolve_streamed_line_grid(
            case,
            block_electrons,
            transport_core=transport_core,
            n_hat=n_hat,
            groove=groove,
            spill=spill,
            _keep_lines=True,
        )
        return grid, record, {
            **replay, "grid_transports": transports + replay.get("grid_transports", 1)
        }
    assert resolved is not None
    grid, record = resolved
    grid_pass: dict[str, Any] = {
        "transported": True,
        "evidence_bytes": evidence.nbytes,
        "spilled": evidence.spilled,
        "spill_peak_bytes": evidence.spill_peak_bytes,
    }
    if case.get("cxr_fiber") is not None:
        grid_pass.update(
            evidence.line_evidence_record(), grid_transports=1 + evidence.tail_replays
        )
    return grid, record, grid_pass


def streamed_fallback_grid(case, refusal, grid_record, block_electrons, **grid_options):
    """``(E_grid, record, grid_pass)`` of the axis a refused streamed default extends to.

    The streamed counterpart of :func:`.bandwidth_fallback.measured_bandwidth_fallback`
    (#1243): the stop comes from :func:`.bandwidth_fallback.fallback_axis_payload`
    (raising ``refusal`` for an explicit bandwidth) and the kinematic-shaped
    axis is resolved like any streamed policy grid, by its own evidence pass.
    The record carries the same ``bandwidth_fallback`` provenance; the case
    keeps its measured payload, so the rescore is audited again.

    Validation: line-grid-resonance-bandwidth
    """
    from .bandwidth_fallback import fallback_axis_payload, fallback_record

    earlier = (grid_record or {}).get("bandwidth_fallback")
    fallback, stop, ceiling, refusals = fallback_axis_payload(case, refusal, earlier)
    E_grid, record, grid_pass = resolve_streamed_line_grid(
        {**case, "line_grid_policy": fallback}, block_electrons, **grid_options
    )
    return E_grid, fallback_record(record, case, stop, ceiling, refusals), grid_pass
