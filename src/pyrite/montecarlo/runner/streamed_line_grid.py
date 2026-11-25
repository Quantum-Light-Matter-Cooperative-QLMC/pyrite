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
every production line (#1257). The first pass therefore keeps the lines for
that resolution too, under any bandwidth; the joined resolver then builds the
same axis from them.

Windowed policies read whole-population seed providers and are refused.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from ..._cxr_grid_policy import OVERSAMPLED_RESOLUTION_POLICY, RESONANCE_BANDWIDTH_POLICY

if TYPE_CHECKING:
    from .line_grid_spill import LineFile, SpillStore

#: Line-grid policy features live streaming cannot resolve from block evidence.
WINDOWED_POLICY_REFUSAL = "windowed line_grid_policy is not supported by live streamed scoring"


def policy_streaming_refusal(case) -> str | None:
    """Why the case's line-grid policy cannot be streamed, or ``None``."""
    payload = case.get("line_grid_policy")
    if payload is not None and payload.get("windows") is not None:
        return WINDOWED_POLICY_REFUSAL
    return None


class BlockGridEvidence:
    """Grid-resolution evidence accumulated over whole-electron blocks.

    Serves the two quantities :class:`.cxr_grid.SegmentGridEvidence` reads from
    a joined population, for the case's single direction ``n_hat``: the sinc
    ``(width, weight)`` samples of every valid line row, and -- for the
    measured ``resonance-population`` bandwidth or a ``sinc-oversampled``
    resolution, whose over-budget axis refines around each line (#1257) --
    every production line's ``(energy, width, weight)``. Evidence is 16 bytes
    per valid line row plus, when lines are kept, three backend reals per
    production line. Whether a ``sinc-oversampled`` axis exceeds its budget
    is known only from the final sinc spacing, so its lines are always kept.

    Under ``spill`` (:class:`.line_grid_spill.GridSpillPolicy`), once the
    evidence held, or projected over the run from the blocks so far, exceeds
    the threshold, it moves to a scratch directory and later blocks are written
    there as they arrive (#1141); :meth:`close` removes it. A projected spill
    the scratch disk cannot hold raises
    :class:`.line_grid_spill.LineGridSpillError` at that block.
    """

    def __init__(self, case, payload, n_hat, Ne, abs_layers=None, groove=None, spill=None):
        from .line_grid_spill import GridSpillPolicy

        self.case, self.n_hat, self.Ne = case, np.asarray(n_hat, dtype=float), int(Ne)
        self.abs_layers, self.groove = abs_layers, groove
        bandwidth = payload["bandwidth"]
        self.band = (float(bandwidth["start_eV"]), float(bandwidth["stop_eV"]))
        self.measured = bandwidth["policy"] == RESONANCE_BANDWIDTH_POLICY
        # Mixed transport scores no lines; its local axis is the backbone alone.
        oversampled = payload["resolution"]["policy"] == OVERSAMPLED_RESOLUTION_POLICY
        self.keeps_lines = self.measured or (oversampled and case.get("transport_mode") != "mixed")
        self.policy = GridSpillPolicy(threshold_bytes=None) if spill is None else spill
        self._electrons = max(int(Ne), int(case.get("Ne_brem") or 0), 1)
        self._width: list[np.ndarray] | None = []
        self._weight: list[np.ndarray] | None = []
        self._populations: list = []
        self._dtypes = None
        self._seen = [0, 0, 0]  # transported electrons, sinc bytes, line bytes
        self._store: SpillStore | None = None
        self._lines: LineFile | None = None
        self._runs: list = []
        self.spill_peak_bytes = 0

    def add(self, segs) -> None:
        """Record one block; commits only after every block quantity succeeded."""
        from ..spectrum.diagnostics import sinc_feature_samples
        from .cxr_grid import _direction_populations

        width, weight = sinc_feature_samples(segs, self.n_hat, electron_limit=self.Ne)
        populations = []
        if self.keeps_lines and int(segs["L_ang"].shape[0]):
            populations = _direction_populations(
                self.case,
                segs,
                self.n_hat,
                self.Ne,
                self.abs_layers,
                self.groove,
                *self.band,
                None,
            )
        if self._width is None or self._weight is None:
            raise RuntimeError("streamed sinc evidence was already reduced")
        if self._dtypes is None:
            self._dtypes = (width.dtype, weight.dtype)
        # The joined samples take the first block's dtypes (_drain).
        width = width.astype(self._dtypes[0], copy=False)
        weight = weight.astype(self._dtypes[1], copy=False)
        seen = [
            self._seen[0] + int(segs.get("Ne", 0)),
            self._seen[1] + width.nbytes + weight.nbytes,
            self._seen[2] + _population_bytes(populations),
        ]
        if self._store is None and self._should_spill(seen):
            self._start_spill(seen)
        elif self._store is not None:
            self._check_disk(self._store, seen)
        if self._store is not None:
            self._spill_block(self._store, width, weight, populations)
        else:
            self._width.append(width)
            self._weight.append(weight)
            self._populations.extend(populations)
        self._seen = seen

    def _projected(self, seen) -> tuple[float, float]:
        scale = self._electrons / max(seen[0], 1)
        return seen[1] * scale, seen[2] * scale

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

    def _spill_block(self, store, width, weight, populations) -> None:
        assert self._lines is not None
        run = _write_run(store, width, weight)
        try:
            spilled = self._lines.append(populations)
        except BaseException:
            store.unlink(run.width)
            store.unlink(run.weight)
            raise
        self._runs.append(run)
        self._populations.extend(spilled)
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
        return int(self._seen[1] + self._seen[2])

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

    def populations(self, n_hat, start, ceiling, profile):
        """Every accumulated production line over the policy band."""
        del profile  # stage profiling covers the joined resolver only
        self._check_direction(n_hat)
        if not self.keeps_lines or (float(start), float(ceiling)) != self.band:
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
    case, block_electrons, *, transport_core, n_hat, groove, spill=None
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
    from . import _announce_transport_pass, _transport_case
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
    evidence = BlockGridEvidence(case, payload, n_hat, case["Ne"], abs_layers, groove, spill=spill)
    _announce_transport_pass("grid")
    try:
        _transport_case(
            case,
            transport_core=transport_core,
            block_electrons=block_electrons,
            block_consumer=evidence.add,
            keep_segments_on_device=True,
        )
        grid, record = _resolve_policy_grid(
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
    finally:
        evidence.close()
    return (
        grid,
        record,
        {
            "transported": True,
            "evidence_bytes": evidence.nbytes,
            "spilled": evidence.spilled,
            "spill_peak_bytes": evidence.spill_peak_bytes,
        },
    )


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
