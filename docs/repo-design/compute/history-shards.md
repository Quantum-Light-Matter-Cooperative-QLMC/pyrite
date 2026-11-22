# History shards

Experimental, CPU only (#1387). Every execution backend parallelizes across
cases, so one expensive case gets no speedup from more hardware. History
shards split one fixed-count case's electron histories into ranges that run on
separate workers, then join them into the result a single run over
`[0, N)` gives, bit for bit.

## How it works

- **Shard.** `runner/history_shards.py::transport_history_shard(case, start,
  stop)` transports electrons `[start, stop)` through the electron-block
  driver (`runner/block_transport.py`). An `ElectronShard` passed as
  `block_electrons` runs only that range, in blocks of its own size, and hands
  the block consumer its unfinalized join: global electron indices, the
  range's additive counts, and no population-wide fields.
- **Publish.** `execution/shards.py::HistoryShardStore` writes the raw rows to
  shared disk under a key derived from the case content key, the range and the
  transport core. Publication is first-writer-wins (exclusive hard link plus a
  digest receipt), like the case inbox in
  [distributed artifacts](../../reference/distributed-artifacts.md).
- **Join.** `run_case_from_shards(case, shards)` passes a `ShardJoin` as
  `block_electrons`: the shards concatenate in electron order, counts add, the
  two population-wide host steps run once, and the case continues as the
  fixed-count run (line grid, spectra, outputs).
- **Schedule.** `execution/dask_shards.py::ShardedDaskExecutor` is a
  `DaskExecutor` with a `ShardPolicy(shard_electrons=...)`. Each case becomes
  one task per unpublished shard plus a join task submitted after all its
  shards finish. The configured `max_in_flight` bounds outstanding shard and
  join futures together. Identical cases share one shard set and join, while
  each requested case receives its own callback. The join publishes the case
  artifact, then deletes consumed shards unless `keep_shards=True`.

## Exactness

Three properties make the join equal the single run:

1. Every random input is addressed by `(seed, electron, draw)`
   (`stream_keys(..., start=)` and the per-electron counter streams), so a
   shard draws exactly its slice of the full run. No new stream or seeding
   scheme exists.
2. The per-electron core emits rows electron-major, so shards concatenated in
   order reproduce the single call's row order; every shard uses the
   population's table energy range.
3. The two population-wide steps are redone once at the join: bunch timing
   (the centroid is a property of the realized population; each raw offset is
   counter-addressed) and hard-photon direction completion (one sequential
   generator over the joined rows). Shards skip their own completion.

`tests/montecarlo/runner/test_history_shards.py` pins segment rows and
spectra against `run_case` for 1, 3 and 4 shards with N = 13, through a pickle
round trip and in reverse run order. `tests/execution/test_dask_history_shards.py`
repeats the comparison through real Dask process workers and checks resume.
Cache hit/miss telemetry can differ between processes; it is excluded from
the comparison, while numerical diagnostics and spectra remain exact.

## What crosses the network

Raw rows, not per-shard spectra. Spectra of disjoint electron blocks add, but
only to floating-point summation order (see
[streamed scoring](streamed-scoring.md)), and several spectrum-phase steps read
the whole population: automatic line grids, reflection selection, coherent
sums. Shipping rows keeps the join exact; the cost is shard size and a serial
spectrum phase at the join.

Only references and indices cross the Dask scheduler. Shards are pickles,
because the transport metadata (tuples, NumPy scalars, aliased arrays) must
round-trip exactly; the join reads one only after its digest matches the
receipt. The trust boundary is the cluster's own, which already exchanges
pickled tasks.

## Resume and retries

Dask `retries` apply per task, so a failed shard reruns alone. A rerun over the
same artifact root delivers published case artifacts without work and never
resubmits a published shard; an interrupted run loses at most the shards in
flight.
If another coordinator finishes the same case and removes its shards during
a read or publication, the worker reuses the completed case artifact.

## Limits

- Refused before staging (`refuse_unshardable`): the lockstep core (one
  generator across electrons; CPU `"auto"` resolves to it, so pin
  `transport_core="per-electron"`), the CUDA core (GPU shards are out of
  scope), adaptive electron counts, secondary cascades, channeling, transition
  radiation, photon transport, GDF beams, and observation directions.
- The equality is with the per-electron run, not the CPU default (lockstep),
  which is a different realization of the same distribution.
- The spectrum phase is serial at the join, so the speedup is bounded by the
  transport share of the case wall time.
- The join holds every row, as the unsharded run does.

