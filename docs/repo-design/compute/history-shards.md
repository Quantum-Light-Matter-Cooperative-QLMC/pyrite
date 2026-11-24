# History shards

Experimental (#1387, #1395). Transport shards run on CPU; the join scores on CPU, or
on one GPU with `ShardPolicy(scoring="gpu")`. With `ShardPolicy(reduction="shard-sum")`
every shard also scores its own spectra (see [Shard-summed scoring](#shard-summed-scoring)). Every execution backend parallelizes across
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

## Numerical identity and stored work

Before keys or resume are evaluated, the executor binds an immutable
`transport_realization` record to each case:

```json
{"schema": "pyrite.transport-realization.v1", "core": "per-electron"}
```

This admitted realization uses CPU per-electron transport and CPU spectrum
scoring, including when a GPU is available in the coordinating process.

`ShardPolicy(scoring="gpu")` binds the device-scored variant instead:

```json
{"schema": "pyrite.transport-realization.v1", "core": "per-electron", "scoring": "gpu"}
```

- **Transport** is unchanged: CPU per-electron shards. They run on any worker,
  including a CPU-only one.
- **The join** is submitted with the Dask resource `GPU=1`, so it needs a
  worker started with `--resources GPU=1` (for example a #1386 GPU worker), and
  scores the spectrum phase on that process's CuPy backend.
- **No silent CPU fallback.** A GPU out-of-memory error raises instead of
  rescoring on CPU. A CPU-only process refuses the spectrum phase of a
  device-scored case before it starts.
- **Exactness.** A device-scored result equals the unsharded marked run with
  device scoring, bit for bit on the same device. It is a different
  realization from the CPU-scored one, and its distinct key keeps the two from
  sharing artifacts.

The CPU record stays implicit (two keys, the original pickle state), so every
CPU-marked key and stored artifact is unchanged.
Conflicting explicit transport-core requests or environment pins fail early.
An ordinary serial run of the same marked case reproduces the same numerical
realization. Shard size, block size, worker count and scheduler placement do
not participate in its final identity.

The marker is part of the canonical case, so existing independent validators
use `case_content_key` directly. Simulation results retain the prepared case
and its canonical digest. A case without the marker retains its historical
mapping, key and default execution behavior; it cannot reuse a marked artifact
or resume into a marked checkpoint. Dataset identity adds the marker only for
this opt-in mode. Checkpoint manifests also record it when no dataset identity
was supplied, and reject mismatched live records, metadata-only completion,
and checkpoint-to-CAS seeding before adoption. The marker survives component,
low-retention and recompute paths.

Shard plans use `pyrite.dask.shards.v2` and validate their realization against
their staged cases. Old experimental plans and unmarked prototype outputs are
not adopted automatically. Existing files are preserved. An unmarked prototype
final cannot be distinguished from an ordinary final by its key alone; keep
known mixed prototype stores separate from ordinary production stores rather
than assuming their provenance can be recovered.

## Shard-summed scoring

Opt-in (#1395): `ShardPolicy(reduction="shard-sum")`. A joined case scores its
whole population in one process, which is almost all of its wall time: in
every case measured for #1387, transport was under 3 % of it. Once the
population-wide choices are frozen, the line, characteristic and continuum
estimators add over electrons, so each shard scores its own rows.
`runner/shard_scoring.py` owns the three steps; the executor runs each as a
Dask task.

| Stage | Task | Reads | Publishes |
|---|---|---|---|
| transport | one per shard | case | raw rows (as above) |
| plan | one per case, after every shard | rows of every shard for a `line_grid_policy` axis (streamed `BlockGridEvidence`); only the shards reaching the selection prefix for an automatic reflection selection | `ShardScoringPlan`: axis, grid record, selected set, digest |
| score | one per shard, after the plan | the shard's rows and the plan | `BlockSpectrumScorer.partial()`: totals, audit sums, diffraction depth weights, fallback rows, coverage records, counts |
| merge | one per case, after every score | the plan and every spectrum payload | the case artifact |

The plan lives under `shard_plan_key(case)`, each spectrum under
`shard_spectrum_key(case, start, stop, plan digest)`. The digest covers the
axis and the selected set, not the grid record, whose speed-cache telemetry
differs between processes. A rerun skips every published stage; the merge
removes rows, spectra and plan unless `keep_shards=True`.

**Exactness.** Each shard commits one block, and the merge absorbs the
partials in shard order exactly as a block scorer commits blocks. The result
therefore equals live streamed scoring (`run_case_streamed`) with
`block_electrons = shard_electrons`, bit for bit, and is bitwise repeatable for
a fixed shard size whatever order the shards were scored in. Against the joined
reduction (`run_case` or the #1387 join) it carries the streamed contract, only
FP64 summation order changes:

- continuum and characteristic spectra are sums of non-negative per-segment
  terms, so a reordering moves each bin by at most `2 γ(n−1)` of the bin (`n`
  the terms in it, `γ(k) = k u / (1 − k u)`); the merge of `K` shard sums adds
  `(K − 1) u`. Measured on the regression case (hopg 30 keV, N = 13, K = 2–5):
  at most 8 `u` per bin;
- line spectra pass through the windowed far-tail reduction (#1191), whose
  rounding is a fraction of the peak, not of the bin: measured at most
  2e-14 of the peak;
- the tests hold both to the streamed tolerance, `rtol = 1e-11` plus
  `atol = 1e-14` of the peak.

The two population-wide transport steps the join redoes are skipped, as
streaming skips them (no shard-summed estimator reads bunch timing or completed
hard-photon directions).

**Identity.** The executor binds
`{"schema": ..., "core": "per-electron", "scoring": "cpu", "reduction": "shard-sum", "shard_electrons": K}`.
The shard size fixes the summation order, so it is part of the identity:
a shard-sum result never shares a key with a joined one or with another shard
size. `run_case` of a marked case runs the same plan, score and merge in
process (bit for bit the Dask result); joined spectrum scoring
(`_spectrum_with_recovery`, the #1387 join) refuses it.

**Supported cases** are those live streaming supports (`streaming_refusal(live=True)`)
with a fixed count and no line roulette (its per-electron line state spans the
population); others are refused before staging.

**Open (#1395).** The plan stage is serial: an automatic selection scores
every candidate shell over the prefix shards in one task (about 1400 s of the
#1320 case at N = 8 on CPU), and a policy grid pass reads every shard. A
refused default measured bandwidth is not extended and rescored as the joined
route does (#1243); the merge raises the refusal.

## What crosses the network

Joined reduction: raw rows, not per-shard spectra. Spectra of disjoint electron blocks add, but
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
  `transport_core="per-electron"`), the CUDA transport core (GPU transport shards are out of scope;
  `scoring="gpu"` moves only the join to a GPU), adaptive electron counts, secondary cascades, channeling, transition
  radiation, photon transport, GDF beams, and observation directions.
- The equality is with the per-electron run, not the CPU default (lockstep),
  which is a different realization of the same distribution.
- With the joined reduction the spectrum phase is serial at the join, so the
  speedup is bounded by the transport share of the case wall time. For
  line-dominated cases (the #1320 thick-target case: transport about 3 s,
  spectrum phase hours on CPU and minutes on one GPU), `scoring="gpu"`
  shortens the join, and `reduction="shard-sum"` scores the shards in parallel.
- The join holds every row, as the unsharded run does.

