# Live streamed scoring

Opt-in (#611). The public, experimental switch is
`KernelTuning.stream_block_electrons` on `pyrite.simulate` (#865; how-to:
[Stream spectra in electron blocks](../../guides/streamed-scoring.md)); every
other `transport_tuning` caller and batch executors refuse it. Internally,
`pyrite.montecarlo.runner.streamed_scoring.run_case_streamed(case, block_electrons, transport_core=...)`
transports a case in electron blocks, scores each block as soon as it finishes,
and releases it. No joined trajectory is ever held, so peak memory follows the
block size rather than the electron count. `run_case` (full trajectory, then
spectra) remains the default and the reference.

## How it works

- Transport reuses the electron-block driver (`runner/block_transport.py`).
  Every random draw is addressed by `(seed, electron, draw)`, so a block draws
  exactly its slice of the full run. `_transport_case(block_consumer=...)`
  hands each block, with global electron indices, to the consumer instead of
  joining it, and returns only the additive transport counts.
- Scoring uses `BlockSpectrumScorer`, the same whole-electron-block reduction
  that `stream_spectrum_from_artifact` applies to saved trajectories. Each
  eligible estimator is a track-length sum normalized by the fixed requested
  electron counts (`Ne` for lines, `Ne_brem` for characteristic and continuum),
  so block sums add. Only the floating-point summation order differs from the
  joined path.
- Live CUDA transport requests resident segment rows wherever
  `case_driver._resident_transport` admits them (#914). Continuation rounds
  join on the device, and the shared staging helper accepts device arrays
  directly. The block driver releases the pool after scoring returns and the
  raw rows go out of scope.
- A resident allocator OOM during transport or scoring replays only that
  electron block with host rows, using the same counter-addressed streams and
  table range. Spectra, row counts and line-coverage records commit only on
  success, so a late scoring failure cannot double-count earlier blocks.
  Host-memory admission precedes replay; a failed host attempt propagates.
  Characteristic scoring additionally recovers allocator OOMs within the
  current block (#915): halve the failed segment-row range, retain successful
  partial sums, and reuse the learned row ceiling for later blocks. Continuum
  recovery splits only escape-geometry evaluation, then concatenates the
  ordered geometry arrays before the original full-population reduction. This
  preserves its fused float32 summation order. Global electron IDs and the
  original population denominator stay fixed.
  Retry count follows the existing GPU OOM policy; exhaustion propagates to
  the host-block replay above. CXR retains complete electron histories and its
  existing spectrum chunks; TR also retains complete histories. Neither goes
  through the row splitter. There is no streamed CPU fallback.
- Streaming skips the two population-wide host steps that the joined block path
  redoes after the join. Bunch-centroid timing is skipped because the eligible
  estimators are invariant to a per-electron time offset, which is a common
  phase within one electron's flight groups. Hard-photon direction
  re-completion is skipped because the coupled BremsLib continuum scores
  expected values on the tracks and never reads completed photon directions.
  Each block's own completion is skipped too (#864): blocks run with the
  internal `_complete_hard_photons=False`, which omits the direction and recoil
  fields. Pair cascades are the exception: conversion reads the directions, so
  their blocks complete them, keyed by lineage (#923). Completion draws from its own generator, so transport rows and
  counts are unchanged bit for bit. If a future consumer needs them, the completion generator can be carried
  across ordered blocks, or `advance()`d by twice the earlier photon count;
  both reproduce the single-call draws exactly.

## Eligibility

Supported: incoherent line `spec`, characteristic lines and the EEDL or BremsLib
continuum, including coupled BremsLib radiative transport, on the per-electron
or CUDA core, with a fixed electron count and either a final line grid or a
supported non-windowed `line_grid_policy` (see [Resolved line grids](#resolved-line-grids)).

`streaming_refusal(case, live=True)` refuses the following before any transport:

| Case key | Reason |
|---|---|
| `coherent_emission`, `temporal_profile`, `photon_transport`, bunch `transition_radiation` | Need every segment or history at once (shared with artifact streaming). |
| windowed `line_grid_policy`, `_diagnostic_line_grid` | Window seed providers and the diagnostic derivation read the whole transported population. |
| `adaptive_precision` with `reflection_selection` or `batch_means_band_eV` | Automatic reflection selection and per-bin batch means read every realized segment (`adaptive_refusal`; prefix selection waits on #1281). Plain adaptive cases stream; see [Adaptive cases](#adaptive-cases). |
| `transition_radiation` | Electron-block transport refuses it; artifact replay still streams per-history TR. |
| `channeling`, `gdf_source` | Refused by electron-block transport. |

The lockstep core shares one generator across electrons and cannot run in
blocks; pin `transport_core="per-electron"` or `"cuda"` (`"auto"` can resolve
to lockstep at small electron counts).

### Resolved line grids

A `line_grid_policy` grid is resolved from the whole line population before
any line is scored, so a streamed policy case runs two transport passes over
the same electron blocks (#1001). Primary streams are addressed by
`(seed, electron, draw)`, so both passes see the same histories. Large
grid-pass evidence spills to disk (below).

1. **Grid pass.** `runner/streamed_line_grid.py::BlockGridEvidence` keeps,
   per block, only what policy resolution reads along the case direction: the
   valid sinc `(width, t_L^2)` samples of every line row (16 bytes each) and,
   for the `resonance-population` bandwidth, every production line's
   `(energy, width, weight)` (three backend reals each). It then calls the
   joined resolver (`cxr_grid._resolve_policy_grid`) with that evidence in
   place of segments, so the grid, its record and its speed-cache entry are
   the joined path's own.
2. **Scoring pass.** Transport again and score each block on the resolved grid,
   as for a final grid.

A speed-cache hit (`cxr_grid.cached_policy_cxr_grid`) needs no trajectories,
so it skips the grid pass. An adaptive policy-grid case runs this grid pass
after its monitored pass ([Adaptive cases](#adaptive-cases)). The output's private `_streamed_line_grid_pass`
records whether the pass ran and how many evidence bytes it held.

Equality: the sinc samples of consecutive whole-electron blocks concatenate to
the joined samples, so the sinc spacing, and with it the uniform grid, is
exact. `resonance-local` seeds are integer coverage counts. The
`resonance-population` stop search sums line masses per collected chunk, and
chunk boundaries follow the blocks, so its sums can differ from the joined
path's in the last ulp. The stop is rounded up to a whole 100 eV step, so the
grid differs only if a tail ratio lands within rounding of the truncation
limit. The truncation audit and `line_grid_resolved` record match the joined
path to summation order.

Memory: segment rows still follow the block size, but the grid-pass evidence
grows with the electron count. Above `GridSpillPolicy.threshold_bytes`
(default `default_spill_threshold_bytes()`, #1453: a quarter of
`host_memory.host_headroom_bytes()`, floor 2 GiB, 2 GiB if unreadable; evidence
held, or projected over the run from the blocks so far), `runner/line_grid_spill.py` moves it to a per-run scratch directory
under the output `cache/line-grid-spill/`, and later blocks are written there
as they arrive (#1141). Host memory is then bounded independent of N; disk is
not. `KernelTuning.stream_spill_dir` / `stream_spill_mib` (#1283) set the
policy's directory and threshold from `pyrite.simulate`
(`transport_options.stream_spill_policy`); unset, the default applies.

- **Blocks.** Each block's sinc samples are two raw arrays, written before the
  block commits, so a block replayed after an allocator OOM is never counted
  twice. Production lines are appended to one raw file (`LineFile`, truncated
  back on a failed block) and served after transport as views of a single
  read-only memory map, in the in-memory block layout. One map per array ran
  out of file descriptors at 64 h-BN blocks (job 13610).
- **Selection first (#1453).** For float64 populations of at least
  `SELECT_MIN_ROWS` rows, `spectrum/spacing_selection.sinc_spacing_by_selection`
  finds the 1 % weighted quantile without sorting: a strided sample picks two
  pivots, one pass sums the weight below the lower pivot and keeps the rows
  between them, and only those are sorted. It reads fixed-size chunks of the
  row stream, so held blocks, a joined array and spilled runs (any block
  layout) give the same bits, and a spill needs no sorted copy on disk. The
  reference's float summation error bounds its quantile index to a weight band
  of `5 * rows * u * limit * total`: the target and the prefix sums near it are
  about `limit * total`, so their rounding scales with the limit. A first
  version omitted the limit factor; on cleo_test (26M-118M distinct widths,
  limit 0.001) that band spanned thousands of rows and selection always fell
  back to the sort (qlmc jobs 15977/15979). The quantile at both band edges is computed, and
  when they are the same width the spacing equals the stable sort's bit for
  bit. When they are adjacent distinct widths the spacing is returned only if
  they differ by at most `SPACING_RTOL = 1e-6` (relative), so it is within
  1e-6 of the sort's, always. Ties, float32, an unclosed bracket or a wider
  jump return None and the exact sort below runs. The aliased fraction is
  summed in another order, so it agrees to rounding (absolute <= 1e-12 in the
  tests), not bitwise. Admission still charges the sort's worst case
  (`sinc_spacing_sort_bytes`) because the fallback may run.
- **Exact quantile (fallback).** `external_stable_sort` distributes rows into value
  buckets at sampled pivots, each bucket in row order, and sorts every bucket
  that fits `SORT_ROWS` with the same stable argsort; larger buckets recurse.
  Equal widths share a bucket, so the result is the global stable order
  whatever the pivots, and pivots inside `(min, max]` make every bucket a
  strict subset. `sorted_sinc_spacing` then reads the sorted rows chunkwise,
  carrying the running sum into each chunk's first element, so the cumulative
  sum, quantile and aliased weight are the in-memory reduction's bit for bit
  (`tests/montecarlo/transport/test_streamed_line_grid_spill.py`, including
  ties and single-valued populations).
- **Disk.** Peak is about twice the sinc evidence (raw blocks, then buckets
  and sorted output) plus the lines. Each spilled block re-projects that peak
  over the run and raises `LineGridSpillError` when the file system cannot
  hold it; the projection starts at the first block, so a short disk refuses
  before the bulk of the transport.
- **Cleanup.** `resolve_streamed_line_grid` removes the directory in a
  `finally`, so success, errors and `KeyboardInterrupt` leave nothing. A
  killed process cannot, so each new store prunes this host's directories
  whose process is gone.

Windowed policies are refused because their seed providers read the whole
population. A `sinc-oversampled` axis over its point budget refines
node-locally around every production line (#1257). Whether it is over budget
is known only from the final sinc spacing, so under that resolution the grid
pass always keeps the production lines (three reals each, spilled with the
sinc samples), whatever the bandwidth (#1280). The joined resolver then builds
the same axis from them; the interval union still holds per-line arrays in
memory, as on the joined path. Mixed transport scores no lines and keeps none.

### Transport activity

In-process transport keeps throttled activity reports within each chunk, but
completed chunk boundaries bypass the throttle. On an adaptive stop, the final
transport report uses the realized population, including a stop inside a chunk.
Before population-wide grid resolution and reflection selection, the activity
changes to `spectrum`. The progress JSON and `pyrite remote job status` show
that phase and clear the transport history counter; spectrum work can continue
without appearing to be stalled transport.

A case may transport its population several times (see "Adaptive cases"), so
each full-population pass names itself (`runner._announce_transport_pass`):
`stopping` (the monitored pass, counted against `max_histories`),
`energy-range replay`, `grid`, `scoring` or `cascade`. Reports then carry
`transport_pass` and its 1-based `transport_pass_index`, and the dashboard
shows `pass N · <label>`. A pass after `stopping` reruns the realized
population, so the adaptive panel marks the stopping decision as final.

### Adaptive cases

Adaptive cases stream since #1280, with memory bounded by the chunk size
instead of `max_histories`. `run_case_streamed` runs the monitored adaptive
transport with a block consumer (#1397,
`runner/streamed_adaptive.py::run_adaptive_streamed`). The stopping monitor
reads each chunk, and the driver's stop-inside-a-chunk and brem-freeze
replays (`[start, cut)`) and the energy-range replay run as on the retained
route. The consumer sees only the chunks the driver keeps, which are the
realized population. Counter-addressed streams make that the population the
retained route scores, so the result equals retained adaptive within
streamed-vs-joined summation order, and `adaptive_sampling` is the same record
(`tests/montecarlo/transport/test_streamed_adaptive.py`,
`test_adaptive_cascades.py`). `adaptive_route` picks one of two routes:

| Route | Case | Monitored pass | Transports |
|---|---|---|---|
| `single` | explicit grid (no `line_grid_policy`) | scores the kept chunks | 1 |
| `measure` | `line_grid_policy`, cascades (#1240), line roulette | measure-only (#1280) | 2, or 3 with a grid pass |

- **Single pass.** The realized counts are unknown until the stop, so blocks
  are scored at a provisional power-of-two count `L >= max_histories`. Every
  estimator's `electron_limit` passes every kept row, and its `sum / L` is
  exact. At the stop, `BlockSpectrumScorer.result(population=(Ne, Ne_brem))`
  multiplies the line totals by `L / Ne` and the brem-population totals
  (continuum, characteristic, DBR, Kossel) by `L / Ne_brem`, which differs
  from scoring by `Ne` only in the final rounding. A chunk past the brem
  freeze scores lines only; no kept chunk straddles the freeze. An automatic
  selection reads `min(Ne_brem, SELECTION_HISTORIES)`: when the prefix is
  complete before the stop, the brem population is frozen or larger than the
  prefix, so the provisional count stands in for it exactly. A prefix still
  held at the stop (`Ne <= SELECTION_HISTORIES`) is selected and scored at
  the realized counts. Only explicit grids qualify: a policy grid needs the
  whole population, and its truncation audit and the line roulette size
  per-electron state by the line population.
- **Coalesced chunks.** Adaptive chunks are small (`AdaptiveChunkPlan`, about
  2,000 for the case below), so the sink joins kept chunks with
  `merge_electron_blocks` into blocks of at least the fixed route's
  `block_electrons` before scoring. It scores early when the next chunk lies
  across the brem freeze, and at the end of the pass. On the 16 GiB RTX 5080
  (`cleo_sub_perf1` HOPG 500 nm, 300 keV, 15°, explicit grid, 55.1M realized
  electrons) the single pass scoring each chunk alone took 126.4 / 122.9 s
  against 100.1 / 94.2 s for the measure route (job 15605). Coalesced, it took
  84.1 / 82.6 s against 94.8 / 96.7 s, 11-15 % faster, with no OOM recovery
  (job 15613). Spectra differ from the measure route by at most 2.8e-11 of
  the peak, because the blocks are partitioned differently for the fp32
  device sums. Brem arrays are equal, and the selected reflection set is
  identical.
- **Policy grids stay on the measure route.** The grid needs the whole
  population, and its speed-cache key reads the realized `Ne`/`Ne_brem`, so
  a cache hit is known only at the stop. Collecting the evidence inside the
  monitored pass (#1397) was tried and measured on the 16 GiB RTX 5080
  (`cleo_sub_perf1` HOPG 500 nm, 300 keV, 15°, 55.1M realized electrons; jobs
  15587, 15589, 15595, 15598). After #1396 a full transport costs less than
  the folded evidence: the folded route was 8-10 % slower on a fresh run
  (2 transports vs 3) and 57 % slower on a cached rerun (2 vs 2). Overlapping
  it with the monitored transport was 5 s slower again, because the adaptive
  chunks are small and the GPU is already busy with CXR. A cache hit
  therefore still runs two transports and a fresh case three.
- **Commit-once.** Each pass calls the consumer's `begin_pass(monitor)`
  (`transport_case_blocks`), which starts a fresh scorer,
  so an energy-range replay never counts twice. A scoring allocator OOM rolls
  the scorer back (`snapshot`/`restore`) and rescores a host copy of the
  chunk once (`commit_once`).

The output's private `_adaptive_route` names the route (`scoring`) and its
full-population `transports`, counting each energy-range replay pass.
`streamed_adaptive.MONITORED_PASS = False` restores the #1280 measure-only
route for explicit grids too, for a same-revision comparison.

Scans (`run_case`, `run_cases`, `pyrite run`) select this route
automatically. A retained adaptive transport projects its kept bytes at
`max_histories` after every chunk (#1279, device rows included). When three
copies exceed host headroom it raises `StreamedAdaptiveRoute` instead of
refusing, and `_spectrum_with_recovery` reruns the case streamed in blocks of
about 2 GiB of rows, with a `RuntimeWarning`. Resident (device) rows also
project their device peak (#1274): the join's two copies, or the joined rows
plus the spectrum chunk budget (`spec_budget_mb`), whichever is larger. When
that growth exceeds `device_headroom_bytes()` the case streams too, before
the device runs out of memory. Rows already on the host (`keep=False`, or
after #1269 per-block recovery) skip the device check. Cases `streaming_refusal`
refuses keep the retained route and its admission refusal, and so does
`PYRITE_STREAMED_ADAPTIVE=off`.

### Cascades

Secondary, pair and positron cascades stream since #923. Their random streams
are lineage-addressed (`transport/lineage.py`, case key
`cascade_rng_model = "lineage-addressed-v1"`). A root's key is its global
primary stream key. Every daughter, pair conversion, annihilation clock and
emission, and hard-photon direction derives from the parent's lineage key,
a process namespace and a parent-local event ordinal. The ordinal is counted
before launch or energy filtering. Dense `track_id`s remain array indices
only, so a primary block runs each history's whole cascade exactly as the
joined run does. Every pass of a cascade builds its tables over one range:
the block driver's population range, else the same range sampled from
`(seed, e)`.

Contracts of the streamed cascade path:

- Rows are scored per block with global `electron_id` (the primary history);
  `Ne`/`Ne_brem` denominators and the transport counts stay primary-only.
- Nested `secondary_tracks` and `pair_production` metadata, `track_id` and
  `parent_id` stay block-local. Scoring never reads them across blocks, and
  the block is released after scoring. Compare histories across runs by
  `secondary_tracks["lineage_key"]`, not by dense IDs.
- The secondary-track budget is run-wide: `1000 N` descendants over the case.
  A block may launch only what committed blocks left, so a run fails iff the
  joined run would, whatever the block size; a replayed block never counts
  twice.
- Joined (non-streamed) block transport of cascades stays refused: merging
  nested metadata, cross-references and generation order across blocks is not
  implemented.

Fixed-seed cascade realizations changed with #923. Cases recorded before it
carry no `cascade_rng_model`; the runner refuses to transport them rather than
replay them under the new scheme.

## Equivalence evidence

`tests/montecarlo/transport/test_streamed_scoring.py` and
`tests/montecarlo/cascades/test_coupled_radiative_runner.py::test_streamed_scoring_matches_the_joined_coupled_run`
compare streamed and joined spectra at `rtol=1e-12` over several block sizes,
with unequal line and continuum populations and with coupled BremsLib straggled
transport. `tests/montecarlo/transport/test_streamed_line_grid.py` compares a streamed
`sinc-nyquist` and a `resonance-local` policy grid with the joined path's: the
grid bit for bit, spectra at `rtol=1e-12`, provenance records at summation
order, and a cached grid skipping the grid pass.
`tests/montecarlo/transport/test_streamed_cascades.py` does the same
for shell cascades, uncoupled (to generation 2) and coupled, at block sizes 1,
a nondivisor, `N` and above `N`, with `Ne` above and below `Ne_brem`.
`tests/montecarlo/cascades/test_cascade_lineage.py` shows that blocked shell,
pair and positron cascades reproduce every joined history's rows, launches,
pair events and annihilation photons bit for bit on CPU, including with energy
spread and per-electron cutoffs. Transport counts and segment totals agree exactly.
`tests/montecarlo/transport/test_streamed_adaptive.py` compares the adaptive
routes with the retained route and the #1280 measure-only route: a stop inside
a chunk, a brem freeze and the energy-range replay on the single pass; an
automatic selection whose prefix completes at the stop, before the brem
freeze and after it; and a `resonance-population` grid at block sizes 7, 13
and 1000, cache miss then hit, whose grid, record and spectra equal the
separate grid pass's bit for bit. It counts the monitored and fixed passes,
and a scoring OOM injected after the scorer mutated its state commits the
chunk once.

## Memory and wall time

`checks/streamed_scoring_memory.py` runs one arm per process on the lab GPU
box (RTX 5080, 16 GiB): 300 keV electrons on a 10 µm slab tilted 30°, coupled
BremsLib midpoint transport, CUDA core, float32 scoring, 400 line and 299
continuum bins, seed 1. Host peak is `ru_maxrss` (about 600 MiB before the
run). Device figures are CuPy pool bytes above the pre-run level: *reserved*
includes cached free blocks, *used* is live arrays. Streamed arms release
cached pool blocks after each block. Jobs 2991 (Si) and 2992 (W), code
`300eb90ba`; single runs.

**Si** (about 74 segments per electron):

| Arm | N | Block | Segments | Host peak (MiB) | Reserved / used VRAM (MiB) | Wall (s) | Transport / scoring (s) |
|---|---|---|---|---|---|---|---|
| joined | 16,000 | | 1.18 M | 1391 | 3182 / 622 | 1.6 | 0.8 / 0.8 |
| joined | 64,000 | | 4.74 M | 2220 | 6193 / 2901 | 3.4 | 1.4 / 2.0 |
| streamed | 16,000 | 4,000 | 1.18 M | 1219 | 458 / 237 | 1.9 | 1.0 / 0.8 |
| streamed | 64,000 | 4,000 | 4.74 M | 1221 | 472 / 317 | 4.8 | 2.6 / 2.0 |
| streamed | 64,000 | 1,000 | 4.74 M | 1165 | 256 / 180 | 11.6 | 6.9 / 4.2 |
| streamed | 64,000 | 16,000 | 4.74 M | 1465 | 3236 / 730 | 3.5 | 1.5 / 1.9 |

**W** (about 1950 segments per electron):

| Arm | N | Block | Segments | Host peak (MiB) | Reserved / used VRAM (MiB) | Wall (s) | Transport / scoring (s) |
|---|---|---|---|---|---|---|---|
| joined | 4,000 | | 7.76 M | 2902 | 9364 / 4602 | 8.2 | 3.0 / 5.2 |
| joined | 16,000 / 32,000 | | | | out of device memory | | |
| streamed | 16,000 | 4,000 | 31.3 M | 3282 | 9599 / 4706 | 28.8 | 9.8 / 18.6 |
| streamed | 32,000 | 4,000 | 62.3 M | 3282 | 9967 / 5075 | 56.1 | 18.3 / 37.1 |
| streamed | 64,000 | 4,000 | 124.6 M | 3284 | 10263 / 4965 | 109.8 | 35.3 / 72.9 |
| streamed | 64,000 | 1,000 | 124.6 M | 1702 | 5690 / 1342 | 170.9 | 91.8 / 75.6 |
| streamed | 64,000 | 16,000 | | | out of device memory | | |

At a fixed block size, host and device peaks stay flat as N grows 16-fold.
The joined W run cannot score 16,000 electrons on this card: characteristic
scoring exhausts the spectrum-phase OOM recovery. Streaming scores
64,000. The peak follows block size × segments per electron, so an electron
block is not a fixed memory budget across materials: 16,000-electron W blocks
also run out of device memory. Reserved pool bytes run about twice the live
bytes in every arm.

Where the joined path fits, it is now faster. Small blocks underfill the GPU
during transport (Si 1,000-electron blocks: 6.9 s against 1.4 s joined), and
each block adds a fixed scoring and release overhead (about 0.13 s for Si).
Scoring itself is linear in segments. Before #286 removed an O(segments²)
coverage copy and #859 made the line optical-data screen opt-in, the joined
scoring phase grew superlinearly, so older streamed-vs-joined wall-time
comparisons (jobs 2932–2945) overstate streaming.

Carrying a learned `seg_capacity` across blocks (a prototype `capacity_hint`)
gave identical wall times and bitwise-identical output on Si and W at
1,000- and 4,000-electron blocks, and was removed: since #860, blocks of
1,000 histories or more already take bounded continuation.

## Block sizing on large-memory GPUs (#907)

`checks/streamed_block_sizing.py` (one arm per process; sbatch:
`checks/streamed_block_sizing.sbatch`, `checks/streamed_block_sizing_ncu.sbatch`)
measures how block size, scratch budget and `electron_batch` set the
continuation population, and what that does to kernel time and memory. Scene: W
1 MeV, 10 µm slab tilted 45°, scalar detector at 90°, 251-bin CXR window, one
reflection family, seed 7, `brem_histories = N // 16`, CUDA core, default
mixed precision, `kernel_threads=32`, 256 continuation rows. W keeps 1,121.6
rows per electron (294.0 M rows at 262,144 electrons). Kernel time is the sum
of CUDA-event brackets around every continuation launch. Each block's
`scratch_mib` and `electron_batch` were sized so the first round holds the
whole block. Devices: Molab RTX PRO 6000 Blackwell Server Edition (188 SMs,
96 GB, 160 GB host) and the lab RTX 5080 (84 SMs, 16 GB; jobs 3680–3681).

**Kernel throughput saturates at about six warps per SM.** Time per history
per round in the first, full-population launch:

| Histories | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 | 131,072 | 262,144 |
|---|---|---|---|---|---|---|---|
| RTX 5080 (µs) | 7.8 | 4.8 | 3.98 | 4.09 | 3.86 | | |
| RTX PRO 6000 (µs) | | | 2.55 | 2.09 | 1.87 | 2.01 | 1.98 |

Below saturation every launch costs a floor of about 28 ms (5080) or 35 ms
(6000), however few histories it holds. Nsight Compute on a full-population
16,384-history launch on the 5080 shows why throughput stops there: the FP64
pipe is 86.5% busy on average (100% on the busiest SM), while only 4% of issue
slots are used. Occupancy is limited by registers (168 per thread, 12 warps
per SM, 25% theoretical); the launch achieves 5.9 warps per SM. Warps have
9.5 of 32 threads active per instruction, and the stalls are short-scoreboard
waits. More resident warps cannot help once the FP64 pipe is saturated. The
saturated rate scales with SM count (6000/5080 ≈ 2.0 against an SM ratio of
2.24), so the population that fills a card is about 6 × 32 × SMs: about
16,000 histories on the 5080 and 36,000 on the 6000, not the 1,536-thread
resident ceiling (129k and 290k).

**Past saturation, larger blocks still cut kernel time by removing pass
tails.** Each pass ends in 25–30 nearly empty rounds at the launch floor, so
kernel time falls with the number of passes. At 262,144 electrons:

| Block | Passes | 6000 kernel (s) | 5080 kernel (s) |
|---|---|---|---|
| 4,096 | 64 | | 43.8 |
| 8,192 | 32 | | 25.0 |
| 16,384 | 16 | 16.4 | 15.0 |
| 32,768 | 8 | 9.6 | 10.2 |
| 65,536 | 4 | 6.0 | 7.4 |
| 131,072 | 2 | 4.4 | |
| 262,144 | 1 | 3.45 | |

A 524,288-history population on the 6000 (one pass, 64 GiB scratch) was
slower: 10.4 s of kernel, with a 5.7 s first launch, and a 92 GiB host peak.
64 threads per block and 1,024 continuation rows changed kernel time by less
than 5%.

**Wall time at code `67d54de6c` (after #904).** RTX PRO 6000, 262,144 electrons;
the split arm adds device synchronizations at every seam:

| Block | Wall (s) | Scoring (s) | Kernel (s) | Host peak (GiB) | VRAM peak (GiB) |
|---|---|---|---|---|---|
| 65,536 | 32.7 | 11.0 | 6.0 | 12.5 | 50.2 |
| 131,072 | 30.7 | 11.1 | 4.4 | 23.4 | 66.7 |

At the 65,536 block, `transport.assemble` (each block's device join and its
download to host) takes 9.6 s, more than the kernel. Scoring then uploads the
same rows again (1.2 s), and estimators take 7.0 s (lines), 2.8 s (continuum)
and 1.4 s (characteristic). cProfile puts 8.1 s in `_continuation_join._drain`.
Before #904, scoring took 62–69 s at every block size and hid the transport
gain: wall was 84–99 s for blocks of 16,384–131,072.

**Memory follows the block, not N.** Host peak is about 150 B per retained row
(about 168 KB per W 1 MeV electron in the block) on both cards: 43 GiB for a
262,144 block at both 262,144 and 524,288 electrons. Transport-only VRAM is
mostly the scratch budget (5080: 3.8, 7.5 and 9.2 GiB at 16k, 32k and 65k
blocks). Streamed scoring is the binding device limit: on the 5080, blocks of
16,384 or more electrons run out of memory in characteristic scoring
(`segment_escape_paths`, 12 GB pool cap), which has no block-level OOM
recovery. 8,192 fits at 11.3 GiB. The joined path at 262,144 electrons runs out
of memory on the 6000 in characteristic scoring (71 GB pool cap), at every
measured code version; streaming scores it.

**Equivalence.** Streamed vs joined at 32,768 electrons: lines within 2.3e-7
per bin relative on both cards; characteristic bitwise identical on the 6000
and within 1.8e-7 on the 5080; continuum bitwise identical. Streamed arms at 262,144 agree with each other to the same
bound across block sizes, threads and row quanta, and are bitwise identical
across the three measured code versions.

## Memory-target block sizing (#868)

`KernelTuning.stream_memory_mib` (how-to: [the guide](../../guides/streamed-scoring.md))
caps the CuPy pool below the target for the run: the target minus the device
memory in use outside the pool at the start and a 0.5 GiB allowance, at most 95%
of the target, with half the target as a floor for small targets. The cap is what bounds
device memory, because scoring adapts to the pool and its peak does not follow
row count. A 1,024-electron first block measures retained bytes per electron;
later blocks give retained rows a quarter of the target and follow the largest
rate seen. Unset `scratch_mib` and `electron_batch` scale with each block.

Re-measured on the RTX PRO 6000 (2026-11-05, `f316e1921` with #868's modules
overlaid), single runs, W 1 MeV / 10 um and Si 1 MeV / 10 um, 262,144 electrons:

| Scene, target | Wall | Driver peak | Blocks after the first |
|---|---|---|---|
| W, 16 GiB | 37.3 s | 15.65 GiB | 21,729 |
| W, 64 GiB | 29.9 s | 61.73 GiB | 86,918 |
| Si, 16 GiB | 5.0 s | 12.57 GiB | 261,120 |

That table's driver peak excludes the process's own CUDA context, so W at 16
GiB actually used about 16.19 GiB. A fixed 1.25 GiB reserve then fixed the RTX
PRO 6000 but left the RTX 5080 0.30-0.43 GiB over (job 3997), because the baseline
after importing CuPy is 1.32 GiB there against 0.55 GiB. Hence the per-run
reading. Whole-card peaks at `5a8c02ca7`, 262,144 electrons, single runs, 5 ms
sampling (RTX 5080: job 3999; RTX PRO 6000: Molab):

| Card | Scene, target | Wall | Whole-card peak |
|---|---|---|---|
| RTX 5080 | W, 4 GiB | 137.8 s | 3.73 GiB |
| RTX 5080 | W, 8 GiB | 43.8 s | 7.78 GiB |
| RTX 5080 | W, 12 GiB | 38.2 s | 11.85 GiB |
| RTX 5080 | Si, 4 GiB | 7.4 s | 3.72 GiB |
| RTX 5080 | Si, 8 GiB | 3.8 s | 7.88 GiB |
| RTX PRO 6000 | W, 16 GiB | 38.5 s | 15.91 GiB |
| RTX PRO 6000 | W, 64 GiB | 30.4 s | 62.27 GiB |
| RTX PRO 6000 | Si, 16 GiB | 5.4 s | 13.06 GiB |

W at 4 GiB took 61.6 s with the larger pool of job 3997, so a target close to the
overhead is much slower.

Fixed blocks at the same commit, W, measured with the #907 harness:

| Block | Wall | Scoring | Kernel | Host peak | VRAM peak |
|---|---|---|---|---|---|
| 16,384 | 28.3 s | 8.2 s | 16.8 s | 1.4 GiB | 27.7 GiB |
| 32,768 | 20.0 s | 8.0 s | 9.9 s | 1.3 GiB | 38.6 GiB |
| 65,536 | 21.7 s | 14.1 s | 6.1 s | 2.0 GiB | 57.6 GiB |
| 131,072 | 20.0 s | 14.6 s | 4.5 s | 2.7 GiB | 65.8 GiB |
| 262,144 | 23.2 s | 18.6 s | 3.5 s | 4.0 GiB | 65.8 GiB |
| joined | 34.1 s | n/a | 3.5 s | 58.3 GiB | 65.6 GiB |

Since #907, device-resident blocks cut the 131k host peak from 23.4 to 2.7 GiB
and its wall from 30.7 to 20.0 s, and the joined 262,144 W run now completes.
Wall is flat from 32k to 131k blocks, so a larger block is not faster here.

## Resident live CUDA blocks (#914)

Lab RTX 5080, implementation `16db8150c`, job 3686; the #907 W scene above
at 65,536 electrons (`Ne_brem=4,096`), seed 7, 8,192-electron blocks,
2,048 MiB transport scratch, 32 kernel threads and 256 continuation rows.
Each arm runs in a fresh process, warms the same 4,096-electron scene first,
and enables `PROBE_SPLIT=1`. Three host and three resident repetitions ran
in the order host/resident/resident/host/host/resident. Driver VRAM is sampled every 5 ms. Values below are
medians with observed ranges, not confidence intervals; split timings add
synchronizations and should only be compared with other split arms.

| Metric | Host blocks | Resident blocks |
|---|---|---|
| Wall (s) | 14.387 (14.366–14.532) | 11.887 (11.859–11.940) |
| Assembly (s) | 1.289 (1.235–1.312) | 0.117 (0.116–0.119) |
| Staging, `score.upload` (s) | 0.771 (0.766–0.846) | 0.039 (0.039–0.039) |
| Host peak (GiB) | 2.563 (2.563–2.602) | 1.184 (1.184–1.184) |
| Driver VRAM peak above warm baseline (GiB) | 9.117 (8.910–9.188) | 10.234 (10.168–10.234) |

Median wall time falls 17.4%. Device assembly avoids the block download;
the remaining staging time includes device casts to scoring precision.
Resident rows consume more VRAM while removing the large host block.
All six saved line, characteristic and continuum arrays are bitwise identical.
Job 3688 repeated the same protocol with 4,096-electron blocks on
`3edb0cc54` (identical implementation payload; added test coverage). Host vs
resident median wall time was 20.992 vs 18.026 s, a 14.1% reduction; observed
ranges were 20.851–21.070 and 17.851–18.052 s. Assembly was 1.492 vs 0.137 s,
and staging 0.974 vs 0.042 s. Median host peak was 1.855 vs 1.184 GiB and
VRAM peak 6.561 vs 6.957 GiB. These six spectra were also bitwise identical
at the same block size. Resident host peak stayed 1.182–1.184 GiB across
both block sizes, whereas host-block peaks rose from 1.855 to 2.563–2.602 GiB
when the block doubled. This establishes the removed row-dependent host
allocation over the measured range, not a bound for arbitrary scenes.

The probe supports a matched host baseline via `PROBE_HOST_SEGMENTS=1`:

```bash
PROBE_SPLIT=1 PROBE_HOST_SEGMENTS=1 uv run python checks/streamed_block_sizing.py streamed 65536 8192 2048 32 0 scratch/host
PROBE_SPLIT=1 uv run python checks/streamed_block_sizing.py streamed 65536 8192 2048 32 0 scratch/resident
```

Run each arm on an allocated remote GPU; the arguments fix the workload and
launch shape. Tests in `test_streamed_scoring.py` check resident CUDA blocks,
bitwise parity with host blocks, joined parity within the 16-float32-epsilon
summation allowance, and block-local replay after injected transport,
characteristic or continuum allocator OOMs. CPU tests exercise the same
recovery decisions and verify that raw rows are dead before pool release.

## Per-estimator scoring split (#872)

`checks/streamed_scoring_memory.py` times each estimator inside
`BlockSpectrumScorer.add` with device synchronizations (`score.*` phases).
Same setup as above, 20,000 electrons in 4,096-electron blocks, 1e5 Å slab
(10 µm), seed 1; jobs 3035 (before) and 3041 (after), RTX 5080, two reps each,
seconds:

| Case | Upload | Lines (`spec`) | Characteristic | Continuum | `add` total | Wall |
|---|---|---|---|---|---|---|
| W before (3035) | 0.73 | 9.25 | 14.6-14.8 | 2.1 | 25.7-25.9 | 37.6 |
| W after (3041) | 0.82 | 9.2-9.4 | 3.85 | 2.1 | 14.9-15.2 | 26.5-26.9 |
| Si before (3035) | 0.03 | 0.61 | 0.2-0.22 | 0.5 | 1.1 | 2.4 |
| Si after (3041) | 0.03 | 0.6 | 0.13-0.14 | 0.5 | 1.04-1.05 | 2.3-2.4 |

Joined W at 4,000 electrons: characteristic 3.09 s before, 0.85 s after (spectrum
phase 5.69 s to 3.59 s, wall 8.6 s to 6.4 s).

Characteristic scoring dominated W (57% of `add`) because it ran one set of
small array kernels per (chunk, line): about 90 lines per chunk, launch bound.
It now scores lines in groups against each chunk (bounded to 4 Mi matrix
elements per group). Per-element arithmetic is unchanged; the matmul and
segment-sum reduction orders are regrouped. In practice streamed spectra
before and after were bitwise identical on W and Si (`compare`, max relative
difference 0), so no tolerance change. The remaining W cost is the incoherent
line reduction (`spec`, 36%), owned by #286; the BremsLib continuum is 8%, the
host-to-device upload 3% and the per-block pool release about 2%.

## Characteristic preparation and launches (#1005)

`checks/characteristic_scoring_1005.py` adds synchronized timers and NVTX ranges
inside `mc_characteristic_spectrum` to the harness above: 300 keV, 20,000
electrons in 4,096-electron blocks, 400 line bins, seed 1, RTX 5080, two reps
(job 4020 baseline; 4022 prepared context; 4025 fused kernels; 4028 column sum).
Seconds; launches and copies from Nsight on a third rep:

| Case and step | Characteristic | Setup | Accumulate | Interpolation | Launches | H2D copies |
|---|---|---|---|---|---|---|
| W, baseline | 3.96-3.98 | 0.24-0.25 | 3.45 | 0.91-0.92 | 79,470 | 14,904 |
| W, prepared context | 3.43-3.45 | 0.08 | 3.07-3.09 | 0.53 | 76,143 | 454 |
| W, fused kernels | 2.85-2.87 | 0.08 | 2.49-2.51 | 0.06-0.07 | 14,977 | 68 |
| W, column sum | 1.53-1.55 | 0.08 | 1.18-1.19 | 0.06 | 17,320 | 68 |
| Si, baseline | 0.137-0.139 | 0.04 | 0.08 | 0.014 | 2,880 | 407 |
| Si, column sum | 0.100-0.202 | 0.02-0.13 | 0.058 | 0.003 | 1,646 | 38 |
| 1 um W on 10 um Si, baseline | 0.94-0.95 | 0.52-0.53 | 0.36 | 0.07 | not traced | not traced |
| 1 um W on 10 um Si, column sum | 0.35 | 0.10 | 0.19 | 0.006 | not traced | not traced |

Setup is per call, so its one-time part shows only in the first block: W spends
108 ms there and 38 ms in each later block before, 71 ms and 4 ms after. The
0.13 s Si setup is an intermittent one-time host gap (115 ms) before the first
CUDA call of the process, not a per-block cost.

W wall drops from 16.7 s to 14.2 s and `add` from 8.4 s to 5.9-6.0 s. Most of
the gain is not setup: interpolation and transmission were launch bound, and
CuPy's `sum(axis=0)` over the `(chunk, line group)` float64 yields took 0.79 s
of GPU time (1.38 s once the fused kernel fed it). A cuBLAS ones-vector product
reads the same array in about 0.03 s.

Every characteristic spectrum, streamed W, Si and layered W/Si and joined W at
4,000 electrons, stayed bitwise identical to the baseline through all steps.
The fused kernels are bit-for-bit by construction; the column sum reorders only
a float64 sum that accumulates into float32 line yields, so a rounding-level
difference is possible on other inputs. There is no model marker change.

The case-local preparation cache (`EmissionBatch.table_cache`) gains nothing
at 400 bins, because setup there is milliseconds per call. Job 4035 repeats the
measurement on 190,000 line bins (1-20 keV), 8,192 electrons in 2,048-electron
blocks, four calls, one rep each, with the cache on and with it withheld
(`C1005_NO_CACHE=1`). Seconds:

| Case | Characteristic, cache / none | Setup, cache / none | Accumulate, cache / none |
|---|---|---|---|
| W 10 um | 4.82 / 6.04 | 0.48 / 1.23 | 4.23 / 4.70 |
| Si 10 um | 0.72 / 0.43 | 0.23 / 0.20 | 0.47 / 0.22 |

On W the cache removes 0.75 s of setup, 12% of the uncached characteristic
time. The Si arm is confounded: the cached arm ran first in a cold job with no
warm-up, and its whole `add` took 22.9 s against 6.4 s for the uncached arm
that followed. It therefore says nothing either way about Si. Spectra with and
without the cache are bitwise identical on both. The cache stays: it clears
the 5% bar on W. At this grid size incoherent line scoring dominates the W wall
(126 s of 132 s), so the cache saves about 1% of W wall.

Spectral-window pruning is not implemented: on W at a 1e-6 in-window mass
tolerance it would keep 79 of 95 lines, and only the about 1 s of per-line
kernels scales with that count, about 1% of wall for an opt-in approximation.
This is an estimate from line counts and the kernel split, not a timed arm.

The remaining W cost is `pyrite_characteristic_line_yield` (0.59 s, float64
`exp` and `expm1`, because the escape paths are float64) and the float64
shell-response product (0.30 s).

## Transport/scoring overlap

Overlap (#862, `montecarlo/runner/stream_overlap.py`) is on by default for
CUDA streamed runs with a fixed block size (`stream_block_electrons`), no early
stop, and at least two full blocks of at least 2¹⁸ (262,144) electrons
(`AUTO_MIN_BLOCK`). `PYRITE_STREAM_OVERLAP=1` forces it on for any fixed-block
CUDA run, and `0` turns it off. The main thread transports block
`k + 1` while one worker thread scores block `k` on a non-blocking CuPy
stream. One block is in flight, so staging is bounded. Blocks are scored in
order by one consumer, so outputs equal serial streaming bit for bit. A device
OOM drains the pipeline and the serial loop resumes. Overlap engages only for
fixed block sizes with no early stop, so the #868 memory sizer and adaptive
stops run serially.

**With CXR lines (the default) overlap does not pay.** All runs below use the
#872 harness (300 keV, 10 µm slab, coupled BremsLib, seed 1) on the lab RTX
5080. Job 13787 measured Si 20,000 electrons in blocks of 4,096: serial wall
21.08 s, overlap 21.04 s, with spectra bitwise identical. The run was not
GIL-bound (GIL waits 0.02–0.07 s per thread). Transport took about 1.75 s of
the 21 s; scoring and reflection selection took the rest. Job 13824 measured
W 8,192 electrons in blocks of 4,096 at 90% GPU busy, with transport 4.7 s of
280 s. Overlap can hide only transport, so the ceiling is single-digit
percent on Si and about 1% on W.

The scoring lever went to bitwise-neutral GPU elementwise trims in batched
line scoring instead (see #862):

- amplitude-only `_cxr_amp_sq_total_core` unless PXR/CBS components or
  attribution are read;
- polarization sums seeded from their first term;
- fused segment-block prologue kernels.

Jobs 13854 and 13859 (three interleaved reps each) cut W 8,192/4,096 serial
streamed wall from 275.8 s to 226.6 s (−17.9%) and Si 20,000/4,096 by 9%.
Pool peak was unchanged. W line scoring has ulp-level run-to-run variance
(≤ 1.8e-16 relative, in one or two bins), and the base commit shows it too.
That is far inside the #611 1e-12 tolerance.

**Brem-only (`cxr_source="none"`) overlap pays at large blocks.** These runs
used a 96 GiB RTX PRO 6000 Blackwell on the same harness with
`STREAM_CXR=none`, measuring production wall (`STREAM_PHASES=0`) with serial
and overlap runs interleaved. Overlap spectra were bitwise equal to serial in
every run.

| Case (electrons / block) | Serial (s) | Overlap (s) | Δ | Pool, serial → overlap (GiB) |
|---|---|---|---|---|
| Si 20,000 / 4,096 | 1.80–1.85 | 1.84–1.90 | +2% | 0.14 → 0.19 |
| W 8,192 / 4,096 | 1.94–2.00 | 2.13–2.23 | +10% | 0.67 → 0.89 |
| Si 262,144 / 8,192 | 3.38–3.50 | 3.65–3.89 | +10% | 0.25 → 0.34 |
| W 262,144 / 8,192 | 6.80–6.92 | 6.59–6.82 | −2% | 1.3 → 1.6 |
| Si 2,097,152 / 131,072 | 4.71–4.78 | 4.06–4.24 | −12% | 1.3 → 1.7 |
| W 2,097,152 / 131,072 | 21.22–21.34 | 19.03–19.50 | −9.5% | 18 → 20 |
| W 2,097,152 / 524,288 | 20.27–20.80 | 18.28–18.54 | −10.4% | 71 → 80 |
| Si 16,777,216 / 1,048,576 | 25.02–25.40 | 17.59–17.75 | −30% | 11.0 → 11.6 |
| W 8,388,608 / 262,144 | 77.16–77.27 | 68.59–69.03 | −11% | 36 → 41 |

At small blocks, the per-block handoff and synchronization make overlap
slower. Gains need blocks of about 10⁵ electrons or more and many blocks.
For brem-only runs, the #868 sizer's largest block that fits is slower than a
modest fixed block (Si 2M as one block 4.8 s vs 4.75 s at 131,072; W 8M at a
64 GiB target 86 s vs 77 s at 262,144). W stays far from max(transport,
scoring) because both phases are GPU kernels on one device. Overlap costs
about one extra block of retained rows plus scoring scratch.

`checks/stream_overlap_brem_862.sbatch` repeated these on the 16 GiB RTX 5080
(jobs 14466 and 14477, three interleaved reps unless noted; min–max).
Spectra were bitwise equal to serial in every brem-only arm; the CXR-on arm
differs by at most 1.8e-16, the W line-scoring jitter above.

| Case (electrons / block, blocks) | Serial (s) | Overlap (s) | Δ (median) | Pool, serial → overlap (GiB) |
|---|---|---|---|---|
| Si 2,097,152 / 131,072 (16) | 4.44–4.53 | 4.16–4.29 | −4.0% | 1.32 → 1.67 |
| W 2,097,152 / 65,536 (32) | 26.57–26.71 | 25.70–26.04 | −3.2% | 9.1 → 10.3 |
| W 8,388,608 / 65,536 (128) | 102.9–108.9 | 99.07–99.58 | −3.8% | 9.1 → 10.4 |
| Si 524,288 / 262,144 (2) | 1.878–1.883 | 1.699–1.727 | −8.7% | 2.6 → 3.1 |
| Si 786,432 / 262,144 (3) | 2.259–2.314 | 1.988–2.015 | −11.8% | 2.6 → 3.1 |
| Si 4,194,304 / 262,144 (16) | 7.513–7.915 | 5.862–5.947 | −23.4% | 2.6 → 3.1 |
| Si 16,777,216 / 1,048,576 (16) | 24.00–24.45 | 16.87–17.29 | −29.7% | 10.7 → 11.6 |
| Si 2,097,152 / 262,144 (8), CXR lines on, 1 rep | 96.79 | 93.80 | −3.1% | 7.7 → 8.2 |

The default threshold follows from these: every arm at 2¹⁸ electrons or more
gained 7–30% (brem-only) on either card, every arm at 2¹⁶–2¹⁷ on 16 GiB gained
at most 4%, and 4,096–8,192 blocks lost 2–10%. Two blocks are enough. With
CXR lines on, overlap hides only transport (3.4 s of 92.8 s here), so its
gain is bounded by that share, but large blocks do not lose. The #868 memory
sizer (`stream_memory_mib`) still runs serially: its block sizes are chosen
one block at a time, and at a 12 GiB target it was never faster than overlap
at 262,144-electron blocks.

## Equivalence on CUDA

On CUDA, scoring runs in float32 unless `PYRITE_FP64=1`, so block sums
differ from the joined sum at float32 rounding. Relative to each spectrum's
peak, streamed and joined differ by at most 3.1e-6 for Si at 64,000 electrons
and 1.9e-5 across the Si block sizes. For W across block sizes, lines and
characteristic agree to 4e-7 and the continuum to 2.9e-5. With
`PYRITE_FP64=1` an earlier Si comparison agreed to 1.75e-15 (job 2937), which
confirms that the float32 differences come from summation order only.

Memory-target blocks split the continuum population differently from joined
runs, so W's float32 continuum differs by up to 3.6e-5 of its peak at 262,144
electrons. With `PYRITE_FP64=1` at 32,768 W electrons, memory-target blocks agree
with the joined run to 5.4e-15 of the peak and a whole-run block is bitwise
identical (2026-11-05).

## Resolved line grids: memory and wall time

`checks/streamed_line_grid_1001.sbatch` (jobs 12054 and 12055, code
`0ca247990`, single runs, CUDA core, float32 scoring) measured the two-pass
route of [Resolved line grids](#resolved-line-grids). The speed cache is
disabled in every arm, so each one resolves its grid. Host peak is
`ru_maxrss` (about 640 MiB before the run). Device figures are CuPy pool
reserved / used MiB above the pre-run level.

| Case | Arm | N | Block | Host peak | Device | Wall (s) | Grid (s) | Grid points |
|---|---|---|---|---|---|---|---|---|
| Si 300 keV, 10 µm, automatic policy | joined | 64,000 | | 1964 | 5965 / 2812 | 19.8 | 1.0 | 68,651 |
| | streamed | 64,000 | 4,000 | 1029 | 1471 / 1272 | 21.2 | 2.8 (pass) | 68,651 |
| | streamed | 64,000 | 16,000 | 1025 | 3364 / 1410 | 20.0 | 2.1 (pass) | 68,651 |
| h-BN 5 MeV, 1 mm, `resonance-local` | joined | 2,048 | | 2329 | 9033 / 4184 | 81.1 | 6.3 | 135,846 |
| | streamed | 2,048 | 512 | 1478 | 7804 / 1425 | 81.3 | 10.2 (pass) | 135,846 |

Every streamed grid is bitwise equal to the joined one, and transport counts
are exact. The largest per-bin relative spectrum difference is 1.9e-5 for Si and
5.3e-5 for h-BN, both in `brem_wide`. That is float32 summation order, as in
[Equivalence on CUDA](#equivalence-on-cuda); CPU tests agree at `rtol=1e-12`.
Wall time costs 1 to 7% on Si and nothing measurable on h-BN, where scoring
dominates. The grid pass replaces the joined path's transport and grid
resolution, and the scoring pass's transport is cheap next to scoring.

The grid pass alone (`grid-pass` mode), against N:

| Case | N | Evidence (MB) | Host peak (MiB) | Device used (MiB) | Wall (s) |
|---|---|---|---|---|---|
| Si, automatic policy, block 4,000 | 16,000 | 18.9 | 798 | 181 | 1.2 |
| | 64,000 | 75.8 | 991 | 177 | 2.9 |
| | 256,000 | 303.3 | 1953 | 181 | 10.0 |
| h-BN, `resonance-local`, block 512 | 512 | 62.5 | 1036 | 1355 | 4.0 |
| | 2,048 | 250.3 | 1476 | 1353 | 10.3 |
| | 8,192 | 1001.1 | 3124 | 3650 | 24.3 |

Evidence grows linearly with N: about 1.2 kB per Si electron (74 segments at
16 bytes) and about 122 kB per h-BN electron, of which roughly 50 kB is sinc
samples and the rest production lines. Host peak above the pre-run level is
about 4.5 times the evidence at the largest N. That run joined the blocks
with a full copy and kept every input while sorting. The grid pass now joins
into one preallocated array, releasing each block as it is copied, and the sinc
reduction releases each input once its sorted copy exists. The result is
unchanged. On 10^7 synthetic samples the reduction's peak fell from 4.1 to 3.0
times the sinc evidence; the stable argsort's own workspace sets the rest. The
grid pass has not been re-measured on the box since this change.
Extrapolated, 10^6 h-BN primaries need about 120 GB of
evidence, so at #879's scale the grid pass, not segment rows, bounds host
memory. Device memory during the h-BN grid pass belongs to the two-node line
evaluation. Its used peak rose at N = 8,192, probably from one heavy block, but
this was not resolved. The measured h-BN stop also moves with N (288 keV at
2,048 and the 966 keV ceiling at 8,192, which needs 368,084 points), as it does
on the joined path.

### Spilled grid pass (#1141)

`checks/streamed_line_grid_spill_1141.sbatch` (job 13622, code `676364694`,
RTX 5080) ran the h-BN case above in `grid-pass` mode at block 512, spilled
(`STREAM_SPILL_MIB=0`) and held (`none`) at each N. Anon is the sampled
`RssAnon` peak above its pre-run level. `ru_maxrss` also counts the spilled
files' mapped pages, which the kernel can reclaim, so it is not the bound.

| N | Evidence (MB) | Anon, spilled / held (MiB) | Spill disk peak (GB) | Wall, spilled / held (s) |
|---|---|---|---|---|
| 2,048 | 250 | 468 / 678 | 0.25 | 6.1 / 5.7 |
| 8,192 | 1,001 | 476 / 2,249 | 1.41 | 20.9 / 19.1 |
| 32,768 | 4,005 | 573 / 7,779 | 5.63 | 80.2 / 72.0 |

Spilled and held grids are bitwise equal at every N, and the scratch
directory is empty after every arm. Across a 16x range of N, held memory
grows with the evidence (about 1.9x it), while spilled memory rises
105 MiB, 2.6% of the added evidence. Disk peak matches the projection:
twice the sinc evidence plus the lines. Spilling costs 7 to 11% wall.

Two earlier runs found defects that are now fixed:

- **Job 13610:** one memory map per spilled line array exhausted file
  descriptors at 64 blocks. Lines now go to one `LineFile`.
- **Job 13619:** the pivot sample held strided views that pinned every
  scanned chunk, 8 bytes per sinc row: spilled anon grew 516 to 993 MiB from
  N = 8,192 to 32,768. The sample is now copied. A synthetic probe of the
  out-of-core sort alone is flat at 363, 392 and 402 MiB for 25M, 100M and
  200M rows.

