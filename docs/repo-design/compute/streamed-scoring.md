# Live streamed scoring

Opt-in, internal (#611). `pyrite.montecarlo.runner.streamed_scoring.run_case_streamed(case, block_electrons, transport_core=...)`
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
- Streaming skips the two population-wide host steps that the joined block path
  redoes after the join. Bunch-centroid timing is skipped because the eligible
  estimators are invariant to a per-electron time offset, which is a common
  phase within one electron's flight groups. Hard-photon direction
  re-completion is skipped because the coupled BremsLib continuum scores
  expected values on the tracks and never reads completed photon directions.
  If a future consumer needs them, the completion generator can be carried
  across ordered blocks, or `advance()`d by twice the earlier photon count;
  both reproduce the single-call draws exactly.

## Eligibility

Supported: incoherent line `spec`, characteristic lines and the EEDL or BremsLib
continuum, including coupled BremsLib radiative transport, on the per-electron
or CUDA core, with a fixed electron count and a final line grid.

`streaming_refusal(case, live=True)` refuses the following before any transport:

| Case key | Reason |
|---|---|
| `coherent_emission`, `temporal_profile`, `photon_transport`, bunch `transition_radiation` | Need every segment or history at once (shared with artifact streaming). |
| `line_grid_policy`, `_diagnostic_line_grid` | The line grid is resolved from the whole transported population. |
| `adaptive_precision` | The electron count, and hence the normalization, is decided during the run. |
| `transition_radiation` | Electron-block transport refuses it; artifact replay still streams per-history TR. |
| `channeling`, `gdf_source` | Refused by electron-block transport. |
| `secondary_threshold_eV`, `pair_production_model`, `positron_transport` | Cascade block locality not yet confirmed. |

The lockstep core shares one generator across electrons and cannot run in
blocks; pin `transport_core="per-electron"` or `"cuda"` (`"auto"` can resolve
to lockstep at small electron counts).

## Equivalence evidence

`tests/montecarlo/test_streamed_scoring.py` and
`tests/montecarlo/test_coupled_radiative_runner.py::test_streamed_scoring_matches_the_joined_coupled_run`
compare streamed and joined spectra at `rtol=1e-12` over several block sizes,
with unequal line and continuum populations and with coupled BremsLib straggled
transport. Transport counts and segment totals agree exactly.

On CUDA, scoring runs in float32 unless `PYRITE_FP64=1`, so block sums
differ from the joined sum at float32 rounding. On the benchmark scene
below, the largest per-bin difference relative to each spectrum's peak was
5.3e-6 (continuum) and 3.0e-7 (lines), and spectrum totals agreed to 4e-7.
With `PYRITE_FP64=1` the same comparison agrees to 1.75e-15 (job 2937),
which confirms that the float32 differences come from summation order only.

## Memory and wall time

`checks/streamed_scoring_memory.py` runs one arm per process on the lab GPU
box (RTX 5080, 16 GiB): Si 300 keV, 10 µm slab, coupled BremsLib midpoint
transport, CUDA core, 400 line and 299 continuum bins, one seed. Host peak is
`ru_maxrss` (about 600 MiB before the run). Device figures are CuPy pool bytes
above the pre-run level: *reserved* includes cached free blocks, *used* is
live arrays. Joined arms are `run_case` (job 2932); streamed arms release
cached pool blocks after each block (job 2940, code `d6fed1bca`; joined arms code `2d4125d08`).

| Arm | N | Block | Segments | Host peak (MiB) | Reserved VRAM (MiB) | Wall (s) |
|---|---|---|---|---|---|---|
| joined | 4,000 | | 0.30 M | 1582 | 940 | 37.8 (cold JIT) |
| joined | 16,000 | | 1.18 M | 1483 | 3182 | 20.2 |
| joined | 64,000 | | 4.74 M | 2240 | 6301 | 264.7 |
| streamed | 4,000 | 4,000 | 0.30 M | 1217 | 940 | 4.2 |
| streamed | 16,000 | 4,000 | 1.18 M | 1243 | 940 | 12.7 |
| streamed | 64,000 | 4,000 | 4.74 M | 1247 | 1101 | 47.5 |
| streamed | 64,000 | 1,000 | 4.74 M | 1195 | 261 | 51.7 |
| streamed | 64,000 | 16,000 | 4.74 M | 1501 | 3236 | 61.3 |

At a fixed block size, host peak stays flat and reserved VRAM rises only 17% as N grows 16-fold;
at fixed N they follow the block size. Before the per-block release, reserved
VRAM grew with N (1925 MiB at 64,000 electrons in 4,000-electron blocks) and
reached 10 GiB at 16,000-electron blocks while live arrays stayed near
0.6 GiB (job 2937): per-block staging and scoring sizes vary, and the caching
pool kept every freed size. Wall times are single runs; the joined 64,000
run's 265 s was not investigated, and transfers, synchronization and reduction
overhead were not profiled separately. CPU-transport/GPU-scoring overlap is
not implemented.
