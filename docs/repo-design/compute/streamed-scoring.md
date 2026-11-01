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
- Streaming skips the two population-wide host steps that the joined block path
  redoes after the join. Bunch-centroid timing is skipped because the eligible
  estimators are invariant to a per-electron time offset, which is a common
  phase within one electron's flight groups. Hard-photon direction
  re-completion is skipped because the coupled BremsLib continuum scores
  expected values on the tracks and never reads completed photon directions.
  Each block's own completion is skipped too (#864): blocks run with the
  internal `_complete_hard_photons=False`, which omits the direction and recoil
  fields. Completion draws from its own generator, so transport rows and
  counts are unchanged bit for bit. If a future consumer needs them, the completion generator can be carried
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

## Transport/scoring overlap

Overlapping CPU transport with GPU scoring is not implemented. Job 3010 timed
CPU per-electron transport alone on the same cases (Ryzen 9950X3D, one
thread). An ideal pipeline is bounded below by the larger of CPU transport
and GPU scoring:

| Case | CPU transport (s) | GPU scoring (s) | GPU-only streamed wall (s) | Ideal overlap (s) |
|---|---|---|---|---|
| Si 64,000 | 4.6 | 2.0 | 4.8 | ≥ 4.6 |
| W 16,000 | 18.1 | 18.6 | 28.8 | ≥ 18.6 |

Si gains nothing. For scoring-heavy W cases overlap could save up to about a third
of the wall time. It needs a producer process or a `nogil` per-electron core
(the current core is `@njit` without `nogil`) and bounded staging buffers.
Overlapping GPU transport with GPU scoring on separate streams has the same
lower bound.

## Equivalence on CUDA

On CUDA, scoring runs in float32 unless `PYRITE_FP64=1`, so block sums
differ from the joined sum at float32 rounding. Relative to each spectrum's
peak, streamed and joined differ by at most 3.1e-6 for Si at 64,000 electrons
and 1.9e-5 across the Si block sizes. For W across block sizes, lines and
characteristic agree to 4e-7 and the continuum to 2.9e-5. With
`PYRITE_FP64=1` an earlier Si comparison agreed to 1.75e-15 (job 2937), which
confirms that the float32 differences come from summation order only.
