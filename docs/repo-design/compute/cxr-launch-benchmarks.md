# CXR reduction launch measurements

RTX 5080 measurements for [issue #200](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/200),
2026-10-13. Retain **512 threads, three energies per block**, the 400,000-row
batch target, and the existing bin-mean pair/tree selection. These bounded
measurements do not justify a production default or hardware-selection change.
A [2026-10-29 MeV capture](#2026-10-29-mev-production-batch) confirms this and
shows that the production MeV reduction is the bin-mean route, not the point
kernel these launch settings tune.

The [machine-readable evidence](data/cxr-launch-2026-10-13.json) records every
configuration, timing sample, deviation, fixture hash, and saved case. The
harness is `checks/cxr_reduction_benchmark.py`. No production arithmetic,
physics policy, RNG, or model marker changed.

## Protocol and workloads

Code snapshot `86db77f5`, isolated `/home/aamador/pyrite-200-cxr-tuning` on `qlmc`.
Successful measurement jobs: SLURM 1244, 1246, 1247 (completed point arms), and
1248. Each requested one RTX 5080, four CPU threads, and 12 or 24 GB host RAM;
allocations were bounded to five or ten minutes and serialized.

RTX 5080: 84 SMs, 16,303 MiB; driver 610.47, CUDA driver API 13.3, runtime 13.2,
CuPy 14.2.0, Python 3.14.6. Reported power limit: 360 W; maximum SM clock:
3,090 MHz. Clocks were not pinned. Replay peak host RSS is recorded; GPU pool
reservation at process end is **not peak VRAM**.

- **Production transport control:** saved `trajectory_demo` HOPG transport,
  30 keV, 2 µm, polar angle 45°, azimuth 180°, 100 electrons, seed 1, four
  reflection rows. Current spectrum replay produced 14,356 lines and 4,286
  resolved nodes. Small slice: first 4,097 lines, every fourth node (1,072).
- **Spectrum-only control:** saved phase-retention HOPG transport, 60 keV,
  5 µm, 200 electrons, seed 362, screened-Rutherford/continuous-stopping.
  Explicitly select the existing incoherent bin-mean evaluator and one `(002)`
  reflection: 12,728 lines and 5,750 bins. This is a diagnostic control, not a
  fresh default-production transport run. Point timings on its saved axis are
  a separate evaluator comparison, not a replacement for bin-mean spectra.
- **Derived size stress:** repeat the production batch's lines 32 times and
  divide weights by 32; refine the diagnostic axis to 100,003 nodes. This gives
  459,392 lines and approximately 45.94 billion pairs. It repeats one population
  rather than representing an independent large production case.

Sweep threads `{128, 256, 512, 1024}` × energies/block `{1, 2, 3, 4}`. One
untimed warmup per arm; point order alternates forward/reverse each repeat.
CUDA events enclose launches, with synchronization and separate wall samples.
Reset, transfers, compilation and artifact I/O are excluded. Production-control
replay has nine repeats; other final arms have five. Bin methods use grouped
order and synchronized wall time, including tree construction and transfers.

## Point results

Times are median **[minimum, maximum] milliseconds**. Best means lowest observed
median, not a statistically established winner.

| Workload | Default 512 × 3 | Best configuration | Best time |
| --- | ---: | --- | ---: |
| Production small slice | 0.07306 [0.07088, 0.09616] | 512 × 2 | 0.07264 [0.07066, 0.07456] |
| Production full batch | 0.21862 [0.21565, 0.22086] | 512 × 4 | 0.21414 [0.21238, 0.24138] |
| Spectrum control small slice | 0.07216 [0.07008, 0.08499] | 512 × 3 | 0.07216 [0.07008, 0.08499] |
| Spectrum control full batch | 0.24701 [0.24070, 0.25837] | 256 × 3 | 0.23875 [0.23789, 0.24173] |
| Derived full stress | 91.69424 [89.31533, 92.28864] | 256 × 4 | 86.73018 [85.52973, 87.22359] |

The production full-batch median improves about 2.0%, with overlapping ranges.
The derived stress median improves about 5.4%, with separated ranges. Neither
establishes a benefit for historical 5 MeV h-BN or an end-to-end speedup.

All point comparisons satisfy the predeclared diagnostic allowance of 100
float32 epsilons times the baseline peak. Maximum peak-scaled deviations across
configurations: 1.25e-7 (production), 1.10e-7 (spectrum control), 4.19e-6 (stress).
The evidence also records absolute deviations, relative deviations with a
1e-6-peak floor, and discrete trapezoidal yield changes computed in FP64. This
diagnostic allowance does not establish the full production accuracy gate.

## Bin-mean results

Synchronized wall medians in milliseconds, including tree setup:

| Control | Pairs | Tree | Auto |
| --- | ---: | ---: | ---: |
| 4,097 lines × first 1,025 original bins | 23.573 | 342.204 | 23.613 |
| 12,728 lines × 5,750 original bins | 112.946 | 768.186 | 112.982 |

Auto correctly retains pairs here; the full tree route is about 6.8 times
slower. Tree/pair peak-scaled differences are at most 1.44e-11. These controls
are below the five-billion-pair crossover and do not calibrate a replacement.
Original adjacent edges are preserved; the small bin control differs from the
subsampled point axis.

## 2026-10-29 MeV production batch

SLURM 2676, code `f877bf3d` (rebased on main `3afd08dc`), fresh clone
`qlmc:~/pyrite-200-r2`. Same RTX 5080, driver 610.47, CUDA driver API 13.3,
runtime 13.2, CuPy 14.2.0, Python 3.14.6; clocks not pinned. Nine repeats per
arm, same protocol. Evidence: [data](data/cxr-launch-2026-10-29.json).

- **Fixture:** `high_energy_demo` h-BN, maximum-energy case (300 keV, 1 mm,
  tilt 5°), 2,000 electrons, seed 392081. The first production flush held
  416,114 lines on 27,303 bin-mean bins, about 1.14e10 line-bin pairs
  (SHA-256 `4ee24c08…`). Capture took 6 min 14 s wall and 3.6 GB peak host RSS.
  Job 1242 timed out at 8,000 electrons and 10 minutes.
- **Point kernel on this batch's axis** (diagnostic: production uses bin-mean
  here). Small slice 4,097 lines × 1,051 nodes: default 512 × 3 0.05718
  [0.05491, 0.08890] ms; best 512 × 2 0.05542 [0.05251, 0.09747] ms, with
  overlapping ranges. Full batch: default 22.336 [22.271, 22.401] ms
  (5.1e11 pairs/s); best **512 × 4 21.567 [21.503, 21.620] ms**, a 3.4% gain
  with separated ranges; 256 × 4 21.657 ms. Every arm is within the
  peak-roundoff allowance; maximum peak-scaled deviation 1.21e-6.
- **Bin-mean reduction (the production route).** Small control (1,025 bins):
  pairs 3.54 ms, tree 16.47 ms, auto 3.55 ms (auto selects pairs). Full batch:
  auto selected the tree route at **8,633 ms**, tree 8,629 ms (synchronized wall, including tree setup and transfers). The forced
  all-pairs arm was skipped above the 5e9-pair budget. That is about 1.3e9
  pairs/s, roughly 400 times slower per pair than the point kernel.
- **HOPG drift check:** the 2026-10-13 fixture replayed on current main gives
  default 0.22186 [0.21622, 0.22650] ms against 0.21862 ms then, and best
  512 × 4 0.21258 ms (4.2%). The ranking and size of the gain are unchanged.
- CUDA bin integration/tree checks: **15 passed in 4.99 s**.

Decision: retain 512 × 3. Four energies per block wins small, repeatable gains
(3–4%) on the full point batches, but the point kernel costs about 22 ms per
MeV flush against about 8.6 s for the bin-mean reduction. Retuning point
launch geometry cannot change MeV end-to-end time materially. Any MeV
reduction work belongs to the bin-mean tree/pair kernels; this measurement
does not justify a change to them either.

## 2026-11-06 MeV reduction speedup

[Issue #1003](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1003).
RTX 5080, SLURM 4021, 4036, 4059 and 4067, frozen grid (133,728 bins for h-BN). Median
of 2 timed replays after one untimed warmup, synchronized wall, FP64 flush sums.
Harness: `checks/cxr_bin_mean_profile_1003.py`. Clocks were not pinned. These
are per-flush reduction times from captured lines, not per-primary wall time.

| Flush (lines × bins) | Tree s | Tree + series | Fused | Fused + series | + edge reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOPG 300 keV (47,698 × 17,100) | 0.751 | 0.615 | 0.165 | 0.122 | 0.092 |
| HOPG 300 keV (220,352 × 34,200) | 3.258 | 2.172 | 1.288 | 0.817 | 0.624 |
| HOPG 300 keV (220,352 × 126,224) | 5.325 | 2.644 | 3.962 | 1.974 | 1.550 |
| h-BN 5 MeV (400,000 × 133,728) | 12.580 | 6.129 | 5.070 | 2.639 | 2.033 |
| h-BN 5 MeV (379,028 × 133,728) | 11.087 | 5.623 | 4.770 | 2.481 | 1.915 |

Edge reuse (job 4067, A/B against the previous kernel in the same job): each
lane evaluates one bin edge's `Si` tail and passes it to the lane below, so an
exact pair costs one tail, not two. Its sums are bitwise equal to the kernel
without it on every flush. With the series on, about 80% of h-BN near pairs take
the series and the remaining 20% are exact.

`auto` before #1003 took 12.63 and 11.14 s on the h-BN flushes, so the fused
route with series and edge reuse is **6.2× and 5.8×**. On the HOPG flushes it
took 0.484, 4.63 and 5.36 s, so 5.3×, 7.4× and 3.5×. Every arm repeats bitwise. Against the series-free tree,
the worst per-bin relative deviation is 3.0e-9 with a floor of 1e-6 of peak, and
the worst integral deviation is 2.0e-12. Below that floor agreement is weaker:
h-BN far-tail bins near 6e-11 of peak reach 2.3e-6 relative. See the
[accumulator limits](../../validation/radiation-physics/sinc-bin-near-far.md).

Split of the h-BN flushes without the series (4.8–5.1 s): near 4.1–4.3 s, far
0.58–0.62 s, host setup about 0.1 s. With the series, near takes 1.80 s on the
second flush; edge reuse brings it to 1.23 s, so the unchanged far tree (0.58 s)
is now 30% of the flush. Near pairs are FP64 bound: on a 50,000-line
slice `ncu` reports compute throughput 88%, FP64 the top pipeline, 27–28 of 32
lanes active. The tree it replaces was imbalance bound at 50%, 27 lanes, 28.6 of
50 occupancy. The two changes combine: the series cuts FP64 work per pair, and
line-major warps keep lanes on one branch without a busiest-bin bound. Rejected:
one warp per bin (3.7–4.1× slower than `auto`, job 4021) and bin-major sparse
near (1.7–1.9× slower).

Decision: `auto` takes the fused route from 1e9 pairs. Below the gate it keeps
all pairs. The 8.2e8-pair flush above suggests a lower gate would win, but one
flush does not establish where the crossover sits.

### ROCm

SLURM 4075 on qlmc-ace (Radeon 8060S, gfx1151, ROCm 7.1, CuPy 14.2 HIP, environment
from `checks/rocm/env.sh`) at `a5ac6c2fd`. The bin-mean kernels compile through
hiprtc with no source or flag changes, and the CUDA-gated bin-mean tests pass
(39). These include fused against the host reference and the series dispatch.
Replay of the h-BN flushes, one timed run after a warmup:

| Flush | Tree s | Fused | Fused + series |
| --- | ---: | ---: | ---: |
| h-BN 5 MeV (400,000 × 133,728) | 15.87 | 6.43 | 4.24 |
| h-BN 5 MeV (379,028 × 133,728) | 15.23 | 6.04 | 3.97 |

Each arm repeats bitwise. Against the RTX 5080 sums of job 4067, per-bin
relative deviation is at most 8.2e-12 for fused plus series and 4.9e-14 for
the tree (floor 1e-6 of peak). Those are FP64 math-library differences, so the
two GPUs do not agree bitwise.

### Warm wall per primary

SLURM 4072, RTX 5080, CUDA transport in fp64, float32 scoring, 128 primaries,
seed 1, 1 mm at 10°/100°. The h-BN run uses the frozen seed-0 grid; the silicon
grid was frozen in the same job. One untimed warmup run, then the median of 2
runs, with each side's own checkout: `main` `9b861a89e` before #1003 against
`14c430d5b`. Harness: `checks/cxr_bin_mean_wall_1003.sbatch`
(`cxr_bin_mean_profile_1003.py grid|wall|compare`).

| Case (bins) | Before s/primary | After s/primary | Speedup |
| --- | ---: | ---: | ---: |
| h-BN 5 MeV (133,728) | 0.196 | 0.036 | 5.4× |
| silicon 5 MeV (45,517) | 0.478 | 0.092 | 5.2× |

Spectra: bremsstrahlung and characteristic are bitwise equal. In every line
bin above 1e-6 of peak the two agree within 2 float32 ulps (h-BN: 76 of 32,432
bins differ; silicon: 145 of 40,159), and the integral within 1.2e-12.

## Limits and reproduction

The h-BN 300 keV, 1 mm, 8,000-electron capture (SLURM 1242) hit its ten-minute
limit before saving a batch. Its stage was not independently profiled; this
is not a measured reduction time. The 2026-10-29 section supplies a
300 keV production batch; the 5 MeV batches are in the #1003 section. No
catalog golden comparison has completed. Large derived bin arms were stopped and excluded from
final evidence; use bounded forced-pair tests and point-only size stress.

Nsight Compute (SLURM 1249) reported `ERR_NVGPUCTRPERM`. Occupancy, registers,
spills, DRAM/L2 and instruction/SFU counters are unavailable; no measured
memory-versus-sine bottleneck verdict follows. Matched Molab RTX Pro 6000 measurements moved to the [#776](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/776) CUDA tuning tutorial, which replays this page's h-BN fixture there.
Tiling, device-specific defaults and batching changes need production evidence.

Raw fixtures/logs remain in `/home/aamador/pyrite-200-cxr-tuning/results/`;
exported local copies are under the issue worktree's ignored
`scratch/lab-results/`. Fixture and source-transport SHA-256 values are in the
evidence. Reproduce **on a lab GPU allocation**, with the project runner:

```bash
uv run --extra nvidia python checks/cxr_reduction_benchmark.py \
  replay results/hopg-batch.npz --output results/hopg-replay.json \
  --revision 86db77f5 --repeats 9

uv run --extra nvidia python checks/cxr_reduction_benchmark.py \
  derive results/hopg-batch.npz --output results/stress-batch.npz \
  --revision 86db77f5 --line-repeats 32 --grid-nodes 100003

uv run --extra nvidia python checks/cxr_reduction_benchmark.py \
  replay results/stress-batch.npz --output results/stress-replay.json \
  --revision 86db77f5 --skip-bin-mean
```

`capture ... --artifact PATH` replays schema-2 transport. For a spectrum-only
artifact, `--incoherent-artifact --quadrature bin-mean` selects the control here.
`profile FIXTURE --revision REVISION` launches the default point kernel twice
for a profiler warmup skip. Heavy commands never run on local WSL.

Existing CUDA bin integration/tree checks: **15 passed in 5.09 s**, through
`pyrite-dev test` with `PYRITE_TEST_BACKEND=cuda`. These check existing numerical
contracts; no kernel/default change was made.
