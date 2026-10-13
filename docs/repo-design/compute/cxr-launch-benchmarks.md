# CXR reduction launch measurements

RTX 5080 measurements for [issue #200](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/200),
2026-10-13. Retain **512 threads, three energies per block**, the 400,000-row
batch target, and the existing bin-mean pair/tree selection. These bounded
measurements do not justify a production default or hardware-selection change.

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

## Limits and reproduction

The h-BN 300 keV, 1 mm, 8,000-electron capture (SLURM 1242) hit its ten-minute
limit before saving a batch. Its stage was not independently profiled; this
is not a measured reduction time. No fresh 5 MeV batch or catalog golden
comparison completed. Large derived bin arms were stopped and excluded from
final evidence; use bounded forced-pair tests and point-only size stress.

Nsight Compute (SLURM 1249) reported `ERR_NVGPUCTRPERM`. Occupancy, registers,
spills, DRAM/L2 and instruction/SFU counters are unavailable; no measured
memory-versus-sine bottleneck verdict follows. Molab remains unmeasured.
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
