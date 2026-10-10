# Phase retention: paired GPU measurements

Task-owner measurements for [issue #365](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/365),
2026-10-09. These evaluate the existing certified Flat omission and its
[observable/cost gate](phase-retention.md), without introducing a physics
criterion. They do not independently validate the radiation model.
The [machine-readable evidence](data/phase-retention-2026-10-09.json) contains
every timing sample, workload and fingerprint; identical decision records are
collapsed with their occurrence counts.

## Protocol and scope

SLURM jobs 1171 (paired matrix) and 1174 (calibration/validation and separate
host profile), isolated `~/pyrite-365-phase-retention` on `qlmc`. Each allocation
requested one GPU, four CPU threads, 12 GB RAM and a ten-minute limit. No other
GPU process was present at the initial hardware query. The live default was
1500 MB per CPU; memory was requested explicitly. Accounting storage is disabled,
so peak process RSS comes from the benchmark process, not `sacct`.

NVIDIA GeForce RTX 5080, 16303 MiB; kernel driver 610.47, NVIDIA-SMI 610.43.02,
CUDA user-mode driver 13.3; CuPy 14.2.0, Python 3.14.6, float32.
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=4`. The calibrated code snapshot was
`6702fd56` plus working-tree changes; exported code digest
`fb97be4a1dd850f12e9b5c9178437bc5cd6177769ab69dd96985c99a30f33a48`.
Those changes add artifact replay, calibration and explicit row/population
scope metadata. The typed row tuples preserve the recorded reciprocal-vector values.

All cases: HOPG, 60 keV, 200 incident samples, physical population 200,
seed 362, 1 mm square footprint, screened-Rutherford/continuous-stopping
lockstep transport, 5 keV transport cutoff, `B_ang2=0.8`. This is a bounded
reducer workload, not a production-profile or realistic large-charge benchmark.
Changing the physical population can change the omission mask and invalidate
these cost estimates.

| Regime | Thickness | Longitudinal RMS | Observation angle | Segments | Certified candidate retained / skipped nodes |
| --- | --- | --- | --- | --- | --- |
| Thin control | 1000 Å | 1e-9 fs | 119° | 443 | 5750 / 0 |
| Thick, short bunch | 50000 Å | 0.001 fs | 119° | 13232 | 1004 / 4746 |
| Thick, longer bunch | 50000 Å | 0.01 fs | 95° | 13232 | 0 / 5750 |

The thin bunch has a positive near-zero RMS, satisfying the spectrum API while
leaving no certified omission. The initial job 1170 passed 96 CUDA tests, then
stopped before benchmark timing because its zero-RMS input was invalid. It
supplies no timing evidence. Job 1171 passed 96 CUDA checks; final job 1174 passed 98, including row/population cost mismatch refusals.

Each regime writes a trajectory artifact, records the spectrum inputs and axis,
then reopens it before evaluating spectra. Subsequent routes replay that artifact
without transport. The harness refuses a changed transport/input fingerprint.
These are benchmark artifacts; their minimal mapping is not a production typed
`Case` for the runner's artifact-recompute interface.

The common axis is 5750 evaluated coordinates, 500 to below 12000 eV, step 2 eV.
SHA-256: `f9a39f182f4d3c93b71181ffbfb3439efa0a9e533ab749b6b6a3fbd89bd2a0fa`.
Transport fingerprints match across routes and between the two measurement jobs:

| Regime | Segment SHA-256 |
| --- | --- |
| Thin control | `70e800f4cd8a9fcae7465424937d06f4c1cefb9e7f3d8d5f97132bdacfc30f87` |
| Thick, short bunch | `dd34105428c1ba9c49c0b7b7145e3cdc11749e0ebb2014aacd03220e798f2658` |
| Thick, longer bunch | `f5c06210ef15325dc9125632ad8b2d65120d2e97afda1254ebf378b7994af03a` |

Every arm gets one warmup. Timed order alternates forward/reverse each repeat.
GPU synchronization bounds each spectrum-only interval. Artifact I/O, transport,
warmup, profile instrumentation and output serialization are outside those
intervals. Host RSS was 724–1127 MiB across processes. GPU pool bytes were queried
at process end; they are not a GPU peak-memory measurement.

## Direct omission: two reflection rows

Job 1171 used `(002)` and `(00-2)`, seven repeats per arm. Streaming shares a
union mask across both rows; eager/JIT decide per row. `Full` disables omission,
`Reduced` uses its existing 1e-4 share without a cost gate, and `Fallback` uses
the policy without cost evidence, retaining full phase. Times are median
**[minimum, maximum] milliseconds**; speed ratio is Full/Reduced medians.

| Regime / route | Full | Reduced | Fallback | Ratio |
| --- | --- | --- | --- | --- |
| Thin / streaming | 7.730 [7.665, 8.624] | 8.220 [8.145, 8.528] | 8.073 [7.871, 8.193] | 0.940 |
| Thin / JIT | 35.197 [35.077, 35.786] | 35.646 [35.390, 35.758] | 35.412 [35.322, 36.426] | 0.987 |
| Thin / eager | 13.031 [12.664, 13.323] | 13.487 [13.342, 13.790] | 13.184 [13.043, 13.738] | 0.966 |
| Thick short / streaming | 20.538 [20.208, 20.634] | 21.837 [21.652, 22.429] | 22.953 [22.736, 23.108] | 0.941 |
| Thick short / JIT | 37.860 [37.678, 39.062] | 42.307 [41.968, 42.610] | 42.645 [42.097, 44.317] | 0.895 |
| Thick short / eager | 127.275 [127.052, 127.352] | 102.246 [101.909, 102.691] | 132.151 [131.707, 132.772] | 1.245 |
| Thick longer / streaming | 19.560 [19.187, 19.791] | 19.988 [19.780, 20.498] | 22.100 [21.756, 22.620] | 0.979 |
| Thick longer / JIT | 37.667 [37.433, 38.070] | 41.216 [40.700, 42.338] | 42.390 [42.069, 42.619] | 0.914 |
| Thick longer / eager | 125.600 [125.102, 125.836] | 93.518 [93.265, 93.938] | 130.313 [129.957, 130.781] | 1.343 |

Eager reduction benefits when it avoids evaluating Flat on most or all nodes.
Streaming and reduction JIT have no measured median benefit here. The small thin
controls have overlapping timing ranges; no work is skipped. Missing evidence
preserves the full spectrum but still pays certificate/mask construction costs.

All nine paired comparisons passed the existing evaluated-node omission bound
plus the declared 100-epsilon peak-scaled reduction-roundoff allowance. Thin and
longer-bunch yield/centroid changes were zero at working precision. Thick-short
relative yield changes were 2.979e-11 (streaming/JIT) and 3.087e-8 (eager);
centroid changes were 2.828e-11 and 7.526e-10 respectively. Every fallback spectrum
was bit-identical to full evaluation. These fixed-axis trapezoid measurements
are diagnostic yield/centroid comparisons, not continuous-integral accuracy
certificates or a shared #350/#362 budget proof.

## Cost-gated validation: one reflection row

Job 1174 used `(002)` alone so one decision scope covers the whole measured
request. Applying whole-request timings independently to several eager rows
would misstate cost. Nine calibration repeats per full/reduced/fallback arm
provided conservative empirical estimates: minimum full time, maximum reduced
time, and nonnegative maximum fallback time minus minimum full time for overhead.
The reduced request timing already includes certification, making this overhead
charge conservative. These extrema are estimates, not guaranteed runtime bounds.
Calibration is local to this process, saved input, axis, backend and row; no
default model or reusable hardware-rate table is installed.

After calibration, full and calibrated arms receive equal warmup and **nine new
alternating repeats**, excluded from fitting. Times use the same units/format as
above. The gate accepts only the two thick eager regimes.

| Regime / route | Decision | Full | Calibrated | Ratio |
| --- | --- | --- | --- | --- |
| Thin / streaming | No certified omission | 7.893 [7.660, 8.178] | 8.004 [7.815, 9.354] | 0.986 |
| Thin / JIT | No certified omission | 35.831 [34.781, 36.597] | 35.465 [35.059, 36.990] | 1.010 |
| Thin / eager | No certified omission | 12.603 [12.240, 14.225] | 12.986 [12.399, 13.972] | 0.971 |
| Thick short / streaming | No net saving | 16.984 [16.693, 17.677] | 19.676 [19.084, 21.414] | 0.863 |
| Thick short / JIT | No net saving | 37.360 [37.126, 39.519] | 40.088 [39.677, 42.081] | 0.932 |
| Thick short / eager | Accept 4746 skipped nodes | 125.709 [124.665, 127.679] | 98.163 [97.367, 98.812] | 1.281 |
| Thick longer / streaming | No net saving | 16.052 [15.589, 16.523] | 18.772 [18.417, 20.163] | 0.855 |
| Thick longer / JIT | No net saving | 36.811 [35.985, 37.885] | 39.832 [38.681, 41.781] | 0.924 |
| Thick longer / eager | Accept 5750 skipped nodes | 123.988 [123.223, 124.872] | 91.036 [90.684, 92.577] | 1.362 |

Accepted outputs exactly match the previously evaluated reduced spectrum;
declined outputs exactly match the full spectrum. Calibrated eager accuracy
therefore inherits the same evaluated-node test. The full empirical timing and
decision records, including row vectors and physical population, are in the
linked evidence. Refused thick workloads still incur about 2–3 ms to construct
the candidate certificate. Use the explicit disabled control for a workload
already known to be unprofitable; these results do not justify enabling a gate
by default on streaming/JIT campaigns. The cost of gathering calibration samples
is also excluded from validation timings and must be amortized before any
end-to-end workflow speedup claim.

## Separate host profile

The eager thick-short one-row replay was profiled separately with `cProfile`,
one repeat, without fitting or accepting its instrumented timings. Seven calls
including warmups and the floor check spent cumulative host time of 0.509 s in
`_accumulate_batched`, 0.421 s in `_batched_coherent_finalize`, 0.188 s in
`_coherent_electron_grouped_row`, and 0.125 s in escape-piece expansion.
`require_resolved_power` accumulated 0.403 s, including device synchronization.
These values overlap and cannot be added; they measure host dispatch/wait, not
exclusive GPU kernel time. They identify coherent reduction, the mandatory
grouped floor, escape preparation and power validation as material costs that
Flat omission alone cannot eliminate.

## Reproduce

Sync to an isolated lab checkout with `pyrite remote sync`, install its locked
CUDA environment with `uv sync --extra nvidia --locked`, then execute inside a
bounded SLURM allocation with the environment above:

```bash
uv run --no-sync python checks/coherent_flat_omission.py \
  --electrons 200 --physical-electrons 200 --thickness-ang 50000 \
  --bunch-rms-fs 0.001 --trajectory results/phase365/thick-short.h5 \
  --phase-policy --repeats 7 --route eager
```

Repeat with `--route auto` and `--route jit` on the same artifact. For calibration,
use a separate artifact and add `--single-reflection --calibrate-policy`, with
`--repeats 9`. Use the regime parameters in the first table for the controls.
The harness verifies saved inputs before replay; a changed row list needs a
separate artifact. Transport and artifact I/O are reported separately.

Coverage remains deliberately bounded: one material, one transport law, one
physical population, one axis, scalar directions and no intra-electron phase
truncation. Other transport/source ensembles, detector/bin error contracts,
automatic campaign calibration and additional supported mechanisms remain
issue #365 work. No scattering-induced decoherence or global coherence-length
switch is inferred from these measurements.
