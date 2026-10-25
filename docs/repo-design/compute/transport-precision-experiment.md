# Radiative transport precision experiment

The RTX 5080 pilot identifies radiative FP64 arithmetic as a material cost.
Selected FP32 helper variants reduce measured transport/emission time by
roughly 1.5–1.9× on the two measured scenes. These are diagnostic timing ratios:
whole-transport equivalence is unresolved and endpoint component probes fail.
Production transport precision and defaults remain FP64.

## Workload and precision boundary

Jobs 1343 (preflight), 1344 (Si) and 1345 (W) ran on
`gpu-node`, partition `gpu`, one RTX 5080, one requested CPU and
24 GiB RAM. SMT allocated two logical CPUs; worker thread counts were one.
Preflight requested five minutes; each pilot requested ten minutes.
All 80 requested samples completed: six scenes × five arms × 64 primaries,
plus two scenes × five arms × five seeds × 5000 primaries. Each process warms
the complete requested workload with a separate seed before measurement.
Pilot seeds are 58001–58005. Pilot scenes are normal-incidence 300 keV
Si/100 µm and W/10 µm, hard-photon cutoff 1 keV, one 5000-primary outer
block and 4096 MiB scratch. The frozen manifest supplies the remaining controls.

The isolated generator copies only the radiative helper module. Radiative
floating tables, helper constants and helper inputs become FP32. Trajectory
state, geometry, stopping, elastic scattering, integer random streams, initial
uniform generation and host photon-angle completion retain their production
precision. Photon-angle completion is included in the transport/emission timer.
This experiment does not measure an entirely FP32 transport engine.

| Arm | Radiative arithmetic | Photon inverse iterations |
| --- | --- | --- |
| `fp64` | Unmodified production source | 52 |
| `rad-fp32-52` | FP32 helpers and radiative tables | 52 |
| `rad-fp32-24` | FP32 helpers and radiative tables | 24 |
| `rad-cell64-24` | FP32 except FP64 cell-integral intermediates | 24 |
| `rad-stable32-24` | FP32 with log1p cell-integral rearrangement | 24 |

The stable cell arm evaluates the same linear-cell integral using `log1p` and a
small-argument correction polynomial. It adds no physical tail/grid model.
Generated CUDA confirms float radiative helpers and double transport state;
the mixed arm promotes only cell-integral intermediates. The exported eV→keV
constant remains double because the transport kernel imports it for debiting
emitted photon energy. Downcasting that interface in initial job 1339 produced
energy-ledger failures. Revision 2 separates the internal FP32 converter;
1339 is retained as failed diagnostic evidence and never pooled with these runs.
Job 1342 was canceled while pending to submit the corrected preflight.

## Measured timing

Times below are warmed synchronized transport/emission seconds, five fresh
processes per arm and scene. Compilation, warmup, setup and scoring are excluded;
full process times are retained separately. Variant order reverses each seed.
Trajectory audits and hashes run outside the timer. Launch timing instrumentation
remains active, so these records stay separate from the production benchmark.
Capacity replays remain part of the workload. PXR/CBS profiling is enabled only
in the small preflight; it is disabled in timed pilots.

| Scene | Arm | Median (s) | Min–max (s) | Reference/candidate |
| --- | --- | ---: | ---: | ---: |
| Si | `fp64` | 1.955 | 1.796–2.063 | 1.00× |
| Si | `rad-cell64-24` | 1.206 | 1.141–1.396 | 1.62× |
| Si | `rad-fp32-24` | 1.110 | 0.952–1.281 | 1.76× |
| Si | `rad-fp32-52` | 1.034 | 0.932–1.144 | 1.89× |
| Si | `rad-stable32-24` | 1.089 | 1.044–1.257 | 1.80× |
| W | `fp64` | 7.258 | 6.976–7.360 | 1.00× |
| W | `rad-cell64-24` | 4.739 | 4.426–4.871 | 1.53× |
| W | `rad-fp32-24` | 4.057 | 3.379–4.495 | 1.79× |
| W | `rad-fp32-52` | 4.098 | 3.781–4.462 | 1.77× |
| W | `rad-stable32-24` | 4.271 | 3.786–4.613 | 1.70× |

The 24-iteration arm is not consistently faster than 52 iterations across
materials. Five-repeat variation does not establish an iteration-count winner.
The machine summary also records exploratory seed-matched timing bootstrap
intervals. Neither these intervals nor the hardware's FP64 throughput ratio
establish equal-accuracy speedup, 100000-primary scaling, or cross-engine ranking.

## Accuracy evidence and limits

All corrected samples passed finite/nonnegative/support checks, unit directions,
exactly one terminal record per incident primary, photon energy debit and
per-primary energy conservation. Maximum errors across warmup and measured blocks:
energy ledger 2.56e-12 keV, photon debit 5.68e-14 keV, direction norm 2.22e-16.


At the actual 1 keV cutoff all arms pass the predeclared 2e-4 component screen.
Across the experimental arms the maximum scaled moment error is
3.43e-07
and maximum inverse-photon error divided by incident energy is
3.17e-06.


Broader probes include table knots, 20 keV interpolation boundaries, low/high
uniforms and cutoffs approaching incident energy. All candidate arms fail the
full endpoint screen: scaled hard-rate errors reach about 1–3.4%. Selective FP64
cell evaluation and the stable FP32 rearrangement improve this stress behavior
but do not remove it. Converting inputs/tables to FP32 before the integral still
loses endpoint information. This is sufficient reason to retain FP64 defaults.

Ensemble scores include transmitted/backscattered/stopped/side-exited fractions,
transmitted energy, photon count bands, photon/continuous/terminal energy per
incident primary and transmitted-angle bins. Approximate normal Bonferroni
intervals screen all 504 scene/population/arm/observable comparisons together;
40 pass, zero fail and 464 remain unresolved. This includes the small preflight;
it is not 504 independent physical validations. Difference standard errors use
the sum of arm standard errors to bound unknown seed covariance. Zero variance
and rare photon bins remain unresolved. Passing requires an entire interval
inside the frozen equivalence margin, not merely a nonsignificant difference.
Normal-interval coverage is itself unvalidated. No arm qualifies for T(epsilon).

## Evidence and reproduction

The [machine summary](../../validation/check-records/equal-precision/precision-experiment-summary.json)
contains timing, observable moments, intervals, component errors and archive SHA.
Raw samples, scores, generated Python/CUDA, source hashes, launch/resources,
allocation-start snapshots and checksums remain in
`/home/user/pyrite-510-precision-20261015/evidence/precision/{1343,1344,1345}`.
The local `scratch/precision-results/precision-final.tar.gz` archive preserves them.
It has not been published. End-of-script checksum manifests verify successfully;
Slurm job records expired before final retrieval and accounting is disabled,
so no final accounting record is claimed.

Run the commands under Radiative precision experiment in `checks/equal_precision/README.md`.
The experiment uses a distinct schema excluded from production benchmark analysis.
CPU checks cover the generator, energy invariants, eligibility gate, interval
classification, corrupt/control/source pooling and prior precision infrastructure.
GPU reference regression: 9 passed. Performance tests: 68 passed. CPU check job
1346 also scanned generated evidence copies during lint; source lint must be run
with raw evidence outside the source tree. No performance claim relies on that
failed artifact lint.

The [issue 200 work](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/commit/7650913643221aea61e1a92a4399e5e0efccd0fe)
informed frozen inputs, interleaved repeats, synchronized timing and explicit
error limits. Its CXR launch settings are not transferred to electron transport.
Future production work needs endpoint-safe inputs/evaluation and a larger,
independently validated ensemble equivalence study before exposing a precision
switch or changing the numerical model marker.


## Endpoint coordinate follow-up

The initial experiment's remaining stress discrepancy begins before the cell
integral. At an incident energy of 23643.540225079392 eV and cutoff
23643.516581539167 eV, the interval is 0.02364354022574844 eV wide. Rounding
both endpoints to FP32 produces a 0.025390625 eV interval, a 7.39% width error.
Promoting those already rounded values inside the cell integral cannot recover
the original interval.

The opt-in `rad-boundary64-24` arm preserves energy/cutoff, incident/nominal/tip
coordinates, node traversal and photon-inverse bounds in FP64. SDCS values,
interpolation weights, per-atom moment accumulation, cell logarithms and the
small-argument correction remain FP32. Layer density scaling/accumulation and
incident-energy interpolation logs remain FP64. Its cell expression uses the
same stable integral, with the dimensionless width formed by double subtraction
before conversion to single. No production module is edited.

The [CPU diagnostic record](../../validation/check-records/equal-precision/endpoint-cpu-summary.json)
executes generated device-helper source through NumPy after removing imports and
device decorators. It uses the same 2000 cases per Si/W table and pinned FP64
reference values retained from job 1343. Table keys/digests and reference-source
hashes must match. This is an arithmetic diagnostic, not a CPU transport arm or
CUDA validation.

| CPU diagnostic arm | Si maximum scaled hard-rate error | W maximum scaled hard-rate error |
| --- | ---: | ---: |
| FP64 control | 6.82e-15 | 6.82e-15 |
| Stable FP32 control | 1.03e-2 | 1.84e-2 |
| FP64 boundaries / FP32 cell | 3.08e-7 | 3.96e-7 |

The new arm passes this CPU component screen, including endpoint stress cases;
its maximum inverse-photon error divided by incident energy is 3.20e-6 (Si)
and 2.10e-6 (W). The old failing control is reproduced. This isolates a viable
precision boundary without establishing GPU accuracy or performance.

The separate frozen `checks/equal_precision/precision-endpoint.json` uses seeds
59001–59005 and one five-minute preflight plus two ten-minute pilots, keeping
the previous workload controls. All three arms need six-scene trajectory
preflight; the new arm also needs the full endpoint component screen. Required
preflight failures make the job fail, blocking dependent pilots. Completed
per-job evidence belongs under `scratch/endpoint/<job-id>`; the actual submitted
batch script is captured alongside source hashes and generated CUDA.

Jobs 1371/1372/1373 completed successfully after the GPU became available earlier
than its scheduler estimate. Draft jobs 1369/1370 were replaced while pending
and consumed no GPU time. All five earlier generated helper controls remain
byte-identical. Focused CPU tests: 32 passed. The complete Linux performance
suite passed 72 tests in CPU job 1374; source lint and Linux type checking passed.
GPU reference regression: 9 passed. The broader Windows run retains three
platform/environment failures (Tk setup, CRLF byte counts, unavailable iowait).

### GPU endpoint and timing result

The [GPU machine summary](../../validation/check-records/equal-precision/endpoint-gpu-summary.json)
records all 48 samples, 24 scene/population/arm groups and zero failures/deferrals.
The new arm passes the full 4000-case Si/W component screen. Maximum scaled
hard-rate errors are 2.87e-7 (Si) and 2.85e-7 (W); corresponding inverse-photon
errors divided by incident energy are 3.20e-6 and 2.10e-6. Generated CUDA confirms
double boundary arguments/subtraction and float cell logarithm/correction,
SDCS interpolation and per-atom moment accumulation. Radiative table dtypes are
four FP64 arrays (density and three coordinate arrays), plus FP32 SDCS values.

All trajectory audits pass across warmups and measurements. Maximum primary
energy-ledger error is 3.13e-12 keV and photon-debit error is 5.68e-14 keV.

Times use the same synchronized warmup/transport boundary and five fresh-process
seeds per arm as the previous experiment, with the separately frozen seeds
59001–59005. Medians below are compared only within this new pilot; prior
allocation timings are not pooled.

| Scene | FP64 median (s) | Boundary64/FP32 median (s) | Reference/candidate | Exploratory bootstrap 95% ratio interval |
| --- | ---: | ---: | ---: | ---: |
| Si 300 keV / 100 µm | 1.682 | 1.553 | 1.083× | 1.046–1.095 |
| W 300 keV / 10 µm | 6.410 | 6.005 | 1.067× | 1.063–1.090 |

The endpoint-failing stable FP32 control has medians 0.889 s (Si) and 3.443 s
(W), ratios 1.89× and 1.86×. The endpoint-preserving arm recovers only a small
part of that performance gain. Generated CUDA still uses double divisions for
cell-width normalization, SDCS coordinate interpolation, reduced photon
coordinates and inverse interpolation. Source inspection identifies these as
candidates for narrower precision boundaries; it does not establish their
instruction-level cost. No new profiler counter claim is made.

Across the 252 simultaneous approximate ensemble comparisons, 10 pass, zero
fail and 242 remain unresolved. These include preflight diagnostics and do not
establish whole-transport equivalence. Timing ratios remain diagnostic;
T(epsilon), larger-population scaling and cross-engine ranking stay unqualified.
Production precision/defaults and numerical-model markers remain unchanged.

Raw evidence is retained under the lab checkout's `scratch/endpoint/{1371,1372,1373}`
and local `scratch/endpoint-results/endpoint-gpu-evidence.tar.gz`, with archive
SHA in the summary. All three per-job checksum manifests verify. Final scheduler
records show successful exits; accounting memory/counters remain unavailable.
For historical analysis pass the captured preflight
`source/precision-endpoint.json` to `--protocol`: byte hashes include line endings.


The subsequent [paired equivalence study](../../validation/beam-transport/transport-precision-equivalence.md)
examines selected FP32 quotients after precise FP64 subtraction, with independent
held-out populations and finite-sample confidence bounds for the frozen scores.
