# Native/CPU and GPU-host audit (#1432)

Evidence report for #1432; the issue body is the decision record and
`README.md` owns the commands. No production numerics, API, RNG, cache or
executor default changed by this audit. Numerical baselines: CPU cases on
main `147400d356812d1f1d0ad39834de347ca1d8555f` (pre-#1281) and
`51a928b25` (post-#1281); GPU-host cases on main `3ab6857a5` and `42413c99c`.
Every row names its job; evidence for each is under `results/job-<id>/`
(index at the end). Cumulative profiler times overlap and are never summed;
RSS is a per-process lifetime high-water mark; one sample per arm unless a
count is given.

## Disposition

| # | Owner | Evidence | Decision |
| ---: | --- | --- | --- |
| 1 | GPU-host idle after #1452-#1455 (scoring-worker drivers, stopping monitor, beam-entry prefetch, continuation host stages, CXR stop search) | 15983, 15986, 15988 | Attributed; `_transport_case` was a thread artifact (1.8 % own). Six proposed slices in "Remaining GPU-host idle", unfiled |
| 2 | CPU node-window line reduction on fixed-grid crystal cases | 15957, 15959 | Adopt-candidate, filed #1446; thread budget input to #1395 |
| 3 | Axial MCWF solver cache and OAM projection | 15944-15946 | Adopt, filed #1437 (cache), #1436 (basis cache, matmul) |
| 4 | Lattice-share preparation | pilot, 15957 | Closed by #1281 (5.5-6x faster complete cases) |
| 5 | Policy-grid sinc spacing, beam entry, cutoffs, characteristic launch overhead | 15962-15983 | Closed by #1452-#1455 and #1453 follow-up; -24.3 % summed wall vs main |
| 6 | Photon transport, characteristic scorer, EEDL/BH scorer, bin-mean host Fourier, coherent finalize | 15936, 15939, 15957, 15958 | Defer: milliseconds in the measured fixtures or no production-sized fixture |
| 7 | Case-level `prange`, BLAS/OpenMP caps for Numba | 15939, 15941, 15959 | No blanket `prange`; set Numba masks explicitly at shard seams (#1395) |

Rejected/negative: more `prange` in the already-parallel BremsLib/CXR
reducers; HOPG 1x4 Numba threads (no gain); per-history FFT batching for
energy-resolved axial MCWF (FFTs are ~6 % of the profile); eager signatures
or a new cache framework for startup (managed disk caches already work).

## CPU findings

**Complete cases.** 16-history cases, one native thread, warm median; the
pilot (15936) ran with all 32 CPUs visible, 15939/15957 on four bound CPUs.

| Case | Pre-#1281 (s) | Post-#1281 15957 (s) | Peak MiB | Leading cost after #1281 |
| --- | ---: | ---: | ---: | --- |
| thin30 | 1.216 | 0.224 | 930 | lattice share 0.16 s |
| thin100 | 1.986 | 0.326 | 690 | lattice share 0.27 s |
| crystal300 | 1.145 (15939) | 0.206 | 690 | lattice share 0.16 s |
| photon-case (Cu 30 keV) | - | 37.0 | 1128 | node-window reducer 29.7 s excl. |
| layered (MoS2/Si 100 keV) | timed out (180 s) | 131.4 | 2085 | node-window reducer 99.6 s excl. |
| mos2_5mev (2 histories) | timed out (180 s) | timed out (600 s) | - | no result accepted |

Pre-#1281, lattice-share preparation was 92-94 % of the HOPG cases
(`SelectionContinuum.share -> share_only_tables -> _corrected_tables`,
bypassing the runner cache); #1435/#1281 fixed it. The first thin30 call
(18 s) is cold JIT. All arms: repeat and profiled outputs bit-identical.

- **Node-window reduction (#1446).** On fixed-grid crystal cases the
  incoherent `accumulate_sincsq_windowed` is 79 % of the Cu case and 74 % of
  the layered case (`production` selection mode scores every shell on the
  production axis). It is already `parallel=True`: 4 Numba threads take the
  Cu case 37.0 -> 15.2 s (2.44x; reducer 29.7 -> 8.1 s), bitwise identical,
  CPU/wall 2.65, +121 MiB (15959). A user-reported `quickstart -m hopg` run
  shows the same sensitivity (39 s serial vs 188 s through a one-thread
  worker, #1422), while HOPG crystal300 1x4 is slower than 1x1: workload
  dependent.
- **Other routes (16 histories, 55 segments; routes, not rankings).**
  Bin-mean 0.757 s (host Fourier fixed cost 0.52 s), coherent 0.037 s,
  coherent at 19 001 nodes 0.589 s (15958). Standalone scorers: photon
  queue 0.051 s (`sample_photoelectric` 0.30 s for 643 events in the opt-in
  Cu case, 0.35 s of 37 s), pair 0.054 s, EEDL 0.015 s, BH 0.024 s, all
  profile-overhead-limited. Characteristic scorer on 4096 rows x 512 nodes:
  4.3 ms multi-element, 2.9 ms layered (NumPy/BLAS only).

**Startup (15937, 3 fresh processes each, README fixture).** Cold JIT first
simulation 34.75 s (compilation 19.99 s, 123-124 compile events), warm disk
14.12 s (zero compile events, 13 cache hits), second call 12.58 / 12.02 s.
Identical result identity in all six; peak RSS 986-992 / 786-794 MiB.
Managed disk caches work for this fixture: no eager-signature or new-cache
case. About 12 s of warm-process work is not compilation and is unexplained.

**Process and thread topology (15939; four identical crystal300 cases per
round, 3 rounds, OS affinity 0-3, exact result hashes).**

| Processes x Numba threads | Round wall (s) | Worker CPU (s) | Sampled peak PSS (MiB) |
| --- | ---: | ---: | ---: |
| 1x1 | 5.753 | 4.87 | 715 |
| 4x1 | 4.738 | 14.71 | 2228 |
| 1x4 | 6.290 | 5.85 | 716 |
| 2x2 | 4.953 (3.94-5.52) | 8.27 | 1269 |

Four processes give ~1.21x batch throughput for ~3x CPU and 3.1x PSS; a
four-thread process is slower than serial on these preparation-dominated
cases. Arm order is fixed and a GPU allocation shares the node, so small
deltas are not stable rankings; cache/bandwidth contention is a hypothesis
without counters. The same crystal300 case on qlmc-queen (same CPU model,
no co-tenant) is bit-identical, 0.956 s vs 1.145 s (15942). **No second CPU
architecture was available.**

**Numba pool budget (15941, local TBB run alike).** Inside
`threadpool_limits(limits=1)` the OpenMP pool reports 1 thread while the
Numba mask stays 4 and a 2048x2048 `_sincsq_reduce` runs at CPU/wall ~4 with
identical hashes. A BLAS/OpenMP cap does not enforce a Numba iteration-team
budget; shard workers must set the mask (or `NUMBA_NUM_THREADS`)
explicitly. Library diagnostic, not a measured oversubscription in a Dask
job (#1395 owns that). `axial_mcwf._fft2/_ifft2` pass SciPy `workers=-1`
independently of those caps (unmeasured).

**Parallel loops and free threading.** BremsLib `_spectrum_tile`/`_band_rows`
and CXR `_cpu_reduce`, `_bin_mean_jit`, `_node_window`, `_fourier_host`
already parallelize independent outputs; `_fourier_host._es_spread`
releases the GIL and bounds its rectangle threads. Electron `cores` bodies
are nopython but have no explicit `nogil`; wrapper GIL overlap is unmeasured.
New `prange` owners are plausible only for private-output loops (packed
characteristic interpolation/line groups, EEDL/BH cells, compatible MCWF
histories); histogram scatter, photon queues, variable-length rejection and
daughter ordering are not safe targets, and no `fastmath`/`nogil` default is
justified. Numba documents experimental free-threading support (0.63);
#1421's matrix could not install a complete free-threaded stack (a
dependency-stack blocker, not universal incompatibility). Shared-state
safety is #1423's audit; nothing here forces the GIL off.

**Energy-resolved axial MCWF (production depth: 5 MeV Si <001>, 32x32,
9-node beam, 1000 A, 1000 histories; queen, 4 bound CPUs).**
Profile (15944, warm 85-88 s, 451 MiB): `_EnergySolvers.get` 46.1 s
(**308 rebuilds**: dissipator 35.3 s + level observer 10.8 s); OAM
`CylindricalProjection.harmonics` 34.8 s (`_fourier_basis` 18.3 s over 7882
calls, `einsum` 16.7 s); split-step `fft2`+`ifft2` only ~4.9 s.

| Arm (15945/15946) | Warm s | Gain | Output vs default |
| --- | ---: | ---: | --- |
| cache 2 (default) | 86.9 / 85.5 | 1.00x | - |
| cache 8 | 41.3 / 42.6 | 2.06x | bit-identical |
| cache 8 + basis cached per `n_theta` | 24.2 / 24.1 | 3.57x | bit-identical |
| + `matmul` contractions | 10.7 / 10.7 | 8.06x | max rel 1.3e-16 (summation order; pinned outputs move) |

Cache 8 vs 2 costs 463 vs 439 MiB (6 builds vs 308); one cached solver
retains ~5 MiB (32x32) to ~9 MiB (64x64), indicative. Filed #1437
(cache sizing) and #1436 (basis cache, matmul with a tolerance/identity
decision). Planar MCWF is not profiled.

## GPU-host findings (cleo_test: HOPG 300 keV / 500 nm, tilts 5-85 deg)

Probe: `gpu_host.py` (existing `_nsys_push` ranges, 10 ms main-thread stack
sampler joined with NVML utilization, harness-only ranges around host
owners) and `nsys_gaps.py` (device busy = union of kernel/memcpy/memset;
idle charged to the innermost main-thread range). RTX 5080, 8 bound CPUs,
policy-grid speed cache cleared per arm. The 45 deg case is refused by the
production grid tolerance in the first profile (fixed by its owner later)
and is left out for comparability. Nsight arms run 4-19 % slower than
sampler arms.

| Sampler wall (s) | main 15966 | stack 15973 | main 42413c99c 15983 |
| --- | ---: | ---: | ---: |
| 5 / 15 / 25 / 35 deg | 63.0 / 106.2 / 62.9 / 70.4 | 54.8 / 89.0 / 53.7 / 60.0 | 46.9 / 79.7 / 48.6 / 54.9 |
| 55 / 65 / 75 / 85 deg | 12.5 / 7.4 / 9.6 / 35.0 | 10.6 / 6.2 / 7.3 / 25.5 | 10.4 / 6.2 / 7.5 / 23.9 |
| **sum** | **367.1** | **307.1** (-16.3 %) | **278.0** (-24.3 %) |
| device busy (traced) | 32 % (15967 trace, 386.7 s) | 37.7 % (324.4 s) | **40.6 %** (299.7 s) |

Deterministic outputs (`E_grid`, `E_grid_brem`, `brem`, `brem_wide`,
electrons, segments) are bit-identical across all three; CXR `spec` varies
with FP64 atomic order; `spec_characteristic` differs from the stack at 6 of
8 tilts in 15983 (digests only; stable across arms within the job and equal
to main at 5 deg; the fused-by-default kernel `63612a907` is the candidate,
not isolated). 15983 also includes `42413c99c`; the gain over the stack is
the sum of both. Single samples; sub-5 % deltas are unresolved.

Idle by innermost range, summed over 8 tilts (seconds), by what fixed it:

| Owner | 15963 base | 15973 stack | 15983 now | Fixed by |
| --- | ---: | ---: | ---: | --- |
| `_resolve_policy_grid` / sinc spacing | 53.3 (+27.6) | 5.4 (+27.6) | 4.7 (spacing gone) | #1453 (+ follow-up) |
| `initial_beam_positions` | 37.5 | 5.1 | 5.0 | #1454 |
| `electron_cutoffs` | 12.7 | 1.8 | 1.8 | #1452 (cutoffs per block; 1e8-draw `arange`, >1.6 GB transient gone) |
| `build_shell_inelastic_tables` | 6.0 | 3.6 | 3.6 | #1452 (one build per case) |
| `BlockSpectrumScorer.add` (characteristic, launch-bound) | 14.8 | 2.9 | 2.9 | #1455 (own chunk; 85 deg: 42 301 launch sets) |

The first profile (15962) showed the same owners at one tilt; the first
Nsight pass (15963) put device busy at 31 % with idle in gaps >=100 ms
(52 %) and <1 ms (25 %). The `cleo_sub`/`cleo_test` profiles were changed by
their owner between runs; only `cleo_test` is compared here.

## Remaining GPU-host idle (15983 traces re-reduced, 15986 sampler)

Device idle is 178.0 s of 299.7 s traced (59 %) over the 8 tilts. Three
views, each with its own blind spot:

1. **Nsight, innermost range on the main thread** (15983 as first reduced):
   `_transport_case` held 34.7 s (11.6 %), the largest owner.
2. **Nsight, innermost range across all threads** (`nsys_gaps.py`, offline
   re-reduction, job 15988; base numbers identical to view 1): the same idle
   seconds charged to the latest-started open range on any thread.
3. **Sampler (15986)**: main-thread stack joined with NVML. NVML reports a
   coarse per-period busy fraction, so it finds 48 s (18 % of samples) against
   Nsight's 178 s; sub-millisecond launch gaps are invisible to it, as is every
   worker thread.

**Finding: the uncovered `_transport_case` idle was a thread artifact.**
Scoring and selection run on worker threads (`pyrite-stream-score`: overlapped
streamed scoring, #862) and the beam-entry prefetch (#1454) on another; the
reducer kept only main-thread ranges, so device idle during worker host work
was charged to whatever the main thread had open, usually `_transport_case`
while it waited. Over all threads `_transport_case` falls from 34.7 s to
**5.5 s** (1.8 %). Its own host time in the sampler is 4.2 s, 4.1 s of it
`_backend.release_memory` (CuPy pool release). Off-main-thread wall over the
8 tilts: `PrefixSelectingScorer.add` 102.5 s, `BlockSpectrumScorer.add`
96.2 s (30 calls each), `initial_beam_positions` 72.7 s (812 calls),
`cuda_fourier_bin_mean_reduce` 52.4 s (159), `BlockGridEvidence.add` 45.9 s,
`sinc_feature_samples` 23.3 s.

Idle seconds by innermost range (main thread only -> all threads), share of
traced wall for the all-thread value:

| Range | Main only | All threads | Share | Thread / mechanism |
| --- | ---: | ---: | ---: | --- |
| `StoppingMonitor.on_block` | 18.4 | 18.4 | 6.1 % | main; per-block CXR mass; 13.3 s host time in the sampler (`to_cpu` 3.6 s, `_ordered_bincount._route` 1.7 s, `expand_escape_pieces` 1.4 s); 40 966 gaps closed by a memcpy, mean 0.16 ms |
| `initial_beam_positions` | 5.0 | 17.6 | 5.9 % | prefetch thread (#1454); the look-ahead does not fully hide it |
| `cuda_fourier_bin_mean_reduce` | 2.4 | 13.0 | 4.3 % | scoring worker; Python-side driver of the device Fourier reducer |
| `BlockSpectrumScorer.add` | 2.9 | 11.9 | 4.0 % | scoring worker |
| `block_tail_sums` | 11.7 | 11.7 | 3.9 % | main; case-start CXR stop search on host-resident populations (`cxr_seeds.lost` -> `_population_tail.py:86`); `DeviceTail` exists but is chosen only for device-resident populations |
| continuation `launch` / `stage` / `assemble` | 11.7 / 11.5 / 10.2 | 11.7 / 11.5 / 9.3 | 3.9 / 3.8 / 3.1 % | main; host syncs per round (`_gather_index`, `_pending_order`, `run_transport_kernel`); 100 602 gaps closed by `cupy_take`, mean 0.04 ms |
| `BlockGridEvidence.add` | 1.1 | 9.1 | 3.0 % | scoring worker (streamed line-grid evidence) |
| `transport.core` | 9.8 | 8.9 | 3.0 % | main |
| `ResidentBlockRecovery.simulate` | 7.3 | 7.2 | 2.4 % | main |
| `_transport_case` (own) | 34.7 | 5.5 | 1.8 % | main; pool release |
| `_corrected_tables`, `_resolve_policy_grid`, `sinc_feature_samples`, `transport.upload` | 4.3, 4.7, 2.6, 4.3 | 5.1, 4.7, 4.4, 4.3 | 1.4-1.7 % | case-start preparation |

What closes the gaps (idle grouped by the first device activity after each
gap and the range open at that moment; the whole gap is charged to the
closing range, so this says what the device waited *for*, not which code ran
during the wait): host-to-device `memcpy` in `transport.upload` closes 30 041
gaps worth 57.5 s (32 % of all idle, mean 1.9 ms): the device waits for the
next block's host-prepared input. Elementwise CuPy kernels in the uncovered
`_transport_case` close 74 659 `true_divide` gaps (2.6 s, mean 0.03 ms) and
142 663 `multiply` gaps (0.85 s): launch-bound eager chains.

Case-start host stalls in the sampler (main thread, 15986): the host CXR stop
search 7.6 s of NVML-idle samples, lattice-share table load 1.7 s,
`xsgen.store.arrays` shell-table loads 1.3 s (`transport.inelastic`) and
0.7 s (`ResidentBlockRecovery.simulate`), `cxr_seeds.__init__` 1.9 s,
main-thread beam entry 5.0 s (equal to the Nsight main-only value),
continuation `_gather_index` 1.8 s.

Disposition (proposed slices, ranked by idle seconds as an upper bound on
what overlap could recover; none is a measured gain, and each must keep
bit-identical deterministic outputs, stream/RNG identity and stopping
decisions):

1. **Scoring-worker host work** (`cuda_fourier_bin_mean_reduce`,
   `BlockSpectrumScorer.add`, `BlockGridEvidence.add`: 34 s, 11.3 %): the
   device idles while the worker's Python drivers issue small launches; the
   bin-mean driver's 1.63 s exclusive Python time (15962) is still
   unexplained.
2. **Stopping monitor's per-block CXR mass** (18.4 s, 6.1 %): a host sync and
   escape expansion per block; cadence or incremental update is bounded by
   the stopping-decision contract (`checks/adaptive_decision_cadence_1253.py`).
3. **Beam-entry prefetch depth** (17.6 s, 5.9 %): #1454 hides 72.7 s of host
   work but leaves 17.6 s of idle; look-ahead depth against the device block
   time of each tilt is the first thing to measure.
4. **Continuation host stages** (launch/stage/assemble + `transport.core`,
   41 s, 13.8 %): per-round host syncs and 100k small `take` gaps.
5. **Case-start host CXR stop search** (11.7 s, 3.9 %): device route for
   host-resident populations, or overlap with transport; the tail sums must
   stay within the existing rounding/bracketing contract.
6. **Case-start tables and pool release** (about 8 s): shell-table and
   lattice-share loads, `release_memory`.

The earlier plan to add finer ranges inside `_transport_case` is moot: its own
idle is 1.8 %. Limits: one sample per tilt, one hardware configuration, one
profile; the all-thread charge picks the latest-started open range when two
threads overlap, so idle that two threads share is assigned to one of them;
the sampler job queued 70 min behind a CPU-only job and ran alone.

## Coverage gaps

Not measured or not qualified, stated rather than inferred: MoS2 5 MeV / 1 mm
(no result at 600 s, 2 histories); planar MCWF; production-sized coherent
and bin-mean CXR; hBN GPU-host case (blocked on #1449); a second CPU
architecture; hardware counters, LLVM/vectorization and wrapper GIL overlap;
aggregate-node memory beyond the sampled PSS in 15939; Dask/shard topology
under a real worker (#1395). Heavy or multi-GiB work stays remote.

## Evidence index

| Job | What | Producer | Retained under `results/job-<id>/` |
| --- | --- | --- | --- |
| 15936 | Unbound pilot (32 CPUs visible), 10 arms | `c0a09186e` | `summary.json` (raw dropped) |
| 15937 | Startup cold/warm (+ affinity-unbound case arms) | `36352a534` | startup JSON, summary |
| 15939 | Bound topology, characteristic scorers | `bfeb31875` | topology and char-* JSON, summary |
| 15941 | Numba pool budget | `4ae34de80` | pool-budget JSON |
| 15942 | Second node (queen) | `4ae34de80` | portability JSON |
| 15944-15946 | Axial MCWF profile, cache, prototypes | `6cc9a6ed3`, `c51030eeb`, `c550bc701` | JSON |
| 15957-15959 | Post-#1281 cases, CXR variants, 4 threads | `86128404c` and kin | case JSON, summary |
| 15962 | GPU-host first look (`cleo_sub`) | `63b4dff43` | profile and sampler JSON, summary |
| 15963-15965, 15973 | Nsight idle attribution and finer splits | `c80e5374a` .. `1cb284d7f` | `*-gaps.json`, summary |
| 15966 | Main baseline for the stack | `0ae722442` | summary |
| 15983 | Current main remeasure | `9035c313b` | gaps and sampler JSON, summary |
| 15986 | Sampler attribution of the remaining idle | `29f9a65fc` | summary (raw sampler JSON kept locally) |
| 15987-15988 | Offline re-reduction of the 15983 traces | `d4b8baec3` | replaces `job-15983/*-gaps.json` (superset) |

Manifests list SHA-256 of retained and dropped (not retained) files; raw
Nsight reports (`.nsys-rep`) and bulky per-arm logs stay on the box under
the isolated checkouts. Jobs 15935 and 15938 never produced results.
