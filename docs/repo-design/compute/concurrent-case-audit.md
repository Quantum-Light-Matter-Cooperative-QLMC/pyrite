# Concurrent cases in one interpreter: audit and thread-scaling probe

Audit for [#1423](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1423),
first taken at main `42413c99c`, rebased onto the #1432 native-CPU audit
(`checks/native_cpu_audit/`), whose kernel inventory and process/thread
topology this page extends. Question: do concurrent cases in one interpreter
offer enough throughput or memory benefit to justify a production executor?

**Recommendation: defer a production executor; threads are not a general
replacement for the process pool.** On the HOPG slab case, four threads nearly
match a four-process pool (2.4-2.6x against 2.5-2.9x serial) at well under half
its PSS, on the workstation and on the remote box. On other case families they
do not. Planar channeling scales only 1.8x on threads against 3.75x on
processes, because its time is Python-level work under the GIL. The layered
MoS2-on-Si case reaches 1.5-1.7x on either arm. Pure transport does not
scale on threads until its kernels release the GIL (#1458). The memory benefit
is real but family-dependent. A thread executor would have to be opt-in per
workload, with the blockers below fixed and a separate reviewed plan. Digests
equalled serial in every arm. No executor, default, tolerance, seed semantics
or checkpoint format changed.

## Measurement

`scripts/concurrent_case_probe.py` runs 8 independent jobs serially and on N
threads or a warmed `spawn` process pool, with Numba, OpenMP and OpenBLAS pinned
to one thread, and reports best-of-3 wall time, peak summed PSS (and, since
2026-11-26, USS) of the process tree, and whether every result digest equals the serial run. Jobs are the
maintained #611 HOPG case (`checks/streamed_scoring_memory.py`, 30 keV, 1 um,
seeds 0-7): ELSEPA elastic, BremsLib continuum with `bremslib-soft-hard`, fixed
400-node line grid, per-electron core. `pyrite tables verify` passes
(186/186) and a `warnings.catch_warnings` check recorded no physics-fallback
warning. Mott was never selected.

| Environment | Value |
|---|---|
| Host | i7-13620H, 16 logical CPUs (hybrid P/E cores), 30 GiB, shared dev workstation, not the remote box |
| Python / GIL | 3.14.10, GIL enabled |
| NumPy / Numba / llvmlite | 2.5.5 / 0.68.1 / 0.50.0; `NUMBA_NUM_THREADS=1`; threading layer `default` |
| Backend | `PYRITE_MC_BACKEND=cpu`, under `systemd-run MemoryMax=8G` |

Full case (`run_case`, 1024 histories per job, 8 jobs):

| Arm | 1 worker | 2 workers | 4 workers | Speedup at 4 | Peak PSS at 4 |
|---|---|---|---|---|---|
| threads, current source | 8.91 s | 5.96 s | 3.37 s | 2.6x | 1276 MiB (1.8x serial) |
| threads, every `@njit` `nogil` (prototype) | 9.87 s | 5.47 s | 3.46 s | 2.9x | 1340 MiB |
| process pool (`spawn`, warmed) | 9.61 s | 5.37 s | 3.07 s | 3.1x | 2917 MiB (4.1x serial) |

Serial PSS is 707 MiB. The `nogil` prototype adds nothing to full cases: their
time is dominated by GIL-releasing NumPy and Numba work, and transport is a small
share at this size. At 128 histories the same arms gave 2.96 s serial, 1.55 s
(GIL) and 1.60 s (`nogil`) at four threads; at 16 histories 4 threads gave
only 1.2-1.4x because jobs are 0.26 s each.

Transport only (`runner._transport_case`, 40,000 histories per job, 8 jobs):

| Arm | 1 worker | 2 workers | 4 workers |
|---|---|---|---|
| threads, current source | 5.81 s | 6.05 s | 9.61 s |
| threads, every `@njit` `nogil` (prototype) | 5.53 s | 5.05 s | 3.57 s |
| process pool | 5.48 s | 3.86 s | 3.39 s |

With the GIL held, threaded transport is no faster than serial and slower at
four threads (an earlier alternating run measured 16.98 s). Releasing the GIL
recovers 1.55x at four threads, close to the 1.62x of processes. Samples are
noisy (the 2-thread GIL arm ranged 6.05-16.59 s), so read ratios, not
decimals. Transport PSS here (6.2 GiB serial, 9.8-12.3 GiB parallel) is
dominated by the 8 retained result sets, not the working set.

**Repeat with bound CPUs and USS (2026-11-26).** Unbound repeats of the full
case on this hybrid-core host varied 6.9-13.5 s serial between identical runs,
so the arms were rerun under `taskset -c 0,2,4,6` (one SMT thread on each of
four P-cores), Python 3.14.10, branch source after the math-mode merge (#1468;
default `reproducible`). Seeds were submitted in shuffled order and digests
compared by seed.

| Arm, 1024 histories | 1 worker | 2 workers | 4 workers | Speedup at 4 | Peak PSS / USS at 4 |
|---|---|---|---|---|---|
| threads, current source | 7.03 s | 4.03 s | 2.74 s | 2.6x | 1330 / 1325 MiB (1.8x serial) |
| process pool (`spawn`, warmed) | 7.13 s | 3.74 s | 2.45 s | 2.9x | 2914 / 2781 MiB (4.2x / 4.0x serial) |

The memory ratio holds under USS: threads share almost no pages a process pool
would count once, and spawned workers duplicate the tables and JIT state
(serial USS about 0.7 GiB). At 128 histories (2.77 s serial) threads reached
1.55x at two workers and no further gain at four, so a thread executor pays
only for cases of roughly a second or more. Digests equalled serial in every
arm and order.

**Stop, drain and exceptions.** `PROBE_STOP=1` makes job 2 of 8 raise; the
coordinator takes results from `as_completed`, cancels pending futures on the
first exception, then calls `shutdown(wait=True)`. Bound as above, 1024
histories:

| Arm | Workers | Started | Cancelled | Completed after error | Drain |
|---|---|---|---|---|---|
| threads | 2 | 4 | 4 | 2 | 0.9 s |
| threads | 4 | 5 | 3 | 4 | 1.4 s |
| process pool | 2 | 7 | 1 | 5 | 2.3 s |
| process pool | 4 | 8 | 0 | 7 | 2.7 s |

The exception reached the coordinator unchanged in both arms and every
completion was observed on the coordinator thread. Threads cancel every job not
yet running. `ProcessPoolExecutor` moves up to `max_workers + 1` jobs into its
call queue ahead of execution, and `cancel()` fails on those, so after a stop
it ran work that had not started: at four workers nothing could be cancelled.
The #928 contract (drain started tasks, start no more) is therefore easier to
meet with threads. The production `cpu_pool` had the same flaw on a bounded
sweep: it submitted every case up front, so after the deadline up to
`max_workers + 1` queued cases still began. It now keeps at most `max_workers`
cases submitted when bounded ([#1478](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1478)),
as `gpu_pipeline` already did. Running jobs cannot be interrupted in either arm; drain time is the
remainder of the running cases.

**Remote box and more case families (2026-11-26).** qlmc-queen (Ryzen 9
9950X3D, 32 logical CPUs), Python 3.14.10, NumPy/Numba as above, math
`reproducible` (the built-in default). Slurm allocated 8 CPUs but did not bind
them (affinity 0-31), so the node was otherwise idle rather than pinned. Jobs
16062 (`a63978c31`) and 16063 (`434508e53`). `PROBE_FAMILY=layered` is the
#1432 MoS2-on-Si stack (100 keV, 4 histories). `PROBE_FAMILY=channeling` is
the admitted planar Si demo (17 MeV, 2 um, plane (1-10), 50 histories) through
`pyrite.simulate`. HOPG rows are best of 3; the other families are one sample.
Process pools were not run at 8 workers for those families, to stay inside the
node's memory.

| Family | Arm | 1 | 2 | 4 | 8 | PSS at 4 |
|---|---|---|---|---|---|---|
| HOPG, 1024 histories | threads | 4.73 s | 3.03 s | 1.97 s (2.4x) | 1.86 s (2.5x) | 1583 MiB |
| HOPG, 1024 histories | processes | 4.74 s | 2.85 s | 1.90 s (2.5x) | 1.62 s (2.9x) | 2922 MiB |
| HOPG, 128 histories | threads | 2.15 s | 1.58 s | 1.34 s (1.6x) | 1.46 s | 1120 MiB |
| layered MoS2/Si | threads | 130.8 s | 93.0 s | 85.7 s (1.53x) | 83.3 s (1.57x) | 5313 MiB |
| layered MoS2/Si | processes | 130.2 s | 85.4 s | 76.9 s (1.69x) | – | 7337 MiB |
| planar channeling | threads | 98.8 s | 66.3 s | 55.6 s (1.78x) | 54.2 s (1.82x) | 870 MiB |
| planar channeling | processes | 99.2 s | 50.2 s | 26.4 s (3.75x) | – | 1977 MiB |

USS tracks PSS within 3 % for threads and 5-10 % for processes. At 8 HOPG
workers the process tree reached 5138 MiB against 2058 MiB for threads.
Channeling's thread arm stalls at about 1.8x: the planar path spends its time
in Python and in NumPy calls too small to release the GIL usefully, so only
processes scale. Layered stalls on both arms, consistent with #1432's finding
that its node-window reducer dominates. Its share after #1469 was not profiled
here. Memory favours threads in every family, by 1.4x (layered) to 2.3x
(channeling) at four workers.

**Not measured, so not claimed.**

- The #1432 audit measured four processes at 1.21x for four identical
  `crystal300` cases on four bound CPUs of a shared GPU node. Its allocation
  differs from this probe's, so the two topology results are not directly
  comparable.
- Adaptive precision, cone split, photon, axial MCWF and GPU paths. Repeated
  samples for layered and channeling.
- More than one Numba thread per job with concurrent launches (see row 10).
- A free-threaded interpreter: none could be installed (#1421).
- Memory sampling is PSS and USS of the process tree at 50 ms; no allocator
  breakdown. Shutdown was measured only for a raised exception, not for
  `KeyboardInterrupt` or a killed worker.
- A first-pass probe that called `_simulate_trajectories` directly ran the
  BremsLib-less fallback and an unrelated attempt that hand-built a case dict
  selected Mott; both were discarded, not used above.

## Kernels, `prange` and SVML

**GIL.** There are 143 `@njit` decorators in `src/pyrite` (an earlier count of 230
also matched CuPy `@jit.rawkernel` GPU kernels, which are not Numba). Only 3 declare `nogil=True`
(`spectrum/_population_tail.py:23`, `transport/kinematics.py:281`,
`spectrum/cxr/_fourier_host.py:62`); 12 are `parallel=True`. The per-electron
entry points that Python calls (`core` in `transport/core_adapters.py:160`, and
its grooved, LUT, cone-split and mixed siblings) are `@njit(cache=True)` and hold
the GIL for the whole batch. The #1432 report states the same ("no explicit
`nogil`; wrapper GIL overlap is unmeasured"); the transport table above is that
measurement. Numba compilation does not imply `nogil`, and #1432 judges a
blanket `nogil` default unjustified: it would need RNG-parity and performance
evidence per kernel. Only the entry points that Python calls matter for thread
scaling; callees compile into them.

**More `prange`.** Not recommended inside a case. The per-electron core writes
one shared `segments` buffer with counters and replay capacity, so a `prange`
over electrons needs per-thread buffers and a deterministic merge (#1432's
inventory: "counters, segment capacity/replay and buffers need redesign").
Existing `parallel=True` loops already cover independent bins and rows
(`_cpu_reduce`, `_node_window`, `_bin_mean_jit`, `_fourier_host`,
`_brem_cpu_fused`), and #1432 measured four Numba threads in one process slower
than serial on preparation-dominated cases. Case-level concurrency obtains the
same parallelism without touching kernel loops, because every case already owns
its buffers. A thread executor plus `nogil` entry points is therefore the
smaller lever for transport; a new `prange` is not.

**SVML and `fastmath`.** Superseded by
[CPU sine vectorization](cpu-sine-vectorization.md) (#1459), which adopted the
portable `vector_sin` and keeps SVML as an opt-in `svml` environment, and by
[math mode](math-mode.md) (#1468), which owns any `fastmath` variant. Neither
changes the threading question: kernels compiled under either mode still hold
the GIL unless declared `nogil` (row 1), and the `fast` guarantee is
run-to-run determinism at a fixed thread count, which concurrent cases with one
Numba thread each preserve. This audit's earlier single-kernel microbenchmark
is kept with its evidence in `~/dev/pyrite-benches/issue-1423/svml/`.

## Shared-state register

Status key: **fixed** = changed in this branch; **blocker** = must change before
a threaded production executor; **ok** = safe for the shape described; **open** =
not evaluated; **n/a** = not reachable by the CPU case path.

| # | State | Owner / reference | Protection and tests today | Missing | Status |
|---|---|---|---|---|---|
| 1 | Hot-kernel GIL | `@njit(cache=True)` entry points (above) | none | `nogil=True` on audited entry points, with RNG-parity and performance evidence, plus a scaling test | **blocker** for transport-only throughput; irrelevant to full cases |
| 2 | Transport-core pin | `runner/adaptive.py` `_adaptive_worker_call` used to swap `PYRITE_MC_TRANSPORT_CORE` in `os.environ` | per-process | now `realization.core_pin_override` (a `ContextVar`) read by `resolve_transport_core` and `required_core`; `tests/scan/test_adaptive_api.py::test_adaptive_worker_pin_is_isolated_between_threads` | **fixed** |
| 3 | Worker-wide env/thread setup | `runner/__init__.py` `_worker_init` (`set_canonical_env`, `numba.set_num_threads(1)`, `_RESOURCE_POLICY.gpu = False`); `scheduling._single_thread_blas` | designed for a spawned process | process-global; a thread cannot opt in per case | **blocker** |
| 4 | Environment-driven configuration | 124 distinct `env_value("PYRITE_...")` reads | stable within a process | per-case values cannot differ between threads | **blocker** for per-case env |
| 5 | LRU memos | `materials/atomic.py` `_HENKE_MEMO`, `materials/photon_cross_sections.py` `_TOTAL_MEMO` (`OrderedDict`) | `get` then `move_to_end` raised `KeyError` when another thread evicted the key between the two calls | now locked; `tests/materials/test_memo_thread_safety.py` fails 3/3 with the locks disabled at a 1 us switch interval | **fixed** |
| 6 | Plain-dict caches | `runner/lattice_share.py` `_CACHE`, `transport/stopping.py` `_CROSSOVER_CACHE` | each `get`, `set` and `clear` is atomic; a race only repeats a deterministic computation | none needed | ok |
| 7 | Locked caches | `transport/shell_transport.py` `_table_cache_lock`; `transport/radiative_moment_transport.py` `_cache_lock`; `materials/catalog.py` double-checked `_DEFAULT_CATALOG_LOCK` | locks present | a concurrent test per cache | ok, untested |
| 8 | Lazy module singletons | `lattice_share._KERNEL`; `_BIN_MEAN_KERNEL`, `_TREE_KERNEL` and similar; `_DUMMY_F32` | idempotent assignment | confirm each initializer is idempotent and import-safe | ok, unverified |
| 9 | `functools.cache`/`lru_cache` tables | `photon_cross_sections._table()` and ~35 others | CPython serializes the dict; two threads may both compute the first value | read-only after build | ok |
| 10 | ContextVar state | tuning, spectrum launch, precision, progress, recovery engine, fallback audit, line-data coverage (`runner/transport_options.py:15-22`, `runner/__init__.py:79-82`, `runner/recovery.py:13`, `_physics_fallback.py:18`, `spectrum/cxr/_coverage.py:45`, `_bin_quadrature.py:801`, `transport/beam_prefetch.py:32`) | a new `ThreadPoolExecutor` thread starts with an empty context. The probe set none of them in the caller, so it does not exercise propagation; `stream_overlap.py` and `beam_prefetch.py` call `copy_context()` explicitly, pinned by `tests/montecarlo/runner/test_stream_overlap.py::test_scoring_thread_sees_the_callers_context` and `tests/montecarlo/transport/test_beam_prefetch.py::test_prefetch_worker_sees_the_callers_context` (both fail with the copy removed) | an executor must copy the caller context per submitted case and own a per-case fallback-audit and progress sink | **blocker** for a case executor; existing worker threads ok |
| 11 | Numba `parallel=True` | 12 kernels | `numba.set_num_threads(1)` only in pool workers. Numba documents that `workqueue` is not thread-safe and aborts on nested parallelism; `default` promises nothing and `threadsafe` needs `omp` or `tbb` | pin `NUMBA_THREADING_LAYER=threadsafe` and test two concurrent launches; the probe ran with one Numba thread | **blocker** |
| 12 | RNG and seed ownership | per-case `seed` consumed as `default_rng(p.seed)` or `SeedSequence(seed).spawn(...)`; no `np.random.seed` or global generator in `src/` | the 8 seeds produced digests equal to serial in every thread and process arm above, also with shuffled submission order | cone-split path; channeling and layered digests matched serial in the probe only | ok for the measured path and adaptive precision: `tests/scan/test_adaptive_api.py::test_adaptive_cases_on_threads_match_serial_and_own_their_arrays` runs three seeds on two threads in reverse order and gets serial outputs and realized counts bit for bit |
| 13 | `_RESOURCE_POLICY` | `runner/oom.py` mutates `pool_peak_bytes`, `cases_since_free`, `pool_limit_*`; `chunking.py` rebinds `gpu` | single-driver design | CPU path reads only; GPU OOM counters are not thread-safe | **blocker** for any GPU use |
| 14 | Host-memory admission | `runner/host_admission.py`, `transport/host_memory.py` read `MemAvailable`/cgroup headroom per call; the docstring says a projection is not a reservation | check-then-act | N threads admit against the same headroom and can over-commit together; needs a shared reservation ledger | **blocker** |
| 15 | Worker-count and memory budget | `host_admission._cpu_pool_workers` (`worker_mem_mb`, about 5.5 GiB per full-case process measured earlier) | sized for processes | a thread budget needs its own measured per-case footprint (this probe: about 0.19 GiB marginal PSS per extra thread at 1024 histories, against about 0.7 GiB per extra process) | open |
| 16 | Device contexts, streams, pools | `runner/stream_overlap.py` one scoring thread on a non-blocking stream | one owner by design; #1411 owns multi-GPU | none for a CPU slice | n/a |
| 17 | Callbacks, stop and drain | `execution/protocol.py` requires `callback` in the coordinating process and draining of started tasks on `should_stop` | #928 contract; `LocalExecutor` delegates to `run_cases` | probe: exceptions reach the coordinator, completions are observed there, and threads cancel every unstarted job, while a process pool runs up to `max_workers + 1` queued jobs after a stop (above); a production executor still needs these as tests; the bounded process `cpu_pool` no longer overruns a stop (#1478, fixed) | ok for threads (probe only) |
| 18 | Checkpoint and result isolation | callback owns persistence; results returned by value | a process boundary copies today | rule: a `run_case` output owns every array in it, sharing no memory with another case's output or with any case input, so the caller may mutate or keep it; the callback that receives it owns persistence. Checked for adaptive outputs by the test in row 12; fixed-N and channeling outputs are unchecked | ok for adaptive outputs |
| 19 | Concurrent tests | `test_montecarlo.py` and `test_gpu_oom_retry.py` replace `ProcessPoolExecutor` with a synchronous fake | none ran two cases at once; this branch adds the two tests named in rows 2 and 5 | a concurrent test per locked cache (row 7); an executor-level stop/drain test (row 17). Thread digest parity (row 12) and ContextVar propagation at existing worker threads (row 10) are now covered | **blocker** until an executor exists |

## Free-threaded Python

No free-threaded arm was run. Its benefit would depend on rows 3, 4, 10, 11 and
14 being resolved and on Numba and extension modules declaring support. The GIL
must not be disabled around extensions that have not.

## Decision and follow-ups

- **Defer** adoption; do not reject. Threads match processes only where the
  case's time is GIL-releasing native work (HOPG slab). There they use about
  half the memory, on two hosts. Channeling, which is Python-heavy, scales half
  as well on threads. An executor would therefore be an opt-in per workload,
  never a replacement default.
- Fixed beside this audit: the bounded `cpu_pool` no longer starts queued
  cases after a stop (#1478).
- Done here, valid without any executor: the core pin no longer touches
  `os.environ` (row 2) and the two `OrderedDict` memos are locked (row 5).
- Before a plan can be written: resolve rows 3, 4, 10, 11 and 14; measure
  adaptive precision; decide whether `nogil` on the per-electron
  entry points is worth its parity evidence (row 1), which matters for
  transport-only workloads and not for full cases (#1458). #1458 also
  covers thread-fed CPU transport for the GPU pipeline. A free-threaded arm,
  the only lever for channeling's Python-level work, waits for #1421.
- Not pursued: a new `prange` over electrons. SVML and `fastmath` now belong
  to #1459 and #1468.
- Raw logs and the SHA-256 manifest are kept locally in
  `~/dev/pyrite-benches/issue-1423/` (not committed).
