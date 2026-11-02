# CPU startup latency

`pyrite-dev startup` measures the README scalar simulation in fresh CPU processes:
HOPG, 30 keV, a 10,000 Å slab tilted by 30°, the default scalar detector,
450 line electrons, 100 bremsstrahlung electrons, and the scalar API's seed 1.
It reports access to `pyrite.Beam`, scene setup, the first simulation, a second
simulation in the same process, and their combined first-use time. Timings exclude
interpreter launch and benchmark-harness imports.

## Reproduce

Install the [pinned physics tables](../development-workspace.md) first. Run from
the checkout being measured, with its locked dependencies and bundled catalog:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache NUMBA_CACHE_DIR=/tmp/pyrite-startup-numba \
  pyrite-dev startup --cache warm --repeats 3 --json \
  --profile /tmp/pyrite-startup.prof > /tmp/pyrite-startup.json
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python -m pstats /tmp/pyrite-startup.prof
```

Warm mode runs an unmeasured priming process before the samples. Cold mode
(`--cache cold`) gives each sample an empty temporary Numba cache and leaves the
user's cache intact. Neither mode clears fetched table files or the operating
system's filesystem cache. Other environment and configuration overrides are
inherited; keep them identical when comparing revisions. Run timing arms
sequentially with no concurrent test suite or simulation.

`--profile` records imports, setup, and the first simulation in a separate fresh
process after the timing samples. Profile times include instrumentation overhead
and must not be compared directly to the unprofiled samples.

JSON stdout contains one `pyrite.startup-benchmark.v1` report: workload, backend,
cache policy, Python/platform metadata, per-stage median/minimum/maximum seconds,
and each sample's timings, dependency versions, peak resident memory, identity
digest, and spectrum-array SHA-256 hashes. Each sample's `numba_jit` record
attributes compilation: wall time spent compiling during the first and the
repeated simulation, compilation counts, disk-cache hits and misses of every
PyRITE Numba kernel, and exclusive compile time per kernel (callee compilation
is not double counted). It comes from Numba's `numba:compile` event, which fires
only on a cache miss, and from each dispatcher's hit/miss counters
(`pyrite._jit_cache`). Peak memory covers both simulations
within the child process. Diagnostics use stderr; runtime failures exit 1,
invalid arguments exit 2, and interruptions exit 130. Samples with inconsistent
identities or spectrum hashes fail the benchmark.

## Implementation

Public scene construction defers execution imports. `campaign.model` loads
lowering only when constructing executable cases, and the cost proxy loads
transport when estimating stopping. `montecarlo` and `instrument` resolve their
existing exports on demand; detector response operators load when requested.
Exports retain their owning implementation objects and appear in `dir()` before
being loaded.

Shell-table construction repeatedly reads the same free-atom and conduction-band
inputs across its energy nodes. Their validated parsing results have bounded
process caches keyed by exact source bytes and source path. Each load still reads
the file, and the default atomic-shell load still verifies the pinned checksum.
Changed contents therefore invalidate the cache. Returned dictionaries and
conduction-band formula mappings remain owned by each caller.

These caches avoid repeated parsing without introducing persistent derived-table
artifacts or changing transport equations, RNG streams, or identity inputs.

## Recorded comparison (2026-10-06)

Baseline: `f3b44c1e`. Candidate: the lazy imports and content-keyed parsing on
`issue-346-startup-latency`. Both ran the same benchmark harness against their
own normally installed checkout. Hardware: AMD Ryzen 9 5900X, 24 logical CPUs,
WSL2 Linux 6.18.33.1, glibc 2.43; Python 3.14.6, NumPy 2.5.3, Numba 0.67.0.
Backend: CPU; no GPU. Fetched physics files and filesystem caches were retained.
Each warm-cache arm used its own Numba cache directory, one unmeasured priming
process, three measured fresh processes, and a separate profile process. The
arms ran sequentially after tests had stopped. Values below are median
(minimum–maximum), in seconds:

| Warm-cache stage | Baseline (s) | Candidate (s) | Median reduction |
|---|---:|---:|---:|
| `import pyrite; pyrite.Beam` | 1.323 (1.310–1.648) | 0.724 (0.722–0.729) | 45% |
| First `simulate` | 8.220 (7.659–8.452) | 4.403 (4.376–4.548) | 46% |
| Combined first use | 9.543 (9.307–9.762) | 5.127 (5.105–5.271) | 46% |
| Second `simulate` in the same process | 1.414 (1.378–1.620) | 0.952 (0.825–1.053) | 33% |

Peak resident memory remained about 1.49 GB in both warm-cache arms. These are
local measurements of this scalar workload, not a GPU or sweep throughput claim.

Three additional cold-cache processes per revision each received an empty
temporary Numba cache, without a priming process:

| Cold-cache stage | Baseline (s) | Candidate (s) | Median reduction |
|---|---:|---:|---:|
| `import pyrite; pyrite.Beam` | 1.441 (1.301–1.504) | 0.791 (0.724–0.820) | 45% |
| First `simulate` | 18.440 (17.872–19.607) | 16.337 (15.548–18.034) | 11% |
| Combined first use | 19.945 (19.174–21.048) | 17.157 (16.339–18.758) | 14% |

Cold-cache peak resident memory was about 1.63 GB in the baseline and 1.62 GB in
the candidate. First compilation remains a substantial cost in cold mode.

The separately instrumented warm-process profiles explain the table-building
cost. Cumulative times overlap and must not be summed:

| Profiled operation | Baseline (s) | Candidate (s) |
|---|---:|---:|
| Shell inelastic table construction | 6.424 | 0.734 |
| Atomic-shell loads (276 calls) | 5.140 | 0.097 |
| Conduction-band loads (186 calls) | 0.506 | 0.017 |
| Numba cache loading (9 calls) | 0.479 | 0.355 |
| Residual Numba compilation | 2.089 | 1.944 |
| Catalog construction | 0.558 | 0.461 |
| CIF loading (49 calls) | 0.494 | 0.411 |

The baseline also spent about 0.029 s resolving ELSEPA layer tables, 0.139 s
resolving SBETHE composition tables, 0.011 s loading the EPDL photon table, and
0.079 s loading/preparing BremsLib tables. Those operations were much smaller
than repeated shell parsing. A persistent derived-table cache was therefore
deferred in favor of bounded process caches for the measured hotspot. Warm mode
does not eliminate every Numba compilation; residual compilation remains visible
in both profiles.

All 12 warm and cold samples across both revisions have identity digest
`6c8b914b1af553fa7cb7bee1c642eeeee1500da2e380694e8da024edcfd9a94a`.
The SHA-256 hashes of `energy_eV`, `spectrum`, `background_energy_eV`, `background`,
and `characteristic_spectrum` also match exactly. `coherent_spectrum` is absent
for this incoherent workload.

## Compilation attribution and persistent kernels (2026-11-01, #910)

Harness: `pyrite-dev startup` with the `numba_jit` attribution above. Hardware:
Intel Core i7-13620H, 16 logical CPUs, Linux 7.0.0, glibc 2.43; Python 3.14.4,
NumPy 2.5.4, Numba 0.68.0. Backend: CPU. Base revision `abd601ae8`. Three
cold processes, then one priming and three warm processes sharing one
`NUMBA_CACHE_DIR`, plus a separate profile process. Values are median
(minimum–maximum) in seconds:

| Stage | Cold cache | Warm cache |
|---|---:|---:|
| `import pyrite; pyrite.Beam` | 0.861 (0.850–0.945) | 0.857 (0.820–0.879) |
| First `simulate` | 32.341 (32.217–32.931) | 3.892 (3.855–3.962) |
| of which Numba compilation | 28.279 (28.127–28.918) | 0 |
| Second `simulate` in the same process | 1.525 (1.510–1.561) | 1.504 (1.499–1.522) |
| Compilations / cache hits / misses (first call) | 117 / 0 / 62 | 0 / 11 / 0 |

A cold first call compiles 62 PyRITE kernel signatures (117 compilations
including Numba's own array helpers). Two specializations of the CPU transport
body dominate: `make_cpu_transport_core.<locals>.body` (10.6 s exclusive) and
`.core` (3.3 s), then `radiative_moments._exact_layer` (1.3 s). Every other
kernel compiles in under 0.8 s.

**Closure cache key fix.** Before this change a warm process still recompiled
`core` (one sample: 3.39 s of compilation, first `simulate` 7.60 s, 11 hits
and 1 miss). Numba keys a closure's disk cache on its pickled closure cells;
`core` captures the `body` dispatcher, which pickles by value with random
per-process UUIDs (its own and those of the ~30 dispatchers its globals reach).
`make_cpu_transport_core` now replaces that cell pickle in `core`'s index key
with the specialization flags plus SHA-256 digests of the sources of `body`'s
module and of every module defining a dispatcher `body` calls
(`pyrite._jit_cache.stabilize_closure_cache`). Bytecode, signature, target CPU
and Numba's own source stamp stay in the key. A warm first `simulate` now
compiles nothing.

The separately profiled warm process (instrumented, overlapping cumulative
times) spends 0.45 s loading 11 cached overloads, 0.27 s importing Numba and
1.12 s building shell inelastic tables, the persistent-table cost deferred in
#346.

All cold and warm samples, before and after the fix, have identity digest
`04d1bb570fc917017a7d6a721cfffa9bd5e564ccb6b44a9665f78119364158a4` and
identical SHA-256 hashes of every returned spectrum array. The cache key change
alters no generated code; it only lets `core` be found on disk.

### Cache locations

`import pyrite` (and so the `pyrite` command and the remote
`python -m pyrite._entry.scan` shim) points unset or empty `NUMBA_CACHE_DIR` and
`CUPY_CACHE_DIR` at `<user cache>/pyrite/numba` and `<user cache>/pyrite/cupy`
before any kernel module is decorated; non-empty user values win. It imports
neither Numba nor CuPy; if Numba is already imported, `numba.config.CACHE_DIR`
is set as well, which is early enough because each dispatcher reads it when it
enables caching. CuPy reads the variable at each kernel compile. The in-tree
`__pycache__` location is a reported, clearable legacy location. `pyrite cache
status` reports each location, its size and, per Numba kernel family, how many
indexes are current (source digest and Numba version match) or stale; `pyrite
cache clear` deletes only PyRITE-owned entries. Numba 0.68 stamps sources by
content digest, so an mtime-only change such as a remote sync keeps hits;
`tests/test_jit_cache.py` pins hit, source-edit and directory-change behaviour.
