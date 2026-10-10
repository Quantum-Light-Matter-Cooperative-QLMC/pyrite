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
digest, and spectrum-array SHA-256 hashes. Peak memory covers both simulations
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
