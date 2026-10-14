# Physics adapter dispatch overhead

`checks/physics_adapter_overhead.py` measures a bounded scalar CPU workload
through the real transport and spectrum phase owners. It runs no sweep,
checkpoint I/O or GPU computation. Use matched source revisions, environments
and installed data; larger/GPU workloads belong on the remote host.

## Method

The pre-refactor parent is `c33fb5adbce20c7aa3a7c1bfb39d241a2a387db1`;
the adapter/provenance runtime is `92a997815426db1085040484d5afe5c04fb80af5`.
The documentation/example/benchmark additions do not change that runtime.
Both revisions use the same lockfile, Python 3.14.6, NumPy 2.5.3, installed
table digests and CPU/NumPy fp64 backend on the same WSL2 host (AMD Ryzen 9 5900X).

The workload is one 30 keV HOPG case: 1000 angstrom slab, 30 degree tilt,
5 mm square footprint, seed 1, 32 line and 32 continuum histories. It uses
default physical selections with 31 photon nodes for each source grid
(1500–4500 eV CXR; 1000–10000 eV continuum), explicit lockstep transport,
and no capture or detector observation. Each run produces 104 segments.
Model/table identity and every output array hash must agree across revisions.

Prime both environments first, then alternate parent/candidate process order.
Each fresh process records import/case initialization and first phase calls,
performs three additional warm-up calls, and collects 40 warmed phase samples.
On-disk Numba caches and filesystem caches are warm; process-local setup is new.
Startup measurements are not cold-compilation or cold-filesystem claims.
Peak RSS is reported per process; the benchmark performs no downloads or
per-segment Python callbacks.

## Results

Twenty alternating process pairs provide 800 warmed samples per revision and
phase. Reported times are medians of process medians. Exploratory 95% percentile
intervals resample paired processes 10,000 times with seed 485; the 40 samples
inside each process are not treated as independent replicates.

| Warm phase | Parent | Candidate | Change | Paired-process interval |
| --- | --- | --- | --- | --- |
| Transport | 16.881 ms | 16.985 ms | +0.61% | −1.79% to +3.56% |
| Spectrum | 11.249 ms | 11.271 ms | +0.19% | −1.32% to +3.95% |

Both measured phases meet the 5% median regression target. The differences are
smaller than process-to-process timing noise. Source keys, table markers,
segment counts and CXR/characteristic/continuum output hashes agree exactly.

Fresh-process medians with warm on-disk caches, kept outside those throughput
comparisons:

| Measurement | Parent | Candidate |
| --- | --- | --- |
| Import and case initialization | 1.425 s | 1.396 s |
| First transport call | 2.225 s | 2.178 s |
| First spectrum call | 101.224 ms | 102.125 ms |
| Process peak RSS, including initialization/compilation | 372 MiB | 455 MiB |

The approximately 83 MiB RSS increase is recorded rather than attributed to
dispatch time. A separate stage probe locates the difference before transport,
in runner import and case setup; peak RSS does not grow further across warmed
repeats after the first scoring call. Only the metadata and emission-adapter
modules are added to the probe's module inventory. This probe does not identify
the allocator responsible for that initialization difference or establish
large-case memory scaling; the raw record includes its stage measurements.

A separate candidate cProfile run (44 phase calls including first/warm-up calls)
attributes spectrum cumulative time mainly to coupled continuum scoring
(0.416 s), CXR scoring (0.196 s) and characteristic scoring (0.058 s).
Adapter selection/catalog lookups and evaluation's own exclusive time are a few
milliseconds in total. Cumulative adapter time includes its native scorer and
must not be interpreted as dispatch overhead. Profiling is separate from the
timing samples; it does not justify a kernel or estimator change.

## Reproduce

Run the same harness file in each checkout through its project environment:

```bash
PYRITE_MC_BACKEND=cpu UV_CACHE_DIR=/tmp/pyrite-uv-cache \
  uv run --offline python /path/to/physics_adapter_overhead.py \
  --repeats 40 --electrons 32 > sample.json
```

Keep each revision's editable installation separate and do not change import
paths. Use installed pinned tables; do not regenerate or substitute tables
between arms. Repeat in alternating order, compare source keys/table markers
and output hashes, then compare medians of process medians with their spread.
The 5% median regression target applies only to the matched measured phases.
Retain first-call/setup samples separately from warmed throughput.

## Evidence scope

The [raw measurements](physics-adapter-overhead.json) record all phase samples,
process startup/RSS, model/table identity and output hashes. These measure
default dispatch on a small CPU case. They do not establish GPU speed/parity,
large-grid memory scaling, remote scheduler overhead, an external plugin ABI
or a physical validity domain. Existing native backend and physics validation
tests retain those responsibilities.
