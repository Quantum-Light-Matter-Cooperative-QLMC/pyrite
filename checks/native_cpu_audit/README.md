# Native CPU and GPU-host audit

Performance evidence for #1432. Production code, models and defaults are
unchanged. The issue body owns progress and decisions; this directory owns the
reproducible protocol, the route inventory, the retained measurements
(`results/`) and their interpretation (`report.md`).

## Protocol

- Run only from a clean committed checkout synced to an isolated remote
  directory (`git archive` + rsync, with `CODE_SHA`); heavy, multi-GiB and GPU
  work never runs locally. Remote pre-flight: `remote-gpu-jobs`.
- CPU arms: one allocation of 4 CPUs, pinned by `bind.sh` (OS affinity to the
  assigned CPU IDs; Slurm/srun binding alone was ineffective on the pilot
  host, and the producer rejects a mismatched affinity count). CPU backend,
  Numba/BLAS/OpenMP threads and the Numba cache directory are set explicitly.
  One process per arm: one first call, warm repeats, one separate cProfile
  call on identical input, array hashes compared between all calls.
- A failed, refused or timed-out arm is a coverage gap, never a zero-cost
  measurement. No JIT disabling, `fastmath`, RNG substitution or silent
  fallback. Cold means an empty JIT cache, not flushed OS/table caches.
- GPU arms: production serial device path and thread defaults (recorded), the
  policy-grid speed cache cleared per arm.
- Cumulative profiler times overlap and are never summed; RSS is a lifetime
  per-process peak, not aggregate node memory.

## Files

| File | Purpose |
| --- | --- |
| `../native_cpu_audit.py` | Case producer (`--case NAME --repeats N --out JSON`) and shared helpers (`environment`, `case_workload`, `profile_rows`) |
| `cpu.sbatch` | CPU job, `MODE` = `cases`, `startup`, `topology`, `pools`, `axial-proto` (usage in its header) |
| `cpu_budget.py` | `topology` (1x1, 4x1, 1x4, 2x2 processes x Numba threads, PSS/USS sampling) and `pools` (Numba mask vs `threadpool_limits`) probes |
| `axial_prototypes.py` | In-process solver-cache and OAM-basis prototype arms behind #1436/#1437; production source unchanged |
| `gpu.sbatch` | GPU job over `MATERIAL:E0_KEV:TILT` cases, arms `warm`, `sampler`, `prof`, `nsys` (`ARMS` env) |
| `gpu_host.py` | One GPU-host arm: production `_nsys_push` ranges, harness-only host-owner ranges (`HOST_SPANS`), a 10 ms stack sampler joined with NVML utilization (`by_range_frame_*` attribute idle to code lines), phase walls, digests |
| `nsys_gaps.py` | Reduces an Nsight SQLite export: device busy (union of kernel/memcpy/memset), idle by innermost range, gap-size histogram, top kernels/API calls |
| `bind.sh` | Sourced by every sbatch file |
| `fastmath.sbatch` | #1469 CPU job: `cases` (complete cases under global fastmath flag sets, a reducer variant, or `asm` vectorization records) and `replay` (sine micro + captured reducer replay); `PIXI_ENV=svml` for SVML Numba |
| `fastmath_arms.py` | One complete case under one #1469 arm (`--fastmath`, `--reducer`, `--asm`, `--arrays`) |
| `fast_variants.py` | #1469 prototypes: reduced-accuracy sines and node-window reducer variants rebuilt from the production source by named edits; not used by production |
| `sin_micro.py`, `reducer_replay.py`, `capture_window_inputs.py` | #1469 sine speed/ulp micro, replay of captured reducer calls per `EDIT:FLAGSET` arm, and the input capture (from #1459) |
| `math_gpu.sbatch` | #1469 math-mode validation on one GPU (gpu-amd by default): each case under `PYRITE_MATH=reproducible` and `fast` |
| `compare_modes.py` | Compares one case's `--arrays` output under the two math modes (bitwise equality, change over peak, sum change) |
| `inventory.json` | Code-referenced routes and explicit unknowns |
| `report.md` | Findings, rank, coverage gaps, evidence index |
| `results/job-<id>/` | Retained evidence per job: `summary.json`, `manifest.json` (SHA-256 of retained and dropped files), selected raw JSON |

```bash
# CPU: complete cases, then the cold/warm startup fixture
sbatch checks/native_cpu_audit/cpu.sbatch cases thin30 crystal300
sbatch checks/native_cpu_audit/cpu.sbatch startup
# GPU: sampler attribution over tilts of the user-catalog profile (PROFILE, default cleo_test)
ARMS="warm sampler" sbatch -J gpuhost1432-<sha> checks/native_cpu_audit/gpu.sbatch \
    runs/sampler "hopg:300:5 hopg:300:85" 1800
```

## Workloads (`native_cpu_audit.py --case`)

- `thin30`, `thin100`, `crystal300`: maintained #611 HOPG slab builder, 1 um,
  16 histories, seed 1432, fixed 400-node line grid. `layered`: 100 keV, 1 um
  MoS2 on 10 um Si. `mos2_5mev`: 5 MeV / 1 mm MoS2, 2 histories
  (resource-bounded). `photon-case`: documented opt-in Cu 30 keV
  `PhotonTransport(source="brem", n_samples=1000)`, seed 341.
- `crystal300-binmean`, `crystal300-coherent`, `crystal300-coherent-hr`: CXR
  route variants (bin-mean; coherent on a laterally infinite slab at 400 or
  19 001 nodes). Routes only, not comparable to the selected incoherent arms.
- `photon`, `pair`: photon-history geometries (Cu 20 keV / W 20 MeV, 128
  histories, seed 341), whole queue with ledgers. `eedl`, `bh`,
  `char-multi`, `char-layered`: standalone scorers on a fixed synthetic
  4096-segment recipe (seed 819); not supported model overrides and never a
  case-throughput claim.
- `axial`: small energy-resolved MCWF smoke (12x12, 2 A). `axial-prod`: the
  documented example (5 MeV Si <001>, 32x32, 9-node beam, OAM 0, 1000 A in 5 A
  steps, 1000 histories, seed 627); `axial-prod-cache8` differs only in
  `max_cached_solvers=8` and must be bit-identical.
- `transport-si300` (#1469): transport only, default physics (shell-soft-hard,
  ELSEPA single scattering), 2048 primaries, 300 keV, 200 um Si, 5 deg, seed
  1469, explicit 5 keV cutoff (engine call, no transport reuse; not scorable).
  `channeling-demo` (#1469): the admitted planar `channeling_demo` case of the
  experimental planar channeling guide through `pr.simulate`.

## Parallelism and free threading

`prange` needs independent writes or a supported reduction; histogram
scatter, queue mutation and RNG/daughter ordering need separate designs.
Native compilation alone does not establish GIL release. Dask workers run
tasks in a thread pool: count nested Numba, BLAS and FFT pools against the
allocation. Free-threaded Python is a separate environment axis: record
`sys._is_gil_enabled()` after native imports and never force the GIL off
(#1421 owns the install matrix, #1423 shared-state safety). See
[Numba parallel loops](https://numba.readthedocs.io/en/stable/user/parallel.html),
[Dask workers](https://distributed.dask.org/en/stable/worker.html) and
[CPython extension support](https://docs.python.org/3/howto/free-threading-extensions.html).
