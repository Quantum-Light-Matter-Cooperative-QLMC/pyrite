(optuna-batch-cxr-chunk-tuning-experimental-682)=
# Tune CUDA execution controls (experimental)

Tune CUDA execution controls for one fixed scene. None of them is a physical
optimization variable. The tuner uses Optuna directly, separately from
`OptimizationStudy`'s scene-parameter API.

## Python API

`pyrite.optimization.tune_kernels` tunes any subset of the transport
`electron_batch`, continuation `schedule` and `continuation_rows`, and the
spectral `cxr_chunk`/`brem_chunk`, and returns a `KernelTuning` to apply
through ordinary `Numerics`:

<!-- verify: skip (requires a CUDA device and the optimization extra) -->
```python
import pyrite as pr
from pyrite.optimization import KernelLattice, tune_kernels

result = tune_kernels(
    scene,  # a pr.Scene whose numerics leave transport_core="auto"
    KernelLattice(
        electron_batch=(None, 4096, 16384),
        schedule=("full-history", "continuation"),
        continuation_rows=(None, 128, 512),
        cxr_chunk=(None, 8192, 32768),
        brem_chunk=(None, 1024),
    ),
    storage="sqlite:///kernel-tuning.sqlite3",
)
print(result.environment)
print(result.transport.speedup, result.spectrum.speedup)
tuned = scene.numerics.with_kernel_tuning(result.best)  # use in a fresh process
```

The two groups are **separate studies**. Transport candidates run only
electron transport ({func}`pyrite.phases.simulate_transport`) and must
reproduce the baseline's transported arrays bit for bit. Spectrum candidates
rescore one kept baseline transport ({func}`pyrite.phases.score_spectra`):
no transport is repeated, and their spectral grids must equal the baseline's
and densities agree within `rtol`/`atol`. Transport controls do not change
the transport, so neither study depends on the other's winner; the searches
cost the sum of the two lattices, not their product. `result.best` joins
both winners. `result.transport` and `result.spectrum` keep every trial,
its timings and any rejection reason. `objective="total"` instead times
whole `simulate` calls over all dimensions jointly.

The baseline is the scene's own `KernelTuning` on the CUDA core. By default
each phase tries every valid combination (a row quantum paired with
`full-history` is skipped); `trials=` smaller than that searches with TPE.
Each candidate gets `warmups` untimed then `repeats` synchronized runs. The
fastest is re-timed against the baseline in `repeats` interleaved pairs and
replaces it only when that confirms `min_speedup` (default 1.02), so noise
alone does not pick a "winner". A candidate whose spectrum needed
allocation-retry chunk halving is rejected rather than timed favorably.

All measurements run in a CUDA worker process. With `isolate="phase"`
(default) the worker serves every candidate of the tuning, so imports, table
loading and kernel compilation are paid once; the device memory pool is
released before each candidate. `isolate="candidate"` starts a fresh worker
per candidate, which is colder and much slower. `on_progress` receives a
`TuningProgress` per candidate and, while one runs, each transport batch and
spectrum stage. The `environment` field records GPU, driver, CUDA/CuPy
versions and precision; winners do not transfer between devices. Persistent
studies refuse to resume with a different workload, lattice, protocol or
device.

## Run the phases yourself

The phase functions are ordinary experimental API, independent of tuning:

<!-- verify: skip (requires a CUDA device) -->
```python
from pyrite.config import KernelTuning
from pyrite.phases import score_spectra, simulate_transport

transported = simulate_transport(scene, progress=print)  # no spectra
first = score_spectra(transported)  # equals pr.simulate(scene)
again = score_spectra(transported, kernel_tuning=KernelTuning(cxr_chunk=32768))
```

`score_spectra` never transports; it changes only spectral chunks. The kept
transport holds every segment, on the device for CUDA, until released. To
persist a transport and rescore it later from disk, use
`pyrite run --trajectories` and `pyrite trajectories score`.
`pyrite.simulate(..., progress=callback)` reports the same
`SimulationProgress` events (transport batches, then spectrum stages) for any
in-process run.

## MoLab example

The older command-line tool tunes two parameters, through environment
variables:

- `PYRITE_MC_ELECTRON_BATCH_MAX` caps CUDA transport electrons per launch
  (`PerElectronTransportConfig.max_batch`); the segment-scratch budget can
  impose a smaller effective limit.
- `PYRITE_MC_CXR_CHUNK` sets the requested **segment** chunk for CXR spectral
  processing (this is *not* the electron count). The VRAM admission policy can
  clamp it to a smaller effective value.

Run on an allocated GPU worker through the
[cluster workflow](running-on-a-cluster.md), from the repository root after
`uv sync --extra nvidia --extra optimization`:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_FP64=1 uv run --no-sync python -m pyrite.perf.batch_chunk_tuner \
  --scene examples/molab_batch_chunk_scene.py \
  --batches 512,1024,2048,4096,8192 \
  --chunks 256,512,1024,2048,4096 \
  --trials 20 --warmups 1 --repeats 3 \
  --study-name molab-hopg \
  --storage sqlite:///molab-batch-chunk.sqlite3
```

The scene factory file must define `make_scene() -> Scene`, resolve to exactly
one case, and **leave `numerics.cxr_chunk=None`**. Use at least several thousand
transported electrons so that the batch candidates are meaningful. Keep
physics, random seeds, detector, energy grids, and numerical precision fixed.

Each candidate is executed in a **fresh subprocess** with an explicit CUDA
backend. Inside that subprocess the case is lowered and the kernel is warmed
before collecting three `perf_counter` measurements (with device synchronization).
Startup/JIT compilation and checkpoint I/O are not included. The objective is
median **end-to-end case compute** time, not isolated kernel latency or Dask
scheduler time. Reference and candidate spectra (line, brem, continuum), grids,
segment counts, backscatter and hit fractions must agree (default `rtol=1e-4`,
`atol=1e-12`) before a candidate is accepted. These tolerances are a tuning
guard, not a substitute for scientific validation.

The baseline is `(electron_batch=1024, cxr_chunk=1024)` unless changed using
`--baseline-batch` and `--baseline-chunk`. It is tested afresh on each resume.
The study's database remembers every trial. `--trials` means **total** trials;
raising it continues the study without repeating completed candidates. Study
identity includes the workload file content, GPU name, precision, candidate
lattices, reference, tolerances and PyRITE version; changes require a new study
name or SQLite database. Only run one tuning coordinator per SQLite study.

Afterward apply the suggested values to a **new** PyRITE process:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
export PYRITE_MC_ELECTRON_BATCH_MAX=4096  # replace with Optuna winner
export PYRITE_MC_CXR_CHUNK=1024          # replace with Optuna winner
PYRITE_MC_BACKEND=cuda PYRITE_FP64=1 uv run python your_simulation.py
```

Execution batch caps may be overridden by the CUDA scratch memory budget, and
explicit `Numerics(cxr_chunk=...)` overrides the CXR environment setting. An
OOM/failed worker is pruned, never assigned an artificially favorable time.
Final winners should be rerun under a representative production workload;
benchmark results do not necessarily generalize to other materials, counts,
energy-grid widths or GPUs.
