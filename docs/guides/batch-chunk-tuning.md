(optuna-batch-cxr-chunk-tuning-experimental-682)=
# Tune CUDA execution controls (experimental)

Tune CUDA execution controls against the median duration of an **uncached
full simulation** (transport and spectra). None of them is a physical
optimization variable. The tuner uses Optuna directly, separately from
`OptimizationStudy`'s scene-parameter API.

## Python API

`pyrite.optimization.tune_kernels` tunes any subset of the transport
`electron_batch`, continuation `schedule` and `continuation_rows`, and the
spectral `cxr_chunk`/`brem_chunk`, for one fixed scene, and returns a
`KernelTuning` to apply through ordinary `Numerics`:

<!-- verify: skip (requires a CUDA device and the optimization extra) -->
```python
import pyrite as pr
from pyrite.optimization import KernelLattice, tune_kernels

result = tune_kernels(
    scene,  # a pr.Scene whose numerics leave transport_core="auto"
    KernelLattice(
        electron_batch=(2048, 4096, 8192),
        schedule=("full-history", "continuation"),
        continuation_rows=(None, 128, 512),
        cxr_chunk=(512, 1024, 2048),
    ),
    trials=24,
    storage="sqlite:///kernel-tuning.sqlite3",
)
print(result.environment, result.speedup)
tuned = scene.numerics.with_kernel_tuning(result.best)  # use in a fresh process
```

The baseline is the scene's own `KernelTuning` on the CUDA core. Each trial
runs `simulate` in a fresh process. Its spectral grids must equal the
baseline's and its densities agree within `rtol`/`atol`, or it is rejected.
`result.trials` keeps every candidate, its timings and any rejection reason.
`result.best` is the baseline unless a candidate was faster. A row quantum
paired with `full-history` is an invalid combination and is skipped. The
`environment` field records GPU, driver, CUDA/CuPy versions and precision;
winners do not transfer between devices. Persistent studies refuse to resume
with a different workload, lattice, protocol or device.

To tune transport and spectral execution separately, use independent studies:

<!-- verify: skip (requires a CUDA device and the optimization extra) -->
```python
from dataclasses import replace
from pyrite.optimization import KernelLattice, tune_kernels

transport = tune_kernels(
    scene, KernelLattice(electron_batch=(2048, 8192), schedule=("full-history", "auto")),
    objective="transport", trials=4, warmups=1, repeats=1,
)
spectral_scene = replace(
    scene, numerics=scene.numerics.with_kernel_tuning(transport.best)
)
spectral = tune_kernels(
    spectral_scene, KernelLattice(cxr_chunk=(None, 8192, 32768)),
    objective="spectrum", trials=3, warmups=1, repeats=1,
)
# The second winner retains the first stage's transport settings.
final = spectral.best
```

`objective="transport"` minimizes synchronized transport-phase wall time,
including trajectory setup and automatic line-grid resolution.
`objective="spectrum"` minimizes synchronized spectrum-phase wall time,
including CXR/background scoring. Both still run the complete simulation
for numerical equivalence checks. The default `objective="total"`
retains end-to-end timing. `on_progress(done, total, state)` receives the
baseline and each finished Optuna trial on the coordinating process.
It cannot report progress inside an individual CUDA launch.

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
