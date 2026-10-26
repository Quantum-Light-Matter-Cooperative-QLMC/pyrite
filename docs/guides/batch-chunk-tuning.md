# Optuna batch + CXR chunk tuning (experimental, #682)

This tool tunes **two execution parameters jointly** against the median duration
of an **uncached full case** (transport and spectra):

- `PYRITE_MC_ELECTRON_BATCH_MAX` caps CUDA transport electrons per launch
  (`PerElectronTransportConfig.max_batch`); the segment-scratch budget can
  impose a smaller effective limit.
- `PYRITE_MC_CXR_CHUNK` sets the requested **segment** chunk for CXR spectral
  processing (this is *not* the electron count). The VRAM admission policy can
  clamp it to a smaller effective value.

Neither parameter is a physical optimization variable. The implementation uses
Optuna directly, separately from `OptimizationStudy`'s scene-parameter API.

## MoLab example

After `uv sync --extra nvidia --extra optimization`, from the repository root:

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
