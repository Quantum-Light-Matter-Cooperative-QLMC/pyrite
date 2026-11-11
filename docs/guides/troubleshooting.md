# Troubleshooting

## A command uses the wrong profile, host, or workspace

Run `pyrite config list`; it reports the winning source for every value. A command option beats `PYRITE_*`, the persistent store, and the built-in default. Remove or change the higher-precedence value.

## The catalog does not load

Use `pyrite material validate [PATH]`. Errors are grouped by TOML location. Common causes are an unknown key, missing required root table, invalid grid descriptor, profile membership naming an unknown material, beam references to an unknown beam, CIF paths outside packaged data, or elements without transport support. See the [catalog schema](../repo-design/materials-catalog-schema.md).

## How line grids are selected

If a run produces an unexpectedly broad or dense line grid, inspect its
resolved policy before changing physical settings:

```bash
pyrite profile numerics cxr-grid show PROFILE
```

Automatic grids bound the case's kinematic bandwidth and resolve the spacing
from sampled trajectories; a bound is not a measured emission spectrum.
The result records `line_grid_policy` and `line_grid_resolved` in provenance,
and changing the policy changes checkpoint identity. A request that exceeds
the point budget or coordinate precision **raises**, rather than silently
coarsening the grid.

For environment overrides, point-budget and ULP limits, nonuniform local
windows and their restrictions, use
[automatic line-grid diagnostics](../reference/campaign-configuration.md#automatic-line-grid-diagnostics).
For the precedence of explicit coordinates, stored rows and automatic policies,
see [photon grids](../reference/campaign-configuration.md#photon-grids).

## Deriving grids on purpose

Use `pyrite numerics energy-grid derive --profile NAME` to measure a bandwidth
and install reusable rows when automatic resolution is too costly. This is
**optional**, not a prerequisite to running a campaign. Do not copy
survey-cropped bounds into the catalog; check the installed rows with
`pyrite-dev energy-grid verify`. The
[derivation contract](../reference/campaign-configuration.md#photon-grids)
covers installation, remote execution and grid identity.

## A run does not resume the checkpoint I expected

Compare each active stem's component metadata beneath the effective `pyrite-output/checkpoints/` root. Fidelity, profile-resolved values, overrides, beam/detector values, emission mode, and crystallography can change the digest and stem. This is intentional isolation, not a cache miss bug. `pyrite checkpoint list` shows active datasets; `pyrite checkpoint archive list` shows the archive shelf. Legacy checkpoints without identity are provenance-incomplete.

## Analysis cannot find output

Confirm the effective `workspace.root`, any `PYRITE_HOME`, and any `--checkpoint-dir` used for the run. Inspect that same active checkpoint root or the analysis app's dataset selector. Analysis needs component checkpoint data; exported HTML alone is not an input dataset.

## Only one spectral component is present

Interrupted runs and component recomputation can leave line or brem alone. Use `pyrite checkpoint recompute cxr|brem MATERIAL` for the missing component. Check identity before merging or restoring data; incompatible identities are correctly rejected.

## The first run of a configuration is slow

PyRITE compiles its CPU (Numba) and GPU (CuPy) kernels on first use, which can
take tens of seconds; later processes load them from disk. `pyrite cache status`
shows where compiled kernels live, their size, and whether each Numba kernel
family is current (loads without compiling) or stale (recompiles on next use,
for example after upgrading PyRITE or Numba). `pyrite cache clear` previews, and
with `--yes` deletes, only PyRITE-owned entries; results never depend on the
cache.

The `pyrite` command, Python sessions and notebooks (`import pyrite`) all store
kernels under the PyRITE user cache directory (`~/.cache/pyrite/numba` and
`~/.cache/pyrite/cupy` on Linux) unless `NUMBA_CACHE_DIR` or `CUPY_CACHE_DIR` is
already set to a non-empty value, which always wins. Older in-tree `__pycache__`
entries beside the PyRITE sources and CuPy's `~/.cupy/kernel_cache` are not
used by default; `pyrite cache status` still reports the in-tree location as
legacy and `pyrite cache clear` removes its PyRITE entries.

To pay the compile cost once, before a job or on a new node, run
`pyrite cache warm`. It simulates one-electron cases and discards the results;
only the kernel caches change. `--profile minimal` (the default) covers the
default physics of the README case in about 20 s on a laptop CPU; `--profile
standard` adds the other CPU transport-core variants (continuous stopping,
uncoupled continuum, secondary transport) and takes about 40 s. A target
without shell data uses continuous stopping, so warm `standard` for such
materials. Cache entries are specific to the CPU instruction set (Numba) and
GPU architecture (CuPy); warm on the node type that runs the job. See
[startup latency](../repo-design/compute/startup-latency.md) for measured
costs.

CUDA backend selection enables NVRTC precompiled headers on NVRTC 12.8+ to
reduce cold compilation time. Set `CUPY_NVRTC_USE_PCH=0` before starting Python
or `pyrite` to opt out (for example, when temporary storage is constrained).
Any explicitly set value is preserved; ROCm and older NVRTC are unchanged.
Headers use temporary storage; the native CuPy kernel cache provides reuse
across sessions. CPU-only imports do not probe NVRTC.

## The GPU is unavailable or runs out of memory

Inspect `pyrite config setup --help` and the active backend environment. CPU is the safe base installation. Do not solve memory pressure by changing scientific inputs invisibly: use documented resource/chunk controls, record them for performance work, and route heavy GPU sweeps through the [cluster workflow](running-on-a-cluster.md).

## Remote execution fails

Verify `pyrite config get remote.target`, then inspect `pyrite remote job list` and `pyrite remote job status JOB_ID`. Use `job logs` for scheduler/runtime failures and `remote` resource commands for connectivity/setup. Detached submission returns before results exist; pull or wait for completion before analysis.

## Results look plausible but differ unexpectedly

Compare full identities, selected coordinates, fidelity, emission mode, detector response, solid-angle scaling, and catalog revision. View source components before detector convolution. Then consult the [validation ledger](../validation/physics-validation-ledger.md); execution success is not physics validation.
