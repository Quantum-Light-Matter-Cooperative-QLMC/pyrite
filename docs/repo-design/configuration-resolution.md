# Configuration and profile resolution

PyRITE has three distinct configuration layers. Keeping them separate prevents surprising runs and makes dataset identity reproducible.

1. **CLI context** selects the current campaign profile, remote target, and workspace root.
2. **Catalog profiles** select campaign grids, membership, beam, detector, and emission policy from the material catalog (`data/catalog/profiles/`).
3. **Fidelity presets** (`full` or `survey`) set workload and reduce resolved grids. Fidelity is not a catalog profile. `--fidelity` is deprecated for removal in 0.6.0; `survey` is retired (issue #215).

## CLI context precedence

Each context value resolves independently in this order:

1. an option supplied for this command;
2. a `PYRITE_*` environment variable;
3. the persistent user config store;
4. the built-in default.

| key | environment | built-in |
|---|---|---|
| `profile.current` | `PYRITE_PROFILE` | `standard` |
| `remote.target` | `PYRITE_REMOTE_HOST` | unset; configure an SSH host alias to use remote commands |
| `remote.gpu_vendor` | `PYRITE_REMOTE_GPU_VENDOR` | `nvidia` |
| `remote.partition` | `PYRITE_REMOTE_PARTITION` | `gpu` |
| `remote.nodelist` | `PYRITE_REMOTE_NODELIST` | `any` (no `--nodelist`) |
| `remote.gres` | `PYRITE_REMOTE_GRES` | `gpu:1` |
| `workspace.root` | `PYRITE_HOME` | current directory |

`pyrite config list` shows both effective values and their sources. `config set` writes atomically to Click's platform-specific user configuration directory.

The workspace resolver uses an explicit command path first, then the effective `workspace.root`. Packaged catalog and CIF data remain package-relative and are never redirected into the workspace.

## Environment-variable reference

This is the maintained inventory of environment variables that PyRITE reads or
writes. A variable supplied through a command option takes precedence where the
option documents it. Variables marked **numerics** change the resolved
calculation or its floating-point representation; those marked **resources**
change scheduling, memory, or profiling only.

### Stable runtime controls

| variable | purpose, values, and default | precedence / owner | effect |
|---|---|---|---|
| `PYRITE_PROFILE` | Campaign-profile name; unset resolves to `standard`. Empty is invalid. | Command option > environment > config store > default; `console.config`. | selects configuration |
| `PYRITE_REMOTE_HOST` | SSH-config host alias; unset means remote commands require configuration. | Command option > environment > config store; `console.config`, `remote.config`. | remote destination |
| `PYRITE_HOME` | Workspace-root path; unset resolves to the current directory. | Explicit workspace/checkpoint path > environment > config store > cwd; `console.config`. | checkpoint/artifact location |
| `PYRITE_MC_BACKEND` | `auto`, `cpu`, `cuda`, `rocm`, or `sycl`; unset is `auto`. An explicit unavailable accelerator, or one whose device lacks native fp64, fails; `auto` falls back to CPU. | Process-wide backend selection; `_backend`. | **numerics**, resources |
| `PYRITE_MC_SYCL_DEVICE` | A dpctl SYCL device selector; unset chooses the first usable GPU. | Used only when the SYCL backend resolves; `_backend`. | hardware choice |
| `PYRITE_FP64` | Set to `1` for float64 accelerator calculations; any other or unset value retains the normal CPU float64/GPU float32 behavior. | Process-wide; `_backend`. | **numerics**, resources |
| `PYRITE_MC_TRANSPORT_CORE` | `lockstep`, `per-electron`, `cuda`, or `auto`; unset leaves the requested/automatic core selection in force. | Process-wide transport pin; `montecarlo.transport.batching`. | **numerics**, resources |
| `PYRITE_MC_RESOURCE_POLICY` | `auto`, `conservative`, `balanced`, or `throughput`; unset is `auto` (`conservative` below 8 GiB, otherwise `balanced`). | Process-wide resource policy; `montecarlo._resources`. | **resources** |
| `PYRITE_ANALYZE_INITIAL` | Initial material key for the analysis app; unset falls through to the persisted app default, then `hopg`. | CLI argument > environment > persisted default > `hopg`; `apps.analyze`. | app startup |
| `PYRITE_VIEWER_INITIAL` | Initial material key for the viewer app; same unset behavior as analysis. | CLI argument > environment > persisted default > `hopg`; `apps.viewer`. | app startup |

### Automatic line-grid policy

These controls apply only when PyRITE resolves an automatic line grid. The
presence of any one outranks a stored catalog grid policy. Invalid values fail
rather than silently coarsening a grid. They change case identity and numerical
output.

| variable | format and unset behavior | owner |
|---|---|---|
| `PYRITE_ENERGY_GRID_RTOL` | Positive relative tolerance fallback. Unset uses the per-observable defaults. | `_line_grid_policy` |
| `PYRITE_ENERGY_GRID_RTOL_<OBSERVABLE>` | Positive relative tolerance for `INTRINSIC_SOURCE` (default `1e-3`) or `DETECTED_COUNTS` (default `1e-2`). This dynamic name outranks the global fallback. | `_line_grid_policy` |
| `PYRITE_ENERGY_GRID_MAX_SPACING_EV` | Positive maximum spacing in eV; unset is `3.0`. | `_line_grid_policy` |
| `PYRITE_ENERGY_GRID_ULPS` | Positive backend-coordinate safety factor; unset is `8`. | `_line_grid_policy` |
| `PYRITE_ENERGY_GRID_MAX_POINTS` | Integer point budget greater than one; unset is `600000`. | `_line_grid_policy` |

### Remote and external-cross-section configuration

| variable | purpose, values, and default | owner | effect |
|---|---|---|---|
| `PYRITE_REMOTE_DIR` | Absolute POSIX path to the remote checkout, or `~/...` (expanded to the remote login home over one cached ssh call); unset is `~/pyrite`. | `remote.config` | remote destination |
| `PYRITE_REMOTE_UV` | Executable name, absolute POSIX path, or `~/...` path for `uv` on the remote host; unset is `~/.local/bin/uv`. | `remote.config` | remote execution |
| `PYRITE_REMOTE_GPU_VENDOR` | `nvidia`, `amd`, or `intel`; unset is `nvidia`. Selects the batch-script prelude and `uv sync --extra`; `intel` fails before batch-script generation. | Environment > config store (`remote.gpu_vendor`); `console.config`, `remote.config`. | remote resources |
| `PYRITE_REMOTE_PARTITION` | SLURM partition name for new jobs; unset is `gpu`. | Environment > config store (`remote.partition`); `console.config`, `remote.config`. | remote resources |
| `PYRITE_REMOTE_NODELIST` | SLURM node list for `#SBATCH --nodelist`, or `any`; unset is `any`. | Environment > config store (`remote.nodelist`); `console.config`, `remote.config`. | remote resources |
| `PYRITE_REMOTE_GRES` | SLURM gres string for `#SBATCH --gres`, such as `gpu:radeon8060s:1`; unset is `gpu:1`. | Environment > config store (`remote.gres`); `console.config`, `remote.config`. | remote resources |
| `PYRITE_XSGEN_BREMSLIB_SOURCE` | Directory containing the BremsLib source tree; unset is `../BremsLib_v2.0.8`. | `console.config`, `xsgen.sources` | external data/tooling |
| `PYRITE_XSGEN_ELSEPA_SOURCE` | Directory containing the ELSEPA source tree; unset is `../elsepa-2020`. | `console.config`, `xsgen.sources` | external data/tooling |
| `PYRITE_XSGEN_SBETHE_SOURCE` | Directory containing the SBETHE source tree; unset is `../sbethe`. | `console.config`, `xsgen.sources` | external data/tooling |
| `PYRITE_XSGEN_FC` | Compiler executable name or path; unset tries `gfortran`, `ifx`, `flang-new`, then `flang`. | `xsgen.toolchain` | external compilation |

### Diagnostic and resource controls

These are operational overrides for profiling, capacity experiments, or job
scripts. They are read once when the runner imports unless noted otherwise.
Positive integer controls ignore blank, malformed, and non-positive values and
use their stated default. They do not change the physical model, but changing
chunking or a backend can alter floating-point reduction order.

| variable | format and unset behavior | owner |
|---|---|---|
| `PYRITE_MC_SPEC_CHUNK`, `PYRITE_MC_BREM_CHUNK` | Positive segments-per-spectrum/brems chunk; unset `0` selects adaptive sizing. Explicit CLI chunk options win. | `runner.chunking` |
| `PYRITE_MC_MIN_CHUNK` | Positive smallest segment chunk the device budget must admit before a case falls back to CPU; unset `1000`. Malformed or non-positive values raise. Lower it to keep very wide grids on the GPU at smaller chunks. | `montecarlo._resources` |
| `PYRITE_MC_SPEC_BUDGET_MB` | Positive MiB spectrum working-set budget; unset is `min(1920, resolved device budget)`. | `runner.chunking` |
| `PYRITE_MC_FREE_EVERY` | Positive case cadence for releasing GPU-pool blocks; unset comes from the resource policy. | `runner.chunking` |
| `PYRITE_MC_FREE_WATERMARK_MB` | Positive MiB free-memory watermark; unset `0` disables the watermark. | `runner.chunking` |
| `PYRITE_MC_GPU_POOL_FRAC` | Floating-point fraction cap for the CuPy pool; unset uses the resolved device-budget fraction, or `0` without a GPU. | `runner.chunking` |
| `PYRITE_MC_GPU_SHARE` | Positive number of co-tenant processes sharing one GPU; unset `1`. Remote parallel-material scripts write it. | `runner.chunking`, `remote._queue_scripts` |
| `PYRITE_MC_GPU_OOM_RETRIES` | Positive retry count after GPU allocation failure; unset comes from the resource policy. | `runner.chunking` |
| `PYRITE_MC_WORKER_MEM_MB` | Positive per-worker MiB budget for CPU full-case workers; unset `6144`. | `runner.chunking` |
| `PYRITE_MC_PIPELINE_WORKER_MEM_MB` | Positive per-worker MiB budget for CPU transport workers behind the GPU pipeline; unset `1536`. | `runner.chunking` |
| `PYRITE_MC_TIMING` | Any nonempty value other than `0` enables phase-timing reports; unset disables them. | `runner` | profiling |
| `PYRITE_MC_DEBUG` | Any nonempty value enables PyRITE debug logging; unset is silent. | package initialization | diagnostics |
| `PYRITE_MC_NSYS_PYSTACK` | Any nonempty value asks generated Nsight scripts for Python stacks/backtraces; unset omits them. Use only with a supported Python/Nsight pair. | `remote._queue_scripts` | profiling |

### Internal handoffs written by PyRITE

Do not set these as persistent user configuration. PyRITE sets or clears them
for a child process or one run; a pre-existing value may be honored where the
owner states it.

| variable | writer and scope | reader / unset behavior |
|---|---|---|
| `PYRITE_MC_NSYS` | `runs.scan` and remote queue scripts set `1` for an Nsight child. | `runner.chunking` enables NVTX ranges; unset disables them. |
| `PYRITE_LOCAL_DASHBOARD` | `runs.scan` sets `1` while its local dashboard runs, then clears it. | `runner.scheduling` enables dashboard progress only for `1`. |
| `PYRITE_MC_SPEC_CHUNK`, `PYRITE_MC_BREM_CHUNK`, `PYRITE_MC_GPU_SHARE` | CLI and remote scripts may write these runtime overrides. | See the diagnostic table above. |
| `PYRITE_MC_TRANSPORT_CORE`, `PYRITE_MC_RESOURCE_POLICY`, `PYRITE_MC_MIN_CHUNK` | Remote queue scripts copy the submitter's values into the job, validated before submission. | See the tables above. |

### Validation, documentation, and test controls

These variables are for maintainers and validation runs, not normal user
configuration.

| variable | purpose and unset behavior | owner |
|---|---|---|
| `PYRITE_BOTE_SALVAT_TABLE` | Path to a local pinned Bote--Salvat validation table; unset uses the PyRITE user-data location. | `validation.shell_ionization` |
| `PYRITE_SELTZER_BERGER_TABLE` | Path to a local pinned Seltzer--Berger validation table; unset uses the PyRITE user-data location. | `validation.brem_sources` |
| `PYRITE_DOCS_SHOW_AUTODOC_WARNINGS` | Set exactly `1` to show otherwise baselined autodoc warnings during a docs build; unset suppresses them. | `docs._warning_baseline` |
| `PYRITE_ONLINE_TESTS` | Set exactly `1` to run tests that fetch external structure data; unset skips them. | pytest marker config |
| `PYRITE_EXTERN_CODES_TESTS` | Set exactly `1` to run ELSEPA, SBETHE, and BremsLib tests when a compiler and source trees exist; unset skips them. | pytest marker config |
| `PYRITE_TEST_BACKEND` | Backend name for test collection/session; unset is `cpu`. | `tests.conftest` |
| `PYRITE_RUN_INTEL_SYCL_TESTS` | Set exactly `1` to enable Intel SYCL hardware tests; unset skips them. | integration tests |

### Exclusions

`_PYRITE_COMPLETE` is Click's completion protocol variable, not a PyRITE-defined
runtime setting. `PYRITE_PI` is a constant inside generated kernel source, not
an environment read. Standard/external variables such as `NO_COLOR`, `TERM`,
`SLURM_*`, `NUMBA_DISABLE_JIT`, `MPLBACKEND`, `MPLCONFIGDIR`, `PATH`,
`MP_API_KEY`, `UV_CACHE_DIR`, and `UV_PROJECT_ENVIRONMENT` belong to their
owning tools or execution environment.

## Catalog-profile resolution

`pyrite run [PROFILE]` uses the positional profile when present; otherwise it uses `profile.current`. The selected `[profiles.NAME]` row supplies shared scan values. `[profiles.NAME.overrides.MATERIAL]` replaces values for one material. An explicit membership list limits the campaign; an absent list means every configured material. `-m/--material` narrows that resolved membership and does not create another profile.

Named beam references resolve to beam values before hashing. Detector settings inherit the selected profile's block, then the `standard` detector block, then the built-in 90-degree scalar geometry. The built-in detector has no spectral response; scoring preserves the source spectrum apart from acceptance scaling. An explicit emission policy overrides the fidelity preset's default. The photon dispersion model is not configurable: the in-medium relation always applies. Energy-grid references are verified and resolved for the selected profile before a material sweep is built.

## Run resolution order

For each material, the run path performs the following conceptual sequence:

```text
CLI context -> catalog profile -> material override -> named beam/detector
            -> resolved detector energy bins -> fidelity preset
            -> explicit run overrides -> cases and identity
```

All values are resolved before the dataset identity is computed. Changing a resolved physics or workload input therefore selects a different dataset; changing only a display label does not. See [Dataset identity and storage](storage/dataset-identity-and-storage.md).

## Inspect before compute

```bash
pyrite config list
pyrite profile show standard
pyrite beam show default
pyrite material show hopg --profile standard
```

Use [Configuration cookbook](../guides/configuration-cookbook.md) for common edits and the generated [CLI reference](cli/cli-reference.md) for exact option contracts.
