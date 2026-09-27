# Troubleshooting

## A command uses the wrong profile, host, or workspace

Run `pyrite config list`; it reports the winning source for every value. A command option beats `PYRITE_*`, the persistent store, and the built-in default. Remove or change the higher-precedence value.

## The catalog does not load

Use `pyrite material validate [PATH]`. Errors are grouped by TOML location. Common causes are an unknown key, missing required root table, invalid grid descriptor, profile membership naming an unknown material, beam references to an unknown beam, CIF paths outside packaged data, or elements without transport support. See the [catalog schema](../repo-design/materials-catalog-schema.md).

## How line grids are selected

Bundled profiles have no stored line-grid rows. PyRITE resolves a case-local line grid automatically — a closed-form kinematic bandwidth, then a sinc-Nyquist resolution measured from the run's own trajectories. The resolved policy, tolerance, coordinates, and configuration source are recorded in the result's provenance under `line_grid_policy` and `line_grid_resolved`, and the policy is part of case/checkpoint identity.

Automatic grids are deliberately conservative — the bandwidth is a bound, not a measurement — so they cost more points than a derived row. Tune them through the usual precedence chain (per-call/API > `PYRITE_*` > stored artifact > built-in):

| Variable | Meaning |
|---|---|
| `PYRITE_ENERGY_GRID_RTOL` | global *fallback* relative tolerance |
| `PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE` | per-observable override (default `1e-3`) |
| `PYRITE_ENERGY_GRID_RTOL_DETECTED_COUNTS` | per-observable override (default `1e-2`) |
| `PYRITE_ENERGY_GRID_MAX_SPACING_EV` | coarsest admissible spacing (default `3.0`) |
| `PYRITE_ENERGY_GRID_ULPS` | backend coordinate-precision safety factor |
| `PYRITE_ENERGY_GRID_MAX_POINTS` | point budget before the run refuses (default `600,000`) |

Setting any of these selects automatic resolution even where an opt-in stored row exists. An explicit `EnergyBins.line` still wins over all of them.

A requested tolerance that cannot be met inside the point budget or the backend ULP floor **raises** and names the unmet tolerance and the correction. The grid is never silently coarsened; raise the budget, relax that observable's tolerance, narrow the bandwidth, or run with `PYRITE_FP64=1`.

Local line windows are opt-in while their convergence is being measured (issue #101). Passing `line_grid_policy={"windows": True}` on a `Sweep` keeps a backbone at the maximum spacing and refines windows around PXR/CBS resonances (from the run's own segments), absorption edges, and characteristic lines. A mapping may set `samples_per_feature` (default `8`), `tail_widths` (default `2.0`), and `providers`; `False` disables windows from a stored policy. The window plan is recorded under `line_grid_resolved`, and enabling or changing windows changes case/checkpoint identity. Windowed grids are nonuniform, so `EagleXO(resolve_energy=True)` and `LegacyEDS(convolve=True)` refuse them.

## Deriving grids on purpose

`pyrite material energy-grid derive --profile NAME` is an inspection and prewarming command, not a prerequisite. It measures a 95%-integrated-coverage **bandwidth** (not an accuracy target) and installs immutable artifacts, repointing that profile automatically; add `--remote` for the configured compute host, then run `pyrite-dev energy-grid verify`. Detached submission returns before installation; attach to the job for progress, then rerun without `--detach` to resume, pull, and install. Do not paste survey-cropped bounds into the catalog; fidelity reduction happens later.

## A run does not resume the checkpoint I expected

Compare each active stem's component metadata beneath the effective `checkpoints/` root. Fidelity, profile-resolved values, overrides, beam/detector values, emission mode, and crystallography can change the digest and stem. This is intentional isolation, not a cache miss bug. `pyrite checkpoint list` lists the archive shelf, not active datasets. Legacy checkpoints without identity are provenance-incomplete.

## Analysis cannot find output

Confirm the effective `workspace.root`, any `PYRITE_HOME`, and any `--checkpoint-dir` used for the run. Inspect that same active checkpoint root or the analysis app's dataset selector. Analysis needs component checkpoint data; exported HTML alone is not an input dataset.

## Only one spectral component is present

Interrupted runs and component recomputation can leave line or brem alone. Use `pyrite checkpoint recompute line|brem MATERIAL` for the missing component. Check identity before merging or restoring data; incompatible identities are correctly rejected.

## The GPU is unavailable or runs out of memory

Inspect `pyrite config setup --help` and the active backend environment. CPU is the safe base installation. Do not solve memory pressure by changing scientific inputs invisibly: use documented resource/chunk controls, record them for performance work, and route heavy GPU sweeps through the [cluster workflow](running-on-a-cluster.md).

## Remote execution fails

Verify `pyrite config get remote.target`, then inspect `pyrite job list` and `pyrite job status JOB_ID`. Use `job logs` for scheduler/runtime failures and `remote` resource commands for connectivity/setup. Detached submission returns before results exist; pull or wait for completion before analysis.

## Results look plausible but differ unexpectedly

Compare full identities, selected coordinates, fidelity, emission mode, detector response, solid-angle scaling, and catalog revision. View source components before detector convolution. Then consult the [validation ledger](../validation/physics-validation-ledger.md); execution success is not physics validation.
