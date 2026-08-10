# PyRITE

**a Python toolkit for Radiation from Interactions and Transport of Electrons**

**Coherent X-ray radiation and electron transport in crystals.**

PyRITE predicts narrow, tunable X-ray lines from ~30–60 keV electrons in
crystals, plus detector-visible flux. Active question: expected line flux and
enhancement at θ_obs = 90° for a 2×2 Timepix3 quad or Raptor Eagle XO CCD.

Research code; absolute predictions remain bounded by
[validation status](docs/validation/physics-validation-ledger.md) and instrument inputs.

## Physics

- **PXR:** virtual photons Bragg-diffract from lattice planes. Line energy
  follows ω = **v·g** / (1 − **v·n̂**) and changes with beam/observation
  geometry.
- **CBS:** crystal periodic potential adds coherent bremsstrahlung peaks.
- **Interference:** observed coherent line is `|A_PXR + A_CBS|²`; channels are
  not separable.
- **Background:** incoherent bremsstrahlung forms smooth continuum.

Pipeline: Monte Carlo electron transport → segment-wise PXR+CBS and
bremsstrahlung → layered self-absorption → instrument response. Default spectra
add segments/electrons incoherently; a profile's `emission` policy
(`incoherent`/`coherent`/`both`) opts into an experimental phased sum using
trajectory and bunch timing, where `both` runs one electron transport and stores
the incoherent and coherent spectra side by side for comparison. That path is
[unverified](docs/physics/radiation-physics/coherent-emission.md) and must not support scientific claims
until its phase convention and bunch-form-factor limits are independently
validated.

Core conventions: beam along +z; detector at φ = 0; positive sample tilt points
toward detector. At θ_obs = 90°, zero tilt self-absorbs photons traveling along
slab faces. See [tilt convention](docs/physics/geometry/tilt-convention.md) before geometry work.

## Install

Requires Python ≥3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
uv sync
uv run pyrite-dev bootstrap  # per-clone local git config (TODO.md merge driver)
uv run pyrite --help
```

`uv sync` installs the root distribution and contributor dependency groups.
For a locked runtime-only installation use `uv sync --no-dev --locked`;
focused contributor and CI commands are documented in
[`docs/development-workspace.md`](docs/development-workspace.md).

`pyrite-dev bootstrap` is idempotent and only sets local git config that cannot be
committed (it registers the `.gitattributes` `TODO.md merge=ours` driver so
merge/rebase conflicts on `TODO.md` resolve to the current branch automatically).
Run it once per clone; worktrees share the config.

Base `pyrite-xray` is CPU-only. Install exactly one accelerator extra in a clean
environment:

| Hardware | Install | Backend |
|---|---|---|
| NVIDIA | `uv sync --extra nvidia` | CUDA CuPy |
| AMD | `CUPY_INSTALL_USE_HIP=1 uv sync --extra amd` | ROCm source-built CuPy |
| Intel | `uv sync --extra intel` | oneAPI `dpnp` + `dpctl` |

Do not combine `nvidia` and `amd`: both provide the `cupy` import. AMD's current
`amd-cupy` wheels only support CPython 3.10, below PyRITE's Python requirement,
so the AMD extra uses upstream CuPy's ROCm source build. ROCm remains
provisional until exercised on AMD hardware.

Use `uv run ...`; bare system Python lacks locked dependencies.
`PYRITE_MC_BACKEND=auto|cpu|cuda|rocm|sycl` selects the array backend. `auto`
tries CuPy, then SYCL, then NumPy; explicit accelerator selection errors if
unavailable. `PYRITE_FP64=1` requires fp64 and falls back to CPU only under
automatic selection.

`PYRITE_MC_RESOURCE_POLICY=auto|conservative|balanced|throughput` controls memory
admission, retry count, release cadence, and host-worker admission. `auto`
uses `conservative` below 8 GiB. Its device budget is
`min(50% of VRAM, VRAM - 2 GiB)`, protecting small GPUs such as the 4 GiB Arc
A370M. Expert chunk/pool environment overrides remain supported but cannot
bypass pre-allocation admission.

> **Distribution warning:** locked `crystals` 1.7.0 dependency is GPLv3. Review
> licensing before distributing source, wheels, binaries, or containers that
> include it.

## Run

```bash
# Small survey run; writes component checkpoints.
uv run pyrite run standard -m hopg --fidelity survey

# Analyze existing checkpoint.
uv run pyrite app analysis launch hopg

# Interactive transport/lattice viewer; no checkpoint required.
uv run marimo run src/cxr_mc/apps/trace_app.py

# Validation dashboard.
uv run pyrite app validation launch
```

Main surfaces:

- `pyrite`: run, analysis, validation, export, checkpoint, profile, material,
  and remote workflows. See generated
  [CLI reference](docs/repo-design/cli/cli-reference.md).
- `src/cxr_mc/apps/scan_app.py`: interactive sweep runner.
- `src/cxr_mc/apps/analysis_app.py`: checkpoint analysis.
- `src/cxr_mc/apps/trace_app.py`: trajectory/lattice viewer.
- `src/cxr_mc/apps/validation_app.py`: literature-validation studies.
- `src/cxr_mc/`: importable physics, results, plotting, and detector library.

Full sweeps are heavy. Use [PyRITE's remote workflow](docs/guides/running-on-a-cluster.md) for lab
GPU work or follow portable SLURM templates there.

## Data and outputs

[`src/cxr_mc/data/materials.toml`](src/cxr_mc/data/materials.toml) is immutable
catalog source for crystals, media, scan profiles, materials, and stacks.
Phase-specific CIFs live under `src/cxr_mc/data/cifs/`; production loading is
offline. Validate edits with:

```bash
uv run pyrite material validate
```

Golden catalog snapshot must be regenerated after catalog/schema changes; use
`regen-golden` skill.

Active checkpoints use
`checkpoints/<stem>/{line,brem}.pkl`; analysis tolerates historical layouts.
Stored spectra are intrinsic unless a detector view applies downstream response.
Timepix3 and Eagle XO geometry/QE are instrument-specific; never transfer
counts or solid angle between setups. See
[detector solid angle](docs/physics/detectors/detector-solid-angle.md).

## Validation and provenance

Analytic core is cross-checked against Feranchuk et al. and Zhai et al. Model
claims, assumptions, and evidence live in:

- [validation ledger](docs/validation/physics-validation-ledger.md)
- [validation method and records](docs/validation/methodology.md)
- [coherent-emission model and validation boundary](docs/physics/radiation-physics/coherent-emission.md)
- [crystal mosaicity](docs/physics/materials/crystal-mosaicity.md)
- [atomic data sources](docs/physics/atomic-physics/atomic-data-sources.md)
- [detector solid-angle treatment](docs/physics/detectors/detector-solid-angle.md)

Important open uncertainty: modeled mosaic broadening lacks measured HOPG
line-width validation; Timepix3 hardware parameters still need final instrument
values. Treat absolute detector predictions accordingly.

Data sources: xraydb atomic scattering factors; NIST SRD 64 Mott transport
cross sections with analytic fallback; bundled phase-specific CIFs; bundled
Eagle XO QE plus computed Timepix Si response.

## Repository guide

- [documentation authoring and maintenance](docs/repo-design/documentation.md)
- [package ownership and dependency map](docs/repo_map.md)
- [Python API](docs/api.md)
- [backlog](TODO.md)
- [agent conventions](AGENTS.md)

## References

- W. Zhai et al., *Nat. Commun.* **16**, 11218 (2025).
- I. D. Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

## Status and license

Academic research code under active development. No project license currently
attached; contact author regarding reuse.
