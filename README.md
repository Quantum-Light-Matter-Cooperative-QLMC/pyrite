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

Requires Python ≥3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
uv sync
uv run pyrite config setup   # optional first-run backend detection
uv run pyrite --help
```

`uv sync` installs the root distribution and contributor dependency groups.
For a locked runtime-only installation use `uv sync --no-dev --locked`;
focused contributor and CI commands are documented in
[development workspace guide](docs/repo-design/development-workspace.md).

`uv run` exposes project commands only for that invocation; it does not make
`pyrite` persistently available to later shell sessions. For a regular user
command and persistent tab-completion, install the checkout as a uv tool:

```bash
uv tool install .
uv tool update-shell
exec "$SHELL"
pyrite config completion install
exec "$SHELL"
```

Contributors who want the persistent command to follow source edits may use
`uv tool install --editable .`. Continue to use `uv run pyrite-dev ...` for
locked development and verification. See the
[shell-completion guide](docs/guides/shell-completion.md) for zsh setup,
generated-file locations, removal, and troubleshooting.

Base `pyrite-xray` is CPU-only. Install exactly one accelerator extra in a clean
environment:


| Hardware | Install                                      | Backend                |
| ---------- | ---------------------------------------------- | ------------------------ |
| NVIDIA   | `uv sync --extra nvidia`                     | CUDA CuPy              |
| AMD      | `CUPY_INSTALL_USE_HIP=1 uv sync --extra amd` | ROCm source-built CuPy |
| Intel    | `uv sync --extra intel`                      | oneAPI`dpnp` + `dpctl` |

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

Use `pyrite run` for resumable profile campaigns that write checkpoints:

```bash
# Small survey run; writes component checkpoints.
uv run pyrite run standard -m hopg --fidelity survey

# Analyze existing checkpoint.
uv run pyrite app analysis launch hopg

# Interactive transport/lattice viewer; no checkpoint required.
uv run pyrite app viewer launch hopg

# Validation dashboard.
uv run pyrite app validation launch
```

Use the Python API for one filesystem-free simulation:

```python
import pyrite as pr

beam = pr.Beam(energy_keV=30.0)
target = pr.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0)
detector = pr.Detector()
result = pr.simulate(
    beam,
    target,
    detector,
    numerics=pr.Numerics(n_electrons=450, n_electrons_brem=100),
)
print(result.provenance["identity_digest"])
```

`simulate` returns intrinsic arrays and provenance without reading or writing a
checkpoint. See the [Python API workflow](docs/guides/python-api-workflow.md)
for scenes, sweeps, detector scoring, and persistence boundaries; use
`pyrite run` when resumability, profiles, remote execution, or analysis apps
matter.

Main surfaces:

- `pyrite`: nine visible user nouns — `run`, `app`, `checkpoint`, `config`,
  `remote`, `job`, `profile`, `material`, and `beam`. App launch/export lives
  below `pyrite app`. See the generated
  [CLI reference](docs/repo-design/cli/cli-reference.md).
- `src/pyrite/apps/scan_app.py`: interactive sweep runner.
- `src/pyrite/apps/analysis_app.py`: checkpoint analysis.
- `src/pyrite/apps/trace_app.py`: trajectory/lattice viewer.
- `src/pyrite/apps/validation_app.py`: literature-validation studies.
- `src/pyrite/`: importable physics, results, plotting, and detector library.

Full sweeps are heavy. Use [PyRITE's remote workflow](docs/guides/running-on-a-cluster.md) for lab
GPU work or follow portable SLURM templates there.

## Data and outputs

[`src/pyrite/data/materials.toml`](src/pyrite/data/materials.toml) is immutable
catalog source for crystals, media, scan profiles, materials, and stacks.
Phase-specific CIFs live under `src/pyrite/data/cifs/`; production loading is
offline. Validate edits with:

```bash
uv run pyrite material validate
```

Golden catalog snapshot must be regenerated after catalog/schema changes; use
`regen-golden` skill.

Active checkpoints use `checkpoints/<stem>/{line,brem}.pkl`. The suffix is a
historical layout token: new artifacts contain `pyrite.result` HDF5 version 1,
while legacy plain, gzip, and zstd pickles remain readable. See the
[result schema](docs/repo-design/storage/result-schema.md).
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

- [published documentation](docs/index.md): guides, physics, validation,
  research, repository design, ADRs, and API reference
- [documentation authoring and maintenance](docs/repo-design/documentation.md)
- [package ownership and dependency map](docs/repo_map.md)
- [Python API](docs/api.md)
- [backlog](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues)
- [agent conventions](AGENTS.md)

## References

- W. Zhai et al., *Nat. Commun.* **16**, 11218 (2025).
- I. D. Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

## Status and license

Academic research code under active development. No project license currently
attached; contact author regarding reuse.
