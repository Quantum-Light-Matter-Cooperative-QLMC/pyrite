# cxr-mc

**Coherent X-ray radiation (PXR + coherent bremsstrahlung) from table-top
electron beams in crystals.**

cxr-mc predicts narrow, tunable X-ray lines from ~30–60 keV electrons in
crystals, plus detector-visible flux. Active question: expected line flux and
enhancement at θ_obs = 90° for a 2×2 Timepix3 quad or Raptor Eagle XO CCD.

Research code; absolute predictions remain bounded by
[validation status](docs/physics-validation-ledger.md) and instrument inputs.

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
add segments/electrons incoherently; `cxr scan --coherent` opts into an
experimental phased sum using trajectory and bunch timing. That path is
[unverified](docs/coherent-emission.md) and must not support scientific claims
until its phase convention and bunch-form-factor limits are independently
validated.

Core conventions: beam along +z; detector at φ = 0; positive sample tilt points
toward detector. At θ_obs = 90°, zero tilt self-absorbs photons traveling along
slab faces. See [tilt convention](docs/tilt-convention.md) before geometry work.

## Install

Requires Python ≥3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc.git
cd cxr-mc
uv sync
uv run cxr --help
```

Use `uv run ...`; bare system Python lacks locked dependencies. GPU is optional:
CuPy falls back to CPU. `CXR_MC_DEBUG=1` shows backend selection;
`CXR_FP64=1` forces reference/validation precision.

> **Distribution warning:** locked `crystals` 1.7.0 dependency is GPLv3. Review
> licensing before distributing source, wheels, binaries, or containers that
> include it.

## Run

```bash
# Small survey scan; writes component checkpoints.
uv run cxr scan hopg --fidelity survey

# Analyze existing checkpoint.
uv run cxr app analysis hopg

# Interactive transport/lattice viewer; no checkpoint required.
uv run marimo run notebooks/trace_app.py

# Validation dashboard.
uv run cxr app validation
```

Main surfaces:

- `cxr`: scan, analysis, validation, export, checkpoint, profile, material,
  and remote workflows. See generated
  [CLI reference](docs/cli-reference.md).
- `notebooks/scan_app.py`: interactive sweep runner.
- `notebooks/analysis_app.py`: checkpoint analysis.
- `notebooks/trace_app.py`: trajectory/lattice viewer.
- `notebooks/validation_app.py`: literature-validation studies.
- `src/cxr_mc/`: importable physics, results, plotting, and detector library.

Full sweeps are heavy. Use [`cxr remote`](docs/running-on-a-cluster.md) for lab
GPU work or follow portable SLURM templates there.

## Data and outputs

[`src/cxr_mc/data/materials.toml`](src/cxr_mc/data/materials.toml) is immutable
catalog source for crystals, media, scan profiles, materials, and stacks.
Phase-specific CIFs live under `src/cxr_mc/data/cifs/`; production loading is
offline. Validate edits with:

```bash
uv run cxr material validate
```

Golden catalog snapshot must be regenerated after catalog/schema changes; use
`regen-golden` skill.

Active checkpoints use
`checkpoints/<stem>/{line,brem}.pkl`; analysis tolerates historical layouts.
Stored spectra are intrinsic unless a detector view applies downstream response.
Timepix3 and Eagle XO geometry/QE are instrument-specific; never transfer
counts or solid angle between setups. See
[detector solid angle](docs/detector-solid-angle.md).

## Validation and provenance

Analytic core is cross-checked against Feranchuk et al. and Zhai et al. Model
claims, assumptions, and evidence live in:

- [validation ledger](docs/physics-validation-ledger.md)
- [validation method and records](docs/validation/README.md)
- [coherent-emission design and validation boundary](docs/coherent-emission.md)
- [crystal mosaicity](docs/crystal-mosaicity.md)
- [atomic data sources](docs/atomic-data-sources.md)
- [detector solid-angle treatment](docs/detector-solid-angle.md)

Important open uncertainty: modeled mosaic broadening lacks measured HOPG
line-width validation; Timepix3 hardware parameters still need final instrument
values. Treat absolute detector predictions accordingly.

Data sources: xraydb atomic scattering factors; NIST SRD 64 Mott transport
cross sections with analytic fallback; bundled phase-specific CIFs; bundled
Eagle XO QE plus computed Timepix Si response.

## Repository guide

- [documentation map](docs/README.md)
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
