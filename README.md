# cxr-mc

**Coherent X-ray radiation (PXR + coherent bremsstrahlung) from table-top electron beams in crystals.**

A physics simulation that predicts the narrow, tunable X-ray lines a low-energy
electron beam (~30–60 keV) generates inside a crystal, and the flux a real
detector would measure. It replicates and extends Zhai et al., *Nat. Commun.*
**16**, 11218 (2025), *"Enhanced tunable X-rays from bulk crystals driven by
table-top free-electron energies,"* with the analytic core cross-checked against
Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

The driving question for the active work: **what line flux and enhancement
should a home-built 2×2 Timepix3 quad (or a Raptor Eagle XO CCD) see at
θ_obs = 90°**, for comparison against the paper's TEM/SEM measurements.

---

## The physics, in brief

A relativistic electron moving through a crystal carries a virtual photon field
that "sees" the periodic electron density and lattice potential. Two coherent
emission channels result, plus an incoherent background:

- **PXR (parametric X-ray radiation):** the electron's virtual photons Bragg-
  diffract off lattice planes into real, narrow X-ray lines. The line energy is
  set by the geometry, ω = **v·g** / (1 − **v·n̂**), so it is *tunable* with beam
  energy and observation/tilt angle rather than fixed like a fluorescence line.
- **CBS (coherent bremsstrahlung):** the periodic crystal potential puts coherent
  peaks on the bremsstrahlung continuum. CBS and PXR radiate into the same modes
  and **interfere** — the measured line is `|A_PXR + A_CBS|²`, never separable.
- **Incoherent bremsstrahlung:** the smooth background the lines sit on.

### How the pipeline computes it

```
   beam (30–60 keV)
        │
        ▼
  ┌──────────────────────┐   CASINO-style single-scattering Monte Carlo:
  │ electron transport   │   Joy–Luo slowing-down + Mott/screened-Rutherford
  │ (montecarlo/)        │   elastic scattering → straight radiating segments
  └──────────┬───────────┘
             │ segments (position, direction, energy, length)
        ┌────┴───────────────────────────┐
        ▼                                 ▼
  ┌───────────────┐               ┌────────────────────┐
  │ coherent line │               │ bremsstrahlung      │
  │ spectrum      │               │ background          │
  │ |A_PXR+A_CBS|²│               │ Born + Elwert       │
  │ finite-t sinc²│               │                     │
  └───────┬───────┘               └─────────┬──────────┘
          └───────────────┬─────────────────┘
                          ▼
                ┌─────────────────────┐  Beer–Lambert self-absorption
                │ self-absorption     │  from each segment to the surface
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐  per-instrument forward models:
                │ detector response   │  Timepix3 (Si, ~1.9 keV threshold),
                └─────────────────────┘  Eagle XO (CCD, solid-angle × QE)
```

Each straight trajectory segment between elastic collisions radiates
independently (incoherent across segments, coherent across reciprocal vectors
within a segment) with the finite-interaction-time lineshape
`|Q|² = t_L² · sinc²(P·t_L)` — the physical replacement for the absorption-limited
delta-function of the closed-form theory.

---

## Repository layout

```
notebooks/scan_app.py     RUNNER:  pick material → Sweep → run_sweep → checkpoints/<material>/{line,brem}.pkl
notebooks/analysis_app.py VIZ:     load that checkpoint → all figures (no sweeps here)
notebooks/validation_app.py CHECK: literature anchors and validation studies
scripts/export_pdf.py  legacy-named shim → cxr_mc.export (analysis app → static HTML)
src/cxr_mc/     importable package: physics modules + the cxr CLI entry point
src/cxr_mc/_entry/  box-invokable `python -m` shims (scan, reproduce_zhai); guarded, thin
src/cxr_mc/data/  materials.toml, cifs/, atomic_scattering_factors/, mott_transport_cross_sections/, *_qe.csv
checks/            validation scripts + notebooks (Feranchuk anchor, Zhai Fig 1c, kinematic audit)
docs/              documentation site: guides, validation records, API and CLI references, design history
checkpoints/       per-material results pickles (gitignored)
results/           exported static HTML and figures (generated artifacts gitignored)
```

Packaged data resolves via `cxr_mc.DATA_DIR`, so imports work from any working
directory and the data travels with an installed wheel. `*.pkl` checkpoints and
`*.png` images are gitignored; notebooks are output-stripped on commit by
`nbstripout` via `.gitattributes`.

### The `cxr_mc` package modules


| Module                   | Responsibility                                                                                                                                                                                                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `materials/`             | Material domain: `catalog.py` validates the immutable CIF-backed declarative catalog; `_cif.py` parses CIF structures; `crystal.py` owns reciprocal vectors, structure factors / `chi_g` (PXR) / `U_g` (CBS), Debye–Waller, absorption length, and physical constants; `atomic.py` supplies atomic responses; `attenuation.py` owns layered self-absorption. |
| `montecarlo/`          | The transport + radiation pipeline: `simulate_trajectories` (MC electron transport), `mc_spectrum` (coherent lines), `mc_brem_spectrum` (Born+Elwert brem), detector helpers, `tilted_geometry`, and the case drivers `run_case`/`run_cases`. **Optional CuPy GPU** with automatic CPU fallback. |
| `sweep.py`               | `Sweep` dataclass + `build_cases`. Every physical knob is a scalar (fixed) or a sequence (swept); cases = the Cartesian product.                                                                                                                                                                |
| `config.py`              | Shared builders imported by both marimo apps: `default_settings()`, `material_sweep(mat)`, and `trajectory_sweep(mat)`. Per-material grids come from `data/materials.toml`; detector/analysis knobs remain here.                                                        |
| `run.py`                 | `run_sweep(...)`: checkpointed, crash-safe, resumable driver around `run_cases`; `load_checkpoint`/`cases_from_results` for the viz side.                                                                                                                                                       |
| `results/`             | `Settings` dataclass, per-record metrics (`peak_flux`, `coherent_flux`, `line_flux`, `line_quality`, …), and ranking helpers (`selection_score`, `top_geometries`).                                                                                                                            |
| `plots/`               | Matplotlib and Altair spectrum, sweep, detector, comparison, and electron-penetration figures.                                                                                                                                                |
| `detectors/timepix_response.py`    | Per-photon forward model of the Timepix3 (Si sensor): photoabsorption, e–h pairs, charge sharing, and the**~1.9 keV counting threshold** (the headline effect — it eats sub-2 keV line flux).                                                                                                 |
| `detectors/eaglexo_response.py`    | Raptor Eagle XO CCD: a clean`solid_angle × QE(E)` operator (windowless direct-detection CCD).                                                                                                                                                                                                  |

---

## Installation

The project is managed by [**uv**](https://docs.astral.sh/uv/) with a committed
lockfile (`uv.lock`) and requires **Python ≥ 3.13**.

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/cxr-mc.git
cd cxr-mc
uv sync          # .venv + locked deps + an editable install of cxr_mc (the cxr CLI)
```

Run anything with `uv run …` (or activate `.venv`). Note that a bare `python` on
your PATH will **not** have the dependencies — always use `uv run python …`.
`uv sync` installs the package, so `import cxr_mc` works with no path hacks and
the `cxr` console script is on the venv PATH (`uv run cxr --help`). See the
checked [CLI reference](docs/cli-reference.md) for every command, option, unit,
default, and side effect.

Run `uv sync` again after pulling or switching to a branch that changes
`pyproject.toml` or `uv.lock`. The CIF catalog requires the locked `crystals`
dependency; invoking `cxr` from a stale environment that predates that dependency
cannot load the bundled catalog.

**GPU is optional.** `cupy-cuda13x` (CUDA 13) is a dependency, but it imports
cleanly even with no usable GPU and the code **falls back to CPU automatically**.
The backend probe logs its result (`No GPU found … Falling back to CPU
execution` or `Using GPU`) at DEBUG level, so it's silent by default; set
`CXR_MC_DEBUG=1` to see it. Set `CXR_FP64=1` to force double precision for
reference/validation runs (the GPU path defaults to fp32).

Launch the marimo apps with:

```bash
uv run marimo run notebooks/scan_app.py
cxr analyze hopg
```

### Docker (CPU)

A `Dockerfile` builds a CPU-only image (uv + the locked deps), so you can run the
sweeps and the test suite without setting up a local environment:

```bash
docker build -t cxr-mc .
docker run --rm cxr-mc pytest -q                                   # CPU safety net
docker run --rm -v "$PWD/checkpoints:/app/checkpoints" cxr-mc cxr scan silicon --quick
```

The container falls back to CPU automatically. A GPU image (NVIDIA runtime +
matching CUDA/cupy) is more involved and not shipped yet; the `Dockerfile` is
structured so a GPU build is a base-image swap (see its header comment).

---

## Quickstart

The workflow is **three marimo apps that share the immutable material catalog** — edit a
material's thickness / energies / tilts / energy grids in
`src/cxr_mc/data/materials.toml` once and all three apps pick it up.

1. **`notebooks/scan_app.py`** (the runner): choose a material, then
   `material_sweep(MATERIAL)` → `build_cases` → `run_sweep`, which writes
   `checkpoints/<material>/{line,brem}.pkl` and streams per-tilt statistics tables live.
2. **`notebooks/analysis_app.py`** (the viz): choose the same material, `load_checkpoint`,
   `cases_from_results`, then `browse` / heatmaps / Eagle XO / Timepix /
   penetration figures. No sweeps run here.
3. **`notebooks/validation_app.py`** (the checks): inspect literature anchors,
   validation studies, and cached validation figures without changing sweep results.

`COLLAPSE_AZIMUTH=True` (in `config.py`) keeps only the best azimuth per
(tilt, energy).

### Headless / non-interactive

```bash
cxr scan <material> [--quick] [--workers N]      # installed console script
uv run python -m cxr_mc._entry.scan <material> [--quick]  # identical, via the module shim
```

`--quick` runs a tiny smoke-test grid into an isolated `<material>_quick.pkl`.
(The entry point has the required `if __name__ == "__main__"` guard — see *Notes*.)

### Running on a cluster

`cxr scan` is headless and writes `checkpoints/<material>/{line,brem}.pkl`, so it
drops into any batch scheduler — install once, then submit one job per material.
See [`docs/running-on-a-cluster.md`](docs/running-on-a-cluster.md) for a SLURM
`sbatch` template (including a job-array sweep over several materials). Pull the
checkpoints back and do all interactive analysis and static-HTML export locally.

> The author's own loop uses a small personal helper, `cxr remote` (see
> [`src/cxr_mc/remote.py`](src/cxr_mc/remote.py)), to push the working tree to
> one GPU box (ssh host `qlmc`, overridable via `CXR_REMOTE_{HOST,DIR,UV}`) and
> pull the checkpoint back. It is optional and specific to that setup; the
> cluster recipe above is the portable path.

---

## Materials

[`src/cxr_mc/data/materials.toml`](src/cxr_mc/data/materials.toml) is the single
declarative catalog for crystal metadata, scan profiles, amorphous media, runnable
materials, and stacks. Each crystal points to a phase-specific bundled CIF under
[`src/cxr_mc/data/cifs/`](src/cxr_mc/data/cifs/). Production loading is offline-only:
it reads packaged files through `cxr_mc.DATA_DIR` and never fetches a structure at
runtime.

The `crystals` library parses CIF and expands crystallographic symmetry. That is the
edge of its responsibility: cxr-mc owns atomic form factors, structure factors,
reflection selection, attenuation, electron transport, and PXR/CBS radiation.

### Adding a material

For a phase whose elements already have transport support, the workflow has two edits:

1. Add a phase-specific CIF as `src/cxr_mc/data/cifs/<phase>.cif`.
2. Add a short crystal row and runnable material row to `materials.toml`:

```toml
[crystals.example]
cif = "cifs/example.cif"
validation_id = "example-structure"
B_ang2 = 0.6
beam_uvw = [0, 0, 1]

[materials.example]
label = "Example phase"
profile = "standard"
```

The `validation_id` must name a row in
[`docs/physics-validation-ledger.md`](docs/physics-validation-ledger.md). Run
`uv run cxr check-config` after editing; an explicit full catalog can be checked with
`uv run cxr check-config path/to/materials.toml`.

Profiles are complete scan templates and do not inherit from other profiles. A material
selects exactly one profile, then fields written on that material replace the profile's
field. Grid descriptors map directly to NumPy: `arange.stop` is always excluded;
`linspace` and `logspace` include `stop` by default and honor an explicit
`endpoint = false`. Use exactly one thickness representation: `thickness_ang` gives
physical Å directly, while `thickness_layers` requires the crystal's positive integer
`layers_per_cell` and resolves as `layers * CIF_c / layers_per_cell`.

Pinned reflections use positive family representatives in `hkl_families`; the catalog
adds both `+hkl` and `-hkl` deterministically. Every pin requires a nonempty
`hkl_reason`. Without a pin, cxr-mc selects dominant reflections using its own structure
factors and ranking policy.

Catalog validation rejects runnable compositions containing elements absent from
cxr-mc's transport model. Missing per-element Mott CSV data is different: it emits a
warning and transport uses the analytic screened-Rutherford fallback, so catalog loading
continues.

The package API exposes the immutable `CATALOG: MaterialCatalog`, typed frozen records
(`CrystalSpec`, `MediumSpec`, `MaterialSpec`, `ScanSpec`, and `LayerSpec`),
`load_material_catalog`, and compatibility projections such as `CRYSTALS`, `MATERIALS`,
and `MATERIAL_LABELS`. `CATALOG.material_keys` is an ordered tuple in the declaration
order of `[materials]`; it drives catalog menus and default analysis iteration.
[`mats_to_sim.toml`](mats_to_sim.toml) is only the smaller user-selected ordered list for
`cxr scan --all` and `cxr remote ... --all`; it neither defines nor reorders the catalog.

> **Distribution licensing:** the locked `crystals` 1.7.0 dependency is GPLv3. A
> licensing review is required before distributing cxr-mc source, wheels, binaries, or
> containers that include this dependency.

> **Note:** graphite is keyed `hopg` — there is no `graphite` key. HOPG is
> fiber-textured, so **only (00l) reflections are coherent** (random in-plane
> grain azimuths); it is treated specially (beam along the c-axis, (00l) only),
> not via `dominant_reflections`.

### Crystal mosaicity (optional — off by default)

Real crystals are mosaic: an incoherent ensemble of misoriented crystallites with a
Gaussian *mosaic spread* η (rocking-curve FWHM; e.g. HOPG ZYA 0.4° / ZYB 0.8° / ZYH 3.5°).
Mosaicity is **off by default**; `mosaic=True` enables it via one of two routes
(`mosaic_route`):

```python
material_sweep("hopg", mosaic=True)                       # "analytic" (default route)
material_sweep("hopg", mosaic=True, mosaic_fwhm_deg=3.5)  # ZYH grade override
material_sweep("hopg", mosaic=True, mosaic_route="mc")    # exact per-orientation MC average
```

- **`"analytic"` (default)** — a mosaic tilt rotates **g**, broadening the line by
  `FWHM_mosaic = E·|tan ψ|·η` (ψ = ∠(v, g)), added in quadrature with the EDS / aperture
  widths in the same `convolve_detector` pass. The **intrinsic** spectrum is untouched (so
  `plot_mosaic_comparison` overlays several grades from one record). Cheap, but
  energy-shift only: amplitudes are held fixed across the cone and `tan ψ` diverges as
  ψ → 90° (capped at the peak energy).
- **`"mc"` (exact)** — a true incoherent average over crystallite orientations **inside
  `mc_spectrum`** (2-D Gauss-Hermite quadrature, `mosaic_nodes` per axis). Broadens **both
  PXR and CBS**, captures the amplitude variation and the (asymmetric) lineshape, and has
  **no grazing divergence**. The scoping check shows this matters for HOPG's real grades
  (the line is mosaic-broad, not Doppler-dominated). Costs `mosaic_nodes²` × the line hot
  loop; the moments converge by ~5 nodes, a smooth broad lineshape needs more —
  [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).

`mosaic=False` is an exact no-op for both routes. The per-crystal η is an optional
`mosaic_fwhm_deg` in `materials.toml` (only HOPG carries one today). Neither route
is **yet validated against measured line widths** (see *Validation*).

---

## Detectors

Detector geometry is **per-instrument — never transfer numbers between setups.**


| Setup                                 | θ_obs    | Δθ     | Ω               |
| --------------------------------------- | ----------- | ---------- | ------------------ |
| Our**Timepix3** quad (28 mm at 0.4 m) | 90°      | ≈1.76° | ≈9.5×10⁻⁴ sr |
| Zhai SEM (JEOL 7800)                  | 119°     | 16.6°   | 0.066 sr         |
| Zhai TEM (JEOL 2010HR)                | ≈112.5° | ≈12°   | ≈0.034 sr       |

The intrinsic spectra are detector-agnostic; the **Timepix3** and **Eagle XO**
forward models apply their own quantum efficiency / response downstream. The
Timepix3 ~1.9 keV counting threshold is the dominant instrument effect for these
soft lines.

---

## Physics conventions (read before touching geometry or amplitudes)

- **Frame:** incident beam along **+z**; detector at azimuth φ = 0 in the x–z
  plane at polar angle θ_obs. At θ_obs = 90° the detector is along +x.
- **Tilt sign (critical):** `tilt_deg` is the sample-normal polar tilt in Zhai's
  convention. Positive tilt points the normal toward the detector (front-exit
  geometry); current catalog grids are nonnegative. Opposite signs can have
  different full-model intensities, but neither sign is universally the
  high-flux branch; see `docs/tilt-convention.md`.
- **At θ_obs = 90°, the tilt must be nonzero** — an untilted slab self-absorbs
  photons travelling along its faces (yaw alone gives identically zero).
- **Coherence:** the measured line is `|A_PXR + A_CBS|²` — never separable.
  Segments add incoherently; reflections within a segment add coherently.
- **Units:** Ångström and eV (electron energies in keV where noted).
- **Relativistic CBS** corrections (the 1/γ braced products) matter ≳100 keV and
  are present at all amplitude sites.

---

## Validation (`checks/`)

- `feranchuk_spence.py` — the Feranchuk–Spence analytic core (PXR+CBS amplitudes,
  flux helpers). A reference, **not** part of the results pipeline.
- `notebooks/validation_app.py` — cohesive marimo dashboard for the anchors and diagnostics below.
- `feranchuk_check_script.py` — LiF absolute-flux discrepancy diagnostic (not a passing anchor).
- `feranchuk_vs_zhai_check.py` — analytic vs. MC pipeline agreement.
- `kinematic_validity_check.py` — DYN/recoil audit + van-der-Waals merit table.
- `cxr_analysis_feranchuk.ipynb` — derivation narrative and paper-figure studies.
- `anchor_figures.py` + the Zhai section of `notebooks/validation_app.py` — **model-vs-theory anchor
  figures**: the MC spectra against the Eq.(10) dispersion-relation line energies,
  the Eq.(12) closed-form flux (single-segment MC/closed ratio ≈ 1), and the
  bulk-vs-film enhancement. Drop a digitized Fig 1c curve in
  `checks/reference_data/zhai_fig1c.csv` (schema in that dir's README) and the
  spectra figure overlays the measured data automatically.
- `marimo check notebooks/validation_app.py` — validates the dashboard's reactive cell graph.

### What remains to be validated / approximated

These are known approximations or unvalidated additions — read before quoting absolute
numbers:

- **Crystal mosaicity** — both the analytic broadening and the exact Monte-Carlo
  orientation average (`mosaic_route="mc"`) are implemented and cross-checked against each
  other (`checks/mosaic_mc_check.py`), but **neither is yet validated against a measured HOPG
  rocking-curve / line-width dataset** — [`docs/crystal-mosaicity.md`](docs/crystal-mosaicity.md).
- **Detector solid angle** defaults to a single observation direction `n̂` with a flat Ω
  flux scale and analytic Gaussian polar-aperture broadening (`aperture_fwhm_eV`), exactly
  as the source papers do. `detector_directions` plus `mc_spectrum_solid_angle` provide a
  validated opt-in face integral for wide-detector studies, but it is not wired into the
  checkpoint/plot pipeline; the default remains appropriate for the small Timepix Ω —
  [`docs/detector-solid-angle.md`](docs/detector-solid-angle.md).
- **Atomic data** is sourced from **xraydb** (Waasmaier–Kirfel `f0` + Chantler/FFAST
  `f', f''`); the migration from the legacy Henke/CXRO + Cromer–Mann tables was adopted and
  re-validated against the Feranchuk/Zhai anchors —
  [`docs/atomic-data-sources.md`](docs/atomic-data-sources.md).
- **Timepix3 hardware** parameters (`SENSOR_THICKNESS_UM`, `BIAS_VOLTAGE_V`,
  `TEMPERATURE_K` in `detectors/timepix_response.py`) are **placeholders** pending the real quad
  values; the detected-spectrum figures inherit that uncertainty.

---

## Data provenance

- **Atomic scattering:** Waasmaier–Kirfel `f0` + Chantler/FFAST `f', f''`, supplied on
  demand by **xraydb** for any element (no per-element table to maintain). The legacy
  Henke/CXRO `.nff` CSVs in `src/cxr_mc/data/atomic_scattering_factors/` are now unused (kept for
  provenance / A-B comparison) — [`docs/atomic-data-sources.md`](docs/atomic-data-sources.md).
- **Elastic transport:** NIST SRD 64 relativistic Mott *transport* cross sections
  (`src/cxr_mc/data/mott_transport_cross_sections/`) calibrate the screened-Rutherford
  α(E) per element; free paths from the Browning fit. Elements without a NIST
  table fall back to the analytic screening with a one-time warning.
- **Crystal structures:** phase-specific `src/cxr_mc/data/cifs/*.cif`, referenced by
  `src/cxr_mc/data/materials.toml`; structures load offline through the `crystals` adapter.
- **Detector QE:** `src/cxr_mc/data/eaglexo_qe.csv`; Timepix Si response computed from Henke `f2`.

---

## Notes & gotchas

- **`__main__` guard required for sweep scripts.** `run_cases` uses a
  `ProcessPoolExecutor`; the workers re-import the entry module (`spawn` on
  Windows, `forkserver` on Linux as of Python 3.14). Any script that drives a
  sweep must be guarded with `if __name__ == "__main__":` or it relaunches itself
  recursively. Notebooks are guarded-equivalent; `cxr_mc._entry.scan` is guarded.
- **GPU spectrum, pipelined CPU transport.** With CuPy active, one main-process
  CUDA context handles spectrum/bremsstrahlung work while a process pool prepares
  electron transport. `max_workers=0` forces fully serial execution. Full-case
  CPU pools are capped by available memory as well as core count; tune their
  per-worker estimate with `CXR_MC_WORKER_MEM_MB`.
- A benign `RuntimeWarning: divide by zero` can appear because the wide brem grid
  starts at 0 eV (λ→∞); the values are clamped downstream. It is not a bug.

---

## References

- W. Zhai et al., *"Enhanced tunable X-rays from bulk crystals driven by
  table-top free-electron energies,"* **Nat. Commun. 16, 11218 (2025).**
- I. D. Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

## Status & license

Academic research code under active development. No open-source license is
currently attached — contact the author regarding reuse.
Contributor/agent working conventions are documented in [`CLAUDE.md`](CLAUDE.md).
