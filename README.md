# PyRITE

A **Py**thon toolkit for **R**adiation from **I**nteractions and **T**ransport of **E**lectrons.

PyRITE is a Monte Carlo electron transport code for crystalline and amorphous
solids. Its transport model largely follows the well-validated PENELOPE code
system. Its main purpose is to predict tunable X-ray lines emitted when
electrons interact coherently with crystals: parametric X-ray radiation (PXR)
and coherent bremsstrahlung (CBS). Neither PENELOPE nor Geant4 models these
natively.

> [!IMPORTANT]
> PyRITE is unvalidated research code under active development. Absolute
> predictions are only as trustworthy as their entry in the
> [physics validation ledger](docs/validation/physics-validation-ledger.md).

The longer-term goal is a general-purpose electron transport and radiation
toolkit for 1 keV–100 MeV electrons, with native GPU acceleration, arbitrary
source, target, and detector geometries, and extensible physics. Against the
FORTRAN-based PENELOPE and the C++-based Geant4, PyRITE trades maturity for a
lower barrier to entry: it is plain Python, fast, and easy to extend. Electron
channeling in crystals below 100 MeV is under investigation.

- Source: <https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite>
- Documentation: <https://pyrite.readthedocs.io/en/latest/>

## Install

Requires Python ≥ 3.14. PyRITE is not on PyPI yet; install it from GitHub with
pip or [uv](https://docs.astral.sh/uv/):

```bash
# Command-line tool in its own environment (recommended):
uv tool install "pyrite-mc @ git+https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git"

# Or into the current environment, for use as a library:
pip install "pyrite-mc @ git+https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git"
```

Append `@<tag>` to the URL (for example `pyrite.git@v0.5.0`) to pin a release.
Once PyPI releases exist, `pyrite-mc @ git+…` becomes plain `pyrite-mc`.

Then fetch the reference data and check the install:

```bash
pyrite tables fetch    # pinned datasets and tables, about 100 MB
pyrite config setup    # optional: detect GPU hardware, save the backend
pyrite --help
```

`pyrite tables fetch` downloads EEDL, EADL, EPDL, SBETHE, ELSEPA, and BremsLib
data from public GitHub Releases; no GitHub token is needed. Run
`pyrite tables fetch --help` to list the individual datasets, or
`pyrite tables fetch NAME --archive PATH` to install one from a local archive.

For tab completion, run `pyrite config completion install` and restart the
shell; with `uv tool`, run `uv tool update-shell` first so `pyrite` is on
`PATH`. See the [shell-completion guide](docs/guides/shell-completion.md).

### GPU acceleration

The base `pyrite-mc` package is CPU-only. Install at most one GPU extra by
adding it in brackets after the package name:

| Hardware | Extra    | Backend                 |
| -------- | -------- | ----------------------- |
| NVIDIA   | `nvidia` | CUDA CuPy               |
| AMD      | `amd`    | ROCm source-built CuPy  |
| Intel    | `intel`  | oneAPI `dpnp` + `dpctl` |

```bash
uv tool install "pyrite-mc[nvidia] @ git+https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git"
```

The `amd` extra builds CuPy from source and needs `CUPY_INSTALL_USE_HIP=1` set
for the install command; ROCm support is provisional until exercised on AMD
hardware. uv refuses to combine GPU extras, but pip does not check: installing
two of them leaves conflicting CuPy builds in one environment.

`PYRITE_MC_BACKEND=auto|cpu|cuda|rocm|sycl` selects the array backend. See
[execution and acceleration](docs/computation/execution-and-acceleration.md)
for backend, precision, transport-core, and memory-policy controls.

### From a checkout

To work from source:

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
uv sync                       # creates .venv/; add --extra nvidia|amd|intel for a GPU
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

Without activating the environment, prefix commands with `uv run` (for example
`uv run pyrite --help`). `uv sync` removes extras it is not told about, so
repeat the same `--extra` on later syncs. Contributor and CI commands are in
the [development workspace guide](docs/repo-design/development-workspace.md).

For saved-shower inspection, `uv sync --extra trajectory-viewer` installs the
optional PyVista/trame viewer; see the [saved trajectory guide](docs/guides/saved-trajectory-viewer.md).

## Run

`pyrite run` executes a resumable campaign over a named profile and writes
checkpoints:

```bash
# Small smoke test; writes component checkpoints.
pyrite run standard -m hopg --quick

# Analyze an existing checkpoint.
pyrite app analysis launch hopg

# Pixel-detector observations, and case / cross-material comparison.
pyrite app pixels launch hopg
pyrite app compare launch hopg

# Interactive transport and lattice viewer; no checkpoint required.
pyrite app viewer launch hopg

# Validation dashboard.
pyrite app validation launch
```

Full sweeps are heavy. Run them through
[PyRITE's remote workflow](docs/guides/running-on-a-cluster.md) or the portable
SLURM templates documented there. GPT `.gdf` electron-beam snapshots can be used
as a beam source; see [GDF beam import](docs/guides/gpt-gdf-beams.md).

### Profiles

A profile is a named campaign: thickness, energy, and tilt grids, electron
counts, beam and detector references, and material membership. Inspect the
bundled ones:

```bash
pyrite profile list
pyrite profile show hopg_short
pyrite profile numerics show hopg_short
```

The bundled catalog ships inside the package, and profile edits write to the
selected catalog. Copy it elsewhere before creating or changing profiles:

```bash
bundled=$(pyrite config get catalog.path)   # before changing catalog.path
mkdir -p ~/pyrite-lab/catalog
cp -a "$bundled"/. ~/pyrite-lab/catalog/
cp -a "$bundled"/../cifs ~/pyrite-lab/catalog/
pyrite config set catalog.path ~/pyrite-lab/catalog

# Clone a profile, then replace its energy (keV) and thickness (Å) grids.
pyrite profile create my_scan --from hopg_short --energy 30:60:10 --thickness 5000,20000
pyrite profile set my_scan --material hopg,hbn
pyrite profile numerics set my_scan --line-trials 2000
pyrite profile show my_scan
pyrite run my_scan -m hopg
```

See [sweep profiles](docs/guides/sweep-profiles.md),
[external catalogs](docs/guides/external-catalog.md), and the
[profile settings map](docs/repo-design/profile-settings.md) for every field.

### Python API

For a single simulation with no files on disk:

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

With the scalar `Detector` shown here, `simulate` returns emitted spectra in
photons per incident electron per eV per sr, plus run identification, and
neither reads nor writes a checkpoint. Detector response is not applied; apply
it to the spectra explicitly if needed. The
[Python API workflow](docs/guides/python-api-workflow.md) covers scenes,
sweeps, detector scoring, and persistence. Prefer `pyrite run` when you need
resumability, profiles, remote execution, or the analysis apps.

The generated [CLI reference](docs/repo-design/cli/cli-reference.md) documents
every command.

## Core physics

Pipeline: sampled electron beam → single-scattering electron transport →
per-segment PXR + CBS, incoherent bremsstrahlung, and characteristic lines →
photon escape → detector scoring.

- **[Electron transport](docs/physics/beam-transport/electron-transport.md):**
  independent piecewise-linear flights with explicit elastic collisions and
  condensed energy loss; deterministic by default, with optional Urban
  straggling. Elastic collisions sample ELSEPA partial-wave cross sections;
  collision stopping uses material-level SBETHE tables. Beamline space charge
  and secondary electrons are not modeled.
- **[Crystal source](docs/physics/materials/structure-factor.md):**
  phase-specific structures, complex atomic form factors, Debye-Waller factors,
  selected reflections, and optional mosaicity define the reciprocal-space
  coupling.
- **[Coherent lines](docs/physics/radiation-physics/coherent-radiation.md):**
  first-order kinematic Born PXR and CBS amplitudes are summed coherently within
  each transport segment, i.e. between elastic collisions that change the
  electron direction $\hat{\mathbf v}$. A collision-free flight may be
  subdivided to resolve CSDA energy and clock evolution; each segment uses one
  representative velocity. The in-medium Bragg resonance is
  $\omega = (\mathbf v\cdot\mathbf g)/[1 - \mathrm{Re}\lbrace n(\omega)\rbrace\,\hat{\mathbf n}\cdot\mathbf v]$.
  Dynamical diffraction and electron channeling are not modeled.
- **[Incoherent continuum](docs/physics/radiation-physics/bremsstrahlung.md):**
  the default is the direction-resolved BremsLib v2.0 double-differential cross
  section (`pyrite tables fetch bremslib`). Without those tables, a run warns
  and falls back to isotropic EEDL bremsstrahlung: the ENDF-6 MF=23/MT=527 total
  cross section times the normalized MF=26/MT=527 photon-energy density, also
  selectable as `bremsstrahlung_model="eedl"`. Unscreened Born Bethe-Heitler
  with an Elwert correction remains an optional backend.
- **[Characteristic lines](docs/physics/radiation-physics/characteristic-radiation.md):**
  electron-impact vacancies from EEDL subshell ionization cross sections relax
  through xraydb fluorescence yields with L-shell Coster-Kronig redistribution.
  Each line carries its natural-width Lorentzian and is scored as a separate
  incoherent component of the line spectrum.
- **[Photon transport](docs/physics/radiation-physics/photon-escape-and-dispersion.md):**
  straight-ray Beer-Lambert escape with the total narrow-beam coefficient
  (photoabsorption plus Rayleigh and Compton removal). Scattered photons leave
  the ray with no build-up, redirection, or re-emission; bulk refractive
  dispersion is passive; interface optics are omitted.
- **[Detector treatment](docs/physics/detectors/detector-response.md):** source
  spectra are evaluated in a fixed far-field direction. Solid-angle acceptance,
  aperture broadening, efficiency, and measured-energy redistribution are
  downstream steps.

By default PyRITE neglects interference between distinct flights and between
incident electrons. The optional
[`coherent`/`both` tracking policy](docs/physics/radiation-physics/coherent-emission.md)
keeps phase along each single-electron trajectory and adds inter-electron terms
weighted by the sampled bunch form factor.

The [physics model index](docs/physics/index.md) lists assumptions, model
variants, and conventions, with links to each implementation-level description.
Geometry follows the [tilt convention](docs/physics/geometry/tilt-convention.md).

## Data and outputs

`src/pyrite/data/catalog/` is the bundled catalog: crystals, media, materials,
beams, detectors, and profiles, one TOML file per object. Phase-specific CIFs
live in `src/pyrite/data/cifs/`; loading is offline. `pyrite material validate`
checks the selected catalog.

Each checkpoint is `checkpoints/<stem>/{line,brem,characteristic}.h5` plus a
`checkpoints/<stem>/meta.json` manifest. Stored source spectra exclude detector
response unless a detector view applies it explicitly.

## Validation

The analytic core is cross-checked against Feranchuk et al. (2000) and Zhai et
al. (2025). Before scientific use, consult the
[validation ledger](docs/validation/physics-validation-ledger.md) for
claim-level status and the [validation methodology](docs/validation/methodology.md).

## References

Primary physics and data sources:

- I. D. Feranchuk, A. Ulyanenkov, J. Harada, and J. C. H. Spence, Parametric
  x-ray radiation and coherent bremsstrahlung from nonrelativistic electrons in
  crystals, *Phys. Rev. E* **62**, 4225 (2000).
  [doi:10.1103/PhysRevE.62.4225](https://doi.org/10.1103/PhysRevE.62.4225)
- Q. Zhai et al., Enhanced tunable X-rays from bulk crystals driven by
  table-top free electron energies, *Nat. Commun.* **16**, 11218 (2025).
  [doi:10.1038/s41467-025-66063-6](https://doi.org/10.1038/s41467-025-66063-6)
- F. Salvat, *PENELOPE-2024: A Code System for Monte Carlo Simulation of
  Electron and Photon Transport*, OECD NEA report NEA/MBDAV/R(2024)1 (2025).
- F. Salvat, A. Jablonski, and C. J. Powell, ELSEPA — Dirac partial-wave
  calculation of elastic scattering of electrons and positrons by atoms,
  positive ions and molecules, *Comput. Phys. Commun.* **165**, 157 (2005).
  [doi:10.1016/j.cpc.2004.09.006](https://doi.org/10.1016/j.cpc.2004.09.006)
- A. Poškus, Double and single differential cross sections of electron-atom
  bremsstrahlung at electron energies up to 30 MeV for Z = 1–100,
  *At. Data Nucl. Data Tables* **166**, 101734 (2025).
  [doi:10.1016/j.adt.2025.101734](https://doi.org/10.1016/j.adt.2025.101734)

The full bibliography (atomic data libraries, stopping powers, form factors,
and numerical methods) is on the
[published references page](https://pyrite.readthedocs.io/en/latest/references.html),
generated from [`docs/references.bib`](docs/references.bib).

## Status and license

Academic research code under active development, distributed under the
[UCLA Academic Software License](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/LICENSE.txt)
for educational or academic research use by academic or nonprofit researchers.
For commercial licensing, contact the author listed in the license.
