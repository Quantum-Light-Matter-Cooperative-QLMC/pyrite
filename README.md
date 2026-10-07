# PyRITE

a **Py**thon toolkit for **R**adiation from **I**nteractions and **T**ransport of **E**lectrons

PyRITE is a simulation tool which implements a Monte Carlo electron transport model in solid
rystalline/amorphous materials. The transport model used is largely derived from that used in
the well-validated PENELOPE toolkit. The primary intention is to simulate the expected x-ray
emission of tunable X-ray lines from coherent interactions of electrons in crystalline materials,
though it is on the path towards a general-purpose Monte Carlo transport toolkit.

```{important}
PyRITE is currently unvalidated research code under active development; absolute predictions
remain bounded by [validation status](docs/validation/physics-validation-ledger.md).
```

PyRITE is intended to eventually become a more general-purpose electron transport and radiation
simulation toolkit for electron transport in the 1 keV - 100 MeV range, with native support for
low-level GPU acceleration, arbitrary source, target, and detector geometries, and extensible physics. Thus, PyRITE is Python-based, fast, and an accessible alternative to the more traditional, well-validated,
but higher barrier-to-entry toolkits such as the FORTRAN-based PENELOPE or the C++-based Geant4. Critically, it also implements physics engines not natively supported by those tools; in particular,
Parametric X-ray radiation (PXR), coherent bremsstrahlung (CBS), and support for electron-channeling
effects in crystals below 100 MeV is under active investigation.

PyRITE is developed in a public GitHub repository. See the
[source repository](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite)
or the [published documentation](https://quantum-light-matter-cooperative-qlmc.github.io/pyrite/).

## Install

Requires Python ≥3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git pyrite
cd pyrite
uv sync
uv run pyrite tables fetch   # every pinned dataset and table (about 100 MB): EEDL, EADL, EPDL, SBETHE, ELSEPA, BremsLib
uv run pyrite config setup   # optional first-run backend detection
uv run pyrite --help
```

PyRITE's table archives are hosted on public GitHub Releases, so `fetch` needs
no GitHub token. To install a code from a local copy instead, use
`pyrite tables fetch CODE --archive PATH`. `pyrite tables fetch --help` lists
the codes.

Focused contributor and CI commands are documented in
[development workspace guide](docs/repo-design/development-workspace.md).

`uv run` exposes project commands only for that invocation; it does not make
`pyrite` persistently available to later shell sessions. For a regular user
command and persistent tab-completion, install the checkout as a uv tool:

```bash
uv tool install . [--editable]
uv tool update-shell
exec "$SHELL"
pyrite config completion install
exec "$SHELL"
```

See the [shell-completion guide](docs/guides/shell-completion.md) for zsh setup,
generated-file locations, removal, and troubleshooting.

Base `pyrite-mc` is CPU-only. Install exactly one GPU accelerator extra if applicable:

| Hardware | Install                                      | Backend                 |
| -------- | -------------------------------------------- | ----------------------- |
| NVIDIA   | `uv sync --extra nvidia`                     | CUDA CuPy               |
| AMD      | `CUPY_INSTALL_USE_HIP=1 uv sync --extra amd` | ROCm source-built CuPy  |
| Intel    | `uv sync --extra intel`                      | oneAPI `dpnp` + `dpctl` |

ROCm functionality remains provisional until exercised on AMD hardware.

`PYRITE_MC_BACKEND=auto|cpu|cuda|rocm|sycl` selects the array backend. See
[execution and acceleration](docs/computation/execution-and-acceleration.md) for
backend, precision, transport-core, and memory-policy controls.

> **Distribution warning:** locked `crystals` 1.7.0 dependency is GPLv3. Review
> licensing before distributing source, wheels, binaries, or containers that
> include it.

## Run

Use `pyrite run` for resumable profile campaigns that write checkpoints:

```bash
# Small smoke-test run; writes component checkpoints.
uv run pyrite run standard -m hopg --quick

# Analyze existing checkpoint.
uv run pyrite app analysis launch hopg

# Stored pixel-detector observations; case basket and cross-material comparison.
uv run pyrite app pixels launch hopg
uv run pyrite app compare launch hopg

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

For the scalar `Detector` shown here, `simulate` returns emitted
spectra in photons per incident electron per eV per sr, plus run identification,
without reading or writing a checkpoint. Detector response must be explicitly applied
to the resulting spectra if required.
See the [Python API workflow](docs/guides/python-api-workflow.md)
for scenes, sweeps, detector scoring, and persistence boundaries; use
`pyrite run` when resumability, profiles, remote execution, or analysis apps
matter.

See the generated [CLI reference](docs/repo-design/cli/cli-reference.md) for the
complete command surface.

Full sweeps are usually heavy. Use
[PyRITE's remote workflow](docs/guides/running-on-a-cluster.md) or the portable
SLURM templates documented there.

## Core physics

Pipeline: sampled electron beam → single-scattering electron transport →
segment-wise PXR+CBS, incoherent bremsstrahlung, and characteristic lines →
photon escape → detector scoring.

- **[Electron transport](docs/physics/beam-transport/electron-transport.md):**
  independent piecewise-linear flights use explicit
  elastic collisions and condensed energy loss, deterministic by default with
  optional Urban straggling. The default samples ELSEPA partial-wave elastic
  cross sections, with material-level SBETHE collision stopping;
  beamline space charge and secondary electrons are not modeled.
- **[Crystal source](docs/physics/materials/structure-factor.md):** phase-specific
  structures, complex atomic form factors, Debye-Waller factors, selected
  reflections, and optional mosaicity define the reciprocal-space coupling.
- **[Coherent lines](docs/physics/radiation-physics/coherent-radiation.md):**
  first-order kinematic Born PXR and CBS amplitudes are coherently summed within
  each transport segment (between elastic scattering events which change an
  electron's direction of travel $\mathbf{\hat v}$). A collision-free
  flight may be subdivided to resolve CSDA energy and clock evolution; each segment
  uses one representative velocity for radiation calculations. The PXR and CBS in-medium
  Bragg resonance is $\omega=(\mathbf v\cdot\mathbf g)/[1-
  \mathrm{Re}\lbrace n (\omega) \rbrace \ \hat{\mathbf n}\cdot\mathbf v]$. Dynamical
  diffraction is not modeled, and electron channeling in crystals is currently unsupported.
- **[Incoherent continuum](docs/physics/radiation-physics/bremsstrahlung.md):**
  the default is the direction-resolved BremsLib v2.0 double differential
  cross section (`pyrite tables fetch bremslib`); without its tables a run
  warns and uses isotropic EEDL bremsstrahlung — the ENDF-6 MF=23/MT=527 total
  cross section multiplied by the normalized MF=26/MT=527 photon-energy
  density, also selectable as `bremsstrahlung_model="eedl"`. Unscreened Born
  Bethe-Heitler with an Elwert correction remains an optional/fallback backend.
- **[Characteristic lines](docs/physics/radiation-physics/characteristic-radiation.md):**
  electron-impact vacancies from EEDL subshell ionization cross sections relax
  through xraydb fluorescence yields with L-shell Coster-Kronig redistribution;
  each line carries its natural-width Lorentzian and is scored as a separate
  incoherent component added to the line spectrum.
- **[Photon transport](docs/physics/radiation-physics/photon-escape-and-dispersion.md):**
  straight-ray Beer-Lambert escape uses the total narrow-beam coefficient
  (photoabsorption plus Rayleigh/Compton removal); scattered photons leave the
  ray with no build-up, redirection, or re-emission, and bulk refractive
  dispersion is passive. Interface optics are omitted.
- **[Detector treatment](docs/physics/detectors/detector-response.md):** source
  spectra are evaluated in a fixed far-field direction; solid-angle acceptance,
  aperture broadening, efficiency, and measured-energy redistribution are
  downstream steps.

By default, PyRITE neglects interference between distinct physical flights and
incident electrons. The optional [`coherent`/`both` tracking policy](docs/physics/radiation-physics/coherent-emission.md) preserves phase across a single-electron trajectory and blends inter-electron
terms using the sampled bunch form factor.

See the [physics model index](docs/physics/index.md) for assumptions, model
variants, conventions, and links to each implementation-level description.
Core geometry conventions are documented in the
[tilt convention](docs/physics/geometry/tilt-convention.md).

## Data and outputs

`src/pyrite/data/catalog/`
is immutable catalog source for crystals, media, scan profiles, materials, and
stacks, one TOML file per object.
Phase-specific CIFs live under `src/pyrite/data/cifs/`; production loading is
offline. `uv run pyrite material validate` checks the bundled catalog.

Active checkpoints use `checkpoints/<stem>/{line,brem,characteristic}.h5`
plus a `checkpoints/<stem>/meta.json` manifest.
Stored source spectra exclude downstream detector response unless a detector view
applies it explicitly.

## Validation and provenance

The analytic core is cross-checked against Feranchuk et al. and Zhai et al.
Consult the [validation ledger](docs/validation/physics-validation-ledger.md) for
claim-level status and the [validation method](docs/validation/methodology.md)
before scientific use.

## Documentation

- [Published Documentation](docs/index.md)
- [Physics Models](docs/physics/index.md)
- [Python API](docs/api.md)

## References

- W. Zhai et al., *Nat. Commun.* **16**, 11218 (2025).
- I. D. Feranchuk et al., *Phys. Rev. E* **62**, 4225 (2000).

## Status and license

Academic research code under active development. No project license currently
attached; contact author regarding reuse.

Native GPT `.gdf` electron-beam snapshots are supported as a local beam source;
see [GDF beam import](docs/guides/gpt-gdf-beams.md) for configuration and CLI examples.
