# TR and CTR demo

Explore transition radiation (TR), film interference, and coherent transition
radiation (CTR) with the packaged marimo notebook and two reference profiles.
The demo reuses the [planar-stack and crossing solvers](../physics/radiation-physics/transition-radiation.md).
It introduces no new radiation model.

## Launch the notebook

From a source checkout:

```bash
uv sync --extra apps
uv run pyrite app transition-radiation launch
```

For an installed distribution, install `pyrite-mc[apps]`, then use the same
`pyrite app` commands without `uv run`. No checkpoints or private catalog are
needed. The initial analytic plots need only packaged material data and the
installed dependencies. The CLI profiles require SBETHE stopping and ELSEPA elastic tables;
`pyrite tables fetch` installs these and the atomic data. The notebook can
use the runtime fallbacks when tables are absent, so quantitative comparisons
with a profile require the same installed tables. EEDL is selected
explicitly for bremsstrahlung; BremsLib and a GPU are not required.

The reference scene is a free-standing **20 nm silicon film**, tilted **5°**
toward a scalar detector at **150° from the 30 keV beam**. The plotted band is
**200–1000 eV**, in 10 eV steps. The 5 × 5 mm transport footprint is much larger
than the idealized **1 nm FWHM Gaussian spot**. The bunch has **1 as RMS**
duration and **0.0001 pC**, about 624 physical electrons. These settings expose
soft-X-ray coherence; they are an illustrative source, not a facility model.

Submit **Update comparison** after editing energy, film thickness, observation
angle, duration, spot size or charge. The first plots show the exact
straight-track film spectrum, a semi-infinite interface comparison, an angular
cut, and the Gaussian intensity form factor. Thickness changes include
interference between the entry and exit fields and absorption. The angular cut
and per-bunch checkbox redraw cached arrays.

**Run transported comparison** explicitly runs two small CPU simulations:
history-coherent TR and bunch-coherent CTR. Each uses 32 line histories and 64
continuum histories; the latter population scores TR. Both use midpoint
integration, continuous inelastic loss, incoherent PXR/CBS, uncoupled radiative transport, and EEDL
bremsstrahlung. These counts illustrate behavior and provide no precision
certificate. The notebook reports measured runtime and captured warnings.
On the development CPU, both runs together took about 5 s with the pinned
tables installed and warm compiled code; initial compilation took longer
(about 33 s in a cold fallback-data run). Startup and data installation are
additional. Individual CLI profile runs reported about 1 s of case execution.

The analytic expectation assumes identical straight tracks with independent
Gaussian time/position offsets. It uses the existing face-projection transverse
form factor, including arrival delays on the tilted film. The production path
samples those offsets and includes scattering and energy loss. It does not
apply an analytic form factor on top of sampled phases.

## Run the reference profiles

```bash
uv run pyrite profile show tr_demo
uv run pyrite profile show ctr_demo
uv run pyrite tables fetch
uv run pyrite run tr_demo
uv run pyrite run ctr_demo
uv run pyrite app analysis launch silicon
```

Both profiles select only silicon and the same film, beam, detector, grid and
fixed counts as the notebook defaults. `tr_demo` uses
`transition_radiation = "history"`; `ctr_demo` uses `"bunch"`. Each dataset gets
its own identity-qualified checkpoint stem and a separate `transition.h5`
component. Select the corresponding checkpoint in the analysis app. The
physical population is controlled by charge, independently of trial counts.
`emission = "incoherent"` selects PXR/CBS coherence; it does not disable CTR.

To change a profile, clone it first:

```bash
uv run pyrite profile create my_ctr --from ctr_demo
uv run pyrite run my_ctr
```

For larger sampling studies use the supported remote workflow, for example
`pyrite run my_ctr --remote`, after configuring the host. Keep the default
notebook demonstration small.

## Units, coherence and sampling

Spectra are photons per incident electron per eV per sr. Selecting per bunch
multiplies by the physical population N. For identical trajectories the
ensemble multiplier is `1 + (N - 1)|F|²` per electron, giving N² per bunch for
an in-phase point bunch and N for a long bunch on average. See
[CTR from transported crossings](../physics/radiation-physics/transition-radiation.md#ctr-from-transported-crossings)
for the estimator, assumptions, and validated limits.

The transported CTR estimate is signed and can have negative bins when the
sample does not resolve the bunch excess. The notebook preserves those bins
on a symmetric log scale and displays the warning. A long bunch with a small
sample shows noise or speckle; it need not match the incoherent spectrum bin
by bin. A lack of warnings is not a convergence certificate.

Production plots show the TR component, other CXR components, and combined
CXR with TR added once. Combined CXR does not include the separately gridded
bremsstrahlung continuum. This example uses incoherent line emission. Combining
TR with **coherent** PXR/CBS requires the documented transverse cross-term
suppression condition; diffracted TR and unsuppressed TR–line interference
are not implemented.

## Export and limits

```bash
uv run pyrite app transition-radiation launch --smoke
uv run pyrite app transition-radiation export --stem tr_ctr_demo
```

The export writes `pyrite-output/results/tr_ctr_demo.html`, containing the
initial analytic plots. Static HTML does not execute controls or transport;
use the live notebook for those. `launch --edit` opens the notebook editor;
`launch --tunnel` binds port 2723 for SSH forwarding.

The solver assumes flat, sharp, laterally infinite interfaces with local
isotropic dielectric response. Side exits produce no TR. The independent-atom
Chantler dielectric response is crude near the plasma energy and absorption
edges. The production continuum floor is max(plasma energy, 12 eV); optical/IR
CTR is outside this demo and the current model. No diffracted TR is included.

Scalar spectra sample the observation direction. Acceptance metadata
(1° polar span, 0.0001 sr) does not resolve detector-integrated fringes.
The dedicated `detected_transition_radiation` planar-detector integrator is
the appropriate tool for pixel/channel counts. The demo grid and angular cut
are visualization samples, not certified integrals. Production TR refuses
adaptive precision, electron-block transport, grooved faces, channeling and
temporal profiles. See the physics page for the full validity boundary.
