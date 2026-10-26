# TR/CTR app controls and execution

Use the [TR and CTR demo](../guides/transition-radiation-demo.md) for default
plots and a worked comparison. The app reuses the maintained
[planar-stack and crossing solvers](../physics/radiation-physics/transition-radiation.md).

## Analytic controls

Submit **Update comparison** after changing the beam/bunch (energy, RMS duration,
spot, charge), film (catalog medium, thickness, tilt, spectrum observation angle),
photon band (start, stop, points, log spacing) or map angles. Angles are signed
lab angles from the incoming beam in the x–z plane. Forward rays leave the exit
face; backward rays leave the entrance face. Gray map cells mark directions
parallel to the film. At normal incidence, TR vanishes exactly along the track
(0° and ±180°).

The app accepts beam energies up to 100 GeV and off-plane forward/backward
observation directions. The photon band must lie inside the medium's dielectric
table and below the beam energy; numerical input admission does not certify
model accuracy. Angular cut, log color and per-bunch controls redraw cached
arrays. The straight-track Gaussian expectation assumes independent time/position
offsets and includes the existing face-projection transverse form factor and
arrival delays on the tilted film. Production instead samples these offsets.

## Transport settings and data

**Run transported comparison** explicitly starts two fixed-count runs.
**Line histories** sets the line population; the continuum population is twice
it and scores TR. **Transport core** offers CPU cores, plus CUDA when launched
with `pyrite app transition-radiation launch --backend cuda`. Both runs use
midpoint integration, continuous inelastic loss, incoherent PXR/CBS, uncoupled
radiative transport and explicitly selected EEDL bremsstrahlung. The app reports
measured runtime and captured warnings. Counts and absence of warnings are not
convergence certificates.

The analytic view needs only packaged dielectric data. CLI profiles require
SBETHE stopping and ELSEPA elastic tables; `pyrite tables fetch` installs them
and atomic data. The app can use runtime fallbacks if tables are absent;
quantitative app/profile comparisons require the same installed tables.
BremsLib is not required. The transport footprint is 5 × 5 mm, much wider than
the default 1 nm FWHM spot. The scalar detector has 1° polar span and 0.0001 sr
acceptance metadata; plotted source densities still sample one direction.

The production continuum floor is max(plasma energy, 12 eV); optical/IR CTR is
outside this comparison. Adaptive precision, electron-block transport, grooved
faces, channeling and temporal profiles are refused. For the exact current
contract, see [run-case TR](../physics/radiation-physics/transition-radiation.md#run-case-component).
Coherent PXR/CBS requires the documented
[TR–line cross-term suppression condition](../physics/radiation-physics/transition-radiation.md#interference-with-pxr-and-cbs).

## Profiles and checkpoints

`tr_demo` and `ctr_demo` select only silicon with the app's default beam, film,
detector, photon grid and 32/64 fixed counts. `transition_radiation = "history"`
selects the former; `"bunch"` selects the latter. Physical charge controls the
bunch population independently of trial counts. `emission = "incoherent"`
selects PXR/CBS coherence and does not disable CTR.

```bash
pyrite profile show tr_demo
pyrite profile show ctr_demo
pyrite profile create my_ctr --from ctr_demo
pyrite run my_ctr
```

Each profile produces an identity-qualified checkpoint stem with a separate
`transition.h5` component. Select that stem in the analysis app. The
[radiation diagnostics reference](../guides/radiation-diagnostics.md#checkpoint-components-and-dtr)
describes component selection, signed plotting and stored band metrics. Use the
[remote workflow](../guides/running-on-a-cluster.md) for larger sampling studies,
for example `pyrite run my_ctr --remote`, after configuring the host.

## Launch and export modes

```bash
pyrite app transition-radiation launch --smoke
pyrite app transition-radiation launch --edit
pyrite app transition-radiation launch --tunnel
pyrite app transition-radiation export --stem tr_ctr_demo
```

Smoke mode checks startup. Editor mode opens the maintained marimo source;
tunnel mode binds port 2723 for SSH forwarding. Export writes
`pyrite-output/results/tr_ctr_demo.html` with the initial analytic plots. Controls
and transported calculations require the live app. The separately triggered DTR
view is experimental and unverified; see
[radiation diagnostics](../guides/radiation-diagnostics.md#checkpoint-components-and-dtr).
