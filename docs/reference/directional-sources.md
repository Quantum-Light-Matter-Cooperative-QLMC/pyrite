# Directional sources and angular-map app

The [visual tutorial](../guides/directional-source-heatmap.md) demonstrates one
seeded PXR/CBS map. This page owns the reusable source and interactive-app
contracts. The canonical imports are `pyrite.radiation.directional_source` and
`pyrite.radiation.directional_sources`; both remain **experimental**. See the
[API reference](../reference/python-api/radiation.md#provisional-directional-source-api) for signatures.

## Components and shared transport

`directional_source` returns the incoherent PXR/CBS total, retaining its
PXR–CBS interference. `directional_sources(..., components=...)` returns
separate `DirectionalSpectrum` objects for the requested components:

| Selector | Escaped source |
| --- | --- |
| `pxr-cbs` | Incoherent PXR/CBS total, including PXR–CBS interference |
| `characteristic` | Isotropic atomic lines; angular structure comes from escape geometry |
| `bremsstrahlung` | Continuum; BremsLib angular model when its tables resolve, otherwise isotropic EEDL |
| `transition` | History-coherent planar-crossing TR; infinite slab required |

One transport serves all directions and selected components. Use equal CXR and
bremsstrahlung counts in `FixedSampling`: the directional calculation uses the
line count for both populations. Characteristic and continuum scoring retain
the live runner's lower-cutoff policy. Components stay separate; the result
does not authorize summing arbitrary sources or adding a TR–line cross term.
Cross-electron coherent and channeling radiation are excluded.

TR requires explicit infinite planar geometry (`footprint=None`) and the
existing per-layer X-ray dielectric band. Finite lateral interfaces and optical
response are not inferred. See [transition radiation](../physics/radiation-physics/transition-radiation.md)
for the physical assumptions and current limits.

## Coordinates and escape

The incoming beam defines lab +z. For observation-plane azimuth φ and signed
angle θ, the lab direction is `(sin θ cos φ, sin θ sin φ, cos θ)`.
Default φ = 0 gives the x–z plane. Target pose rotates lab directions into the
sample frame; target angles retain full precision. The map is a plane slice,
with no azimuth or angular-acceptance integration.

Angles increase within [−180°, +180°] without a duplicated periodic endpoint.
Plotting may duplicate the first sample to close the circle. Densities are
photons/(incident electron · eV · sr); `rate(current_na)` gives ph/s/eV/sr.
Omitting current uses the beam's mean current. The conversion and its
independent-electron assumption are defined once in
[spectral observables](../physics/radiation-physics/spectral-observables.md#detector-free-directional-rates)
and checked by the [normalization validation](../validation/radiation-physics/directional-source-rate-normalization.md).

Each emitting segment is attenuated to its first crystal face, including side
faces of a finite footprint, with segment-mean transmission. Finite targets
evaluate every direction. Only an infinite slab marks face-parallel directions
(`abs(n_sample,z) <= 1e-10`) unsupported: NaN density plus `supported=False`.
Zero/underflowed density is distinct from unsupported geometry. Independent
records: [finite transverse crystal](../validation/geometry/finite-transverse-crystal.md)
and [segment escape averaging](../validation/radiation-physics/segment-escape-average.md).

## Photon grids and limits

Supply one explicit, strictly increasing photon-energy **point grid**, at least
two points, starting at or above 10 eV and ending at or below the incident kinetic
energy. No narrow automatic single-direction window is substituted. Refine
angles, energy spacing, reflection count and histories for the intended study;
small statistical errors do not establish grid convergence.

`provenance["line_window_ceiling_eV"]` records the kinematic PXR/CBS line bound,
capped at beam energy. A stop below that bound can truncate forward lines;
this is especially important for relativistic beams. The app uses it as its
automatic stop. Continuum and TR can extend beyond the line band; choose and
refine a suitable explicit window. The app uses beam energy as automatic stop
when TR is selected. See the
[kinematic bandwidth claim](../validation/ledger-core-coherent-physics.md#line-grid-kinematic-bandwidth)
and [emission ceiling claim](../validation/ledger-core-coherent-physics.md#line-grid-emission-ceiling).

Flat slabs and stacks, finite or infinite, are accepted under existing geometry
restrictions. Grooves, direction-targeted longitudinal policies and adaptive
precision are rejected. `direction_batch` bounds scoring scratch; it does not
change normalization. Results retain the resolved case, seed, numerics,
plane azimuth, grids, line ceiling, component and package/model/table identity.
Explicit regeneration after relevant model changes is required; a tutorial's
source hash does not detect numerical drift in the library.

## Interactive app

```bash
pyrite app angular-heatmap launch
pyrite app angular-heatmap launch --smoke
pyrite app angular-heatmap export
```

The default scene is 30 keV point electrons through 2000 Å silicon with a
finite 5 × 5 mm footprint, target tilts 30° and 45°, 128 incident histories per
tilt, 72 periodic angles and 1025 energies from 100 eV to the line ceiling.
Set footprint dimensions to the sample. No campaign, detector or checkpoint is
required.

**Beam**, **Target**, **Footprint**, **Radiation**, **Photon energy** and
**Execution** controls define the calculation. Tilt and target-azimuth lists
accept comma-separated values and inclusive `start:stop:step` ranges
(`5:80:5, 45`). Each polar/azimuth pair requires one transport; submit
**Run source maps** to calculate. The app refuses duplicate orientations and
result arrays exceeding its roughly 1.6 GB guard; it is not a compute-budget
limit.

**Display** controls redraw cached arrays: component, mean current, logarithmic
color/range, orientation browsing or comparison, cut energy and cut angle.
Heatmaps show rates on hover; the angular cut and spectrum compare the displayed
orientations. Panels share axes and color scale. The status rail reports
backend, transport core, elapsed time/cache hit, footprint, grid, line ceiling
and histories. The four-entry calculation cache belongs to the app process.

**Download results (.npz)** retains per-orientation/per-component density and
current-scaled rate, support masks, lab/sample directions, shared coordinates
and JSON metadata (submitted configuration, current, units, ceilings, identity
digests). Exported static HTML contains the default example and cannot launch
new transport calculations.

### Internal Cherenkov view

**Cherenkov (internal bulk)** is a separate homogeneous Frank–Tamm source.
Declare constant refractive index, kinetic energy, straight path length and
photon band. The band must be transparent with nearly constant index, with
photon energies far below electron kinetic energy. Target pose, catalog
material and footprint from the transported view do not enter this source.

Its full-azimuth polar bins about the velocity hold
**photons/(electron · eV · angular bin)** integrated over the path, rather than
pointwise per-steradian density. Below threshold the yield is zero. No
interfaces, refraction, total internal reflection, absorption or finite-target
escape are included, and the result is never added to TR. Downloads include
energies, polar centres/edges, per-bin/path and per-metre yields, threshold mask
and declared source settings. Model: [Cherenkov radiation](../physics/radiation-physics/cherenkov-radiation.md).

### Backends and molab

The array backend is fixed at import. Select CUDA before launch with
`pyrite app angular-heatmap launch --backend cuda`; the status rail reports the
resolved backend. CPU transport cores remain available; the CUDA core is offered
on a CUDA launch. Heavy scripted studies belong on the GPU host through
[remote execution](../guides/running-on-a-cluster.md).

[Open the maintained app in molab](https://molab.marimo.io/github/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/src/pyrite/apps/angular_heatmap_app.py).
Its inline script metadata declares dependencies. Enable the GPU before the
first run; changing backend later requires a kernel restart. The first cell
fetches pinned tables into the session if missing. Check for `Backend: cuda`
and download useful results before the session ends. Static previews have no
computed outputs; use a live session. The distribution is `pyrite-mc`;
the unrelated `pyrite` package is not PyRITE.
