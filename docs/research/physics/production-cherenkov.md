# Production Cherenkov integration contract

Proposed production boundary for [#692](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/692).
The implementation checklist and stage acceptance live in that issue. This
page defines the physical and data contracts to test before enabling a public
selector. Production Cherenkov transport, escape and observations are not yet
implemented. The [bulk Frank–Tamm source](../../physics/radiation-physics/cherenkov-radiation.md)
remains an internal homogeneous-medium diagnostic.

## Supported first geometry

Start with one homogeneous, isotropic, nonmagnetic weak-loss dielectric
rectangular slab, vacuum on both flat faces, and explicit thickness, width,
height and pose. The dielectric is a separate, explicit response associated
with that medium. A composition or material name cannot infer optical data.
The charged-particle transport material must describe the same physical
sample; a crystalline quartz transport description cannot silently stand in
for isotropic fused silica.

Electron and positron histories may change speed and direction inside the
slab. First support downstream finite scalar apertures and planar observation
cells in vacuum. Restrict the accepted source/ray domain to paths that reach
a supported flat face. Refuse cases with nonzero unmodelled edge contributions;
do not discard those rays and call the remainder the full slab prediction.
Grooves, coatings, multilayers, anisotropy, resonant/strong-loss media, focusing
optics and secondary optical cascades remain outside this boundary.

This is an incoherent photon expectation from transported charged tracks,
not bunch-coherent Cherenkov emission. The first scorer should use deterministic
position/energy/cone-azimuth quadrature. It must consume no transport random
draws. Photon counting noise is a later observation operation with an explicit
seed contract.

## Source, escape and instrument are different observables

| Quantity | Normalization and meaning |
| --- | --- |
| Internal source | photons/(incident primary eV), integrated over the entire emission cone and in-material paths |
| Downstream escaped | photons/(incident primary eV), after absorption and flat-face boundary propagation, before detector acceptance |
| Collected | photons/(incident primary eV), integrated over an explicitly finite aperture or planar cell |
| Detected | calibrated photoelectrons/counts, or energy/readout per pulse according to the explicitly selected detector model |

Each quantity retains its own photon-energy coordinates and records whether
values are node densities or bin integrals. Preserve polarization until the
last polarization-dependent boundary or detector factor. Optical attenuation
is derived from the explicit optical response; atomic attenuation and X-ray QE
cannot be substituted. Calibration/support must cover the whole requested
band, including integration intervals, even where the source happens to be zero.

A delta cone has no finite pointwise angular density. Existing scalar and
directional per-steradian spectra cannot receive a cone by sampling a single
direction or inserting an arbitrary angular width. Use finite angular cells
or physical aperture-integrated observations, with their own units and
identity. An average per steradian, if offered, must name and retain the
solid angle used in that average. Reapplying acceptance to collected photons
would count it twice.

Track normalization uses the original incident-primary denominator, including
zero-yield primaries. Secondary electron/positron contributions keep their
parent-history weights; the number of emitted tracks is not a replacement
denominator. A physical pulse population multiplies this expectation once;
there is no coherent population-square factor.

## Local-source validity is a release gate

Reuse the anchored Frank–Tamm equation for the internal bulk source. Integrate
actual in-material arclength and instantaneous speed/direction, splitting at
medium changes and resolving dispersive threshold crossings. Never use slab
thickness as path length for an oblique or scattered track. Flight/substep
representations must not count the same passage twice. Resolve both numerical
segmentation convergence and errors in the underlying transported path.

[Geant4's Cherenkov reference](https://geant4.web.cern.ch/documentation/dev/prm_html/PhysicsReferenceManual/electromagnetic/xray_production/cerenkov.html)
provides an independent bulk-yield comparison. It cannot alone validate the
finite-boundary field or the local-source approximation in this geometry.

The delta-cone bulk limit is not a uniformly local approximation at a boundary.
A wavelength-to-thickness ratio alone does not establish its accuracy at the
Cherenkov pole. Establish a quantitative validity envelope by comparing
integrated finite-acceptance predictions against an independent finite-track,
finite-slab electromagnetic reference. Vary thickness, energy, track angle,
acceptance and speed near threshold. Record numerical convergence separately
from physical-model discrepancy. A production gate must refuse outside that
envelope; neither a free user assertion nor discarding an arbitrary layer near
each face establishes the missing boundary contribution.

Track-scored optical radiation is initially a perturbative tally. Integrate
its emitted energy over the declared full source band and compare against the
charged-history energy budget before detector loss. Specify and validate the
maximum neglected feedback, along with single-photon recoil validity. An
accepted detector may see little energy while the total emitted energy is
large; collected energy cannot justify neglecting feedback. Refuse unsupported
feedback regimes rather than subtracting an unvalidated mean loss.

## Escape and reflection contract

The boundary scorer must propagate polarized source photons with absorption,
Snell refraction, Fresnel **power** coefficients and total internal reflection.
Conserve photon number in the declared band and close the lossless energy
ledger. Preserve source position and path to the boundary so that finite
detector acceptance is geometrically meaningful.

The initial supported observable may be the explicitly labelled *first-pass
downstream* escape. In that mode reflected/TIR photons remain in a separate
retained tally; they are not absorbed, lost, or silently reassigned to later
downstream escape. Close the source partition with upstream first-pass escape
and absorption as well, even when only downstream observations are requested.
A total downstream observable requires validated repeated
reflection transport or a quantitative bound on every omitted contribution.
For finite lateral dimensions, later reflected rays can encounter unsupported
edges. A retained ledger alone does not solve their eventual fate.

Validation must include subcritical, critical and supercritical incidence,
zero loss, finite absorption, polarization, oblique charged paths, nonnormal
detector planes and aperture clipping. A transparent radiator can produce
internal photons while its direct downstream cone is totally internally
reflected. A nonzero source therefore does not guarantee a nonzero detector
signal; the production example must test its actual escape geometry.

## TR combination policy

Adopt refusal for the first production version: enabling bulk Cherenkov and
any planar-stack TR/history/bunch/two-beam scorer for the same case is invalid.
Apply the same validation at API lowering, campaign resolution, replay and
recompute, before starting transport. Different numerical energy grids are
not proof of disjoint physical fields.

The [existing TR boundary](../../physics/radiation-physics/cherenkov-radiation.md#scope-boundary)
already identifies the Cherenkov pole in a Cherenkov-active layer. No
intensity subtraction, guessed cross term or sum of the two source laws
defines a validated separation. A future joint-field model needs its own
derivation, model identity and evidence. The proposed
`cherenkov-tr-component-accounting` claim should initially anchor only the
refusal policy, not advertise a combined-field prediction.

## Identity and portable response records

Every selected medium needs explicit photon bands, optical data, source
provenance, convention, interpolation rule and response digest. Geometry,
component model, local-validity policy, reflection mode, source normalization,
grid semantics and observation calibration also participate in their owning
case/result/observation identities. Disabled defaults must preserve existing
numerical behavior and case identities.

{source}`src/pyrite/materials/optical.py::OpticalResponse.to_record` and
{source}`src/pyrite/materials/optical.py::OpticalResponse.from_record` provide
a provisional self-contained `pyrite.optical-response-v1` JSON record. Each
segment stores energy nodes in eV and separate real/imaginary relative
permittivity columns. Its endpoints define one closed supported band; gaps
remain gaps. Metadata and arrays retain the existing response digest.
Restoration validates schema, interpolation policy, metadata, passivity,
support and content digest without fetching source URLs or reading catalog
files. The digest detects stale/corrupt content, not malicious replacement
of both data and digest. This is a response record, not a production Case or
checkpoint schema.

### Fused-silica reference input

{source}`src/pyrite/materials/optical.py::fused_silica_franta_response` supplies
the pinned CC0 [Franta complex-index record](https://github.com/polyanskiy/refractiveindex.info-database/blob/c5c2f188e848453def5970e347399d653df2ffc2/database/data/main/SiO2/nk/Franta.yml)
(SHA-256 `89e3065c48040e9347fb8707371d5528ab45ff4e05be5fd6644658b588ce458e`).
The source describes a fitted fused-silica plate, 0.405 mm thick, at room
temperature without a numeric temperature. It combines measurements and
published optical data; the reference is
[Franta et al., Proc. SPIE 9890, 989014 (2016)](https://doi.org/10.1117/12.2227580).
Its tabulated extinction is retained, including tiny and zero fitted values;
these are not measured absorption bounds. The broad table also includes
absorbing regimes and is not wholly a valid weak-loss Cherenkov band.

The existing scalar index conversion and linear-epsilon interpolation apply.
`OpticalResponse.require_band(lower_eV, upper_eV)` checks an entire closed
interval inside one supported segment, including its endpoints. Checking only
grid nodes would permit integration across a gap. Coverage alone establishes
neither weak loss nor source/escape/detector validity. For example:

```python
from pyrite.materials.optical import fused_silica_franta_response

response = fused_silica_franta_response()
response.require_band(1.5, 3.0)  # response coverage only, energies in eV
portable_response = response.to_record()
```

This is a portable reference input, not a production configuration. Binding it
to a matching amorphous transport material, explicit per-medium bands and
calibrated full-band detector support remains required. Crystalline quartz
cannot substitute for fused silica. Any index-only lossless reference must
explicitly identify zero absorption as an assumption rather than a measurement.
No new production physics claim is validated by the reference-data adapter.

## Integration owners and evidence boundary

Paths below are relative to `src/pyrite/`. They identify current seams, not
existing Cherenkov support.

| Concern | Existing owners and required extension |
| --- | --- |
| Public selection and grids | `_numerics.py::Physics`, `energy_grid/component_support.py`: opt-in configuration and an independent supported optical grid; retain the shared X-ray floor |
| Response | `materials/optical.py`: serializable per-medium response and pinned transparent reference; existing Au data are not that radiator |
| Case lowering/identity | `montecarlo/case.py`, `campaign/transition_cases.py`, `campaign/identity_options.py`: validate supported combinations, record optical inputs and model pins; follow the existing optional-component seam |
| Track scoring | `montecarlo/spectrum/cherenkov.py`, `montecarlo/runner/emission.py`, `montecarlo/runner/transition_emission.py`: use one retained transport, explicit history populations and a separate source/escape scorer |
| Angular observations | `montecarlo/runner/directions.py`, `montecarlo/runner/directional_source.py`, `results/spatial.py`, `results/directional.py`: finite acceptance and cone-compatible units, not a pointwise cone density |
| Component results | `_spectral_components.py`, `results/model.py`, `results/radiation.py`: distinct optical component/grid; never append it blindly to line-grid additive keys |
| Optical detector | `detectors/optical_energy.py`, `observations/plan.py`, `observations/store.py`, `instrument/`: calibrated response after collected photons; preserve response identity and prohibit X-ray fallback |
| Stored/replayed runs | `checkpoints/_result_v2.py`, `checkpoints/trajectory_scoring.py`, `montecarlo/trajectories.py`, `checkpoints/recompute.py`: store all scorer inputs and reject captures missing required path/population information |
| Campaign/remote/backend | `campaign/`, `runs/`, `cli/`, `remote/transport.py`, CPU/CUDA transport and scoring: one resolved case contract, portable data records, device-resident scoring and real-hardware evidence |

Before implementing track integration, audit the stored segment schema for
path endpoints, speed evolution, direction, material membership, particle
species, parent IDs and statistical weights. A capture missing those inputs
must fail explicitly; replay may not reconstruct them from a new transport
run. Live, batched, directional and replay paths must call the same scorer.

The proposed track-integration, optical-escape and TR-accounting claims in
#692 have no validated implementation or new ledger status from this contract.
Require fresh-context derivations and the issue's independent anchors before
claiming `anchored`. CPU/CUDA parity requires identical stored rows as well
as a bounded native production smoke. Existing optical/TR hardware evidence
does not establish Cherenkov parity. No human sign-off is assigned here.
