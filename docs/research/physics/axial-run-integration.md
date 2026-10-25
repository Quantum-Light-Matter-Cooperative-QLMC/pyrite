# Axial channeling in standard runs: proposed contracts

This is the design gate for [#548](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/548),
not a description of supported runs. `Physics(channeling_model="axial")`
remains refused. The [diagnostic API](../../guides/experimental-axial-channeling.md)
supports periodic-grid evolution and line diagnostics; it supplies neither a
physical escape distribution nor a persisted source spectrum.

## Resolved input and identity

Use a discriminated `Case.channeling` record. Preserve the existing planar
record and omitted off record byte for byte; do not add defaults to either.
The proposed axial variant contains the following resolved fields. Names here
are schema proposals, not accepted Python keywords or CLI options.

| Field | Contract |
| --- | --- |
| `model`, `schema` | `axial`, explicit integer schema version |
| `axis_uvw` | Nonzero integer direct-lattice direction, reduced by a positive greatest common divisor; preserve its sign |
| `temperature_K` | Positive finite temperature within the selected thermal data's range |
| `frame` | Derived right-handed axial-to-lab rotation and projected-cell origin, in the existing crystal orientation convention |
| `entry` | Versioned serializable state recipe described below |
| `cells`, `shape` | Two positive integer cell counts and two even FFT dimensions, with an integer number of samples per projected cell |
| `dz_ang`, `zone_max_ang` | Positive finite axial depth step and maximum axial propagation depth in Å; refuse a cap that truncates the required slab path |
| `ensemble_electrons`, `seed` | Positive fixed MCWF sample count and nonnegative integer seed; distinct from the native transport count |
| `mechanisms` | Canonical ordered selection of TDS and electronic jumps; standard runs require both |
| `boundary` | Versioned physical escape prescription and parameters; no default inferred from a numerical sink |
| `source` | Versioned emission prescription, basis resolution, bandwidth/cluster thresholds and line-shape parameters |
| `budgets` | Explicit maximum grid points, propagation steps, ensemble samples and working bytes; reject excess before solver allocation |

Include every output-affecting value, thermal/potential data fingerprint,
resolved model marker and recipe version in axial case/dataset identity and
physics provenance. Execution-only memory admission limits belong in execution
metadata; include a limit in scientific identity only if it changes a result
instead of refusing it. OAM display origin and plotting controls are analysis
metadata unless they select the emitting state or string.

Use one pure resolver for Scene and profile input. `campaign.lowering` and
`campaign.single_scene` must reach the same `campaign.channeling_cases`
resolver; profile editing must not implement a second physical interpretation.
Extend `CHANNELING_KEYS`, case validation, profile identity and provenance as
one change. Add an axial-specific model marker rather than changing the
existing planar marker merely to add axial support.

### Frames, depth and eligibility

`axial_profile` returns transverse and axial **rows** in `profile.axes` in
crystal Cartesian coordinates. The lattice metric converts `[uvw]` to a
Cartesian direction; an integer triple is not a Cartesian unit vector.
Compose the transpose of those rows with the existing crystal-to-lab rotation
from the resolved `CrystalCut`. Do not accept a second user rotation matrix.
Derive mean carrier tilt from the incident lab beam in that frame. Recipe
tilts are explicitly relative offsets from that carrier, never another copy
of the lab beam tilt.

The real-space solver's depth is along the selected axis. Native segment
length, lab slab-normal depth, stopping loss and time of flight need explicit
conversion for tilted incidence. Entrance phase, transverse drift and the
physical exit plane must use the same transform. A first tier can refuse
non-normal incidence; it cannot silently call solver depth a lab path length.

Proposed first eligibility envelope: electrons in a single, flat, nonmosaic
silicon or diamond slab, isotropic sourced displacement data, rectangular
projected cells, 1–30 MeV, fixed counts and CPU execution. This is an engineering
restriction, not a validated axial energy range. Require a nonempty usable
transition catalog and convergence evidence for the selected axis/energy.
The existing Si peak-energy evidence covers specific ⟨110⟩/⟨111⟩ cases around
2–3 MeV; it does not establish general yields. W at 300 keV remains a
diagnostic anchor, outside this proposed first run tier. Layered materials,
anisotropic axial jumps, positrons, coupled radiative loss, secondaries,
stacks, grooves, GDF inputs and temporal/coherent emission stay refused.

## Serializable entry states shared by planar and axial runs

Introduce a versioned recipe, separate from the executable `EntryState`
objects in `montecarlo.channeling.states`. Both Scene and profile files carry
the recipe; only the solver boundary constructs wavefunctions. Accept scalar,
finite, JSON/TOML-compatible values with strict tagged fields. Reject unknown
keys, booleans as integers, NaN/Inf, zero norms and unsupported dimensions.
Expand defaults before hashing; preserve ordered mixture components.

The first common subset is `gaussian` and `hermite_gauss` in one or two
dimensions, plus 2D `laguerre_gauss` and `bessel_gauss`. A mixture is an ordered
list of positive weights and pure recipes, normalized to sum to one. No
coherent superposition is implicit. Further presets and complex superpositions
need their own explicit parameter schema before admission. Arbitrary arrays,
callables, phase plates and abTEM objects remain diagnostic-only; a file path
is not an immutable state identity.

| Recipe part | Contract |
| --- | --- |
| `version`, `kind`, `dimension` | Positive recipe version, preset tag, dimension 1 or 2 |
| `waist_ang` | Positive finite amplitude waist, retaining the existing preset convention |
| `m`, `n` | Nonnegative Hermite indices; one index for 1D, two for 2D |
| `ell`, `p` | Signed integer winding and nonnegative radial index for LG |
| `ell`, `radial_k_inv_ang` | Signed winding and nonnegative transverse wave number for Bessel–Gauss |
| `impact_ang` | Finite coordinate vector relative to the declared projected-cell origin, in Å |
| `tilt_offset_rad` | Finite relative carrier-angle vector, in radians |
| `impact_average` | `fixed` or an explicit weighted set of impact nodes; no hidden uniform averaging |
| `divergence` | Zero or a declared weighted tilt quadrature, with nodes in radians and normalized nonnegative weights |

For example, a proposed 2D pure recipe is
`{version: 1, kind: gaussian, dimension: 2, waist_ang: 0.5,
impact_ang: [0.2, 0.0], tilt_offset_rad: [0.0, 0.0]}`; omitted averaging
resolves to fixed impact and zero divergence. This is schema notation, not
an invocation of today's API.

Planar lowering calls `planar_entry_density`: it accepts 1D input or the
existing 2D partial trace over in-plane momentum, preserving transverse
coherences as specified by `cr-entry-state-projection`. Axial lowering requires
2D and samples each component through `RealSpaceGrid.sample`. Reuse that
established planar reduction rather than inventing an amplitude projection.
The legacy planar entry remains unchanged when no recipe is supplied. An
explicit recipe owns microscopic impact and angular spread; reject concurrent
nonzero beam emittance/divergence until their joint distribution is defined.
Macroscopic beam spot and microscopic position within a cell are distinct.

The diagnostic currently periodizes and normalizes each sampled component.
That normalization is not a measured entrance transmission. A run adapter must
record pre-normalization norms and refuse unresolved entries, with declared
convergence tolerances; it must not turn missing Fourier weight into physical
entry rejection. Compare increased momentum resolution and larger supercells
separately: a periodized state's norm can differ from one through coherent
image overlap as well as truncation. Rejected entry flux, when physically
defined, needs a separate positive launch distribution.

## Physical escape within the selected entrance-zone architecture

**Maintainer decision (2026-10-17): entrance-zone handoff.** Full-slab quantum
propagation is not the standard-run architecture for #548. The
[escape-operator derivation](axial-escape-instrument.md) separates physical
matching from numerical truncation and defines the positive accounting that
the joint outgoing instrument must satisfy.

The MCWF construction averages outer products of independent trajectories to
obtain a density operator{cite:p}`castin2008`. Its particular numerical
unravelling does not by itself identify a physical measurement of escape.
`AxialMCWF.run` currently records sink depths, total jump counts and averaged
observables. It does not return sink electron IDs, outgoing momenta or physical
escape states. The TDS sink represents omitted transfer outside a finite
band; electronic jumps instead change the tracked carrier momentum. Neither
mechanism implements a common physical free-state boundary.

The planar study [#522](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/522)
also separates spectral truncation from irreversible escape. A band edge, a
potential saddle or a critical angle may define a useful classification;
none alone derives an absorbing two-dimensional quantum-to-classical map.
Sen and Lynn explicitly introduce a phenomenological planar loss model and
discuss rechanneling{cite:p}`sen2014channeling`; it supplies no axial outgoing
distribution. Do not transplant the planar 300 eV margin or band-edge angle.

The architecture decision distinguishes these two routes:

| Architecture | Required evidence and consequence |
| --- | --- |
| Full-slab quantum evolution (not selected) | Retain tracked above-barrier states to the physical exit; no in-solid handoff or artificial absorbing boundary. Numerical sinks invalidate a run unless bounded as numerical error by independent convergence. Would require a different exit-state/stopping/source contract. |
| Entrance zone with physical escape (selected) | Derive a positive escape instrument with a retained map and outgoing measure jointly in depth, transverse position, 2D momentum and energy. Preserve population accounting and quantify omitted rechanneling. Requires differential transfer information for overflow; the current sink rate alone is insufficient. Preserves the intended zone/native architecture. |

**Maintainer decision (2026-10-18): staged depth budget.** Targets range
from hundreds of nanometres to millimetres. The
[return-flux study](axial-escape-instrument.md) shows that a fixed-energy
handoff omits rechanneling yield that grows with depth. #548 therefore ships
the entrance zone with a declared per-case depth budget, derived from the
return-flux deficit, and refuses thicker cases before any work or writes. A
re-entry hybrid is a separate follow-up issue: native transport between fresh
quantum entrance zones whenever multiple scattering returns the electron
inside the matching cone. Classical bound orbits are not used, because the
bound regime has only a few transverse states.

Derive and anchor the entrance-zone map before exposing the selector. Use
full-slab diagnostics to study convergence, without presenting them as an
integrated run or substituting a different architecture.

The finite-grid research helper `channeling.axial_phase_space` now supplies a
positive, correlated position/momentum measurement after an outgoing projector
has been chosen. Its finite Weyl-frame resolution and conditional sampler are
[independently anchored](../../validation/channeling/cr-axial-phase-space-instrument.md).
This solves the measurement algebra, not selection of a physical matching
surface. It neither treats numerical sinks as escape nor supplies stopping,
on-shell launch directions, omitted-coherence bounds or a return-flux budget.

For the zone architecture, the proposed accounting contract is incident weight
equals retained quantum weight plus cumulative physical handoff weight plus
separately reported unresolved numerical weight. Report the signed balance
residual as well. A numerical tolerance never grants permission to relabel
unresolved weight as physical dechanneling. Handoff must retain correlations
between depth, position, energy and both direction components; independent
sampling of marginal distributions needs its own justification.

An outgoing record must carry incident electron ID, event/fate, axial depth,
lab position in Å, kinetic energy in eV, unit lab direction, clock and native
stream key. Derive the launch energy including the declared zone stopping
approximation and reject negative energies. Position, energy and clock must
match the preceding zone row. Seed domains for entry selection, quantum jumps,
handoff and native continuation must be versioned and disjoint; native stream
keys remain tied to incident IDs. Retrying a budget failure cannot silently
resample the ensemble. The existing backend restarts IDs at zero and is not a
resumable per-electron handoff interface.

## Expected-value spectral source

Reuse the [axial polarization derivation](../../validation/channeling/cr-axial-helicity-emission.md)
for a source operator linear in the incident-normalized level density.
`LevelDensity` preserves within-level coherence. Average density operators or
per-electron intensities; never sum amplitudes across incident electrons and
square them. Do not condition the source on survival or multiply an already
incident-normalized observable by survival a second time.

The proposed output is photons/eV/sr/incident electron on the native line grid,
stored once as `spec_channeling`. To reach that output, a new source must define:

1. A complete declared emitting subspace. The present flat-band, one-string
   catalog is not a partition over all strings in a supercell. Summed local
   orbital projectors need an overlap/orthogonality check. Omitted dispersive
   levels and free-to-bound capture need a quantitative bound or scoring;
   planar capture evidence forbids assuming their yield is negligible.
2. A frequency-cluster and line-shape prescription, including finite target
   length, scattering/dephasing and its normalization. A list of delta-like
   lines or the 0.1 eV flat-band gate is not a physical linewidth model.
3. Full observation azimuth and the crystal/axis/beam/lab frame mapping.
   `helicity_pattern` gives helicity components; it cannot replace the full
   polarization matrix at arbitrary azimuth when cross terms survive.
4. Depth-dependent emission placement, photon escape path and attenuation in
   the physical slab. Weight each entry path consistently with the quantum
   history; do not reuse a separable mean path without a declared approximation.
5. Depth/energy-bin quadrature, finite/nonnegative checks with roundoff policy,
   MCWF uncertainty and convergence diagnostics. Fixed ensemble budget and seed
   affect output identity even if transport uses a different electron count.

Use `Result.channeling` and optional `channeling.h5` only after this source
contract is anchored. Common component combination adds it once; brem-only
repair preserves it. Stokes/OAM diagnostics are separate from the scalar photon
component and must not be added to totals.

`channeling.quantum_regression` provides a bounded dense reference for the
full complex two-time correlation of a supplied finite GKSL model. Its
[independent derivation and analytic anchors](../../validation/channeling/cr-quantum-regression-correlation.md)
pin current-component interference, positive spectral phase, finite residence
time, damping and amplitude-level attenuation. It returns a current quadratic
form, not photons/eV/sr. This avoids assuming that a finite-time line shape
times an integrated local yield reproduces a two-time source: an evolving
ensemble and depth attenuation generally make that factorization invalid.
Material-specific currents, Hilbert-space and scattering convergence, photon
prefactors/frames, matching and persisted source remain to be established.

## Consumer audit and initial support policy

The following is an implementation inventory, not a claim that guards already
exist for an axial case. Today the model selector rejects axial globally.
Once a selector is admitted, every downstream path needs variant dispatch.

| Surface and owner | Current planar behavior | Proposed first axial policy |
| --- | --- | --- |
| Scene/profile lowering: `_numerics`, `campaign.channeling_cases`, `single_scene` | Planar-only record and resolution | One shared axial resolver after physical contracts pass |
| Fixed CPU run: `runner.channeling`, `channeling.handoff` | Planar solve, sampled handoff, native launch | Dedicated axial solve/launch adapter; never call planar `zone_solver` |
| Source: `runner.channeling_emission`, `emission_adapters` | Planar expected-value scorer | Dedicated axial source, registered in the existing single channel slot |
| Scalar API/result: `api`, `results.model` | Exposes stored channel separately | Same units/shape and exactly-once combination |
| Directions/physical detectors: `runner.directions`, `observations` | Supported via planar source rescoring | Early axial refusal until azimuth/frame/absorption scoring is validated |
| Replay: `runner.artifacts` | Reconstructs planar source from the case | Early axial refusal before artifact iteration or writes |
| Recompute: `checkpoints.recompute` | Line repair rescores planar CR; brem repair preserves it | Refuse axial line repair; permit brem repair only after byte-preservation coverage |
| CUDA continuation: `runner.channeling` | CPU quantum zone followed by native CUDA | Early axial refusal until the joint launch contract is covered on both backends |
| Adaptive precision: `runner.adaptive` | Refused | Refused until an independent per-electron CR variance contract exists |
| Capture/blocks: `runner._check_channeling_transport` | Refused | Keep refusal; a block-local MCWF solve changes the source ensemble |
| Checkpoint/dataset/provenance: `checkpoints.persistence`, `campaign.profiles`, `_physics_provenance` | Optional channel storage and planar marker | Tagged axial identity, atomic compatible persistence, no reuse of a planar cache |
| Remote submission: `remote`, CLI validation | Forwards planar resolved settings | Refuse unsupported settings locally before sync/submission |

Refusals must run before table preparation, quantum solves, native transport,
checkpoint reservations or writes. Validate raw `Case` callers as well as the
public Scene/profile path. Existing planar support and refusals are regression
contracts, not permission to enable those consumers for axial.

## Claim boundaries and validation requirements

Reserve `cr-axial-zone-handoff-physical` for the new physical escape map,
joint launch distribution, flux balance and continuity. Reserve
`cr-axial-persisted-channel` for the new source estimator, line shape,
normalization, frame/absorption mapping and exactly-once persistence.
The handoff accounting row `cr-axial-zone-handoff-accounting` is `rederived`
with a diagnostic counterexample; the physical row is `unverified`, with no
production implementation or physical verdict.
Register the source claim when its model is selected; attach source derivations
and in-code markers before physics code lands. Both must
reach `anchored` through fresh-context review and independent regression
oracles before #548 is complete.

Reuse `cr-axial-2d-eigenproblem`, `cr-axial-realspace-propagation`,
`cr-axial-realspace-dissipator` and `cr-axial-helicity-emission` only for their
existing scopes. They do not validate the new boundary or integrated source.
`cr-zone-handoff-accounting` and `cr-persisted-channel` remain planar claims;
do not widen them silently.

The eventual bounded CPU anchor must compare equivalent Scene/profile cases,
reach the chosen axial solver exactly once, produce finite nonzero channel
mass and round-trip it through optional storage. Separate anchors must check
zero thickness, vanishing scattering, mixed-state linearity, rotations,
population balance and the joint launch distribution. Test unsupported
consumers with sentinels proving no solver/transport/write occurs. Preserve
existing planar/off identities and values. Numerical grid/depth/ensemble
convergence and any heavy oracle comparisons belong on remote compute.
