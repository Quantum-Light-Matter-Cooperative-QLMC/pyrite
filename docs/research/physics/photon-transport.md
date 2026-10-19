# Optional photon transport

Status: architecture agreed for [issue #341](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/341);
recorded in [ADR-0020](../../adr/0020-photon-transport-source-and-scoring.md).
The photon tracker described here is not implemented or validated. Current
production behavior remains [straight-ray escape](../../physics/radiation-physics/photon-escape-and-dispersion.md).

## Existing interfaces and missing data

The present interfaces provide useful building blocks, but not a complete
photon transport model:

| Owner | Available behavior | Extension required |
| --- | --- | --- |
| `materials/photon_cross_sections.py` | Five EPDL2025 integrated per-atom cross sections, including separate nuclear and electron-field pair channels | None for channel competition |
| `materials/photon_differential.py` | Pinned EPDL2025 subshell photoionization (EADL designators and binding energies), form factor `F(x, Z)` and incoherent scattering function `S(x, Z)` (`epdl-photon-differential-data`) | Compton profiles if Doppler broadening is added; EPDL2025 File 27 has none |
| `montecarlo/transport/pair_production.py` | Stack/prism first interaction and pair daughter sampling | Return every interaction channel and element; resume photons after scattering |
| `montecarlo/transport/photon_navigation.py` | Standalone CPU all-channel/element first-flight reference, vacuum entry/gaps and exact stack/prism boundaries (`photon-channel-first-interaction`) | Called per flight by `photon_histories`; `scene_first_interactions` adds disjoint oriented boxes (`photon-scene-navigation`); production pair/annihilation paths still use their historical entry point |
| `montecarlo/eadl_relaxation.py` | EADL transition energies and probabilities; deterministic expected vacancy cascade | Pass validated records to the sampled cascade for event emission |
| `montecarlo/transport/relaxation.py` | Standalone CPU individual vacancy cascades with explicit draws, transition ancestry, terminal holes and signed energy defects (`eadl-sampled-relaxation`) | Isotropic emission and queue handoff live in `photon_histories`; no production caller yet |
| `montecarlo/transport/photon_scattering.py` | Standalone CPU Rayleigh and Compton events, outgoing photon directions and free-electron recoil (`photon-rayleigh-scattering`, `photon-compton-scattering`) | Optional Doppler/binding model needs new data and validation |
| `montecarlo/transport/photoelectric.py` | Standalone EPDL subshell absorption, Sauter–Gavrila photoelectrons and sampled EADL cascades (`photon-photoelectric-sampling`) | None for the host queue; production integration remains |
| `montecarlo/transport/photon_histories.py` | Standalone CPU photon queue over all five channels with lineage-keyed streams, isotropic relaxation emission, charged handoff rows, an absorbing sample-frame aperture and per-history energy closure (`photon-history-queue`) | Source adapters and normalization; positron handoff |
| `montecarlo/transport/photon_coupling.py` | Host photon/electron round loop: photon-born electrons through the shell cascade from an explicit launch, hard bremsstrahlung returned to the queue, coupled per-history ledger (`photon-charged-coupling`) | Positron transport/annihilation handoff; vacancies from charged impact ionization; soft radiative emission; batch-invariant charged generations ≥1 |
| `montecarlo/transport/secondaries.py` | Charged-particle generation queue, pair launches, energy balance | One owner of photon interactions and charged daughter handoff |
| `montecarlo/transport/annihilation.py` | At-rest/in-flight two-photon emission and first-interaction sink from #295 | Hand off emission rows before sink scoring, preserving event lineage and rest-energy accounting |
| `montecarlo/runner/emission.py` | Directional continuum and characteristic estimators with escape attenuation | An unattenuated source adapter with energy, position, direction, and statistical weight |
| `instrument/` | Finite detector and filter geometry, ballistic filter transmission | Tracked photons use `photon_box_from_pose`/`PhotonAperture.from_pose` (`photon-scene-navigation`); ballistic transmission must be bypassed once photon mode is enabled |
| `_physics_contract.py`, `runner/case_tables.py` | Immutable model descriptors and native elastic preparation | Declare photon requirements and prepare numeric tables; implement queue/samplers in native transport |
| `runner/emission_adapters.py` | Selected batch producers in existing spectral slots, with streaming/adaptive preflight | Separate unattenuated source sampling and tracked scoring from directional attenuated arrays |

`scripts/release_epdl_table.py` exports the five integrated MF=23 channels
as the attenuation table, and with `--table differential` a separately
versioned table (`pyrite tables fetch epdl-differential`) of MF=23 MT 534–599
subshell photoionization and MF=27 MT 502/504 `F(x, Z)` and `S(x, Z)`, with
`x = sin(θ/2)/λ` in Å⁻¹{cite:p}`epdl2025`. The attenuation table and its digest
are unchanged. Every upstream section used declares lin-lin interpolation.
EPDL2025 subshell sets and binding energies are identical to EADL2025's for
all 100 elements, so photoelectric vacancies index the relaxation data
directly. Upstream `EFL` is zero, and File 27 holds no Compton profiles; the
MT 505/506 anomalous scattering factors are not packaged.

EADL's `vacancy_cascade` returns expected vacancy visits. The standalone
`sample_relaxation_cascade` consumes `EADLSubshellRelaxation` transition
records, retaining individual transitions, explicit terminal vacancies and
the energy residual policy of `relaxation_energy_budget`. It normalizes the
rounded EADL transition probabilities for categorical sampling, consumes one
explicit uniform per decay and visits daughters in depth-first order. It
does not yet emit directions or hand particles to transport. Existing
characteristic scoring combines EEDL, EADL, and xraydb; that scorer's expected
output is not a list of photon events.

The standalone interaction references sample unpolarized isolated atoms.
Rayleigh uses $(1+\cos^2\theta)F(x,Z)^2$, with $x=\sin(\theta/2)/\lambda$,
and unchanged photon energy. Compton uses Klein–Nishina times $S(x,Z)$
and free-electron recoil; binding and Doppler broadening are omitted, so
it creates no vacancy. Photoabsorption uses EPDL subshell probabilities and
$T=E-B_i$, followed by the sampled EADL cascade. Its Sauter–Gavrila K-shell
angular approximation is applied to all photoelectron shells. These choices
do not model polarization, crystal coherence, anomalous Rayleigh amplitudes
or multiple-vacancy corrections. Their conditional angular laws are normalized
separately from the integrated EPDL rates used for interaction competition.

All samplers consume explicit uniform sequences and fail on exhausted tapes
or rejection budgets. Their input energy coverage follows the EPDL data
domain; numeric endpoint checks are not a physical validation of the models
across that full range. The references do not yet enable a photon tracker.

`track_photon_histories` is the host ordinary-hit reference. Each queued
flight owns a Philox stream keyed by `(seed, history, event)` and then by
daughter ordinals, so order, chunking and batch position do not change a
history. A flight draws two navigation uniforms, then the selected sampler's
draws, then two isotropic-direction draws per relaxation particle. Photons
and electrons at or below their cutoffs deposit locally; relaxation stops at
shells bound at or below the smaller cutoff. Electrons above cutoff and every
positron become charged handoff rows with `2 m_e c²` booked as pending rest
energy. They are not transported here, so this is the photon-only descendant
stage, not the full MeV coupled model. A rectangular sample-frame aperture is
an absorbing boundary scored when its forward distance precedes the sampled
collision; it must lie outside material.

Filter plates enter as disjoint homogeneous oriented boxes in the same flight
competition, so photons cross vacuum gaps, scatter or fluoresce in a filter
and re-enter the target. A separating-axis check refuses boxes that overlap
the stack (infinite slabs included) or each other, and an aperture that
enters material. `photon_box_from_pose` and `PhotonAperture.from_pose` map a
lab-frame `PlanarPose` [mm] into the sample frame [Å] with
`v_sample = Rᵀ v_lab`, `R = sample_to_lab_R(tilt, azimuth)`, and no
translation, since the lab origin is the sample entrance origin. A lineage that reaches the
generation cap is booked as unresolved, never absorbed. Per weighted history,
incident energy equals escaped + detected + deposited + terminal binding +
signed transition defect + charged kinetic + positron rest + unresolved to
roundoff. `detected_spectrum` returns weighted aperture hits per primary per
eV, an aperture-integrated count, not the existing per-steradian spectrum.

`track_coupled_histories` closes the electron half of the MeV loop. Each
round hands the previous round's electrons above the stage threshold and
inside a non-empty stack layer to a charged stage; electrons born in filter
boxes deposit locally because the charged transport knows only the stack.
`cascade_charged_stage` runs the existing shell soft/hard cascade from an
explicit generation-0 `LaunchState` (beam sampling bypassed; mode-off
callers unchanged), with pair production and positrons left to the photon
queue so conversion has one owner. Its per-history
`secondary_energy_balance` splits each launch into escaped, deposited,
reserved binding and radiated energy. Every hard radiative row becomes a
photon at its row endpoint with key `daughter_stream_key(charged key,
emission ordinal)`, the parent's weight, history and root source label;
photons at or below the photon cutoff deposit at their vertex. A round cap
marks remaining launches unresolved.

Limits of this stage: soft radiative loss below `radiative_cutoff_eV` stays
in the continuous deposit; vacancies made by charged impact ionization keep
their binding reserved and emit no fluorescence; positron kinetic energy is
pending until the #295 emission rows are separated from their sink.
Gen-0 tracks use lineage keys, but the cascade keys later generations and
hard-photon directions on batch-wide track ids and generation seeds, so
those depend on the charged batch composition (an existing cascade
property). Charged clocks start at zero.

## Ownership and input contract

Place the tracker under `montecarlo/transport/`, with material data loaders
under `materials/`. Material loaders must not import the tracker or CLI.
`runner/emission.py` assembles source adapters and requests scoring; the
charged cascade owns physical daughter launches. Share the analytic navigation
primitives for slabs, layer stacks, and finite rectangular footprints.
This preserves [ADR-0008](../../adr/0008-no-arbitrary-target-geometry.md).

### Integration with extensible physics

The implemented [in-tree extension contract](../../adr/0017-in-tree-physics-extension-contract.md)
and [extension guide](../../repo-design/compute/extending-physics.md) govern this
work. They provide preparation and batch-scoring boundaries, not a generic
collision callback, photon scheduler, or external plugin ABI.

- `Physics.validate_transport` in `_numerics.py` remains the public compatibility
  authority. Campaign lowering/building resolves data-only selections across
  all layer elements into `Case`; do not introduce a parallel public plan or
  resolve `auto` again during scoring.
- Add immutable, family-qualified photon descriptors to `_physics_contract.py`
  when the implementation exists. Declare datasets, geometry/energy coverage,
  charged coupling, and actual CPU/device support. Metadata cannot certify a
  physical validity domain or implement an interaction. Keep material loaders,
  arrays, queues, callables and device resources outside the descriptor and case.
- Prepare tables through `runner/case_tables.py` and the material owners,
  preserving units and ordered layer compositions. Native transport owns
  event competition, navigation, queue scheduling, RNG and energy accounting.
  Select preparation outside collision loops; the existing elastic adapter
  map is a proof of this boundary, not a photon execution hook.
- `EmissionBatch` borrows staged electron segments and runtime resources;
  `EmissionOutput` returns an existing native-grid spectral slot. Neither is
  the proposed photon batch below. Existing characteristic and continuum
  producers already apply directional escape, so wrapping their arrays cannot
  produce the required unattenuated position/energy/direction measure.
  Introduce that source contract explicitly, sharing lower-level emission
  physics where valid and retaining the existing single upload/retry ownership.
- Tracked continuum or characteristic scoring may replace an existing slot
  only with matching units, grid quadrature, angular measure and per-primary
  normalization. Preserve CXR alternatives and the one characteristic addition
  in `_spectral_components`; never add tracked and ballistic estimates of the
  same source. A separately persisted fluorescence/annihilation component or
  event-response artifact needs explicit results/checkpoint/observation schema
  design; an arbitrary output dictionary does not supply that migration.
- Integrate selection consistently through live, full/bounded replay,
  directional, recompute and adaptive consumers. Initially refuse unsupported
  routes before transport, uploads or writes. Streaming requires an additive
  whole-primary reduction with a fixed population denominator and complete
  descendant lineage; adaptive support requires per-primary observables.
  Existing adapter declarations alone do not establish either capability for
  resampled photon histories. Preserve current coherent/temporal refusals.

Electron segment positions are in Å and electron energies/cutoffs in keV at
the emission boundary; native photon grids are in eV. The proposed photon batch
uses laboratory-frame directions, whereas emission batches use sample-frame
directions. Source adapters must perform explicit frame/unit conversion once,
with lineage and population normalization preserved.

The proposed photon batch is a structure of arrays:

| Field | Meaning |
| --- | --- |
| `position_ang[N, 3]` | Emission/interaction point in the laboratory frame, Å |
| `direction[N, 3]` | Unit propagation vector in the laboratory frame |
| `energy_eV[N]` | Photon energy, eV |
| `weight[N]` | Expected photon count represented by a scoring history; one for a physical event |
| `history_id[N]`, `track_id[N]`, `event_id[N]` | Stable primary, charged source, and photon lineage identifiers |
| `source_kind[N]` | Bremsstrahlung, primary characteristic, secondary fluorescence, or annihilation |
| `role[N]` | Physical cascade event or weighted scoring history |

The entry point validates finite positions, positive energies and weights,
unit directions, compatible shapes, and data coverage. It must not silently
clamp an outside source into a target. Empty batches are valid. Positions
outside material require explicit vacuum navigation, including entry into a
filter. Geometry and material tables are immutable inputs shared by batches.

Each interaction returns the channel, element, region, position, outgoing
photon rows, charged daughter rows, and an energy accounting record. Escaped,
deposited, below-cutoff, and unresolved histories have separate terminal
reasons. A step or queue limit produces an explicit incomplete result; it
cannot be interpreted as absorption or a complete spectrum.

## Source handoff and prevention of duplicate contributions

### Sampling the continuous source after electron transport

For the mean photon spectrum, the continuous-source adapter is staged:
run the existing electron transport, retain the unattenuated emission measure
along its segments, sample weighted photons from that measure, then transport
each photon and its required descendants. This does not require analog
emission events along every original electron track. Sampling is conditional
on the chosen segment and location within it, retaining the joint energy and
direction distribution. A spectrum integrated over the target depth or
already attenuated into one detector direction cannot serve as this source.
Sampling segments on demand also avoids materializing a dense spatial,
energy, and angular histogram.

This is a design inference for a fixed material and a linear mean-response
tally. PENELOPE provides a relevant implementation precedent: its source can
read precalculated particle phase space containing energy, position,
direction, weight, and shower lineage (2018 manual, Section 7.2.2, printed
pp. 325–326){cite:p}`salvat2019penelope`. That interface supports staged
simulation; it does not by itself validate PyRITE's proposed emission
estimator.

The primary electron stage must still use an appropriate radiative loss and
recoil model. Drawing photons afterward cannot repair a primary trajectory
whose transport neglected important radiative losses. Conversely, weighted
scoring photons do not debit that trajectory a second time.

Photon-only descendant tracking can cover scattering and photoelectric
fluorescence under a stated local charged-energy deposition approximation.
For the full MeV model, photon-born charged particles need transport when
their range, radiation, or annihilation affects the tally. They can be
processed in a later queue and their photon descendants returned to the
tracker; the original primary electron simulation need not be rerun or
interleaved in wall-clock order. This weighted descendant cascade must
include secondary bremsstrahlung and charged-particle-induced vacancies,
besides direct relaxation photons.

A mean source supports a mean spectrum. Coincidences, pulse-height response
to several correlated photons, and physical energy closure per original
electron require additional event correlations. Resampled source histories
also share the finite electron sample's uncertainty; increasing the photon
sample count alone does not eliminate it. Preserve primary lineage for
uncertainty estimates.

### Physical events and weighted estimators

Two source roles must remain distinct. Physical hard bremsstrahlung events
already remove energy from the electron and can launch pair daughters.
Weighted source histories estimate a spectrum on the resulting charged
tracks. In the current coupled route, `_coupled_brem_from_segments` estimates
the continuum over both soft and hard energies; it deliberately does not
histogram sampled hard events. Replacing this with a hard-event histogram
would lose the soft continuum and change the estimator's variance.

In photon mode, physical event photons enter the shared photon/charged queue
once. Disable the old `convert_hard_photons` first-interaction pass for those
events, since the tracker now owns conversion. Photoelectrons, Compton recoil
electrons, relaxation electrons, and pair daughters use the existing charged
launch boundary when above its threshold. Below-threshold kinetic energy and
terminal vacancy energy enter deposition accounting. Positron rest energy
remains explicit until transported or annihilated. #295 has landed:
`annihilation.py::score_annihilation_photons` returns two rows per event with
`k_keV`, `direction`, `origin` and event indices, then applies a first-interaction
sink. Photon mode must separate emission from that sink and hand off the
unattenuated rows once, converting keV to eV and preserving stable parent
lineage. Mode off retains its existing host/device emission streams and draw
order. This is an integration prerequisite, not missing annihilation physics;
the [current positron model](../../physics/beam-transport/positron-transport.md)
still follows no scattering or absorption daughters from those photons.

Weighted histories use an independent stream and a documented unattenuated
source measure over emission position, photon energy, and solid angle.
Importance sampling must carry the corresponding source/proposal weight;
the current spectrum evaluated in one observation direction is insufficient
to construct that measure. Primary characteristic and bremsstrahlung adapters
must expose emission before applying target transmission. All descendants
retain their root source label, with secondary fluorescence also identified
as an interaction product.

Scoring histories must not create a second physical charged cascade or debit
the primary energy ledger. Their response estimator must include radiation
from interaction daughters, through weighted descendant transport or a
separately derived conditional estimator. That estimator needs its own
validation before it can replace the existing continuum. A physical-event
histogram and a weighted estimate of the same radiation must never be added
together.

Source sampling must cover the full emission solid angle, including initial
directions outside the detector cone: their photons can scatter into its
acceptance later. Sampling energy and angle independently is permissible only
where the source model actually factorizes. A detector-biased angular proposal
must retain support wherever the source can contribute through subsequent
transport and carry the matching importance weight. Truncating the proposal
to the detector cone cannot recover scattered-in radiation.

## Coherent and incoherent line boundary

Keep PXR/CBS on the existing line path initially, including the route called
incoherent in the emission policy. In the coherent route, sum amplitudes with
the existing formation absorption and propagation phase before scoring the
intensity. Do not turn individual segment amplitudes into independent photon
histories. Do not pass the already attenuated line spectrum through another
target escape factor.

This initial mode transports bremsstrahlung, characteristic, fluorescence,
and annihilation photons. Coherent line scattering and fluorescence induced
by absorbed PXR/CBS photons remain outside that mode and must be reported as
limitations. Extending them requires a separately derived spatial source and
intensity estimator; the escaped coherent spectrum does not determine the
absorbed-photon source. Proposed mode metadata must identify this component
policy explicitly.

## Navigation and detector scoring

Implement ordinary photon histories and detector-hit scoring as the reference
first. This follows PENELOPE's detailed photon transport, secondary stack, and
impact-detector boundary tallies (2018 manual, Sections 6.5 and 7.1)
{cite:p}`salvat2019penelope`. Coupled descendants are processed through queues;
the scheduling order does not have to reproduce simultaneous physical time.
PENELOPE's documented variance-reduction package uses splitting, Russian
roulette, interaction forcing, and Woodcock tracking (Section 7.1.4).
The conditional detector estimator below is a proposed extension, not a claim
about a built-in PENELOPE method. Validate it against the ordinary-hit
reference before enabling it.

Generalize the existing first-interaction navigator without changing its
mode-off callers. Preserve the separate EPDL channel totals when selecting
the absorbing/scattering element in a mixture. After a scattering event,
resume from its position and new direction/energy, recomputing the crossed
regions. Shared boundaries need a deterministic ownership rule that avoids
repeated zero-length crossings and skipped material. Initial supported
targets are planar stacks and finite rectangular footprints. Grooves must
raise for photon mode until arbitrary scattered directions and material
re-entry are handled; the existing groove escape only supports its working
facet observation direction.

Finite filter plates join the same navigation sequence, including vacuum
gaps and possible return to previously crossed material. The current
ballistic filter transmission must be bypassed for tracked photons. A first
target-exit record is useful diagnostically, but becomes a terminal escape
only after the photon has cleared all modeled material and detector scoring.
Detector response remains downstream of photon transport and is applied once.

An analog photon has zero probability of hitting an exactly specified
direction. Physical detectors therefore score aperture/pixel intersections.
For the existing scalar directional result, derive a conditional detector
estimator, or use a documented finite angular bin with solid-angle
normalization and a convergence check. Do not compare a raw aperture photon
count with the existing per-solid-angle spectrum. Source adapters and scorers
must share the same angular measure, energy-bin convention, and per-primary
normalization. No target or filter Beer--Lambert factor is applied again to
a photon whose interactions have already been sampled.

For small apertures, investigate a conditional detector estimator at emission
and after each scattering or fluorescence event. It scores the expected
contribution of the next outgoing flight into the aperture using the relevant
angular distribution, outgoing energy, and survival through the remaining
material, while the sampled history continues. Derive this estimator before
implementation; its conditional survival is distinct from reattenuating an
already scored analog escape. Analog hits and conditional scores for the same
contribution must be alternative estimators or combined with a derived
weighting scheme, never simply added. Verify scattered-in radiation with a
source directed away from the detector and a geometry that permits a
scattered path to its aperture.

## Interaction models and validation boundary

Use EPDL integrated totals for channel competition. Photoelectric shell
selection needs the new partial tables and a documented edge/binding-energy
alignment with EADL. Compton energy/angle sampling needs the incoherent
scattering function; Rayleigh needs the form factor. Doppler broadening is
an explicit optional model, not implied by the word Compton. Reuse the
existing pair sampler's documented PENELOPE model{cite:p}`salvat2024penelope`;
retain separate nuclear/triplet channel provenance and its existing
approximation policy rather than inventing a new triplet recoil model here.

This note supplies architectural contracts, not new validated sampler
equations. Each implemented sampler requires a source derivation, units,
normalization, limits, a `Validation:` marker, a ledger row, and independent
fresh-context verification. Proposed new claims cover photoelectric shell
selection, sampled EADL relaxation, Compton, Rayleigh, and the shared photon
queue/accounting. Existing pair claims apply only where their actual model
and inputs are reused. No claim status changes follow from this design.

## CPU/GPU layout and random streams

Implement a host reference over numeric batches first, with explicit
uniform draws at the sampler boundary. Stage ragged material/subshell tables
as contiguous values plus offset arrays so the same data can later serve
JIT/device kernels. Active queues use numeric channel and region identifiers;
they carry no Python material objects into device execution.

Photon stream keys derive from seed, primary ID, source track/event ID,
interaction ordinal, daughter ordinal, and a dedicated channel salt.
Neither chunk position nor queue compaction order may define identity.
Rejection draws advance a local interaction stream. Photon streams must not
consume the existing electron stream. Test reordering, chunking, and resume
behavior before introducing device queues. The host reference and device
implementation must agree on deterministic accounting and distributions;
any required bitwise agreement must be specified separately from statistical
agreement. An initially supported CPU implementation must explicitly reject
unsupported device requests rather than conceal a host fallback.

## Opt-in identity and provenance

Propose `Physics.photon_transport_model=None` as the default; this selector
does not exist yet. `Physics` owns the physical model, component policy and
Doppler/charged feedback choices. `Numerics` owns execution controls and
sampling budgets. Lower the resolved enabled values through campaign settings
into `Case`, retaining the existing deprecated-input compatibility path.
Every value that changes the source arrays joins both dataset identity and
per-case content identity when enabled. Add the new photon data digests and
reused relaxation data digest to enabled provenance. Preserve existing
default payloads: absent mode must add no new serialized keys, change no
constant model marker, consume no random draws, and load no new photon data.

Use `campaign/profiles.py`'s canonical dataset identity and
`case_content_key`/API source-key owners. A dataset records the union of
consumed tables; a case records its subset. Adding an unused photon descriptor
or changing descriptive capabilities must not rekey results. Enabled numerical
changes need a selected model-generation marker and the applicable data tags
or digests, rather than hashing the whole descriptor catalog.

Extend the data-only resolved snapshot through `_physics_provenance.py` and
`campaign.physics_view`, and expose it through `results/provenance.py` and
new trajectory captures. Replay must check the enabled model/data snapshot
before scoring or writes, while preserving legacy captures without a snapshot.
Remote allocations must compare staged dataset/table identities with submission
before reuse; the snapshot is descriptive and does not replace canonical hashes.

Keep detector response and read-time aperture selection under the existing
observation identity rules. If photon transport is pre-scored for one
aperture or filter stack, that selection and its geometry must enter the
stored artifact's identity; it cannot be reused as a response-free source
for another detector. Persist enough source/exit information to distinguish
those two artifact kinds. Resuming an incomplete photon queue requires its
lineage counters and model/data identity, not just the primary seed.

## Required evidence before enabling the mode

The independent full-track comparison uses pinned PENELOPE or Geant4 input,
geometry, material definitions, data/model versions, beam normalization,
cutoffs, seeds, histories, tally definitions, and uncertainty estimates.
The comparison must use the same detector acceptance and photon component
policy on both sides. Thick W/Si bremsstrahlung over 300 keV–10 MeV, a binary
alloy fluorescence case, and a 511 keV escape case are the #341 targets.
Heavy comparisons run through `pyrite remote`, with reusable scripts under
`checks/` and compact reference artifacts for regression tests.

Additional focused checks cover vacuum/zero-thickness escape, channel and
element frequencies in mixtures, edge/threshold behavior, layered boundary
navigation, repeated scattering, finite detector normalization, filter
re-entry, sampled relaxation yield against the existing expected cascade,
per-history energy closure, and stream stability. Mode-off goldens and
identities must remain identical. Sampler claims require `rederived` and
full-track comparisons `anchored`; human sign-off remains separate.
