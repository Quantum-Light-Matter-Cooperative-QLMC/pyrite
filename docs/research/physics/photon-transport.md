# Optional photon transport

Status: architecture agreed for [issue #341](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/341);
recorded in [ADR-0016](../../adr/0016-photon-transport-source-and-scoring.md).
The photon tracker described here is not implemented or validated. Current
production behavior remains [straight-ray escape](../../physics/radiation-physics/photon-escape-and-dispersion.md).

## Existing interfaces and missing data

The present interfaces provide useful building blocks, but not a complete
photon transport model:

| Owner | Available behavior | Extension required |
| --- | --- | --- |
| `materials/photon_cross_sections.py` | Five EPDL2025 integrated per-atom cross sections, including separate nuclear and electron-field pair channels | Pinned subshell photoionization cross sections, scattering functions, and form factors |
| `montecarlo/transport/pair_production.py` | Stack/prism first interaction and pair daughter sampling | Return every interaction channel and element; resume photons after scattering |
| `montecarlo/eadl_relaxation.py` | EADL transition energies and probabilities; deterministic expected vacancy cascade | Sample individual vacancy decays and daughter photons/electrons |
| `montecarlo/transport/secondaries.py` | Charged-particle generation queue, pair launches, energy balance | One owner of photon interactions and charged daughter handoff |
| `montecarlo/runner/emission.py` | Directional continuum and characteristic estimators with escape attenuation | An unattenuated source adapter with energy, position, direction, and statistical weight |
| `instrument/` | Finite detector and filter geometry, ballistic filter transmission | Track filter interactions before applying the detector response |

`scripts/release_epdl_table.py` exports only five integrated MF=23 channels.
Its pinned table cannot supply photoelectric shell probabilities or the
angular distributions requested by #341. Extend the data release with a
separately versioned artifact; retain the existing attenuation artifact and
its digest. Inspect the full upstream EPDL2025 sections and declared
interpolation laws before specifying the new parser{cite:p}`epdl2025`.

EADL's `vacancy_cascade` returns expected vacancy visits, not individual
transitions. A sampled cascade must consume `EADLSubshellRelaxation` transition
records, retaining explicit terminal vacancies and the energy residual policy
of `relaxation_energy_budget`. Existing characteristic scoring combines EEDL,
EADL, and xraydb; that scorer's expected output is not a list of photon events.

## Ownership and input contract

Place the tracker under `montecarlo/transport/`, with material data loaders
under `materials/`. Material loaders must not import the tracker or CLI.
`runner/emission.py` assembles source adapters and requests scoring; the
charged cascade owns physical daughter launches. Share the analytic navigation
primitives for slabs, layer stacks, and finite rectangular footprints.
This preserves [ADR-0008](../../adr/0008-no-arbitrary-target-geometry.md).

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
remains explicit until transported or annihilated. Annihilation source rows
must use the eventual #295 interface; its currently open work cannot be
assumed to have landed on `main`.

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

Propose `photon_transport_model=None` as the default. Resolve the enabled
model and its cutoffs, source sampling policy, component policy, and any
Doppler/charged feedback choices into `Numerics`, `Settings`, and `Case`.
Every value that changes the source arrays joins both dataset identity and
per-case content identity when enabled. Add the new photon data digests and
reused relaxation data digest to enabled provenance. Preserve existing
default payloads: absent mode must add no new serialized keys, change no
constant model marker, consume no random draws, and load no new photon data.

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
