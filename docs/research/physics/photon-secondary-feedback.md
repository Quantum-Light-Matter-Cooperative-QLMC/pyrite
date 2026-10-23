# CPU photon/charged feedback reference

Status: experimental implementation for #341, validation claim
`photon-secondary-feedback` **unverified**. This is not a production
`Physics` selector or independently validated MeV detector model.

## Queue boundary

`montecarlo/transport/photon_feedback.py::FeedbackChargedStage` extends the
electron-only `CascadeChargedStage`. It reuses the shell cascade and its
charged transport models, with positron SBETHE stopping, Bhabha scattering,
positron ELSEPA tables when selected, and scaled BremsLib radiation.
Positron flights are truncated at the existing sampled in-flight annihilation
point **before** secondary electrons and hard photons are harvested.
Subthreshold positrons deposit their kinetic energy at birth and annihilate
at rest. Escaped positrons retain the pair rest-energy reservoir.

Both annihilation photons are returned at the emission vertex, with no
first-collision sink or straight-ray attenuation. The ordinary photon queue
then follows scattering, photoabsorption, relaxation and any new charged
descendants. For each charged launch,

$$
T+2m_ec^2\,\mathbf{1}_{e^+}
=E_{\rm esc}+E_{\rm dep}+B_{\rm rem}+\sum_\gamma k_\gamma.
$$

The photon queue originally reserves $2m_ec^2$ at pair creation.
`ChargedStageResult.rest_energy_eV` transfers exactly that reservoir to the
positron-capable stage once. The stage's escaped term includes rest energy
for an escaped positron; otherwise the reservoir enters the returned photons.
The queue rejects a stage claiming rest energy for an electron or the wrong
amount for a positron. Incident photon weights multiply every term once.

Charged-impact inner-shell holes sample the existing individual EADL cascade.
Photons are emitted isotropically; Auger/Coster–Kronig electrons above the
charged threshold re-enter charged transport, while smaller energies deposit.
The original binding reservation is reduced by the emitted particle energies,
not by a second debit to the already transported electron. The remaining
binding term includes terminal holes, signed EADL transition defects, and any
transport/EADL binding-energy difference. Particle energies are not clamped
to remove those tabulation differences.

Internal relaxation rounds have a finite cap; exceeding it raises rather than
silently dropping Auger descendants. Electron and positron calls use distinct
hashed stage seeds, as do subsequent relaxation rounds. Vacancy streams derive
from the input charged key and a per-history vacancy ordinal. Returned photon
ordinals are unique per input charged row across all emission types. The
existing shell cascade's later-generation batch dependence remains; this
does not establish whole-shower chunk invariance or CPU/GPU parity.

## Scene diagnostic

```python
from pyrite.photon_transport import simulate_photon_continuum

observation = simulate_photon_continuum(
    scene,
    edges_eV=[1000, 10_000, 100_000, 1_000_000],
    n_samples=100,
    source_ceiling_eV=1_000_000,
    charged_feedback=True,
)
```

In addition to the existing Scene requirements, this mode requires
`inelastic_model="shell-soft-hard"`. It prepares the charged stage before
running primary transport. The supplied `electron_cutoff_eV` owns the
charged threshold and cutoff. The observation identity adds the feedback
model and actual positron stopping/elastic table manifest digests. Leaving
`charged_feedback=False` preserves the photon-only observation payload.

The primary electron transport still runs once. Its continuous, weighted
BremsLib source is scored through the coupled queue; it is not summed with
the same primary hard-photon event histogram. Normalization remains
photons/(original incident electron eV), integrated over the physical aperture.

Supported charged geometry is a nonempty contiguous material stack from
$z=0$, with an optional shared rectangular footprint. Charged filter-box
transport, charged re-entry after leaving the target and vacuum gaps are
unsupported. The Scene entry point and coupled queue explicitly refuse
filter boxes with this stage. Photon-only filter navigation remains available.
Mott electron tables are not reused for positrons. Soft radiative losses retain
the underlying charged transport's deposition approximation.

## Physical characteristic and annihilation sources

These low-level adapters expose separate physical sources. They do not add
new primary source components automatically to `simulate_photon_continuum`:

- `annihilation_emissions(kinds, kinetic_keV, directions, origins, keys)`
  returns unattenuated, back-to-back at-rest or Heitler in-flight photons.
  Its `parent` indexes the supplied annihilation event population.
- `primary_annihilation_sources(result, ...)` adapts the original #295
  recorded annihilation photon table directly, preserving its sampled
  directions/energies and original electron ownership. Its old `absorbed`
  flags are ignored: originally absorbed photons also enter full tracking.
- `primary_relaxation_sources(result, keys, stage, seed=...)` samples the
  recorded charged-impact holes in a supplied transport result and follows
  their Auger daughters. Its photons include characteristic radiation and
  radiation from those additional Auger tracks. The original arrays are
  unchanged. The returned per-history budget obeys
  $B_{\rm reserved}=E_{\rm esc}+E_{\rm dep}+B_{\rm rem}+\sum k$.
- `emission_sources(emissions, history_id, weight=..., source_kind=...,
  event_offset=..., photon_cutoff_eV=...)` converts either emission table
  into the common queue source batch. Physical photon weights default to one;
  the original primary population remains the detector-score denominator.
  Sources at or below the cutoff retain their weighted energy in
  `local_energy_eV`, grouped by `local_history_id`; they do not enter tracking.

The emission sampler lives in `transport/photon_feedback.py`; the physical
source adapters live in `transport/photon_event_sources.py`. A source batch exposes `track_args` and
`source_kind` for `track_coupled_histories`. Use disjoint event-ID ranges
when combining batches under the same seed/history domain. The caller must
select the intended original electron population and must not combine two
representations of the same bremsstrahlung-driven shower. In particular, the
Scene continuous-source path still refuses primary physical pair conversion.

## Explicit event-source aperture scoring

`montecarlo/runner/photon_emission.py::score_photon_sources` accepts these
physical source batches directly. It follows the same photon-only or coupled
queue and ordinary-hit reducer used by `score_bremslib_continuum`, without
running primary transport or drawing a continuous source again:

```python
from pyrite.montecarlo.runner.photon_emission import score_photon_sources
from pyrite.montecarlo.transport.photon_event_sources import primary_annihilation_sources

sources = primary_annihilation_sources(primary_result, photon_cutoff_eV=1000)
score = score_photon_sources(
    sources,
    layers,
    aperture,
    [1000, 100_000, 600_000],
    n_primaries=original_incident_population,
    seed=17,
    photon_cutoff_eV=1000,
    electron_cutoff_eV=1000,
)
```

`layers` and `aperture` use the existing sample-frame ångström geometry.
The denominator includes incident electrons with no emitted photons. The
returned historical `PhotonContinuumScore` type also represents this explicit
event tally: `n_proposals` defaults to the number of tracked input photon
rows, including zero for an empty batch, and never supplies a normalization.
Continuous callers instead retain their original proposal count, including
zero-yield trials. Source history IDs, including local-only histories, must
fall within the declared primary population. Event and tracker cutoffs must
match so omitted emissions cannot silently disappear from a lower-cutoff run.

Sub-cutoff energy remains in `score.sources.local_energy_eV`; it is outside
`score.histories.ledger`, which covers tracked photons only. Primary energy
reservations remain owned by the adapters and original charged transport.
Pass a `charged_stage` instead of `electron_cutoff_eV` for coupled tracking.
The caller still owns disjoint contributions and event-ID ranges. This API
does not combine primary physical sources with a weighted continuum or add
Scene source selection, enabled-only production identity, or persistence for
physical event tallies. Secondary-feedback claims remain **unverified**.

## Evidence and remaining validation

`tests/montecarlo/test_photon_feedback.py` checks annihilation energy and
momentum, sampled impact-vacancy emission, real silicon electron/positron
energy closure, escaped rest energy, Auger feedback/caps, source population
and cutoff bookkeeping, and removal of pending pair rest energy in the
coupled photon queue. Scene tests cover enabled-only orchestration/identity.
A real one-primary, eight-proposal silicon Scene probe also prepares the
positron tables and closes its photon ledger to roundoff. These are bounded
implementation checks. Fresh-context rederivation, external thick-target and
511 keV comparisons, filter charged transport and production persistence
remain separate work.
