# Issue #341 — explicit Scene-to-photon reference scoring boundary

**Status:** experimental CPU diagnostic, not enabled in the ordinary
`pyrite.simulate`/scan/recompute/checkpoint/observation pipeline. Source:
`src/pyrite/photon_transport.py`. Validation label:
`photon-continuum-scene-handoff` (`rederived` for the bounded population,
geometry, identity and orchestration contract; independently anchored
full-transport comparisons still needed). See the
[independent derivation](../../validation/radiation-physics/photon-continuum-scene-handoff.md).

## Purpose and why the API is deliberately separate

The existing `Result.brem` and physical-detector reconstruction are based on
**photons per incident electron, per eV, per steradian**, evaluated in one
observation direction, with straight-ray attenuation. The new tracker reports
**photons per incident electron, per eV, integrated over the entire physical
detector aperture**, including photons scattered or fluoresced in from outside
that direction. Copying the latter into `Result.brem` or combining the two
would be dimensionally invalid and would double-count a continuum contribution.

The new entry point `pyrite.photon_transport.simulate_photon_continuum` can be
used on a public `Scene` in a CPU NumPy installation. It lowers the existing
case, uses the regular primary-electron transport once, samples the joint
unattenuated BremsLib photon source from the existing tracks, and scores
ordinary hits with the already implemented photon-history queue. It does not
run the directional spectrum pipeline and therefore cannot accidentally add
its weighted continuum to the old directional BremsLib result or to a physical
hard-photon event histogram. It neither subtracts energy from primary electron
tracks nor re-launches their physical hard-event cascade.

The primary electron transport **must** resolve BremsLib coupled radiative
loss (`bremsstrahlung_model="bremslib"`,
`radiative_model="bremslib-soft-hard"`), since a weighted photon source after
transport cannot correct primary electron recoil and energy-loss physics.
Both resolved models are checked before any transport is started.

## Reference use

```python
from pyrite.photon_transport import simulate_photon_continuum

# `scene` is an existing pyrite.Scene with:
# - a physical, unpixelated PlanarDetector (ideal response)
# - Physics(brem_source="mc", bremsstrahlung_model="bremslib",
#           radiative_model="bremslib-soft-hard", radiative_cutoff_eV=1000)
# - Numerics(energy_model="midpoint")
# - CPU backend selected *before importing pyrite*:
#   PYRITE_MC_BACKEND=cpu
observation = simulate_photon_continuum(
    scene,
    edges_eV=[1_000, 10_000, 50_000, 100_000],
    n_samples=50_000,
    photon_cutoff_eV=1_000,
    electron_cutoff_eV=1_000,
    source_ceiling_eV=100_000,
)
print(observation.score.total_per_primary_eV)  # photons / electron / eV
print(observation.score.direct_per_primary_eV)
print(observation.score.redistributed_per_primary_eV)
print(observation.observation_identity_digest)
```

The bin edges determine the detector **tally** only; `source_ceiling_eV` bounds
the sampled emission measure, not just the displayed energy range. Photons
above the highest displayed edge can scatter *into* an energy bin if included
in the source range. Unbinned detector hits are reported separately.

## Supported boundaries

- Analytic single slab and multilayer stacks, rectangular crystal footprint,
  explicit `PlanarDetector` pose and optional `FilterPlate` material boxes.
  Everything is transformed from lab millimetres to sample-frame ångströms
  **once**, using the transport geometry's `sample_to_lab_R` rotation.
- CPU NumPy only. No implicit transfers of CuPy arrays. Fixed primary electron
  population and reference independent Philox source proposals.
- Only unpolarized, weighted BremsLib bremsstrahlung emission. No PXR/CBS,
  original primary characteristic, transition radiation or beam channeling.
- Explicit rejection of grooved entrances, physical-event pair conversion,
  positron transport, adaptive precision, pixelation, detector response,
  acquisition and unsupported material/radiation modes.
- Photon interactions include existing EPDL collision sampling and sampled
  EADL relaxation. By default photon-born charged daughters are booked as
  handoffs. Explicit `charged_feedback=True` follows electrons, positrons,
  charged-impact relaxation, Auger descendants and annihilation photons in a
  contiguous target stack, and requires shell-soft-hard inelastic transport.
  That mode refuses filter boxes. Its enabled-only observation identity adds
  the feedback model and positron table pins; the photon-only payload is
  unchanged. See [CPU secondary feedback](photon-secondary-feedback.md).
  Neither mode is a validated complete MeV thick-target response.

## Source and observation identities

The **source identity** is the existing `case_content_key`, including the
resolved input and BremsLib table markers. Nothing changes on the default
`Case` path, so existing goldens/cache keys are unaffected.

A separate SHA-256 **observation identity** includes source digest, model
version, pinned EPDL integrated, EPDL differential and EADL digests, proposal
seed, proposal count, source/cutoff ranges, full sample-frame detector/filter
geometry and photon energy-bin edges. This means rerunning the same electron
source with different filter geometry, binning or statistical precision cannot
mistakenly reuse the same observation. Identical input remains deterministic.

The score is never automatically persisted into `Result`, sweep checkpoints,
or directional artifacts. Its return type explicitly reports the aperture
normalization and the distinction between weighted scoring histories and the
physical electron energy ledger.

### Explicit reduced-observation persistence

```python
from pyrite.photon_observation import load_photon_continuum

observation.save("photon-aperture.npz")
tally = load_photon_continuum("photon-aperture.npz")
print(tally.total_per_primary_eV)  # photons / electron / eV, aperture-integrated
```

The provisional `pyrite.photon-continuum-tally.v1` artifact stores bin edges,
total/direct/redistributed spectra, detected and out-of-band counts, primary
and proposal populations, charged-feedback selection, both identities and
the complete observation provenance. Arrays retain their floating-point dtype
and values. JSON metadata and NumPy arrays require no pickle. The loader checks
the content checksum, observation identity, bin/category/count consistency and
aperture normalization. Original data pins are retained without consulting the
current table installation. The checksum detects accidental corruption; it is
not an authenticity signature.

Saving uses the exact supplied filename and atomically replaces it only after
validation and successful writing; the parent directory must already exist.
Loading returns `PhotonContinuumTally`, a reduced observation without source
rows, histories or `Case`. It performs no transport or table fetch. This is an
experimental analysis artifact, not a production checkpoint, source replay or
transport restart. The production Result/checkpoint/replay work below remains
open.

## Remaining steps before exposing `Physics.photon_transport_model`

1. Extend the bounded CPU secondary-feedback rederivation to generalized
   charged target/filter re-entry, and support shower-dependent source
   correlations where needed. Physical primary characteristic/annihilation
   source adapters now have a separate
   [Scene physical-event boundary](photon-secondary-feedback.md#scene-physical-event-source-boundary);
   the Scene continuum call remains a weighted BremsLib source.
2. Add a separate physical-aperture/pixel observable to the public result and
   checkpoint schema; never overload the directional `Result.brem` array.
3. Make source versus detector identity and data-pin provenance part of the
   complete public model descriptors, CLI profiles and replay contracts.
4. Pin independently derived sampler/normalization checks, a **mode-off**
   bitwise regression, and full-track comparisons against PENELOPE/Geant4
   before declaring a production photon transport mode validated.
