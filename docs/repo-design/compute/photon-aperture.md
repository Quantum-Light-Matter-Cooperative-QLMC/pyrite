# Opt-in photon aperture observations

`Physics(photon_transport=PhotonTransport(...))` connects the bounded CPU
ordinary-hit reference to `simulate`, scalar Scene lowering, the case runner,
checkpoint components and full trajectory replay. This is experimental model
selection, not full-track validation. The interaction, weighted-source and
bounded charged-feedback claims retain their existing validation states; no
new collision equation or default numerical model is introduced here.

## Selection and units

```python
from pyrite import Physics, PhotonTransport, simulate

physics = Physics(
    bremsstrahlung_model="bremslib",
    radiative_model="bremslib-soft-hard",
    photon_transport=PhotonTransport(
        edges_eV=(1000, 3000, 10_000, 100_000),
        source="brem",
        n_samples=1000,
        seed=341,
    ),
)
result = simulate(beam, target, ideal_planar_detector, physics=physics, numerics=numerics)
counts = result.aperture_dataframe()
flux = result.aperture_dataframe(current_nA=1.0)
```

The detector must be physical, ideal and unpixelated, without a scorer or
acquisition. The selected source is either weighted continuous `brem`, physical
`characteristic`, or physical `annihilation`. A continuous source requires
positive `n_samples`; event sources require `n_samples=None`. Event sources
require equal positive line/continuum primary populations, shell-soft-hard
transport and a secondary threshold equal to `electron_cutoff_eV`. Annihilation
also requires the existing primary pair/positron selectors; the other two
sources refuse them. Characteristic sources always prepare Auger transport.
`charged_feedback=True` enables photon-born charged descendants for any source.
The existing [source and feedback contracts](../../research/physics/photon-secondary-feedback.md)
still apply, including contiguous charged target stacks and no charged filter,
gap or re-entry transport. Photon-only continuum and annihilation support
photon filter boxes.

`Result.aperture` is a `PhotonContinuumTally` or `PhotonEventTally`. Its densities
are photons/(original incident electron eV), integrated over the physical
aperture. `aperture_dataframe` retains exact lower/upper histogram edges and
direct, redistributed and total densities and bin counts. Supplying mean
current in nA multiplies by electrons/s, giving photons/(s eV) and photons/s.
No solid angle, detector response, second filter attenuation, or proposal
normalization is applied. Out-of-bin detections remain separately available
on the tally. Event tallies retain sub-cutoff source energy and characteristic
relaxation budgets.

The ordinary directional spectra, their per-sr units, and `spectral_flux()`
remain separate. Their characteristic/continuum families are alternative
views of contributions also represented by an aperture observation. No method
sums those views. Primary transport runs once; both scoring routes reuse its
rows. The aperture sampler and feedback use their own seeded streams without
changing the primary loss ledger.

## Identity and execution

Mode off adds no Case key, output key, checkpoint component or identity marker.
Mode on appends a resolved `Case.photon_transport` record. It pins the original
source content key and exact Case snapshot, source table manifests, histogram,
source selection, seed, cutoffs, physical aperture/filter geometry, photon data
and enabled feedback tables. The complete enabled record enters the existing
case-content key; the underlying source and aperture observation digests stay
separate. The descriptor `observation.photon/cpu-reference` documents CPU,
fixed-population and whole-segment requirements without certifying a validated
physical domain.

Preparing a selected case checks the source snapshot and installed pins before
primary transport or output capture. Changing the resolved Case after lowering
requires lowering the Scene again. Unsupported backends, CUDA, grooves,
channeling, TR, adaptive/block transport, sweeps and pixel/acquisition requests
are refused. The scalar Python API is the selection boundary in this slice;
catalog/CLI configuration and remote scheduling are not added.

## Checkpoints and replay

Raw case outputs and `store_result` carry a separate `photon_aperture` payload
with original reduced-artifact schema, array bytes, checksum and provenance.
The content-addressable cache preserves it. Component checkpoints write
optional `photon.h5` before the authoritative line index, alongside the existing
directional components. Mode-off stores keep their historical layout. Saves
validate the observation against its Case before replacing any component;
loads refuse missing, corrupt or mismatched selected companions. No current
photon tables are needed to read a finished observation.

```python
from pyrite.checkpoints.persistence import load_checkpoint
from pyrite.results.aperture import aperture_from_record, aperture_dataframe

records = load_checkpoint("silicon", checkpoint_dir=checkpoint_root)
tally = aperture_from_record(records[case_name][beam_energy_keV])
counts = aperture_dataframe(tally)
```

Full `spectrum_from_artifact` replay regenerates the selected aperture from
retained primary rows without rerunning primary transport, checking the
original Case/data pins. Bounded replay refuses photon mode because weighted
source selection requires the full segment population. Partial field/dataset
projections must retain `photon_aperture`; directional-only partial transfers
are refused. Ordinary full slim/archive paths preserve it, and directional
float32 slimming leaves the checked aperture payload at its original dtype.

Independent thick-target W/Si, alloy fluorescence and 511 keV full-track
comparisons, pixel observables and generalized charged navigation remain open.
