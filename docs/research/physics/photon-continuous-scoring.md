# Weighted continuum to ordinary-hit scoring (issue #341)

Status: **CPU reference**, not a released photon-transport model. This slice
follows [weighted BremsLib sources](photon-continuous-sources.md) into the
existing host photon queue and its ordinary absorbing detector aperture.
The opt-in reference entry point is
`pyrite.montecarlo.runner.photon_emission.score_bremslib_continuum`.
It has no call site in `run_case`, `simulate`, the CLI, or the checkpoint replay
path. The existing Beer--Lambert attenuated spectra and their identities are
unchanged. Do **not** mistake this entry point for a validated production mode.

## Contract and physical measure

1. Simulate electron segments separately with an **appropriate radiative-loss
   and recoil treatment**. Their `r_mid`, `v_hat`, `L_ang` and `E_repr_keV`
   (or `E_keV`) are the spatial track-length emission quadrature. The segment
   array has one original primary owner `elec_id` and, for multilayers, a
   material index `layer`. Radiation from tracks below the source's energy
   cutoff is excluded by the proposal support.
2. Supply the same material layers for BremsLib emission and photon navigation,
   and material-matched BremsLib tables. The reference source samples an
   **unattenuated** joint energy/solid-angle distribution on every segment;
   even photons initially traveling away from the detector can scatter in.
3. Track every sampled source history through the EPDL channel/atom competition
   and associated interaction samplers. Geometry includes source layers,
   disjoint oriented filter boxes, a finite sample footprint, and one physical
   aperture. The tracker owns all subsequent attenuation. **Do not multiply
   the result by a Beer--Lambert transmission factor or separately score the
   same hard-bremsstrahlung analog events.**
4. Normalize the aperture tally by the *original electron count* `n_primaries`
   and by the energy-bin width, **once**. The resulting unit is
   `photons / (primary electron * eV)` integrated across the aperture, *not*
   `photons / (primary electron * eV * sr)`. Comparison to PyRITE's directional
   spectrum requires an independently derived solid-angle estimator.

The weighted source integral for an electron segment $i$ is

$$
Y = \sum_i L_i \int_{k_{\rm min}}^{k_{\rm max}(i)} dk
    \int_{4\pi} d\Omega\,j_i(k,\Omega),
$$

where $j_i$ has units photons/Å/eV/sr and is evaluated jointly in photon
energy and direction. For $M$ proposals with joint density $q(i,k,\Omega)$,
the photon weight is $w=L_i j_i/(M q)$; all charged and photon descendants
inherit the same statistical weight. The detection score is

$$
\hat{S}_b = \frac{1}{N_e\,\Delta k_b}
  \sum_{h\in\mathrm{hits}} w_h\,\mathbf{1}(k_h\in b).
$$

These histories estimate a **mean response** and must not debit the primary
electron energy account a second time. Sampling $M$ photons on a fixed set of
electron tracks does not create $M$ independent primary histories; photon-only
statistical uncertainty cannot stand in for electron transport uncertainty.

## Direct and redistributed responses

`PhotonContinuumScore.total_per_primary_eV` is partitioned into
`direct_per_primary_eV` and `redistributed_per_primary_eV` without double
counting. Direct hits have never collided (`generation == 0`) and belong to
photon queue round 0. All other hits are redistributed, including Rayleigh,
Compton, fluorescence and photons emitted by returned charged descendants.
For coupled photon/charged histories, a newly emitted bremsstrahlung photon
has photon `generation == 0` but `round > 0`: it is **not** a direct source hit.
`unbinned_detected_per_primary` exposes photons that hit the aperture but are
outside the chosen reporting energy bins.

### Programmatic diagnostic

The caller must construct the model's finite aperture and any oriented filter
boxes in the sample frame. For example (the table mapping and source segments
must come from a correctly configured electron-transport calculation):

```python
from pyrite.montecarlo.runner.photon_emission import score_bremslib_continuum
from pyrite.montecarlo.transport.photon_histories import PhotonAperture

aperture = PhotonAperture(
    (0, 0, 1e9),
    (1, 0, 0),
    (0, 1, 0),
    1e8,
    1e8,
)
score = score_bremslib_continuum(
    segments=segments,
    layers=layers,
    tables=bremslib_tables,
    aperture=aperture,
    edges_eV=energy_bin_edges,
    n_primaries=number_of_primary_electrons,
    n_samples=50_000,
    seed=341,
    photon_cutoff_eV=1_000.0,
    source_ceiling_eV=300_000.0,
    electron_cutoff_eV=1_000.0,  # explicit photon-only diagnostic
)
print(score.total_per_primary_eV)
```

To follow photon-born electron feedback instead, pass `charged_stage=` and
omit `electron_cutoff_eV`. Construct the stage with the matching material stack,
cutoffs and electron transport physics. The electron-only stage deposits
filter-born electrons locally and keeps positrons pending. The newer
`cascade_feedback_stage` follows in-stack positrons, annihilation and
charged-impact relaxation, and refuses filter boxes. See
[CPU secondary feedback](photon-secondary-feedback.md). Neither reference is
a fully coupled validated MeV cascade.

## Validation and outstanding production work

`tests/montecarlo/transport/test_photon_continuous_scoring.py` checks energy-bin/primary
normalization, disjoint detection categories, coupled-round reclassification,
source-to-tracker parameter/identity handoff, and fail-closed options without
requiring released atomic archives. Fresh-context verification independently
derived the weighted tally, category partition and edge convention;
`photon-continuous-source-scoring` is `rederived`. See the
[independent derivation](../../validation/radiation-physics/photon-continuous-source-scoring.md).
A vacuum finite-square regression checks the absolute source-to-aperture scale
against the analytic solid angle. Neither establishes full-track physics;
`#341` still requires independent PENELOPE/Geant4 comparisons.

Remaining before a `Physics` selector can be offered:

- Add immutable model descriptor and case identity/provenance only for enabled
  photon transport. Avoid introducing new default keys and keep old spectrum
  goldens identical when the mode is absent.
- Decide mutually exclusive scoring policies for continuous weighted sources
  and physical hard-bremsstrahlung events; do not sum both for one continuum.
- Add primary characteristic and positron-annihilation unattenuated source
  handoffs, physical detector configuration and replay/checkpoint data support.
- Complete charged-descendant/positron handoffs, then compare full tracks and
  detector counts with independent references. Support CPU first; reject GPU,
  adaptive, groove and unsupported replay boundaries rather than downgrade.
- Validate detector solid-angle normalization separately before comparing to
  directional CXR/Bremsstrahlung spectra, and derive conditional scoring before
  enabling it as an alternative to ordinary hits.
