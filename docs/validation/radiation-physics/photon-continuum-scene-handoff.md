# Scene handoff for the photon continuum diagnostic

Validation: `photon-continuum-scene-handoff`.

## Independent boundary derivation

Fresh-context verification on 2026-10-22. This section was written from the
ledger row, signatures/docstrings and
[ADR-0020](../../adr/0020-photon-transport-source-and-scoring.md) before reading
`photon_transport.py` implementation bodies. The claim concerns population,
coordinate, identity and orchestration bookkeeping, conditional on the
underlying source, geometry and transport models.

If primary transport supplies a shared population larger than the continuum
population $N_{e,\mathrm{brem}}$, only rows whose primary identifiers belong
to that continuum population may enter its source measure. Their denominator
must remain $N_{e,\mathrm{brem}}$, including primaries producing no segments or
no hits. Neither line-only primaries nor proposal count may replace it:

$$
S_b=\frac{\sum_{h\in\mathrm{det}}w_h\mathbf{1}(k_h\in b)}
{N_{e,\mathrm{brem}}\Delta k_b}.
$$

For an orthogonal sample-to-laboratory rotation $R$, the inverse rotation is
$R^{\mathsf T}$. Since $1\,\mathrm{mm}=10^7\,\mathrm{Å}$, lab positions and
lengths expressed in mm require

$$
\mathbf r_s=10^7R^{\mathsf T}\mathbf r_{\rm lab},\qquad
\hat{\mathbf n}_s=R^{\mathsf T}\hat{\mathbf n}_{\rm lab}.
$$

Unit tangents/normals receive rotation only. Half widths receive the length
conversion only. Orthogonality preserves lengths, dot products and aperture
area orientation. All target, detector and filter geometry must share this
sample frame; rotating only the detector centre is insufficient.

Primary transport must run once; selected continuum segments must feed one
source/scoring pass with no added physical-event histogram. The observation
is aperture-integrated photons/(primary eV), separate from directional default
spectra. Its identity must include the primary source identity, physical
acceptance/filter geometry, photon-data pins, energy edges, cutoffs, proposal
count and random seed. Identical resolved inputs must give identical identity;
changing any response-defining input must change it. This is an explicit
diagnostic and must not alter default simulation selection on import.

Cheap filters: inverse rotation followed by forward rotation recovers each
point; zero hits gives zero score; increasing a line-only population leaves
the continuum denominator unchanged. No new physics equation beyond the
weighted tally and coordinate conversion is asserted.

## Implementation comparison

Inspection after the derivation found one `_transport_case` call, followed by
`_sample_segments` with `Ne_brem`, then one `score_bremslib_continuum` call
with that same `Ne_brem` denominator. `_sample_segments` retains the original
nonnegative primary IDs below that threshold. It selects all aligned source
geometry, representative energies and layer labels together. Line-only rows
are excluded without renumbering the retained primaries.

`_scene_geometry` uses one `sample_to_lab_R` for the detector and every filter.
`PhotonAperture.from_pose` rotates centres and axes by its transpose; centres
and half sizes receive $10^7$ Å/mm. `photon_box_from_pose` uses the same
centre conversion, row-vector axis multiplication by $R$ (equivalent to
column-vector multiplication by $R^{\mathsf T}$), and converted half extents.
Target layers remain in their existing sample-frame Å coordinates, while the
optional transverse dimensions convert from mm to Å.

The source digest calls the existing `case_content_key` with resolved
`case_table_markers`. The observation payload includes that digest, the
diagnostic model/schema, integrated/differential EPDL and EADL data hashes,
all source proposal parameters, transformed aperture/filter geometry and
composition, and energy-bin edges. Sorted JSON with nonfinite values refused
is hashed with SHA-256. No missing response-defining field was found at this
boundary; correctness of the inherited case/table identity is outside this
claim.

Preflight rejects unsupported detector response/pixelation/acquisition,
non-CPU execution, adaptive populations and non-MC source choices. Resolved
case checks require coupled BremsLib radiative transport and reject grooves,
channeling, transition-radiation histories and positron/event-pair options.
The returned diagnostic states aperture normalization and charged handoff-only
limitations explicitly. This entry point performs no default spectrum write
or physical-event histogram addition.

## Evidence and verdict

The focused scene-handoff tests exercise coordinate rotation and units,
continuum population selection, single transport/scorer calls, repeatable
identity with sensitivity to changed acceptance/proposals, and refusal paths.
These are orchestration anchors, not independent full-track physics evidence
or a bitwise mode-off spectrum regression.

- **Filters:** units pass; limits pass; signs/conventions pass.
- **Re-derivation:** matches within the explicitly bounded population,
  geometry, orchestration and identity scope.
- **Verdict:** `rederived`.
- **Suggested ledger change:** set `photon-continuum-scene-handoff` to
  `rederived`, link this page, retain the recorded full-track/charged-feedback
  and default-spectrum-regression limitations.
- **Human sign-off:** pending.
