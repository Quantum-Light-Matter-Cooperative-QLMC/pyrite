# Slice A — field disposition

This decision is against the detector-scorer dependency at `6b14200`. Its
canonical detector already owns acceptance, photon-energy bins, and read-time
response; its canonical target already owns geometry. “Identity” below means
the logical dataset identity. Direct `Numerics` fields do not enter it;
`Numerics.convergence` does.

## Legacy `Sweep`

| Legacy field | Canonical owner | Identity | Disposition |
| --- | --- | --- | --- |
| `material` | `Scene.target.material` | Yes | Remove the mirror; derive it from the target. |
| `beam` | `Scene.beam` | Yes | Keep the existing frozen `BeamSpec` unchanged. Beam energy becomes an ordinary dotted-path axis. |
| `target` | `Scene.target` | Yes | Keep `Slab` / `Stack` as the physical geometry owner. New scene objects contain scalar geometry; legacy sequences are converted to axes. |
| `detector` | `Scene.detector` | Acceptance/bins yes; response no | Acceptance and bins determine intrinsic arrays. The response remains read-time/rescorable and is excluded, matching `case_content_key`. |
| `thickness_ang` | `Scene.target.thickness_ang` or `Scene.target.layers[0].thickness_ang` | Yes | D7 construction shim only; a legacy sequence becomes a sweep axis. |
| `tilt_deg` | `Scene.target.tilt_deg` | Yes | D7 shim; a legacy sequence becomes a sweep axis. |
| `tilt_azim_deg` | `Scene.target.tilt_azim_deg` | Yes | D7 shim; a legacy sequence becomes a sweep axis. |
| `groove_spacing_ang` | `Scene.target.entrance_face.spacing_ang` | Yes | D7 shim; physical entrance-face state. |
| `crystal_width_mm`, `crystal_height_mm` | `Scene.target.footprint` | Yes | D7 shims; physical target bounds. Legacy sequences become axes over footprint fields. |
| `substrate`, `substrate_thickness_ang`, `stack` | `Scene.target.layers` | Yes | Existing deprecated flat sugar remains a D7 ingress path to `Stack`; no parallel state. |
| `allow_normal_incidence` | `Scene.target.allow_normal_incidence` | Yes | Physical/model-domain target choice. |
| `mosaic` | `Scene.target.mosaic` | Yes | Already canonical target state. |
| `mosaic_fwhm_deg` | `Scene.target.mosaic_fwhm_deg` | Yes | Move the catalog-value override beside `mosaic`; it changes the modeled crystal. |
| `theta_obs_deg` | `Scene.detector.observation_angle_deg` | Yes | D7 shim. Detector-scorer proves the nested detector is canonical. |
| `dtheta_obs_deg` | `Scene.detector.polar_acceptance_deg` | Yes | D7 shim. No independent reconciliation survives after the window. |
| `domega_sr` | `Scene.detector.solid_angle_sr` | No for intrinsic content; yes in dataset provenance | D7 shim. It scales a result but does not change stored intrinsic arrays, so content reuse remains valid. |
| `E_grid_line` | `Scene.detector.energy_bins.line` | Yes | D7/config ingress shim; it fixes stored line-array coordinates. |
| `E_grid_line_by_energy` | `Scene.detector.energy_bins.line_by_energy` | Yes | D7/config ingress shim; it fixes stored line-array coordinates. |
| `E_grid_brem` | `Scene.detector.energy_bins.brem` | Yes | D7/config ingress shim; it fixes stored continuum coordinates. |
| `e_grid_eV` | none | No | Dead as object state. Current source only accepts it in `material_sweep` as a legacy alias for `EnergyBins.line`, and identity synthesizes `None` solely to preserve v1 hashes. Retain a warning alias through D7, then delete. |
| `beam_uvw` | `Scene.target` film orientation | Yes | Physical crystal orientation, not a cost knob. Add an explicit target orientation field rather than leave a loose scene field. |
| `n_electrons`, `n_electrons_brem` | `Numerics` | No | Scalar sampling budgets. Legacy one-element grids collapse to scalars; multi-value legacy grids are represented only by the D7 conversion path needed to reproduce expanded cases, not by new `ScalarOrSeq` fields. |
| `spec_chunk`, `brem_chunk` | `Numerics` | No | Execution batching only; current content-key denylist already excludes them. |
| `n_families`, `max_reflections` | `Numerics.convergence` | Yes | Reflection-family/truncation convergence controls; current identity already resolves both. |
| `mosaic_nodes`, `mosaic_route` | `Numerics.convergence` | Yes | Quadrature resolution/algorithm controls. Analytic-vs-MC disagreement is evidence for a physics issue, never a refactor adjustment. |

`Sweep` itself becomes only `base: Scene` plus ordered named axes. Paths are
validated at construction, not expansion: delaying validation would make typos
dependent on when a run is started and would permit an invalid public object.
Case labels use one mechanical `path=value` suffix in axis insertion order.

## Legacy `Settings`

| Legacy field | Canonical owner | Identity | Disposition |
| --- | --- | --- | --- |
| `beam_current_na` | `Analysis` | No | Compatibility/reporting scale for old checkpoints. Resolved beam charge/rate remain on `Scene.beam`. |
| `apply_detector_qe` | `Analysis` compatibility lens | No | D7 spelling translated to the existing read-time `LegacyEDS` response. It never changes intrinsic arrays. |
| `convolve_with_det` | `Analysis` | No | Presentation-time convolution toggle/override. |
| `brem_source` | `Scene` | Yes | Chooses the physical background source (`mc`, `external`, `none`), so it changes the result rather than its presentation. |
| `n_electrons`, `n_electrons_brem` | `Numerics` | No | Sampling budgets; removes loose `build_cases` parameters. |
| `emission` | `Scene` | Yes | Physical emission model; current identity already hashes non-default values. |
| `xray_dispersion` | `Scene` | Yes | Physical photon-dispersion model; current identity already hashes non-default values. |
| `coherent_emission` | none | Derived | D7 read-only compatibility property derived from `Scene.emission`. |

`Analysis` reaches plots through the seam that already exists: result-table and
plot helpers receive `settings` explicitly today. Those parameters change type
to `Analysis`; a deprecated `Settings` adapter supplies the same properties for
one D7 window. No global analysis state and no plotting dependency enters the
simulation core.

## Evidence and resolved questions

- `Sweep._normalize_detector` currently proves that the three flat detector
  inputs are aliases onto `Detector`; no second owner exists.
- `campaign.config.material_sweep` is the only source owner that still consumes
  `e_grid_eV`, immediately translating it to `EnergyBins.line`.
  `_identity_v1` emits `e_grid_eV = None` only for historical hash shape. There
  is no live `Sweep.e_grid_eV` field or runtime reader.
- `results.store.detected_background`, `_detected_background_wide`, and
  `results.tables` already accept a settings-like value explicitly. That is the
  migration seam for `Analysis`.
- `_CONTENT_KEY_DENYLIST` documents that solid angle, current metadata,
  analytic mosaic broadening, and chunk sizes do not alter intrinsic stored
  arrays. The disposition preserves that reuse boundary while recording the
  full resolved scene/numerics in provenance.
- Catalog profiles currently use one-element electron-count grids. The D7
  converter must still handle multi-value grids because profile editing and
  regression fixtures permit them; the canonical object model does not add a
  sweep-typed count field.

