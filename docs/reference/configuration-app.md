# Configuration app: controls and persistence

This page documents the configuration app's form controls, staged edits, saved
catalog behavior and export semantics. For a shorter task recipe, see
[Edit a campaign profile interactively](../guides/configuration-notebook.md).

Install the optional apps extra, then launch:

```bash
pyrite app configuration launch
```

The notebook edits the catalog selected by `--catalog`, `PYRITE_CATALOG`, or
your configured catalog path, with the normal bundled/user-layer behavior.
For an independent complete catalog:

```bash
pyrite --catalog /path/to/catalog app configuration launch
```

`--edit`, `--watch`, `--tunnel` (port 2726), and `--smoke` follow the other
application launchers. File paths refer to the notebook host. No simulation runs
when controls change. Use **Open catalog** to load another catalog and profile;
explicitly allow discarding any unsaved draft first.

## Create and edit a profile

In **Overview**, select an existing profile to inspect it, or enter a new name
and choose a duplication source. Bundled objects are read-only; duplicate them
before editing. A duplicate follows `pyrite profile create --from`: it copies
profile settings but excludes material overrides. Creating without a source
uses the packaged standard sweep ranges and applicable overrides. Attach an
explicit detector before saving if the new profile has none.

All changes are staged in one in-memory catalog draft. Switching profiles keeps
staged changes. Closing the notebook loses changes that were not saved.

**Override previous** means that a field is written into the configuration.
Unchecking it removes the field and restores existing preset/default resolution.
Unchecked suggested values do not change the simulation. Apply each form to
stage its values; validation feedback appears above the sections. Correct any
errors before saving.

**Scene and Materials** selects catalog targets and edits kinetic energy in
keV, thickness in Å or supported layer counts, incidence and azimuth in degrees.
Choose one thickness definition: angstroms or crystal layers. Applying the form
keeps only the selected thickness representation. Comma-separated values create
the existing profile sweep grids. A
material-specific override form edits or removes ranges that supersede the
profile's common ranges. The resolved review shows actual material geometry,
cuts and stack definitions. Profiles represent campaigns, rather than arbitrary
API `Scene` assemblies: creating new body shapes or crystal cuts is outside this
editor. Existing ordered filters remain intact.

**Beam** loads an existing named beam or an inline beam. Attach it unchanged, or
create a private named copy with edits. The copy never modifies other profiles.
Templates support Gaussian spots, Twiss phase space and GDF sources, with point,
duration or supported longitudinal policies. Units are displayed in field names:
spot FWHM mm, bunch duration fs, emittance mm·mrad, Twiss beta m, repetition Hz,
charge pC and fractional RMS energy spread. Energy remains a profile sweep field.
For GDF, specify a host path and lab origin m; time s and screen-position m
selection are mutually exclusive. Normal source validation remains authoritative.

**Detectors** adds named scalar or pixel detectors, duplicates/removes IDs, and
edits profile-local copies. To create a named detector, enter a new name before
applying its geometry. To add more IDs to a legacy profile, choose the explicit
`migrate` action first. This preserves geometry but can change observation
identity. Keep at least one explicit detector. Inline pixel edits require an ID
collection or the legacy physical-detector slot. Scalar angular acceptance and
pixel dimensions cannot be combined. Optional scorer, response and acquisition
sections require their normal companion fields; the catalog validator rejects
incomplete combinations instead of substituting another detector.

**Physics** and **Numerics** attach named presets and edit explicit overrides.
Advanced controls expose supported model/cutoff, reflection, sampling, precision
and photon-grid settings. Fixed count grids and adaptive precision cannot mix;
coherent emission requires compatible fixed counts. A PXR/CBS line scorer is a
joint mechanism, so there is no independent PXR-versus-CBS switch. This editor
does not change physical models or add profile keys for API-only settings.

## Outputs, save and run

The output operations remain distinct:

| Operation | Control and effect |
|---|---|
| Computation | Physics/emission and numerical profile fields select existing calculation behavior. |
| Scoring | Detector geometry, scorer, response and acquisition configure supported observations. |
| Persistence | Normal runs retain component checkpoints and configured detector observations. Trajectory recording adds `--trajectories=DIR` and HDF5 storage flags to the generated run command. |
| Export | Load an existing checkpoint by its exact saved stem, select one record and stored source components, then download CSV or NPZ. This filters exported arrays without changing computation or checkpoint retention. |

Checkpoint/capture destinations and backend choice are execution-only controls.
They generate an existing CLI command and can be downloaded as a shell command;
they are not saved as profile fields. Save the profile before executing that
command. Histories use compact or throughput HDF5 storage. Cached cases are not
transported again: use an explicit `--recompute` run when captures are needed
for those cases. Existing `pyrite trajectories export` produces VTK afterward.

Source-array export lists only components actually stored, such as `spec`
(incoherent PXR/CBS), `spec_coherent`, `spec_characteristic`, `spec_transition`,
`spec_channeling`, `brem` and `brem_wide`. Separate energy grids are retained.
Downloads contain original stored values; no response, normalization or bin
conversion is applied. Optional metadata includes the originating case and
scale. Detector-specific pixel measurements remain available in the Pixels app.
Independent checkpoint-retention subsets, arbitrary observable switches and
independent PXR/CBS exports are not supported by the profile pipeline.

In **Review and Save**, inspect validation, resolved settings, TOML and diff,
then choose **Save catalog changes**. Saving validates the selected and changed
profiles, protects bundled objects and rejects stale edits. It writes all staged
objects through existing catalog persistence. If another process changed the
catalog, reopen it and reapply your edits.

The TOML download is a complete catalog snapshot. Relative CIF/GDF paths and
immutable energy-grid references still need their assets; transfer these as
well when moving catalogs. The resolved JSON download is for inspection, not a
new executable configuration format.

## Related contracts

See [profile settings](../repo-design/profile-settings.md),
[campaign configuration](campaign-configuration.md), and
[catalogs and workspaces](catalogs-and-workspaces.md) for the exact on-disk
and resolution behavior. For a runnable Python scene rather than a campaign
profile, use [Python scene recipes](python-scene-recipes.md).
