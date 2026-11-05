# 0022 — Named target and downstream filter presets (first implementation slice)

- **Status:** Proposed
- **Date:** 2026-10-31
- **Context:** [#665](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/665), [#401](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/401), ADR-0008, ADR-0016, ADR-0021

## Decision and scope

Reuse the *catalog name -> typed value -> scene assignment* pattern of ADR-0021;
do not conflate a material, a body, and its role in a scene. Share bounded
rectangular-body descriptions only through the existing `SubstanceRef` and
`RectangularBody` adapters. Keep `Slab`/`Stack` and `FilterPlate` as distinct
supported interaction roles; a downstream plate does **not** receive electron
transport or secondary-generation physics because it is catalogued.

The first implemented slice supports named **single downstream plates** under
`[filters.NAME]` or `filters/NAME.toml`, with existing units and validators.
Definitions include material, thickness, size, and a default pose; `description`
is display metadata. There is no definition `name`: names belong to instances.
The ordered `[[profiles.PROFILE.filters]]` entry may contain `ref = "NAME"`
and any optional plate-field override, including pose or a scene-local display
name. References are resolved before the existing profile filter parser and
removed from the physical value that enters observation identity. `ref` does
not change transport physics, checkpoint identity, or observation hashing;
changing effective plate values or order retains existing invalidation rules.

Definitions can be loaded as immutable existing `FilterPlate` objects through
`pyrite.geometry.load_filter(NAME)` and listed/created/cloned/edited/renamed/
deleted using `pyrite filter`. Renames update all placements, while deletion
of a referenced preset fails. Bundled definitions are read-only. The CLI
`pyrite filter attach PROFILE NAME` appends a reference without rearranging
existing filter positions. Existing inline plates are unchanged.

**Deferred to follow-up slices:** named slab/stack targets, full filter *stacks*
as a unit, optional standalone placement definitions, and potential typed
scorer/response presets. Targets require care: the runnable `MaterialSpec`
recipe is not a homogeneous `SubstanceRef`, and target placement cannot turn
filter plates into electron navigation geometry. The accepted #401 review and
ADR-0016 favor specialized role wrappers over a general geometry class.

## Proposed extension: target catalogs

The remainder of this ADR is a **design proposal**, not implemented schema or
an accepted amendment to ADR-0016/ADR-0021. Human review must settle the
reference shape and geometry precedence before dependent implementation.
The existing single-plate implementation above remains the first slice.

### Evidence and alternatives

`campaign.geometry.Slab` and `Stack` already own the closed geometry set and
constructor validation. `Layer` preserves beam-entrance ordering; only the
first layer may sweep thickness. `Stack` requires at least two layers and
cannot acquire a grooved entrance face. `Scene` scalarizes geometry, whereas
`Sweep` expands its supported sequences. `campaign.flat_geometry` owns the
legacy flat-field adapter; `campaign.profiles` projects targets back to the
historical identity fields. These are the lowering boundaries to reuse.

`materials._parse._parse_materials` currently combines a material recipe
(crystal plus optional substrate/stack) with selected-profile scan defaults
and per-material overrides. Profile membership can contain many materials;
omitting membership means all catalog materials. Thus a material recipe is
neither a homogeneous substance nor an independently positioned target.

| Alternative | Benefit | Cost / recommendation |
| --- | --- | --- |
| Keep geometry only in profiles/material recipes | No new resolution path | Leaves the requested geometry reuse dependent on copying; reject as the long-term answer |
| Single `profile.target = NAME` | Simple one-target scene vocabulary | Would redefine multi-material campaign membership; defer rather than silently narrow it |
| Per-material target references | Preserves existing campaign membership and attaches a complete supported target | Requires explicit conflict rules; recommended bounded extension |
| Universal body registry | Could name all scene bodies uniformly | Duplicates role-specific contracts and exceeds ADR-0008; reject |

### Proposed schema and API

Use `[targets.NAME]` / `targets/NAME.toml`, containing a discriminated
`kind = "slab" | "stack"` and only fields supported by existing constructors.
Do not add subclasses, arbitrary imports, recursive target references, or
free-form positioning/navigation. A slab's `material` is an existing runnable
material key; stack layers retain the existing entrance-material versus
underlying material/medium restrictions. Do not reinterpret either namespace
as a raw crystal key. Reject a preset whose explicit layers contradict an
entrance recipe's substrate/stack; never concatenate the two implicitly.

Proposed example (not currently loadable):

```toml
[targets.thin_film]
kind = "slab"
material = "silicon"
thickness_ang = 20000.0
tilt_deg = 30.0
tilt_azim_deg = 0.0

[targets.film_on_support]
kind = "stack"
tilt_deg = 30.0
tilt_azim_deg = 0.0
layers = [
  { material = "silicon", thickness_ang = 20000.0 },
  { material = "silicon", thickness_ang = 100000.0 },
]

[profiles.example.targets]
silicon = "thin_film"
```

The example's material keys must exist in the selected catalog. A profile mapping key must
match the referenced target's entrance material and its explicit membership,
when membership is restricted; the mapping never adds members. Omitted
membership retains its existing all-material meaning. Other materials keep
legacy geometry resolution. A future single-scene convenience API can attach
one loaded target directly without changing campaign membership.

Require finite positive thicknesses, finite supported angles, paired finite
transverse dimensions when present, and explicit supported variant fields.
Store lengths in Å and angles in degrees for compatibility. Optional footprint,
crystal cut, mosaic and slab-only groove fields must map to the corresponding
existing typed values and preserve their restrictions. Crystal cut/orientation
is distinct from downstream lab-frame pose. Initial definitions are scalar;
existing profile sweeps remain supported without references. Adding sweep
arrays to definitions needs a separate choice about campaign-template reuse.

A reference owns **all geometry** for that material: inherited profile geometry
is not applied on top. Inspection must mark those profile defaults as unused
for that material. Reject per-material geometry overrides alongside a target
reference, rather than silently ignoring them. Beam energy, Physics, Numerics,
energy grids and observation configuration remain profile-owned. Attaching or
detaching a reference must preview the effective geometry change; detaching
restores legacy profile/material defaults. Reference-free profiles are unchanged.

Propose `pyrite.geometry.load_target(NAME, *, catalog=None) -> Slab | Stack`,
returning existing immutable types with the same validation used by profile
resolution. Keep `Target`'s homogeneous rectangular-body adapter available;
do not force stacks into it. Scalar presets work in both `Scene` and `Sweep`.
No separate target-result class or duplicate material registry is needed.

### Lifecycle, identity and compatibility gates

Follow ADR-0021 object storage: reserve names case-insensitively across layers,
keep bundled objects read-only, clone into the user layer, and reject unknown
keys/references before writes. Validate unused definitions too. Put lifecycle
and resolution services below CLI; CLI commands only adapt arguments/results.
Rename must atomically update profile target mappings; deletion must refuse
while referenced. Preserve material/substance references through existing
rename/delete services as well, including references inside target layers.

Freeze resolved typed values before execution. Keep preset name, description,
source layer and definition digest in inspection/provenance only, outside
scientific identity. Lower through the existing target adapters and historical
identity projection. Equal resolved material, layer order, dimensions,
orientation and supported features must yield identical cases, checkpoints
and observations, whether inline or named. A name edit cannot invalidate
transport; a physical target edit can. Do not change model markers or stored
schemas merely to record a reference. Remote catalog transfer must include
new object files and resolve to the same submitter/host identity.

## Proposed extension: reusable filter stacks

A repeated ordered set of plates across profiles is a concrete reuse need in
#665. Recommend supporting it after target/reference review, using a **flat
ordered sequence of existing `FilterPlate` values**, not a layered homogeneous
medium, target `Stack`, or new solver. Two separately posed plates must retain
their gap and poses; no averaging compositions or thicknesses is valid here.

Prefer an explicit `kind = "sequence"` variant in the existing `filters` table
instead of a second filter registry. Keep existing untagged single plates
compatible. A proposed definition and placement are:

```toml
[filters.window_pair]
kind = "sequence"
plates = [
  { material = "silicon", thickness_mm = 0.1, size_mm = [20.0, 10.0], distance_mm = 50.0 },
  { material = "silicon", thickness_mm = 0.2, size_mm = [20.0, 10.0], distance_mm = 60.0 },
]

[[profiles.example.filters]]
ref = "window_pair"
```

Initially require complete inline plate members and reject nested references,
empty sequences, and sequence-placement field overrides. This avoids recursive
resolution and an ambiguous choice between absolute and relative group pose.
Single-plate placement overrides keep their existing meaning. Expand sequences
in place, preserving member order and order relative to surrounding inline or
referenced plates. Inspection must show both the group reference and expanded
indices; editing one expanded member requires an explicit detach-to-inline
operation. Reference-safe rename/delete operates on the group reference.

Keep `load_filter` a single-plate loader; reject a sequence with a diagnostic
pointing to proposed `load_filters(NAME) -> tuple[FilterPlate, ...]`. The plural
loader accepts either variant and returns a one-element tuple for a plate.
This prevents changing the existing return type. Expanded plate values alone
enter observation identity, exactly like equivalent inline plates. Original
per-instance display-name semantics remain unchanged; catalog group names
are not copied into members. Sequence labels belong only to provenance.

## Assessment: scorer, response and acquisition presets

Recommend **no independent catalogs in this issue**. Named detectors already
own optional scorer, response and acquisition sections; the existing detector
parser validates those sections and profiles reuse detector references. This
provides concrete reuse of a whole instrument configuration today. A separate
registry would require precedence and compatibility rules for detector geometry,
response sensor parameters, pixel scorer shape and acquisition edges without a
specified case needing independent reuse. Distinct concerns in ADR-0016 do not
by themselves justify distinct catalogs.

The motivating case for reconsideration would be one measured response or
acquisition policy deliberately shared across several independently positioned
detectors, with edits propagated while geometry stays independent. Before a
future proposal, supply that use case, identify which typed object owns each
field, and compare cloning existing named detectors against a new reference.
Do not conflate source scoring or numerical response-matrix settings with the
physical detector response. This assessment defers catalogs, not the existing
typed scorer/response/acquisition APIs.

## Review and implementation gates

Human review is requested on the proposed per-material target mapping,
complete-geometry ownership, initial scalar-only definitions, and flat filter
sequence variant. These are recommendations, not runtime contracts or an
assertion that human acceptance has occurred. Review may choose a smaller
slice without changing accepted ADR-0016 boundaries.

After review, implementation should cover these compatibility gates in owning
source and tests (the issue remains the progress record):

- Add target codecs/resolution adjacent to catalog parsing, lower through
  `campaign.geometry` / `flat_geometry`, and expose the existing types through
  `geometry`. Check scalar validation, slab/stack restrictions, material
  namespaces and profile membership/conflicts before adding CLI wiring.
- Extend existing catalog layout/layer services and material reference lifecycle;
  verify directory and single-file catalogs, read-only bundled objects, cloning,
  atomic rename/delete, failed-write rollback and remote catalog transfer.
- Exercise `tests/scan/test_target_geometry.py`, `test_scene_model.py`,
  affected profile/catalog tests and external-catalog identity tests. Add
  inline-versus-named case/checkpoint identity comparisons and name-only rename
  comparisons. Preserve existing observation identity coverage.
- Add sequence expansion beside the current named-filter resolver, preserve
  legacy plate parsing and `load_filter`, then test ordering, mixed placements,
  plural loader behavior, group inspection and reference lifecycle in
  `tests/materials/test_named_filters.py` and `tests/cli/test_named_filters.py`.
- Update API ownership/inventory and canonical schema/how-to docs only when
  implemented; generate CLI references from actual commands. Run focused tests,
  API checks/imports, complexity and the documentation build. No physics kernel,
  default numerical output or general geometry/navigation change is implied.
