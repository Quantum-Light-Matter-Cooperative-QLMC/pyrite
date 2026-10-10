# Target and filter object semantics

**Status:** Proposed design for [#401](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/401).
No API shown as proposed below exists yet. Implementation requires design review.

**Recommendation:** share substance resolution and bounded immutable body
components, while retaining `Slab`/`Stack` and `FilterPlate` as role-specific
interfaces. Material is a body parameter; a campaign recipe is a different
object. Placement constrains a role but does not select its enabled physics.
[Proposed ADR-0016](../../adr/0016-target-filter-body-components.md) records this
choice without superseding [ADR-0008](../../adr/0008-no-arbitrary-target-geometry.md).

## Current boundaries and terminology

This study inspected `dfc8a717`, the issue branch's base on `main`. Source links
identify owners; the statements below describe that revision, rather than
assuming that similar field names have identical meaning.

| Concern | Current owner/type | Meaning and constraints |
| --- | --- | --- |
| Crystalline substance | `materials/_schema.py::CrystalSpec` | CIF-derived lattice, basis, elemental number densities, optical/transport inputs, cut and reflection defaults; contains some historical run defaults too |
| Amorphous substance | `MediumSpec` | Elemental number densities in atoms per cubic Å; no crystalline radiation structure |
| Runnable recipe | `MaterialSpec`, `MaterialCatalog.material`, `_parse.py` | Crystal reference, profile-resolved scan grids, optional substrate/stack, display identity and validation state |
| Target description | `campaign/geometry.py::Slab`, `Stack`, `Layer` | Thickness, target tilt, footprint, crystal orientation and specialized layer/groove lowering; legacy campaign inputs can contain sequences |
| Scalar scene | `campaign/model.py::Scene` | One scalar target; one observation model; ordered filters; rejects implicit multiple target thicknesses/angles |
| Campaign variation | `Sweep`, `campaign/lowering.py` | Explicit path axes expand scenes; legacy adapters retain product ordering, quantized angles, names and seeds |
| Downstream body | `instrument/model.py::FilterPlate` | Homogeneous composition, finite box dimensions, lab-frame centre/axes, optional display name |
| Placement | `_planar_geometry.py::PlanarPose`, target lowering | Filter pose is independent of substance; target has sample-relative specialized orientation, not arbitrary lab translation |
| Interaction capability | Target lowering versus `instrument/attenuation.py` | Source electron transport/radiation versus removal of primary photons; a shared shape supplies neither solver |
| Display role | Target/substrate/filter labels | Useful descriptions; behavior follows the receiving interface, never the label text |

Relevant code: [catalog schema](../../../src/pyrite/materials/_schema.py),
[target geometry](../../../src/pyrite/campaign/geometry.py),
[scene model](../../../src/pyrite/campaign/model.py),
[filter model](../../../src/pyrite/instrument/model.py), and
[observation identity](../../../src/pyrite/instrument/observation.py).

### What “material” means at each boundary

`pyrite material`, profile material membership, and `[materials.KEY]` select
runnable recipes. They are not a homogeneous-substance registry. Recipe
resolution merges profile defaults and per-material overrides before producing
`MaterialSpec`; its entrance `crystal_key` refers to `[crystals.KEY]` and its
substrate-side layers refer to media or crystals.

`Slab.material` and the first `Stack.layers[].material` normally name the
radiator recipe. `campaign.geometry.crystal_params` also accepts a direct
crystal key when the configured photon grid exists. It resolves a recipe to
its entrance crystal; this lookup alone does not install the recipe's stack.
Catalog campaign construction must separately build the corresponding target.
Layers beneath the entrance film use media or crystal keys through
`substrate_composition` and `layer_radiator`; despite `Layer`'s broad field
wording, a runnable recipe is not generally accepted there.

`FilterPlate.material` and `linear_attenuation_inv_mm` accept crystal/media
keys or an explicit `MediumSpec`. A key present only in `[materials]` is
rejected. Where the same spelling occurs in both namespaces, such as
`silicon`, the receiving boundary decides its meaning. Filters resolve only
homogeneous elemental composition, not the crystal's cut or radiation model.

Profile filter rows pass through
`materials/_beam_detector_parse.py::_parse_filter_rows`, then
`campaign/observation.py::filter_from_config` constructs the plate and pose.
`resolve_profile_observation` combines these with detector/scorer/acquisition;
geometry-only profiles need not enable counting observations. The selected
catalog is loaded through `materials.catalog`; construction and several
downstream helpers consult their imported `CATALOG`. A new shared resolver
must receive the selected catalog explicitly and must not fall back to the
bundled catalog for unresolved external keys.

### Useful sharing versus necessary separation

Composition lookup occurs in `substrate_composition`,
`materials.attenuation._resolve_composition`, `FilterPlate` validation,
`api._resolved_composition`, and `instrument.observation._resolved_composition`.
They differ in case handling, validation and allowed inputs. This is avoidable
duplication; extraction must preserve each legacy boundary's accepted inputs,
error behavior and ordered composition output before tightening any rules.

Positive dimensions, finite values, and homogeneous material validation can
share primitive checks. Target tilt quantization, groove-angle constraints,
layer ordering, filter overlap and downstream-volume checks remain specific.
Serialization can consume the same resolved description while retaining
separate case, observation and trajectory encoders. Equal-looking field names
are insufficient reason to replace those encoders.

## Three bounded designs

| Design | API and catalog example | Benefit | Cost and decision |
| --- | --- | --- | --- |
| A. Specialized types, clearer terminology | Keep `Slab("silicon")`, `FilterPlate("silicon", ...)`, `[materials]` recipes and `[crystals]`/`[media]` substances; explain namespaces in help/docs | No new body API or persistence migration | Leaves composition duplication and makes two physical instances hard to describe once; acceptable fallback |
| B. Shared components, specialized wrappers | Proposed `RectangularBody(substance, dimensions_mm)` plus separate pose; explicit target adapter and primary-attenuator adapter; existing catalog rows lower through these components | Material, dimensions and placement can be reused without changing solver ownership | Requires checked unit/frame/orientation adapters; **recommended**, with resolver extraction first |
| C. One closed body set with assignments | Proposed `Scene(bodies={"plate": body}, target=TargetAssignment("plate"), filters=(AttenuatorAssignment("plate"),))`; proposed `[bodies.plate]` and role references | One scene inventory and explicit assignment policy | New identity/reference/serialization rules, duplicate-assignment checks and lowering for infinite/grooved/layered targets; too much migration for current use cases |

All three keep geometry closed. C need not imply a general navigator, but
adding one would require coordination with #19 and a superseding ADR-0008.
Issue #341's optional photon transport is separate from this description
decision; neither scattering nor fluorescence becomes supported by choosing B.

### Component contract and concrete examples

The first implementation slice needs no new public body class. After that,
the proposed component vocabulary is:

- `SubstanceRef(kind="crystal" | "medium", key=...)`, resolved against an
  explicit catalog; an inline medium carries validated number densities.
- `RectangularBody(substance, size_mm=(width, height), thickness_mm=...)`, a
  scalar immutable homogeneous solid. It contains no campaign grids or name.
- Separate `PlanarPose` for a placed instance. A target adapter accepts only
  the existing supported sample placement/orientation and crystalline entrance
  material; it does not promise arbitrary translation or rotation support.
- Role adapters that explicitly select existing source-target behavior or
  `primary-attenuation-only`. Names belong to instances/display metadata.
  Target lattice orientation is additional state, not inferred from pose.

These are interface sketches for review, not executable constructors:

```text
silicon = SubstanceRef(kind="crystal", key="silicon")
plate = RectangularBody(silicon, size_mm=(10, 10), thickness_mm=0.1)
target = TargetAdapter(plate, tilt_deg=30, lattice_cut="catalog")
filter = PrimaryAttenuatorAdapter(plate, pose=downstream_pose, name="window")
```

The adapters lower to existing types and payloads. This shares the *body
description*, not one in-scene instance: the same solid cannot be counted as
both target material and a downstream plate at the same location. Initial
target adapters reject amorphous entrances, layers, grooves, infinite lateral
extent and arbitrary poses; legacy target constructors continue to support
their own existing variants. Avoid introducing a second representation for
those variants until a concrete adapter needs it.

Existing catalog spelling remains valid, for example:

```toml
# Recipe: silicon is resolved to the crystal of the same key.
[materials.silicon]
display_name = "Silicon"

# Homogeneous filter substance, with independently authored dimensions/pose.
[[profiles.example.filters]]
name = "window"
material = "silicon"
thickness_mm = 0.1
size_mm = [10.0, 10.0]
distance_mm = 200.0
polar_deg = 90.0
```

This is an excerpt, not a complete catalog/profile. Under B there is no new
`[bodies]` table and no flag-day rename of `[materials]`. Under C the equivalent
proposed excerpt would have `[bodies.window]` with an explicit crystal
reference and dimensions, and a profile filter assignment with `body="window"`,
pose and interaction policy. That reference system is deliberately deferred.

| Use case | Today | Under the recommendation |
| --- | --- | --- |
| Same homogeneous silicon plate as irradiated target or filter | Construct `Slab("silicon", thickness_ang=1_000_000, footprint=Footprint(10, 10))` versus `FilterPlate("silicon", 0.1, (10, 10), pose)`; target also needs its tilt/cut | Reuse the rectangular description; adapters still choose different solvers |
| Two silicon instances with different poses | Two `FilterPlate` objects share the key/composition and have independent poses/names | Shared immutable substance/body values, separate placed instances; no coupling when one moves |
| Crystalline film on amorphous substrate | `Stack((Layer("mos2", 1000), Layer("sio2", 2850)))`; film drives entrance crystallography; amorphous layer has no coherent radiator | Keep ordered target layers; no averaged “stack material” |
| Multilayer downstream filter | Adjacent non-overlapping `FilterPlate` boxes, one per homogeneous layer; touching boxes are permitted; `Stack` itself is not a filter | Keep the explicit plate collection initially; a future layer adapter must compute each layer's pose/chord, preserve order and reject overlap |
| Plate outside the central ray | Valid if its full volume satisfies downstream checks; pixel centre rays determine partial coverage or a complete miss | Preserve finite-box geometry; do not replace it with one infinite attenuator |
| Nominal filter intersected by electrons | Filters never enter electron transport; downstream validation does not trace electron trajectories | Reject an explicit request for electron interactions; use a supported target stack if appropriate, otherwise require new navigation/physics work |

The current downstream validator checks source/detector half-spaces, normal
direction, unique filter names and pairwise volume overlap. It does **not**
prove that electrons miss a plate. A future API must not advertise such a
guarantee based on pose alone. Geometric checks of a nominal beam axis are not
a substitute for checking scattered electrons.

## Role and interaction policy

Placement alone cannot choose role. A downstream crystal can remove primary
photons under the current filter approximation without becoming a coherent
radiator. A “substrate” may be crystalline and contribute through supported
target-layer lowering. Renaming either object must not change behavior.

The initial shared adapter exposes only the two existing role contracts. It
rejects requested filter fluorescence, diffraction, photon redistribution,
secondary production or electron transport before computation. It rejects
layered-to-homogeneous coercion and conflicting target/filter assignments.
The removal coefficient can include scattering losses without modeling the
scattered photons as downstream signal; see the existing
[attenuation validation record](../../validation/detectors/positioned-filter-attenuation.md).
This proposal adds no equation or new physics claim.

Lattice cut (`surface_hkl` or `beam_uvw`), reflection selection, layer-relative
azimuth and mosaic state describe crystallography. Pose describes where a
finite solid sits and its local axes. For targets these are currently coupled
by specialized lowering; an adapter must preserve that convention explicitly.
For filters, the attenuation model consumes composition and finite chords,
so changing a hypothetical lattice cut does not enable diffraction.

## Units, frames and scalar expansion

Keep substance densities in atoms per cubic Å. Keep current target thickness
and groove spacing in Å, footprints in mm, public angles in degrees, and
downstream dimensions/positions in mm. A rectangular adapter uses mm and
converts thickness explicitly: `1 mm = 10_000_000 Å`. It must not round or
quantize dimensions during conversion. Existing target angle quantization
remains in target lowering.

The lab origin is the target reference entrance origin, not the slab centre.
Filter `pose.center_mm` is the box centre; the normal points downstream with
the source in its negative half-space. Target tilt determines the supported
sample-to-lab rotation; a target-body export must include the entrance-to-centre
offset rather than reinterpret entrance coordinates as centre coordinates.
Trajectory scene records already state the transform and reference; see
[trajectory scenes](../../repo-design/storage/trajectory-scenes.md).

All new body components are scalar. Expand `Sweep` path axes before resolving
body instances. Continue accepting legacy sequence-valued target specifications
through their current campaign adapter; preserve Cartesian-product ordering,
deduplication and seed assignment. Do not convert a sequence to a body by
silently taking its first element. A layered object is ordered physical regions
with thickness and substance per region, never one effective medium by default.

## Resolved content, identity and stored schemas

Normalize through the existing owners, not by hashing new dataclass reprs.
Resolve catalog references to the physical data each solver actually uses;
omit display names and catalog filesystem locations from physical identity.
For a new explicit reference, record kind as well as key during resolution to
avoid namespace ambiguity. Do not change old string lookup precedence.

Preserve the existing order of composition pairs, layers and filters in v1
payloads. Sorting composition or filters, merging equal-element entries, or
adding new policy tags may change persisted digests even if a calculation
appears equivalent. Such normalization needs a separately reviewed identity
version and compatibility reader, not an incidental resolver refactor.

| Change | Required invalidation/reuse |
| --- | --- |
| Target thickness, cut/orientation, footprint, grooves, layer substance/order or other resolved source inputs | Rebuild affected cases/content keys and source output |
| Filter composition, dimensions or pose with source inputs held fixed | Preserve source case; invalidate true-spatial observation and dependent observation results |
| Detector pose/acceptance/angular sampling | Re-resolve source observation directions as well as spatial output; the current case includes a scalar detector projection, so source reuse is not universally guaranteed |
| Response only | Preserve true-spatial identity; change response and linked observation identity |
| Exposure, measured edges or acquisition realization only | Preserve source and true-spatial layers; change acquisition and linked observation identity |
| Filter display name only | Preserve physical observation identity; trajectory scene metadata/digest can change because snapshots retain names |
| Filter reordering | Preserve current ordered observation payload identity semantics; do not canonicalize order merely because ideal primary attenuation commutes |

`campaign.lowering.build_case` passes the scalar target and detector projection
to existing case construction; filters do not enter that source case.
`campaign.profiles.case_content_key` includes resolved case data, seed and
model/table markers while excluding display metadata through its denylist.
Do not broaden that denylist as part of terminology cleanup. Stable recipe
keys still participate in persisted case names and selection even where they
are excluded from a physical digest; renaming them is not a free migration.

`instrument.observation.true_spatial_payload` hashes resolved filter
composition, dimensions, pose, scorer, source digest and attenuation arrays.
Names are excluded; filter order is retained. Its frozen digest fixtures are
a compatibility gate. The legacy physical-observation path in `api.py` has its
own encoder and must also remain unchanged. Response and acquisition have
separate identity layers. Stored observations under `observations/` and
source HDF5/CAS payloads stay readable without conversion.

`instrument.scene.scene_payload` retains ordered geometry, names, material
references and numeric corners in `pyrite.trajectory-scene.v1`; it is a
snapshot, not a new transport input. Preserve it and the optional scene group
for old captures. Visualization must consume stored geometry, never
reconstruct a historical scene from a changed catalog. New body descriptors
should lower to this payload rather than persist Python objects.

The design-only change has no numerical or schema effect. Future adapters
must produce equal cases, payloads and arrays for equivalent old inputs. A
change in default numerical behavior requires the owning model marker and
physics review; a descriptor refactor is not authority for that change.

## Bounded migration after review

The issue body remains the executable checklist. These compatibility gates
define the durable migration contract; they are not a second task tracker.

1. **Terminology and homogeneous resolver.** Owners: `materials/_schema.py`,
   `materials/attenuation.py`, `campaign/geometry.py`, `instrument/model.py`,
   `instrument/observation.py`, `api.py`. Document recipe versus substance;
   extract a material-layer resolver accepting an explicit catalog and allowed
   namespaces. Preserve legacy wrappers, case handling, validation, error
   behavior and composition ordering. Keep `MaterialSpec` and its historical
   `pyrite.materials.catalog` pickle identity; defer any `TargetRecipe` alias
   until a public API naming change is separately approved. Acceptance:
   external catalog selection, explicit media and all legacy accepted/rejected
   namespaces yield equal resolved data and identities.
2. **Scalar rectangular adapter.** Owners: a small import-light description
   module below campaign/instrument, `campaign/geometry.py`,
   `campaign/lowering.py`, `instrument/model.py`, `_planar_geometry.py`.
   Finalize its name/export location at design review. Add immutable
   homogeneous dimensions/substance components and explicit role adapters;
   keep old constructors/keywords and `Target`'s closed variants. Acceptance:
   silicon example, two independent poses, exact unit conversion, supported
   entrance/cut convention, and clear rejection of unsupported shape,
   material, pose and interaction requests. No general navigator or automatic
   role inference.
3. **Catalog/CLI adapters and persistence parity.** Owners:
   `materials/_parse.py`, `_beam_detector_parse.py`, `campaign/observation.py`,
   `campaign/profile_edit.py`, `cli/commands/material.py`, `_profile_filters.py`,
   `observations/store.py`, `instrument/scene.py` and trajectory consumers.
   Keep schema-version-1 tables/keys, complete external-catalog layouts,
   profile filter order and CLI command/JSON/help contracts. Describe
   `material` as recipe in material commands and as crystal/medium in filter
   help; regenerate the CLI reference if help changes. Do not introduce a
   `body` command, `[bodies]` registry or rename recipe keys. Acceptance:
   equal resolved catalog/case/observation/scene payloads, old stored files
   readable, and no changed digest fixtures. Version any unavoidable new
   stored semantics explicitly; no bulk rewrite.

A layered-filter convenience adapter, universal scene inventory, electron
intersection handling, and extra photon interactions are subsequent proposals
with their own acceptance and review. #23/#32 provide positioned-plate context;
#19 governs broader geometry evaluation; #341 governs optional photon transport.
They are not blockers for the resolver/description study.

### Focused regression gates

| Owner/tests | Required evidence for later implementation |
| --- | --- |
| `tests/scan/test_target_geometry.py`, `test_scene_model.py` | Equal legacy lowering, scalar rejection, sweep product order/seeds, layers/radiators, angle/groove constraints and adapter unit/frame parity |
| `tests/instrument/test_model.py` | Explicit/cross-catalog material resolution, unsupported policies, full downstream volume, overlapping versus touching plates, off-centre validity |
| `tests/instrument/test_geometry.py`, `test_attenuation.py` | Equal normal/oblique/side-escape chords, partial shadows, zero-filter identity, multilayer collection parity; retain existing numerical oracles |
| `tests/instrument/test_observation.py` | Frozen identity fixtures unchanged, names ignored, order preserved, invalidation confined to owning layers |
| `tests/materials/test_material_catalog.py`, `test_external_catalog_identity.py`, `test_catalog_layout.py` | Namespace ambiguity, stack resolution, explicit selected-catalog behavior, no path-dependent physical identity, schema/pickle compatibility |
| `tests/cli/test_profile.py`, `test_profile_user_layer.py` | Filter CRUD/JSON/order, recipe/profile vocabulary, complete external catalogs and user-layer writes preserved |
| `tests/observations/test_store.py`, `test_produce.py`, `test_sweep.py` | Old schema reads, unchanged observation reuse, source reuse after filter-only edits |
| `tests/montecarlo/test_trajectory_scene.py`, `tests/plots/test_trajectory_builders.py` | Stored scene reference/transform, old scene-less capture reads, historical geometry preserved after catalog edits |

Design review must approve the component boundary, adapter limits and parity
gates before dispatching implementation. It need not approve a universal body
API, new interaction engine or catalog migration to accept the narrower choice.
