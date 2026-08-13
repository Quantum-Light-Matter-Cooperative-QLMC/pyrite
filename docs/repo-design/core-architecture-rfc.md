# Core architecture RFC: scene objects, scorers, and API surface

* **Status:** Proposed
* **Date:** 2026-08-13
* **Supersedes:** nothing
* **Expected outcome:** one ADR per accepted section (see {ref}`sec-core-arch-adrs`)

This RFC proposes a set of changes to PyRITE's object model, simulation entry
point, result persistence, and command surface. It records the current state
with evidence, the target state with interface sketches, and a migration path
for each change. It does not itself change behavior.

The changes are motivated by a comparison against codes solving a similar
problem — Monte Carlo transport of a configurable beam into a configurable
target, observed by a configurable detector. The relevant comparison set is
[OpenMC](https://docs.openmc.org/), [abTEM](https://abtem.readthedocs.io/),
[xrt](https://xrt.readthedocs.io/), [TOPAS](https://topas.readthedocs.io/),
[EGSnrc/egs++](https://nrc-cnrc.github.io/EGSnrc/doc/pirs898/),
PENELOPE/PENGEOM, and McStas/McXtrace.

## Summary

```{list-table} Proposed changes, in dependency order.
:name: tbl-core-arch-changes
:header-rows: 1

* - #
  - Change
  - Primary benefit
  - Physics touched
* - 1
  - Public single-shot simulation API
  - Composable entry point; unblocks the rest
  - No
* - 2
  - Split `Sweep` into `Scene` / `Sweep` / `Numerics` / `Analysis`
  - One owner per field; every field sweepable
  - No
* - 3
  - Typed `Case` record and versioned dataset identity
  - Ends the divergence-only-key tax
  - No
* - 4
  - Consolidate the geometry surface behind `Target`
  - Four flat key families become one object
  - No
* - 5
  - Detector becomes a scorer; energy grid becomes a detector field
  - Multi-detector runs; demotes `energy-grid`
  - Binning only
* - 6
  - Array results in HDF5/Zarr; MCPL export
  - Archival safety; external interoperability
  - No
* - 7
  - Reduce the CLI noun surface
  - Smaller user vocabulary
  - No
```

Every change is intended to be behavior-preserving for existing runs. Where a
change would alter a stored digest or a stored payload, the migration path is
stated explicitly.

## Motivation

### Fields have no principled owner

`Sweep` ({file}`src/pyrite/campaign/sweep.py` line 344) currently carries, in
one dataclass:

* scene inputs — `material`, `thickness_ang`, `tilt_deg`, `tilt_azim_deg`,
  `beam`, `detector`, `substrate`, `stack`;
* sweep axes — every field typed `ScalarOrSeq`;
* numerics — `spec_chunk`, `brem_chunk`, `n_electrons`, `n_electrons_brem`;
* physics-model switches — `mosaic`, `mosaic_route`, `mosaic_nodes`,
  `n_families`, `max_reflections`;
* tally binning — `E_grid_line`, `E_grid_line_by_energy`, `E_grid_brem`;
* compatibility debris — `e_grid_eV`, and the `theta_obs_deg` /
  `dtheta_obs_deg` / `domega_sr` `InitVar` reconciliation in `__post_init__`.

`Settings` ({file}`src/pyrite/results/store.py` line 69) is documented as
"analysis / detector / unit knobs shared by post-processing and plots" but owns
`emission`, `xray_dispersion`, `n_electrons`, `n_electrons_brem`, and
`brem_source` — all of which change the simulation result, not its
presentation.

The diagnostic symptom is the `build_cases` signature
({file}`src/pyrite/campaign/sweep.py` line 625):

```python
def build_cases(sweep, n_electrons=450, n_electrons_brem=100,
                coherent_emission=False, xray_dispersion="vacuum"):
```

Four run-affecting parameters have to be re-threaded as loose keyword
arguments because they live on the wrong object. Every future run-affecting
option inherits that pattern.

### Sweepability is a property of the field type

Because a field is swept by being typed `ScalarOrSeq`, only fields that were
declared that way can be swept. Today it is not possible to sweep
`stack[0].thickness_ang`, a per-layer orientation, `beam.transverse.eps_n`, or
`detector.observation_angle_deg`, and adding any of them means a new field, new
`_seq()` plumbing, and a new case-name suffix rule — compare the `ne=` suffix
introduced for the electron-count grids.

### The case dict is the real contract and it is untyped

`build_cases` emits plain dicts of roughly forty keys. That dict is
simultaneously the transport input, the spectrum-kernel input, the checkpoint
resume key, and the CAS content key. Its schema exists only in the `run_case`
docstring ({file}`src/pyrite/montecarlo/runner/__init__.py` line 453).

The **divergence-only key rule** compounds this. A run-affecting key is
*omitted* from the payload when it holds its historical default, so that
previously computed digests stay valid — see the `coherent_emission` and
`xray_dispersion` blocks in `build_cases`, and the `Settings.xray_dispersion`
comment. The consequence is that the payload schema is a function of the
project's commit history rather than of the physical configuration, and each
new option adds another permanent conditional.

### The detector is not one object

Two disjoint notions share the name:

* `DetectorSpec` ({file}`src/pyrite/detectors/spec.py` line 54) supplies
  `observation_angle_deg`, `polar_acceptance_deg`, and `solid_angle_sr` to the
  case. Its remaining fields — `response_model`, `qe_curve`, `pixel_pitch_um`,
  `sensor_thickness_um`, `distance_mm`, `threshold_eV` — are documented as
  inert and are read by nothing.
* The forward models `TimepixResponse` and `EagleResponse`
  ({file}`src/pyrite/detectors/`, 1 645 lines) are applied after the fact, in
  `store_result` and in the plotting layer.

There is therefore no object representing a detector end to end, and a run can
have exactly one observation geometry.

Related: the emission binning, which every comparable code treats as a property
of a tally or monitor, has instead become `E_grid_line` / `E_grid_brem` on
`Sweep`, per-material entries in the catalog, and a top-level CLI noun with its
own content-addressed artifact store, provenance, garbage collection,
verification, and golden files — {file}`src/pyrite/energy_grid/`, 4 542 lines.

### There is no first-class Python entry point

The documented entry points are the CLI and the marimo apps. `Sweep`,
`build_cases`, and `run_sweep` are importable, but the API is sweep-shaped: a
single simulation must be expressed as a one-element sweep, and the result is
retrieved through a checkpoint rather than returned. Every comparable code that
supports configurable beams, targets, and detectors made the object model the
primary interface and the CLI a shell over it.

### The command surface is large relative to the modelled physics

Thirteen top-level groups and 88 documented commands. A substantial fraction is
store maintenance surfaced as user vocabulary — `checkpoint gc|rm|merge|slim|
recompute|archive|restore`, `energy-grid add|rm|verify|gc|regen-golden`,
`performance rm`, `remote gc|prune-jobs`. A further part is a TOML editor
implemented as a command group: {file}`src/pyrite/cli/commands/profile.py` is
1 169 lines over {file}`src/pyrite/campaign/profile_edit.py` at 465 lines.

The CLI's *contracts* are strong and are not in question here: exit-code
discipline, the versioned JSON envelope, destructive-operation previews, the
deprecation registry, and the generated reference should all be retained
unchanged. The proposal concerns the number of nouns, not their quality.

## Change 1 — public single-shot simulation API

### Target state

A user-facing function that takes a scene and returns a result, with no sweep,
no profile, no checkpoint, and no catalog mutation:

```python
import pyrite as pr

beam = pr.Beam(
    energy_keV=45.0,
    transverse=pr.CourantSnyder(eps_n=1e-6, beta_m=0.5),
)
target = pr.Slab(pr.material("hopg"), thickness_ang=2e4, tilt_deg=30.0)
detector = pr.Detector(
    observation_angle_deg=90.0,
    energy_bins=pr.arange_eV(100.0, 5000.0, 3.0),
    response=pr.Timepix3(),
)

result = pr.simulate(beam, target, detector, numerics=pr.Numerics(n_electrons=450))
result.spectrum          # counts or photons per bin, units documented on the object
result.provenance        # resolved scene, identity digest, backend, versions
```

`pr.simulate` is a thin composition of the existing pieces: it builds one
`Case`, calls the existing `run_case`, and wraps the returned arrays. It must
not duplicate physics.

### Migration

* Add the module; do not remove `build_cases` or `run_sweep`.
* Reimplement `runs.scan` and `runs.blaze` as loops over `pr.simulate`'s
  internals once Change 2 lands, so the sweep driver and the single-shot path
  share one code path rather than two.
* Port the marimo apps to the public API. The trace app already runs transport
  directly from catalog scan grids and is the natural first consumer.

### Acceptance

* A documented example runs end to end in fewer than twenty lines with no
  filesystem writes.
* `pr.simulate` on a scene equivalent to an existing catalog case reproduces
  that case's stored spectrum bit for bit.
* {file}`docs/api.md` documents the public names; the export-freeze tests are
  extended to cover them.

## Change 2 — split `Sweep`

### Target state

Four objects with distinct lifetimes:

```{list-table} Object responsibilities after the split.
:name: tbl-core-arch-objects
:header-rows: 1

* - Object
  - Owns
  - Enters dataset identity
* - `Scene`
  - `beam`, `target`, `detector` — the physical configuration
  - Yes
* - `Sweep`
  - a base `Scene` plus named axes over paths into it
  - Yes, per expanded scene
* - `Numerics`
  - electron counts, chunk sizes, transport core, backend policy
  - Only where a value changes results
* - `Analysis`
  - unit scaling, presentation-time convolution, plotting knobs
  - No
```

Axes are addressed by dotted path rather than by field type:

```python
sweep = pr.Sweep(
    base=scene,
    axes={
        "beam.energy_keV": [30.0, 45.0, 60.0],
        "target.tilt_deg": [15.0, 30.0, 45.0],
        "target.layers[1].thickness_ang": [2.85e3, 5.0e3],
    },
)
```

This removes `ScalarOrSeq` from the object model, makes nested and per-layer
fields sweepable without new plumbing, and gives case naming a single
mechanical rule — the axis path and its value — instead of the current
hand-written label concatenation.

`Settings` dissolves: its run-affecting fields (`emission`, `xray_dispersion`,
`brem_source`) move to `Scene` or `Numerics` by whether they describe the
physical configuration or the sampling of it; its presentation fields move to
`Analysis`.

### Migration

* Introduce the new objects alongside `Sweep`/`Settings`.
* Provide `Sweep.from_legacy(old_sweep, settings)` and keep the old classes as
  deprecated shims through one support window under the existing D7 harness.
* Resolution order must remain: catalog profile → per-material override →
  explicit argument, unchanged from
  [configuration resolution](configuration-resolution.md).

### Acceptance

* Every existing catalog profile round-trips through the new objects to an
  identical expanded case list.
* At least one previously unreachable axis — per-layer thickness is the
  suggested target — is demonstrated in a test.
* No new `ScalarOrSeq` field is introduced anywhere.

## Change 3 — typed `Case` and versioned identity

### Target state

`Case` becomes a frozen, fully populated, canonically serializable record. No
key is omitted to preserve a digest. Dataset identity gains an explicit
version, and historical digests are preserved by a recorded mapping rather than
by permanent conditionals in `build_cases`:

```python
IDENTITY_MIGRATIONS = {
    1: _identity_v1,   # divergence-only rule, reproduces every stored digest
    2: _identity_v2,   # canonical full-payload digest
}
```

Checkpoints and campaign locks record `identity_version`. Readers dispatch on
the recorded value; writers emit the current one. Old checkpoints continue to
resolve because their digests are recomputed under `_identity_v1`, not because
the payload keeps pretending the option does not exist.

### Migration

1. Land `Case` as a typed wrapper that serializes to today's dict exactly, and
   assert equivalence against the existing golden cases.
2. Add `identity_version` to the lock and checkpoint metadata, defaulting to 1
   when absent.
3. Introduce `_identity_v2` behind a profile-level opt-in; it is a
   recompute-from-scratch boundary for any dataset that opts in.

Step 3 is separable and may be deferred indefinitely. Steps 1 and 2 alone stop
the accumulation of new divergence keys and cost nothing.

### Acceptance

* `Case` validates required fields at construction and rejects unknown keys.
* Every stored checkpoint continues to resume with no recomputation.
* The `run_case` docstring schema is replaced by the type; the docstring is
  reduced to semantics and units.

## Change 4 — consolidate the geometry surface

### Scope

This change is about **how geometry is expressed**, not about supporting new
shapes. Arbitrary geometry is explicitly out of scope; see
{ref}`sec-core-arch-nongoals`.

### Current state

The four configurations that exist today are expressed as four unrelated flat
key families threaded through `Sweep` and the case dict:

* a single slab — `thickness_ang`;
* a layer stack — `substrate` plus `substrate_thickness_ang`, *or* `stack`, and
  the derived `abs_layers` and `layer_radiators` case keys;
* a bounded rectangular footprint — `crystal_width_mm` and `crystal_height_mm`,
  which must be supplied together or both omitted;
* a blazed sawtooth entrance face — `groove_spacing_ang`, valid only under a
  conjunction of constraints enforced by `_reject_invalid_groove_geometry`.

The constraints between these families are real and are currently enforced by
scattered validation: `substrate` and `stack` are mutually exclusive; grooves
require `tilt_azim_deg == 180`, `0 < tilt_deg < 90`, `theta_obs == 90`, no
stack, and no finite footprint.

### Target state

One `Target` object holding a **closed, named set of variants**. No dispatch
protocol, no navigator interface, no constructive solid geometry:

```python
target = pr.Slab(material, thickness_ang=2e4, tilt_deg=30.0, tilt_azim_deg=0.0)

target = pr.Stack(
    layers=[pr.Layer("mos2", 2.85e3), pr.Layer("sio2", 2.85e3), pr.Layer("silicon", 5e6)],
    tilt_deg=30.0,
)

target = pr.Slab(material, thickness_ang=2e4, tilt_deg=30.0,
                 footprint=pr.Footprint(width_mm=5.0, height_mm=5.0))

target = pr.Slab(material, thickness_ang=2e4, tilt_deg=30.0,
                 entrance_face=pr.BlazedGrooves(spacing_ang=1e4))
```

`Target` owns its own validity: the mutual exclusions and the groove
preconditions become `__post_init__` checks on one object with one error
vocabulary, rather than conditions distributed across `build_cases`,
`_reject_invalid_groove_geometry`, and `stack_layers`.

**Transport internals are untouched.** `Target` lowers to exactly the case keys
that exist now — `abs_layers`, `layer_radiators`, `crystal_width_mm`,
`crystal_height_mm`, `groove_spacing_ang` — so
{file}`src/pyrite/montecarlo/transport.py` (3 949 lines) and its CUDA twin
{file}`src/pyrite/montecarlo/transport_jit_kernel.py` (1 047 lines) keep their
current branches, their current arithmetic, and their bit-for-bit guarantees.
The change is a boundary reshape at the campaign layer only.

### Why the surface is worth consolidating even without new shapes

* Validity becomes checkable at construction rather than at case expansion, so
  an invalid target fails in the user's script instead of after profile
  resolution.
* `Target` is what a sweep axis addresses (`target.layers[1].thickness_ang`),
  which is the mechanism Change 2 depends on.
* It is what a scene viewer would render, and what a serialized scene would
  contain.
* It removes the `substrate`-versus-`stack` sugar duplication: `substrate="x"`
  becomes a `Stack` constructor helper rather than a parallel field pair.

### Acceptance

* Every existing sweep configuration is expressible, and expands to an
  identical case list.
* All geometry validation lives in `Target`; `build_cases` contains no geometry
  conditionals.
* No change to any transport code path, kernel, or golden spectrum.

## Change 5 — detector as scorer

### Target state

A `Detector` owns its acceptance, its binning, and its response, and a run may
carry more than one:

```python
detector = pr.Detector(
    observation_angle_deg=90.0,
    polar_acceptance_deg=...,
    solid_angle_sr=...,
    energy_bins=pr.arange_eV(100.0, 5000.0, 3.0),
    response=pr.Timepix3(),          # or None for the intrinsic spectrum
)
```

Two structural consequences:

* `DetectorSpec`'s currently inert fields either become live through the
  `response` object or are deleted. An inert field that never becomes live is a
  liability in a hashed payload.
* The photon-energy grid becomes `detector.energy_bins`. Today's separate line
  and bremsstrahlung grids remain, expressed as two binnings on the detector —
  the physical reason for the split (fine and narrow for the coherent lines,
  coarse and wide for the smooth continuum) is unchanged and should be
  documented on the object.

### Effect on `pyrite energy-grid`

The derivation machinery is genuinely expensive and its artifacts are worth
keeping content-addressed; this change does not propose deleting
{file}`src/pyrite/energy_grid/`. It proposes demoting it from user vocabulary:
the derived grid becomes an input to a detector, and the artifact store becomes
an implementation detail of catalog resolution. The `derive`, `verify`, and
`gc` verbs move under a maintenance noun (Change 7) or under `pyrite-dev`.

### Migration

* Land `Detector` as a superset of `DetectorSpec`, with `DetectorSpec` retained
  as a deprecated alias.
* Multi-detector support is a second step: a single-detector run must remain
  bit-for-bit, and the stored record layout for one detector must not change.
* Reuse of one transport pass across several detectors is the payoff and should
  be designed for, but is not required in the first step.

### Acceptance

* A single-detector run reproduces its stored spectrum bit for bit.
* No inert field survives on the public detector object.
* `E_grid_line` / `E_grid_brem` are absent from the scene objects.

## Change 6 — result persistence and interchange

### Current state

Checkpoints are pickles — `checkpoints/<stem>/{line,brem}.pkl` — plus
content-addressed case blobs, campaign locks, and manifests. The store design
is sound; see
[checkpoint case store](storage/checkpoint-case-store.md) and
[dataset identity and storage](storage/dataset-identity-and-storage.md).

### Problem

Pickle is version-fragile across Python and NumPy releases, unsafe to accept
from a third party, and unreadable by any other tool. It is a poor archival
format for results that back scientific claims.

### Target state

* Array payloads move to HDF5 or Zarr under a documented, versioned schema.
  The CAS layout, the sharding, the atomic-write discipline, and the lock model
  are retained unchanged; only the leaf encoding changes.
* A reader shim keeps existing `.pkl` datasets loadable indefinitely. Migration
  is opportunistic — rewrite on next save, as the store already does for the
  legacy flat layout.
* Add MCPL export for the emitted photon list, and optionally for the transport
  segment list. [MCPL](https://mctools.github.io/mcpl/mcpl.pdf) is the
  interchange format shared by Geant4, MCNP, McStas, and McXtrace, and it is
  the pragmatic answer to "can another code model our detector?" — it makes
  that possible without PyRITE owning a general detector geometry.

### Acceptance

* A result file is readable by `h5py` with no PyRITE import.
* Every stored `.pkl` dataset loads unchanged.
* An exported MCPL file passes `mcpltool` validation.

## Change 7 — reduce the CLI noun surface

### Target state

Six top-level nouns: `run`, `profile`, `material`, `job`, `app`, `cache`.

```{list-table} Disposition of the current top-level groups.
:name: tbl-core-arch-cli
:header-rows: 1

* - Current
  - Disposition
* - `run`, `job`, `app`, `profile`, `material`
  - Retained
* - `checkpoint`, `performance`, `energy-grid` maintenance verbs
  - Consolidated under `cache`
* - `energy-grid derive|show|defaults`
  - Retained under `material` / `profile` as grid inputs
* - `config`, `setup`, `completion`
  - Retained; candidates for `pyrite config` consolidation
* - `remote`
  - Retained as resource management only; `--remote` stays the run modifier
* - `beam`
  - Folded into `profile` unless named beams prove independently useful
```

Two narrower proposals:

* Keep `profile show`, `profile list`, `material show`, and `material validate`.
  Reconsider the interactive `create` / `rename` / `delete` flows: they are the
  bulk of the 1 634 lines across
  {file}`src/pyrite/cli/commands/profile.py` and
  {file}`src/pyrite/campaign/profile_edit.py`, and the catalog is already
  self-validating, so edit-then-validate is a defensible alternative to
  prompt-driven mutation.
* Keep every existing contract. Exit codes, the JSON envelope, `-o/--output`
  semantics, destructive previews, the deprecation registry, and the generated
  reference are not in scope for change.

### Migration

All removals go through the existing D7 deprecation harness with a hidden
warning alias for one support window, and
{file}`docs/repo-design/cli/cli-deprecations.md` is regenerated. The
cli-reference freeze test guards each step.

### Acceptance

* Root help lists at most six primary nouns.
* No retired spelling breaks without a warning window.
* The generated reference and its freeze test are updated in the same change.

(sec-core-arch-sequencing)=
## Sequencing

```{list-table} Dependency order and independence.
:name: tbl-core-arch-sequence
:header-rows: 1

* - Step
  - Change
  - Depends on
  - Separable
* - 1
  - Typed `Case`, steps 1–2 only (Change 3)
  - —
  - Yes
* - 2
  - `Target` consolidation (Change 4)
  - —
  - Yes
* - 3
  - `Detector` object (Change 5, single-detector)
  - 2
  - Yes
* - 4
  - `Scene` / `Sweep` / `Numerics` / `Analysis` split (Change 2)
  - 2, 3
  - No
* - 5
  - Public `pr.simulate` (Change 1)
  - 4
  - No
* - 6
  - Result format and MCPL export (Change 6)
  - 1
  - Yes
* - 7
  - CLI noun reduction (Change 7)
  - 5
  - Partially
```

Steps 1, 2, 3, and 6 are independently valuable and independently landable.
Steps 4 and 5 are the structural core and should be treated as one project.
Step 7 should follow the public API so that removed commands have a documented
replacement.

## Compatibility

* No change in this RFC may alter a stored spectrum. Every step lists
  bit-for-bit reproduction as an acceptance criterion.
* Object-model changes ship with deprecated shims for one support window under
  the existing D7 policy.
* Command changes go through the deprecation registry and regenerate the
  reference.
* The existing export-freeze tests
  ({file}`tests/montecarlo/test_exports.py`, {file}`tests/results/test_exports.py`,
  {file}`tests/plots/test_exports.py`) are extended rather than relaxed.

## Risks

* **Scope creep from Change 2 into physics.** The split touches the object that
  every physics module reads. Mitigation: land the typed `Case` first, so
  equivalence is machine-checkable at the boundary before the objects above it
  move.
* **Identity migration.** Step 3 of Change 3 is a recompute boundary. It is
  opt-in and deferrable; the first two steps carry the benefit without the cost.
* **Two result formats during migration.** Mitigate with an opportunistic
  rewrite-on-save, matching the existing legacy-layout migration.
* **CLI removals annoying existing users.** Mitigate with the standard support
  window and by ensuring the public API covers every removed workflow first.

(sec-core-arch-nongoals)=
## Non-goals

### Arbitrary target geometry

PyRITE will not gain a general geometry system — no constructive solid
geometry, no surface/region algebra, no navigator protocol, no imported meshes.
The reasoning, recorded so it is not re-litigated:

* **There is no internal demand.** The physics of interest is coherent emission
  from crystalline slabs and layer stacks. The four configurations enumerated in
  Change 4 cover the current and foreseeable scientific programme.
* **The GPU cost is the binding constraint.** The transport hot path is a CUDA
  kernel maintained as one algorithm with its CPU twin. A general geometry
  requires per-step boundary queries against an arbitrary region set, which on a
  GPU means an acceleration structure and a ray-tracing traversal. The
  AdePT/Celeritas experience is the reference point: porting realistic detector
  geometry to GPUs made geometry the bottleneck through thread divergence and
  register pressure, and the response was a
  [GPU-friendly surface model](https://www.epj-conferences.org/articles/epjconf/abs/2025/22/epjconf_chep2025_01207/epjconf_chep2025_01207.html)
  in VecGeom — a substantial project in its own right.
* **The cheap escape hatch already exists.** MCPL export (Change 6) lets a code
  that already solved general geometry consume PyRITE's emitted photons.

If the constraint ever changes, the seam is narrow and is worth naming now: the
transport core needs `locate(r) -> region`,
`distance_to_boundary(r, v) -> (s, region)`, and
`escape_path(r, v) -> per-region path lengths`. Three implementations of that
interface already exist in specialized form — layer crossing by depth, prism
face intersection, and groove facet intersection. A future general
implementation would have to be a flat, bounded-depth, device-representable
region table interpreted by one loop on both host and device, never a
polymorphic object graph. That note is the whole of the forward plan; nothing
in this RFC builds toward it.

### Graphical user interface

No GUI is proposed. Two constraints are recorded for whenever the question
returns:

* If a scene editor is ever built, it must be a *script generator* over the
  public object model, in the manner of xrt's xrtQook — one input path, never a
  parallel and less-tested way to configure a run.
* The marimo apps are held to the same rule: they should call the public API
  from Change 1 and must not construct case payloads directly.

The useful near-term visual investment is a **scene viewer** — rendering a
resolved `Target`, beam, and detector acceptance from the same objects the
simulation consumes — not a scene editor.

### Unchanged by this RFC

The materials catalog schema, the validation ledger and its methodology, the
backend abstraction, the remote SLURM orchestration, the CAS and lock model,
and every CLI *contract* are all out of scope and are retained as they stand.

(sec-core-arch-adrs)=
## Follow-up decision records

Each accepted section should produce a short ADR recording the decision and its
consequences, per the convention in the
[architecture decision records index](../adr/index.md):

* scene object model and the dissolution of `Settings` (Changes 2, 4, 5);
* dataset identity versioning and the retirement of divergence-only keys
  (Change 3);
* result format and interchange (Change 6);
* CLI noun-surface reduction, amending [ADR-0002](../adr/0002-cli-surface-redesign.md)
  (Change 7);
* the arbitrary-geometry non-goal, which is a decision in its own right and
  should be recorded even though it produces no code.

This RFC may be retired once those ADRs exist and the reference documentation
describes the implemented system.
