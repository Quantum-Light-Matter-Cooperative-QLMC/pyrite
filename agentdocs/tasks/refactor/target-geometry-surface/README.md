# Consolidate the geometry surface behind `Target`

Branch: `refactor/target-geometry-surface`
Source: [core architecture RFC](../../../../docs/repo-design/core-architecture-rfc.md),
Change 4 and the arbitrary-geometry entry in
[Non-goals](../../../../docs/repo-design/core-architecture-rfc.md) — sequencing step 2.
Depends on: nothing. Blocks: `refactor/detector-scorer`,
`refactor/scene-object-model`.

## Problem and scope

This task is about **how geometry is expressed**, not about supporting new
shapes. Arbitrary geometry is an explicit non-goal of the RFC, for three
recorded reasons: no internal demand, a binding GPU cost (a general region set
means an acceleration structure and ray traversal in the CUDA hot path), and
MCPL export as the cheap escape hatch. **Do not build toward it.**

The four configurations that exist today are expressed as four unrelated flat
key families threaded through `Sweep` and the case dict:

- single slab — `thickness_ang`;
- layer stack — `substrate` plus `substrate_thickness_ang`, *or* `stack`, and
  the derived `abs_layers` / `layer_radiators` case keys;
- bounded rectangular footprint — `crystal_width_mm` and `crystal_height_mm`,
  which must be supplied together or both omitted;
- blazed sawtooth entrance face — `groove_spacing_ang`, valid only under a
  conjunction of constraints enforced by `_reject_invalid_groove_geometry`.

The constraints between them are real and are currently enforced by scattered
validation: `substrate` and `stack` are mutually exclusive; grooves require
`tilt_azim_deg == 180`, `0 < tilt_deg < 90`, `theta_obs == 90`, and no stack.

> **Correction (slice A).** The sentence above, and
> [RFC line 398](../../../../docs/repo-design/core-architecture-rfc.md), also
> claimed grooves forbid *a finite footprint*. They do not. `sweep.py:592-599`
> documents at length that a finite footprint **is** allowed with grooves and is
> the default 5x5 mm, because the sub-micron groove phase and the mm-scale
> footprint are independent in transport. The stale claim is also repeated in
> the `groove_spacing_ang` field comment at `sweep.py:379-382`. Code is
> authoritative; the RFC line, the field comment, and this paragraph were wrong.
> Slice F/G must fix the RFC and the field comment rather than propagate them.

In scope: one `Target` object holding a **closed, named set of variants**, owning
its own validity, lowering to exactly today's case keys — plus the `mosaic`
switch, which is a property of the target (the catalog already carries
per-crystal `mosaic_fwhm_deg`). `mosaic_route` and `mosaic_nodes` do **not**
come along: they are quadrature choices and land on `Numerics.convergence` in
`refactor/scene-object-model`.

Out of scope, deliberately:

- Constructive solid geometry, surface/region algebra, a navigator protocol,
  imported meshes, STL/STEP. Non-goal.
- **Any transport change.** `src/pyrite/montecarlo/transport.py` (3 949 lines)
  and its CUDA twin `transport_jit_kernel.py` (1 047 lines) keep their current
  branches, arithmetic, and bit-for-bit guarantees. This is a boundary reshape
  at the campaign layer only.
- New sweep-axis machinery. Making `target.layers[1].thickness_ang` addressable
  is `refactor/scene-object-model`; this task only has to make the object shape
  that such a path could address.

## Target state

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

`Target` lowers to `abs_layers`, `layer_radiators`, `crystal_width_mm`,
`crystal_height_mm`, `groove_spacing_ang` — the keys that exist now.

## Why this is worth doing without new shapes

- Validity becomes checkable at construction, so an invalid target fails in the
  user's script rather than after profile resolution.
- `Target` is what a sweep axis addresses — the mechanism
  `refactor/scene-object-model` depends on.
- It is what a scene viewer would render and what a serialized scene would hold.
- It removes the `substrate`-versus-`stack` sugar duplication: `substrate="x"`
  becomes a `Stack` constructor helper rather than a parallel field pair.

## Implementation path

Likely owners:

| Concern | Location |
| --- | --- |
| Geometry fields on the sweep | `src/pyrite/campaign/sweep.py:344` (`Sweep`) |
| Lowering to case keys | `src/pyrite/campaign/sweep.py:625` (`build_cases`) |
| Groove precondition check | `_reject_invalid_groove_geometry` |
| Stack expansion | `stack_layers` |
| Profile-side geometry | `src/pyrite/campaign/config.py`, `src/pyrite/campaign/profiles.py` |
| Downstream readers (do not change) | `src/pyrite/montecarlo/transport.py`, `runner/__init__.py`, `src/pyrite/runs/run.py`, `blaze.py` |

## Slice A — constraint inventory (done)

Every geometry constraint enforced today, with its site. Note that
`Sweep.__post_init__` (`sweep.py:461-510`) contains **no geometry validation at
all** — it only normalizes legacy flat detector inputs onto `detector`. All of
the following fire at `build_cases` time, which is exactly the "fails after
profile resolution rather than in the user's script" problem.

| # | Constraint | Site | Owner after refactor |
| --- | --- | --- | --- |
| 1 | footprint pairing: both `None` (infinite slab) or both supplied | `sweep.py:748-752` | `Footprint` type (pairing becomes unrepresentable) |
| 2 | footprint values finite and positive | `sweep.py:753-759` | `Footprint.__post_init__` |
| 3 | `substrate=` and `stack=` mutually exclusive | `sweep.py` build_cases, "give either substrate= or stack=, not both" | `Stack` constructor helper (slice D) |
| 4 | `groove_spacing_ang > 0` | `_reject_invalid_groove_geometry` | `BlazedGrooves.__post_init__` |
| 5 | grooves require `tilt_azim_deg == 180` (all cases) | same | `Slab.__post_init__` |
| 6 | grooves require `0 < tilt_deg < 90` (all cases) | same | `Slab.__post_init__` |
| 7 | grooves require `theta_obs_deg == 90` | same | **cross-object — see below** |
| 8 | grooves forbid substrate/stack | same | type constraint: `entrance_face` on `Slab` only |
| 9 | polar `tilt_deg == 0` banned unless `allow_normal_incidence` | `_reject_banned_angles` | `Target` (**not in the original checklist**) |
| 10 | `tilt_azim_deg == 90` banned unconditionally | `_reject_banned_angles` | `Target` (**not in the original checklist**) |

Two things the original checklist A did not anticipate:

- **Constraints 9 and 10 are geometry constraints too.** The banned-angle rule
  (`issue_notes.md` #1: the tilt=0 zero-coherent-line degeneracy and the azim-90
  ranking bug) is angle validity on the target, and its `allow_normal_incidence`
  escape hatch is currently a `Sweep` field. They belong in the same error
  vocabulary as 5 and 6 or the vocabulary is split for no reason.
- **Constraint 7 cannot live in `Target.__post_init__`.** It couples target tilt
  to `detector.observation_angle_deg`, an object `Target` does not and should not
  hold. Proposal: `Target` validates what it owns at construction, and the
  target×detector leg becomes a single `target.validate_against(detector)` call
  in `build_cases`. That still satisfies "no geometry *conditionals* in
  `build_cases`" — one unconditional validation call, no `if` on geometry — but
  the acceptance wording should be read that way deliberately rather than
  discovered during review.

Ordering constraint for slices B/C: `_quantized_angles` is applied to
`tilt_deg` / `tilt_azim_deg` *before* the angle checks. Any `Target` that owns
angles must preserve quantize-then-validate ordering, or constraints 5/6/9/10
change which inputs they accept.

## Checklist

- [x] A — Inventory every geometry constraint currently enforced, and where.
      Include the groove conjunction, the footprint pairing rule, and the
      `substrate`/`stack` exclusion. Each becomes a `__post_init__` check with
      one error vocabulary.
- [x] B — Land `Slab`, `Stack`, `Layer`, `Footprint`, `BlazedGrooves` with
      construction-time validation and a `lower()` producing today's case keys.
      Landed in `src/pyrite/campaign/geometry.py`; `build_cases` is untouched, so
      B is additive and C is a pure rewire.
- [x] C — Route `build_cases` through `Target.lower()`. `build_cases` must end
      with **no geometry conditionals**.
- [x] D1 — Land `Sweep.target` as the canonical spelling; the flat geometry
      fields become deprecated shims that construct it, and the `config.py`
      override path rebuilds the target instead of `replace`-ing flat fields.
      See "Slice D1 — what landed" below.
- [x] D2 — `substrate=` becomes a `Stack` constructor helper; retire the
      parallel field pair behind a deprecated shim under the existing D7
      harness. Move `mosaic` onto `Target` in the same slice, leaving
      `mosaic_route` / `mosaic_nodes` where they are for
      `refactor/scene-object-model`.
- [x] E — Equivalence sweep: every existing catalog profile expands to an
      identical case list.
- [x] F — Write the ADR recording the arbitrary-geometry **non-goal**. The RFC
      states this is a decision in its own right and should be recorded even
      though it produces no code. Include the narrow future seam
      (`locate`, `distance_to_boundary`, `escape_path`) and the constraint that
      any future implementation be a flat, bounded-depth, device-representable
      region table — never a polymorphic object graph.
- [x] G — Update `docs/repo_map.md` ownership rows and the geometry-facing
      physics/guide pages.

## Slice B — what landed

`src/pyrite/campaign/geometry.py` owns the variant set. It sits **below**
`sweep.py` in the import graph, which forced one structural choice: lowering
needs `substrate_composition` / `layer_radiator` / `stack_layers` / `crystal_params`
/ `fmt_thickness`, and importing those from `sweep.py` would be a cycle. They
moved down into `geometry.py` (a verbatim move, no behaviour change) and
`sweep.py` re-exports them, so `pyrite.campaign.sweep.crystal_params` and the
rest keep working for every existing importer.

Two monkeypatch sites had to follow the move —
`tests/montecarlo/test_surface_orientation.py` patched `sweep_module.CATALOG` and
`sweep_module.substrate_radiator`, which no longer reach the definitions. The
CATALOG test now patches both modules, because `build_cases` still reads
`CATALOG` for the case label.

Contract for slice C:

- `Target.lower(cp, *, label, beam_uvw, n_families) -> tuple[LoweredTarget, ...]`
  in exactly today's `product(thickness, tilts, azimuths, footprints)` order, so
  the `seed=1000 * i_c + ...` enumeration is unchanged.
- `LoweredTarget.name` is the whole geometry half of the case name, and
  `LoweredTarget.case_keys()` is the payload fragment — it **omits**
  `groove_spacing_ang` for an ungrooved target, which is what keeps existing case
  payloads bit-for-bit. `build_cases` spreads both, so it needs no `if`.
- `target.validate_against(detector)` is the single unconditional cross-object
  call carrying constraint 7.
- `tests/scan/test_target_geometry.py` already pins `lower()` against
  `build_cases` output for seven configurations (default, infinite slab, full
  four-axis product, both substrate sugars, a 3-layer stack, grooved). Those tests
  are the equivalence evidence C must keep green.

Two constraints beyond the slice-A inventory, both from making the object shape
total rather than from new policy:

- `Stack` requires the film plus at least one layer beneath it; a single-layer
  stack is a `Slab`.
- Only `Stack.layers[0]` (the film) may sweep `thickness_ang`. Layers beneath it
  are single-valued today — `stack_layers` takes a float thickness per layer —
  and making them sweepable is the per-layer axis work in
  `refactor/scene-object-model`, not this task.

Not in B, deliberately: `mosaic` (slice D), the `Sweep.material` /
`substrate=` deprecation shim (slice D), and any `build_cases` change (slice C).

## Slice C — what landed

`build_cases` now holds no geometry conditional. `_reject_invalid_groove_geometry`
is gone; `_target_from_sweep(sweep)` maps today's flat fields onto a `Slab` or
`Stack`, and the body reduces to three lines: construct, `validate_against(
sweep.detector)`, `lower(...)`. The loop spreads `geometry.case_keys()` into
`Case`, which is a kw-only dataclass with an explicit `_CASE_KEY_ORDER` for
serialization, so the kwarg reshuffle cannot move a payload key.

Three checks stayed in `_target_from_sweep` rather than moving into `Target`,
and all three exist *only* because the flat fields are still separate: footprint
pairing, the `substrate`/`stack` exclusion, and grooves-forbid-a-stack. Slice D
deletes the first and third by construction (one `Footprint`, `entrance_face` on
`Slab` only) and turns the second into the `Stack` helper.

Validation ORDER moved: footprint and angle rejection now fire just before the
case loop rather than before the mosaic-route and electron-count checks. No test
pins a two-error precedence, and every groove/footprint test asserts `ValueError`
without a message, so the reworded messages are free.

One behaviour narrowing worth recording: `stack=()` used to build a case with an
empty `abs_layers` stack and a trailing `" on "` in the name; `Stack` now rejects
it (film plus at least one layer). Nothing constructs it.

Not touched, deliberately: `Sweep` keeps every flat geometry field with its
current type and default, so no caller in `config.py`, `profiles.py`,
`profile_edit.py`, `runs/`, or the CLI moved. Evidence: full `pyrite-dev test`
green (3099 passed, 61 skipped), lint and typecheck clean.

## Slice D1 — what landed

`Sweep.target` is the canonical geometry owner. Every flat geometry input is an
`InitVar` normalized onto it in `__post_init__`, so geometry now fails at
`Sweep(...)` (and at `pr.Slab(...)`) rather than inside `build_cases`.
`_target_from_sweep` is gone; its body is now
`geometry.target_from_flat(material, **flat)`, a keyword function carrying the
historical `Sweep` field defaults, joined by two inverses:
`target_flat_fields(target)` (the flat projection) and
`target_replace(target, **flat_changes)` (rebuild-and-revalidate). The override
paths use those instead of `dataclasses.replace` on flat fields:
`config.material_sweep` splits `_TARGET_OVERRIDE_KEYS` exactly as it already
split `_BEAM_OVERRIDE_KEYS`, and `FidelityPreset.apply_sweep` down-samples
through `target_replace`.

Deviations from the handoff, each deliberate:

- **No forwarding properties.** The handoff suggested keeping `sweep.material`
  readable as a property onto `target.material`. A `@property` cannot share a
  name with a field or `InitVar` in the same dataclass body — the property
  object silently becomes that field's *default value*. A base/subclass split
  works but makes `replace()` round-trip flat values back through the
  properties, which would defeat the "target= plus flat input is a conflict"
  rule. So `material` stays a real field that `__post_init__` keeps
  `== target.material`, and the ~33 `sweep.material` readers did not move. The
  other flat names have no reader outside tests.
- **Reads of a retired flat name return `None`, they do not raise.**
  `dataclasses` leaves an `InitVar`'s default as a class attribute, and
  `dataclasses.replace` needs it (`getattr(obj, name)` for every `InitVar` with
  a default), so it cannot be deleted. `sweep.tilt_deg` is therefore `None`
  rather than an `AttributeError`. This silently weakened three assertions
  (`tests/materials/test_profiles.py`, `tests/cli/test_blaze.py`); all were
  found and rewritten to read through `sweep.target`. D2's
  `DeprecationWarning` work should treat "make the read loud" as part of its
  charter.
- **`UNSET` sentinel for the footprint pair.** `crystal_width_mm=None` IS the
  explicit infinite slab, so `None` cannot also mean "not mentioned". The other
  flat inputs have no such collision and use `None`.
- **Identity payload keeps the flat spelling.** `_jsonable` walks
  `dataclasses.fields()`, which excludes `InitVar`s, so the hashed sweep payload
  lost every flat geometry key and gained a nested `target` — breaking every
  pinned `parameter_sha256` (and therefore checkpoint identity). `_identity_v1`
  now pops `target` and splices `target_flat_fields(...)` back in, the same
  legacy projection the beam and detector already get. All pinned digests are
  bit-for-bit unchanged.
- **`target_flat_fields` inverts a single default-oriented sub-layer to
  `substrate=`.** That is the definition of the sugar, it makes the projection
  lossless both ways, and it keeps the identity payload's historical spelling.
  Checked: no catalog material is misprojected — `materials.toml` has one
  `stack` entry and it has two layers.
- **Silent shims,** per the handoff's own recommendation; `config.py` still
  emits the flat spelling, so warning would fire on the repo's own calls. D2
  warns.

One test fixture changed behaviour rather than spelling: the synthetic catalog
in `tests/materials/test_material_catalog.py` had `tilt_deg` starting at `0.0`,
which `Slab` bans for emission sweeps. It used to survive because nothing built
a target until `build_cases`; it now fails at `material_sweep(...)`. The fixture
grid starts at `5.0` (9 sites) and the one assertion pinning the expanded grid
follows. That is the acceptance criterion "an invalid target raises at
construction" doing its job.

Evidence: `pyrite-dev test` green (3099 passed, 61 skipped), `verify` green,
`docs` builds, lint and typecheck clean.

## Slice D2 — what landed

`Stack.on_substrate(material, thickness_ang, substrate, substrate_thickness_ang)`
is the sanctioned one-layer spelling, and `target_from_flat` routes the sugar
through it, so the "substrate = one `Layer` beneath the film" identity has a
single construction site. Two duplicated defaults got names in the same pass:
`DEFAULT_FOOTPRINT` (was an identical `default_factory` lambda on `Slab` and
`Stack`) and `DEFAULT_SUBSTRATE_THICKNESS_ANG` (was a bare `5e6` in four places).
`target_from_flat` also hoists its two flat-only rules above the variant branch,
so `substrate`/`stack` exclusion and grooves-forbid-a-stack each have one site.

`mosaic` is target state: a field on `Slab`/`Stack`, `Sweep.mosaic` is an alias
`InitVar`, and `build_cases` reads `target.mosaic`. `mosaic_route` /
`mosaic_nodes` stay on `Sweep` for `refactor/scene-object-model`, and so does
`mosaic_fwhm_deg` — it is a run-level substitution for the catalog value, not
target state, and the resolution chain is unchanged. The identity digest is
bit-for-bit: `mosaic` left `dataclasses.fields(Sweep)` but rejoined the payload
through `target_flat_fields`, and `_identity_v1` serializes with
`sort_keys=True`, so the key's position never mattered.

`substrate=` / `substrate_thickness_ang=` on `Sweep` now emit a
`DeprecationWarning` naming `Stack.on_substrate`. Nothing in the repo emits them
any more: `config.material_sweep` and `config.trajectory_sweep` build the target
with `target_from_flat(...)` and pass `target=`, which is honest — the catalog
speaks the flat vocabulary and that function is its documented projection, not a
user-facing shim.

Decisions taken here rather than discovered in review:

- **Only the substrate pair warns.** The other flat geometry inputs
  (`thickness_ang`, `tilt_deg`, the footprint pair, ...) are *not* retired by
  this task — they are the sweep-template vocabulary until `scene-object-model`
  makes `target.layers[1].thickness_ang` addressable. Warning on them would fire
  on ~30 live call sites for a spelling with no replacement yet.
- **`target_replace(substrate=...)` stays silent.** That is the CLI/profile
  *override* vocabulary, not the constructor; `runs/blaze.py` clears a stack
  through it. Retiring the override keys is a CLI-surface question, and
  `substrate` reaches no CLI flag today, so there is no
  `cli-deprecations.md` row to add.
- **The catalog TOML `substrate` key stays**, per the handoff recommendation: it
  is a data vocabulary, not the object model.
- **Retired flat reads are loud.** `_RetiredFlatInput` replaces `_UnsetType` /
  `UNSET` and is the default of every flat geometry `InitVar`. It still has to be
  *readable* — `dataclasses.replace` does `getattr(obj, name)` for every `InitVar`
  with a default — but every use of the value other than `isinstance`/`is` raises
  `AttributeError` naming `sweep.target`. So `sweep.tilt_deg == 45.0` now fails
  instead of silently comparing against `None`, which is the D1 hazard that
  weakened three assertions.
- **"Not supplied" is now the sentinel alone.** An explicit `None` is a
  statement and reaches the target: `crystal_width_mm=None` IS the infinite slab,
  `substrate=None` IS a free-standing film. Consequence: `Sweep(target=...,
  substrate=None)` is now a conflict rather than a silent no-op, and
  `allow_normal_incidence=False` counts as mentioned. Nothing in the repo does
  either.

Evidence: `pyrite-dev test` green (3105 passed, 61 skipped — six new tests, no
new warnings), `verify` green, `docs` builds, lint and typecheck clean.

## Slice D — handoff (D1 landed above; D2 landed above)

### Why D splits in two

The checklist read narrowly (retire `substrate=`, add a `Stack` helper, move
`mosaic`) does not reach the acceptance check *"an invalid target raises at
construction, with a message naming the violated constraint"*. After C, geometry
is still validated only when `build_cases` calls `_target_from_sweep`, which is
the same "fails after profile resolution" timing the task exists to remove. The
decisions section already assumes the wider shape — "`Sweep.material` retires as
a deprecated forward to `target.material` in slice D" only parses if
`Sweep(target=...)` exists. **Decided: go wide, split D1/D2.** D1 lands the
surface; D2 is the sugar retirement and `mosaic`, and is small once D1 holds.

### D1 — the seam

`_target_from_sweep(sweep)` (`sweep.py`, added in C) is the whole conversion and
is the thing to invert. D1 moves the call into `Sweep.__post_init__`, stores the
result on a `target` field, and `build_cases` reads `sweep.target` instead of
constructing one. Geometry then fails at `Sweep(...)` — and at `pr.Slab(...)`
for a caller who builds the target directly.

The repo already has the exact precedent for "legacy flat inputs normalize onto
one nested object": `theta_obs_deg` / `dtheta_obs_deg` / `domega_sr` are
`InitVar` (`sweep.py:266`), consumed in `__post_init__`, normalized onto the
`detector` field, and rejected when they conflict with a supplied nested
`DetectorSpec`. Follow it exactly rather than inventing a second pattern:

- the flat geometry inputs (`material`, `thickness_ang`, `tilt_deg`,
  `tilt_azim_deg`, `crystal_width_mm`, `crystal_height_mm`, `groove_spacing_ang`,
  `substrate`, `substrate_thickness_ang`, `stack`, `allow_normal_incidence`)
  become `InitVar`;
- `target: Target | None = None` is the field, defaulted in `__post_init__` from
  whatever flat inputs were supplied;
- supplying both a `target=` and a conflicting flat input is an error, same
  wording shape as the detector conflict message.

**Trap, and the reason the InitVar route is safe:** `dataclasses.replace()` does
not preserve `InitVar` values — it re-runs `__post_init__` with their defaults,
silently dropping what the original constructor was given. That is survivable
only because the normalized result lives on a real field: `replace(sweep, ...)`
carries `detector` today and would carry `target` after D1. Any flat geometry
input that is *not* also reconstructible from `target` would be lost by the first
`replace()`, so nothing may stay flat-only.

**The one real work item is the override path.** `config.py:194` does
`replace(sweep, **overrides)` with CLI/profile-supplied keys, several of which
are geometry. Once those are `InitVar`, that call drops them. `config.py:196`
already shows the fix pattern for exactly this problem: `_BEAM_OVERRIDE_KEYS`
splits beam-addressed overrides out and re-applies them through
`beam_replace(sweep.beam, **beam_over)`. D1 adds the same split for
target-addressed keys with a `target_replace` (or `dataclasses.replace` on the
variant — note `Slab` and `Stack` are frozen, so `replace` is the natural spelling
and re-runs `__post_init__`, preserving validation). Watch the two-variant case:
an override that changes `substrate` changes which variant the target *is*, so
the helper rebuilds rather than field-replaces.

Sweep construction sites to sweep after the change: `campaign/config.py`,
`apps/anchor_figures.py`, `devtools/package_smoke.py`,
`detectors/eaglexo_response.py`, `plots/mpl/detectors.py`, plus tests.
`sweep.material` is read ~33 times across `campaign/` and `runs/` alone — keep it
readable as a property forwarding to `target.material`, so readers do not move.

### D2 — sugar retirement and mosaic

`substrate=` is not only a `Sweep` field. It is a catalog material key
(`materials.toml:1244` and `:1375`, both `substrate = "sapphire"`), validated in
`catalog.py` — allowed-keys set at `:1312`, must-reference-a-crystal-or-medium at
`:1367-1372`, "cannot define both substrate and stack" at `:1383`, dependency
refs at `:1214` — and it reaches `Sweep` through `MaterialSpec.substrate` /
`MaterialSpec.stack` (`catalog.py:250-251`) via `config.py:105-107`, `:184`,
`:238`, and `:264`. So D2 either keeps the catalog spelling and retires only the
`Sweep` pair (recommended: the TOML key is a data vocabulary, not an object
model), or it becomes a catalog migration too, which is out of this task's scope.

`Stack.on_substrate(film_material, thickness_ang, substrate, substrate_thickness_ang=5e6)`
is the helper; it is the `LayerSpec` construction that `_target_from_sweep`
already performs, given a name. Note `Stack` requires the film plus at least one
layer beneath, so the helper is the only sanctioned one-layer spelling.

`mosaic` moves onto `Target` as a field; `build_cases` then reads `target.mosaic`
where it reads `sweep.mosaic` today, keeping the resolution chain unchanged
(`sweep.mosaic_fwhm_deg` override, else `CATALOG.crystal(cp["crystal"])
.mosaic_fwhm_deg`). `mosaic_route` and `mosaic_nodes` stay on `Sweep` for
`refactor/scene-object-model`, per the RFC.

### Open question for the implementer

Whether the shims emit `DeprecationWarning` immediately. Recommendation: silent
in D1 — `config.py` and the profile surface are still *emitting* the flat
spelling at that point, so warning would fire on the repo's own calls — then warn
in D2 once the canonical spelling exists end to end, with a row in
`docs/repo-design/cli/cli-deprecations.md` if any of it surfaces in CLI or
profile vocabulary. **Resolved: took the recommendation.** D1 is silent; D2
owns the warning, and must also make a retired flat *read* loud (see the D1
deviations above — it currently returns `None`).

## Slice E — what landed

The acceptance evidence for B through D2: a cross-commit equivalence sweep
against the merge-base `64e429a` (the branch point; `main` has since moved on
with unrelated docs commits, so the merge-base is the honest baseline).

Method. A throwaway script fingerprints each expanded case list with a
streaming SHA-256 over a canonical encoding of `dict(case)` — `ndarray` by
shape plus `tobytes()`, floats by `repr(round(v, 12))`, dicts key-sorted. A
plain JSON dump was the first attempt and is not viable: 185,835 cases carrying
full energy grids is gigabytes. Digest-per-profile keeps the comparison to a
few hundred lines while still being byte-exact on every field. The script ran in
this worktree and in a throwaway detached worktree at `64e429a` (removed
afterwards); the two outputs were diffed.

Coverage. 98 `material_sweep` profiles — all 49 `CATALOG.material_keys` at both
fidelities, 185,835 cases total — plus 196 `trajectory_sweep` profiles: every
material across four variants (defaults, explicit `thickness_ang`, an
`n_tilts`/`tilt_span`/`azim_deg` span, and a grooved variant). `trajectory_sweep`
is covered because D2 rewrote it too, and it is the only caller that exercises
`groove_spacing_ang`, `crystal_width_mm/height_mm`, and
`allow_normal_incidence=True` together.

Result. **Every successfully expanded case list is byte-identical.** The 49
`spanned` variants reject identically on both sides (the azim-90 ban). The only
diffs in the whole sweep are the rejection *messages* of the 49 invalid grooved
variants, all of which reject on both sides:

- 46 materials: `grooves require ...` → `blazed grooves require tilt_azim_deg
  == 180 for every case`, a pure rewording from the `BlazedGrooves` rule.
- 3 substrate-backed profiles (`mos2-on-sapphire`, `mos2-on-sio2-si`,
  `mote2_product`): now `grooves are v1 single-slab only (no substrate/stack)`.
  Same rejection, raised earlier and for the more accurate reason — the target
  refuses grooves-plus-substrate at construction before any azimuth is
  quantized. Baseline rejected the same configuration at azimuth 180 (see
  `test_build_cases_groove_rejects_substrate`), so no configuration changed
  from accepted to rejected or the reverse.

Conclusion: the geometry consolidation is behaviour-preserving for every
profile the repo ships. Identity digests are unaffected, which the `mosaic`
splice in `_identity_v1` already covers by test.

## Slice F — what landed

`docs/adr/0008-no-arbitrary-target-geometry.md`, Accepted, plus its `index.md`
toctree and table rows. It records the non-goal, the GPU reasoning
(AdePT/Celeritas, VecGeom's surface model), MCPL export as the sanctioned
interoperability answer and why it stays a required dependency, and the named
seam (`locate`, `distance_to_boundary`, `escape_path`) with the
flat/bounded-depth/device-representable constraint. It cross-references the RFC
non-goals section by label rather than restating it; the docs build resolves
the link and is warning-free.

Consequences section ties the non-goal back to this branch's code: the closed
variant set is what lets `__post_init__` reject at construction time and lets
`Target.lower()` stay total with no dispatch protocol.

**Deferred to `main`:** the TODO reconciliation. `TODO.md` line 205, "Complex
geometry and interoperability", still reads as if arbitrary-shape support were
planned. ADR-0008 scopes that item to the interoperability half (WarpX, MCPL);
STL/STEP import as PyRITE *simulation* geometry is now explicitly out of scope.
`TODO.md` is authoritative on `main` and branch copies are disposable, so the
edit is not made here — the owner should reword that bullet on `main` and cite
ADR-0008.

## Slice G — what landed

`docs/repo_map.md`, three rows:

- `campaign/geometry.py` — added the flat-projection trio
  (`target_from_flat` / `target_flat_fields` / `target_replace`) and why it
  exists (catalog data, profile overrides, and identity digests still speak the
  flat vocabulary), `Stack.on_substrate`, the two named defaults,
  `retired_flat_input`, `mosaic` in the lowered key list, construction-time
  validity, and the ADR-0008 citation for the non-goal.
- `campaign/sweep.py` — a new bullet stating that `target` is the only geometry
  state on `Sweep`, that `build_cases` reads geometry only through
  `target.lower()`, that the flat arguments are `InitVar` aliases following the
  `detector` precedent, that a retired flat *read* raises, and that the
  substrate pair is deprecated in favour of `Stack.on_substrate`.
- `campaign/config.py` — both builders project through `target_from_flat`; added
  `geometry` to its dependency list.

Physics pages:

- `physics/materials/crystal-mosaicity.md` — both wiring lines respelled to
  `Sweep(target=Slab(…, mosaic=True), …)`, with the split stated once: `mosaic`
  is target state, `mosaic_fwhm_deg` / `mosaic_route` / `mosaic_nodes` are the
  model and numerics knobs and stay on `Sweep`.
- `physics/materials/multilayer-materials.md` — `Sweep(substrate="silicon")` →
  `Stack.on_substrate(...)`, and the status note's `substrate=None` → "a bare
  `Slab` target".

Checked and left alone: `docs/guides/` and `docs/api.md` carry no geometry
spellings, so nothing there was stale. The RFC's `sec-core-arch-nongoals`
section is the ADR's context source and stays as the proposal record; its
"Current state" inventory is deliberately historical. The stale RFC
groove/footprint claim and the `groove_spacing_ang` field comment were already
corrected earlier on this branch.

## Decisions and open questions

- **Decided:** closed variant set. No dispatch protocol, no plugin registry, no
  user-supplied geometry classes.
- **Decided:** transport untouched. A diff under `src/pyrite/montecarlo/` that
  is not an import fix is out of scope.
- **Decided (A): `Target` owns `material`.** The stated objection does not
  survive contact with the code: `Sweep.material` is `str` (`sweep.py:365`), a
  **scalar, not a sweep axis**. Sweeping material means constructing one `Sweep`
  per material via `config.material_sweep(material, ...)` (`config.py:112`), so
  there is no collision with a primary sweep axis. `Stack` already owns per-layer
  materials through `LayerSpec.material`, and lowering needs material and
  thickness together (`stack_layers(cp["composition"], thickness, stack)`).
  Film material on `Sweep` while substrate material sits on the stack is exactly
  the asymmetry this task exists to remove.
  - Constraint on B: `Target.lower()` must **receive** resolved crystal params
    rather than call `crystal_params` itself. `crystal_params(material,
    n_families)` needs `n_families`, and `beam_uvw` / `surface_hkl` take a
    `Sweep`-level override — those are numerics and orientation, not geometry.
    A `Target` that reaches into `CATALOG` at construction would also pull the
    campaign layer into materials at the wrong point.
  - `Sweep.material` retires as a deprecated forward to `target.material` in
    slice D, not in B/C — it is required-with-no-default and very widely read.
- **Decided (A): `tilt_deg` / `tilt_azim_deg` go on `Target`,** per the RFC
  sketch and the `"target.tilt_deg"` dotted axis at RFC line 273. The groove
  coupling to `theta_obs` does not pull them back onto the beam relationship; it
  splits constraint 7 out as a cross-object check (see inventory above).
- **Decided (A): fields stay `ScalarOrSeq` in this task.** This is the sequencing
  point that the RFC's target-state sketch hides. RFC Change 2 removes
  `ScalarOrSeq` from the object model by addressing axes as dotted paths — but
  Change 2 is `refactor/scene-object-model`, sequencing step **4**, and this task
  is step **2**. Today `thickness_ang`, `tilt_deg`, `tilt_azim_deg`,
  `crystal_width_mm`, `crystal_height_mm` are all `ScalarOrSeq` and form a
  Cartesian product in `build_cases`. If `Target` took scalars now, every
  existing multi-value sweep would become inexpressible and acceptance check
  "every existing sweep configuration is expressible" would fail immediately.
  So `Target` is a **sweep template** in this task, holding `ScalarOrSeq`, and
  `scene-object-model` narrows it to scalars when the axis machinery lands.
  The RFC sketch is the post-Change-2 end state, not this task's end state.
- **Decided (A): `footprint` on both `Slab` and `Stack`; `entrance_face` on
  `Slab` only.** Nothing forbids a footprint with a stack, and a footprint with
  grooves is explicitly allowed and is the default (see Correction above).
  Grooves *do* forbid a stack, so putting `entrance_face` on `Slab` alone turns
  constraint 8 into a type constraint and deletes a runtime check. With only two
  target classes this duplicates nothing.

## Delegation slices and required skills

- A → `lead-task`; the material-ownership question shapes every later slice.
  Not `one-shot`.
- B, C → `implement-task`; `scientific-library`. Review together — C is what
  proves B is complete.
- D → `implement-task`; `cli-ui-ux` if the deprecation surfaces in CLI/profile
  vocabulary.
- E → `implement-task`; `catalog-golden` + `regression`.
- F → `implement-task-lite`; `documentation-maintenance`. `one-shot`: the
  decision is already made and reasoned in the RFC; this slice transcribes it
  into an ADR.
- G → `implement-task-lite`; `documentation-maintenance` + `repo-orientation`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
```

- Every existing sweep configuration is expressible and expands to an identical
  case list.
- All geometry validation lives in `Target`; `build_cases` contains no geometry
  conditionals.
- **No change to any transport code path, kernel, or golden spectrum.**
- An invalid target raises at construction, with a message naming the violated
  constraint.
- The non-goal ADR exists, is listed in `docs/adr/index.md`, and states the GPU
  reasoning.

## Related

- `refactor/scene-object-model` — consumes `Target` as the thing a dotted sweep
  axis addresses.
- `refactor/detector-scorer` — RFC sequencing step 3, gated on this task.
- TODO "Long-term plans → Complex geometry and interoperability" describes
  arbitrary shapes and STL/STEP import. The RFC now records that as a non-goal
  for the core object model; slice F should reconcile the two statements rather
  than leave them contradicting each other.
