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
- [ ] C — Route `build_cases` through `Target.lower()`. `build_cases` must end
      with **no geometry conditionals**.
- [ ] D — `substrate=` becomes a `Stack` constructor helper; retire the parallel
      field pair behind a deprecated shim under the existing D7 harness. Move
      `mosaic` onto `Target` in the same slice, leaving `mosaic_route` /
      `mosaic_nodes` where they are for `refactor/scene-object-model`.
- [ ] E — Equivalence sweep: every existing catalog profile expands to an
      identical case list.
- [ ] F — Write the ADR recording the arbitrary-geometry **non-goal**. The RFC
      states this is a decision in its own right and should be recorded even
      though it produces no code. Include the narrow future seam
      (`locate`, `distance_to_boundary`, `escape_path`) and the constraint that
      any future implementation be a flat, bounded-depth, device-representable
      region table — never a polymorphic object graph.
- [ ] G — Update `docs/repo_map.md` ownership rows and the geometry-facing
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
