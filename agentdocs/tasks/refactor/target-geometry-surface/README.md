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
`tilt_azim_deg == 180`, `0 < tilt_deg < 90`, `theta_obs == 90`, no stack, and no
finite footprint.

In scope: one `Target` object holding a **closed, named set of variants**, owning
its own validity, lowering to exactly today's case keys.

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

## Checklist

- [ ] A — Inventory every geometry constraint currently enforced, and where.
      Include the groove conjunction, the footprint pairing rule, and the
      `substrate`/`stack` exclusion. Each becomes a `__post_init__` check with
      one error vocabulary.
- [ ] B — Land `Slab`, `Stack`, `Layer`, `Footprint`, `BlazedGrooves` with
      construction-time validation and a `lower()` producing today's case keys.
- [ ] C — Route `build_cases` through `Target.lower()`. `build_cases` must end
      with **no geometry conditionals**.
- [ ] D — `substrate=` becomes a `Stack` constructor helper; retire the parallel
      field pair behind a deprecated shim under the existing D7 harness.
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

## Decisions and open questions

- **Decided:** closed variant set. No dispatch protocol, no plugin registry, no
  user-supplied geometry classes.
- **Decided:** transport untouched. A diff under `src/pyrite/montecarlo/` that
  is not an import fix is out of scope.
- **Open:** does `Target` own `material`, or does material stay a sweep-level
  concern? `Stack` clearly owns per-layer materials; a `Slab` owning one
  material is symmetric but collides with `Sweep.material` being the primary
  sweep axis today. Resolve in slice A — it constrains
  `refactor/scene-object-model`.
- **Open:** does `tilt_deg` / `tilt_azim_deg` belong to `Target` or to the
  beam-target relative geometry? The RFC sketch puts them on `Target`. Confirm
  against how `theta_obs` and the groove precondition couple them.
- **Open:** whether the entrance-face and footprint slots are `None`-able fields
  on `Slab` only, or shared by `Stack`. Today grooves forbid a stack; encoding
  that as a type constraint is cleaner than a runtime check, if it does not
  duplicate the class list.

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
