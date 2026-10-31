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
