# 0019 — CLI configuration ownership

- **Status:** Accepted
- **Date:** 2026-10-17
- **Amends:** ADR-0010 command organization
- **Context:** [CLI cleanup, #551](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/551)

## Decision

Keep Click, lazy command loading, noninteractive workflows, and the existing
catalog resolver. Configuration has one owner:

| Object | Owns | Persistence in this migration |
| --- | --- | --- |
| Material | Composition, density, lattice and intrinsic response data | Existing materials/media/crystals |
| Target | Thickness, orientation, layers, placement | Existing target API; legacy campaign geometry remains in profile grids |
| Beam / Detector | Source distribution / observation geometry and response | Existing named catalog objects |
| Physics | Interaction models, emission selection and physical cutoffs | Existing profile fields; no second preset database |
| Numerics | Trial counts, integration, reflection/mosaic sampling, precision and photon grids | Existing profile fields and policy sections |
| Scene | One resolved beam + target + detector + physics + numerics setup | Public scalar API; resolved from a profile only when scalar |
| Sweep | Base scene plus typed variation axes | Public Sweep API; existing profile grids retain their meaning |
| Profile | Named campaign ranges, object references, material membership and run defaults | Existing catalog profile is the sole editable campaign record |

Root `physics` and `numerics` commands take an explicit PROFILE argument. They
are views/edits of that campaign, not independently named objects. Do not add
standalone scene/target/sweep stores until there is a versioned serialization
and reference-migration design. Profile inspection owns effective ranges and
their material-specific overrides; material inspection owns intrinsic data.
Legacy material recipes may still contain cut metadata used by the resolver;
this change does not reinterpret or erase it.

Detector creation exposes `scalar`, `planar`, and `clone`. Pixels, scoring,
response and acquisition are capabilities of the planar detector, not another
competing geometry type. Both interfaces use the same domain constructors and
validators. Material authoring distinguishes crystal and amorphous medium
requirements and reuses `materials.authoring`.

Remote-only lifecycle operations belong at `remote job`; GDF file operations
belong at `beam gdf`. Numerical energy-grid derivation/inspection belongs under
`numerics energy-grid`; `pyrite-dev` retains artifact maintenance. Ordinary
runs keep automatic grids and do not require specialist derivation.

Use `-o/--output table|json` on ordinary commands; profile summaries and ephemeral
runs additionally retain `wide`, where it has distinct unwrapped/compact output.
Aligned tables are human output, the existing
versioned JSON envelopes remain automation contracts. Do not rename the output
flag merely to match a design sketch. Help describes syntax and side effects;
`config keys`, `list [PREFIX]`, `get KEY`, and `explain KEY` expose configuration
metadata and effective provenance. Internal tuning environment variables need
not become persistent settings.

Renamed command paths keep hidden compatibility routes for two minor releases,
registered in the existing deprecation registry and warning on stderr. This
cohort starts at 0.15.0 and is removable at 0.17.0; the package version does not
change here. Where the syntax is retained but reorganized (typed creation), a
compatibility dispatcher warns and calls the same service. An obsolete human
format value (`wide`) may be rejected as a usage error once its equivalent
`table` replacement is documented; JSON data is not affected. Changed material
JSON content receives a distinct schema name; the old profile-resolved payload
remains available through explicit range inspection during migration.

## Consequences

No numerical defaults, saved profile keys, catalog precedence, checkpoint
identity or run behavior change. CLI names describe ownership even while the
storage schema retains legacy keys. Create/clone/patch/validate/diff/save logic
belongs below Click, with atomic persistence and no writes on validation error.

A future optional detector TUI consumes the same services, edits an in-memory
draft, previews its TOML diff, and saves only on explicit action. Plotting stays
in marimo. Broader editors depend on demonstrated usability, not this ADR.

Representative contract workflows are type-specific detector/material creation,
cloning a bundled detector, inspecting profile ranges for a MoS2 campaign,
editing that profile's physics/numerics, previewing typed Sweep axes through
the API, and exporting the selected catalog to reproduce a CLI run. Commands
and tests, rather than illustrative future syntax, document availability.
