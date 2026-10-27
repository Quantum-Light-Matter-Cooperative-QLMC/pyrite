# 0021 — Named Physics and Numerics presets

- **Status:** Accepted
- **Date:** 2026-10-23
- **Amends:** ADR-0019 CLI configuration ownership (Physics and Numerics rows)
- **Context:** [Reusable Physics and Numerics presets, #663](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/663)

## Context

ADR-0019 kept physics and numerics settings as profile fields with "no second
preset database". Sharing one physics configuration across campaigns therefore
means copying fields between profiles, and there is no place to publish a
documented reference configuration. Beams and detectors already show the
pattern that works here: a named catalog object, layered bundled/user storage,
and a profile reference that the catalog resolves to values before any
consumer sees it.

## Decision

Physics and Numerics configurations may be named catalog objects. They are
data-only configurations of existing, supported model selectors; they are not
a model registry and add no runtime dispatch.

| Object | Catalog table | Profile reference | Owned keys |
| --- | --- | --- | --- |
| Physics preset | `[physics.NAME]` (`physics/NAME.toml`) | `physics = "NAME"` | `emission` plus every profile key that is a `Physics` field |
| Numerics preset | `[numerics.NAME]` (`numerics/NAME.toml`) | `numerics = "NAME"` | Scalar trial counts and the profile numerics keys that are not `Physics` fields, plus `precision` and `line_grid_policy` tables |

Both tables follow the existing object rules: one file per object, file stem
is the name (`[A-Za-z0-9][A-Za-z0-9._-]*`), names are reserved
case-insensitively across the bundled and user layers, and bundled objects are
read-only. A preset may be partial; an omitted key keeps its built-in default.
`description` is display metadata only. A preset is never a sweep: trial
counts are single values, and grids, material membership and beam/detector
references stay in the profile. Trial counts and `precision` form one sampling
policy: a profile that sets any of them takes none of them from its numerics
preset, so fixed counts and adaptive precision never mix.

**Resolution.** One path, lowest to highest precedence:

1. built-in defaults of the typed object;
2. fidelity preset (sampling counts only, unchanged);
3. the referenced named preset;
4. explicit profile fields (existing inline keys, retained as overrides);
5. invocation overrides.

The catalog parser applies steps 3–4 and validates the merged values through
`Physics`/`Numerics` for every profile, so an invalid combination, unknown
reference, or unknown key is rejected before any write or run. Every inspected
field reports its source: `built-in`, `fidelity`, `preset NAME`, `profile`, or
`run`; preset listings report whether NAME is bundled or user-owned.

**Legacy profiles.** A profile without a reference resolves exactly as before;
nothing migrates implicitly. Attaching a preset to a profile that still has
inline physics fields keeps them as visible profile overrides. Explicit
adoption (`pyrite physics preset adopt PROFILE NAME`) creates a user preset
from the profile's inline fields, attaches it, and removes those fields, so
the effective values are unchanged by construction; `--dry-run` shows the TOML
diff.

**Identity.** Names and descriptions are metadata. The catalog hands consumers
only resolved values, exactly as it does for named beams, so dataset, case and
checkpoint identities depend on effective values and resolved model/data
markers, never on the preset name. Renaming a preset, adding an unused preset,
or moving inline fields into an equivalent preset leaves identities unchanged;
a scientifically meaningful preset edit changes the effective values of every
referencing profile and therefore their identities.

**Immutability and provenance.** Bundled presets are immutable from the
ordinary CLI and are cloned with `create NAME --from SOURCE`. User presets are
editable; editing affects future resolutions only. A resolved configuration is
the frozen `Physics`/`Numerics` values recorded with the run; later catalog
edits cannot reach it. Profile runs record each referenced preset's name and
SHA-256 definition digest as an unhashed `presets` label beside the dataset
identity and in the case manifest. Remote jobs ship user preset files with the
rest of the user catalog layer, and the existing submitter/host dataset-digest
comparison covers their resolution. Strict refusal of `auto` model fallbacks
remains open in #663: fallbacks are reported by warnings of mixed categories
across campaign, transport and table code and need one audited category first.

**Commands.** `pyrite physics preset {list,show,create,set,reset,validate,
rename,delete,adopt}` (and the numerics equivalent) manage the objects; the
existing `pyrite physics|numerics {show,set,reset,validate} PROFILE` commands
keep their meaning. `pyrite profile set PROFILE --physics NAME` attaches and
`pyrite profile remove PROFILE --physics` detaches. Delete refuses while a
profile references the preset; rename updates references atomically. All
logic lives in a CLI-independent service in `pyrite.campaign`.

**Python API.** `pyrite.config.load_physics(NAME)` (and `load_numerics` in phase 2)
returns the existing frozen typed object built from the same catalog
resolution and validation, so API users can start from a named configuration
and adjust it with `dataclasses.replace`. Direct construction stays available;
there is no competing `Physics`/`Numerics` hierarchy.

**Reference presets.** Bundled presets restate supported configurations and
document intended use, required tables, and limitations. A preset is described
as validated only when its exact combination has ledger evidence.

## Boundaries

- ADR-0017 (#485) still owns process registration, adapters, backend
  capabilities, model applicability and validation records. Presets reference
  its selectors and add none.
- ADR-0008 is unchanged. Target and filter presets (#663 phase 4) need their
  own design, coordinated with ADR-0016/#401.
- No arbitrary Python imports, plugins, or callbacks enter configuration.

## Consequences

The ADR-0019 table rows for Physics and Numerics now read "existing profile
fields, optionally over a referenced named preset". No numerical default, saved
profile key, or checkpoint identity changes for existing catalogs.
