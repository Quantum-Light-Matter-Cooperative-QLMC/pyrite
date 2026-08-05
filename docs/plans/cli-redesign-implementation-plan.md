# Implementation plan: CLI redesign & repo restructure

- **Status:** Active — tracks three Accepted RFCs
- **Created:** 2026-08-01
- **Spans:** [`../cli-redesign-rfc.md`](../cli-redesign-rfc.md) (surface),
  [`../cli-artifact-model-rfc.md`](../cli-artifact-model-rfc.md) (data model),
  [`../package-structure-rfc.md`](../package-structure-rfc.md) (structure).
- **Decisions:** ADRs [0002](../adr/0002-cli-surface-redesign.md) /
  [0003](../adr/0003-content-addressed-artifact-model.md) /
  [0004](../adr/0004-package-and-repository-structure.md) /
  [0005](../adr/0005-energy-grid-schema-decisions.md).

The RFCs each hold the *why* for one layer; each only sees its own slice. This
plan is the *order* — the linearized cross-cutting sequence with gates and
acceptance criteria, so an agent starts in the right place and never lands a
slice before its prerequisite. Ephemeral: update it as slices land, retire it
when the last one does. Each slice is a `tasks/<branch>/` via the normal
triage → dispatch flow; `TODO.md` stays authoritative for status.

## Sequence

| # | Slice | Source | Gate |
|---|---|---|---|
| 0 | docs/TODO reorg | pkg P4/P5 | ✅ landed 2026-08-01 |
| 1 | command-home → `src/cxr_mc/cli/commands/` | pkg P1 | ✅ landed 2026-08-04 |
| 2 | energy-grid module rename | pkg P2 | ✅ landed 2026-08-04 |
| 3 | noun→verb, remote-as-modifier, `-o` contract | redesign D1–D3 | ✅ landed 2026-08-05 |
| 4 | verb collapse **+** recompute/prune module fold | redesign D4 + pkg P3 | ✅ landed 2026-08-05 (ahead of 3, see below) |
| 5 | vocab controls + deprecation rollout | redesign D5–D7 | ✅ landed 2026-08-05; closure audit completed after 3 |
| 6 | content-addressed store, lockfile, gc | artifact RFC (phase 5) | implemented 2026-08-05 on `feature/cli-artifact-model`; unmerged, awaiting slice 5 landing on `main` |

Slices 4 and 5 are gated on 3 in this linearization, but the redesign RFC's own
migration plan ([§4](../cli-redesign-rfc.md#4-migration-plan)) sequences the
*additive* vocab work — D4/D5 canonical verbs and flags, plus the D7
deprecation harness — as phase 1, ahead of the D1–D3 noun reshuffle, precisely
because it changes no existing spelling. Slice 4 and the D7 harness landed on
that basis, out of the order below. What genuinely needs slice 3 first is the
`-o/--output` contract (its row above) and D6's `--wait`/`--detach`, which
depend on the D2b job noun.

## Resolve first (blocking open questions)

Do not start the dependent slice until its decision is pinned:

1. ~~**P1 destination shape**~~ — resolved in slice 1 as `cli/commands/`: the
   existing `cli/` held 16 modules and P1 added 13, past the flat threshold.
2. ~~**`profile` vs `context` naming**~~ — redesign §6 Q3, blocked slice 3
   (D2c precedence chain). **Resolved (2026-08-02), landed 2026-08-04:**
   `profile` is fixed as the
   campaign term; no `context` rename. The overload was fidelity (`full`/`survey`)
   squatting on "profile" — moved onto its own `fidelity` namespace in branch
   `refactor/fidelity-namespace`. See `tasks/refactor/fidelity-namespace/README.md`
   for the full five-sense taxonomy (A fidelity renamed; B campaign keeps
   `profile`; C performance-profile, D named_profile/dataset-identity, E
   line-shape all distinct and kept).

## Slices

### Slice 1 — command-home consolidation (pkg P1)

Every `cxr` subcommand's argparse/Click wiring lives under `src/cxr_mc/cli/`;
pure domain logic stays in its domain module. Move the loose command modules;
split fused ones (`scan.py`, `blaze.py`) — CLI wiring → `cli/`, driver logic
stays. Do **not** move non-command domain modules; public imports unchanged.

**Accept:** `docs/cli-reference.md` freeze test green; export-freeze guards
green; no public import path changed; `docs/repo_map.md` regenerated;
lint/typecheck/tests green. Pure internal refactor, zero surface change.

### Slice 2 — energy-grid module rename (pkg P2)

Collapse `line_grid/` + `_energy_grid.py` into one `energy_grid/` package to
match the surface noun; align `tests/test_line_grid_*.py`.

**Accept:** mechanical rename, no behavior change; imports updated;
`docs/repo_map.md` regenerated; tests green.

### Slice 3 — surface reshuffle (redesign D1–D3)

Noun→verb ordering, remote-as-modifier, and the `-o/--output` contract (`json`
the sole automation-bound format). Old spellings kept behind D7 deprecation
shims.

**Accept:** old spellings warn+redirect (D7); `docs/cli-reference.md`
regenerated and its freeze test updated deliberately; `json` output contract
test; deprecation warnings emitted; tests green.

### Slice 4 — verb collapse + module fold (redesign D4 + pkg P3)

Land the canonical lifecycle verbs (`add`/`verify`/`gc`/`recompute`/`rm`) and
retire `rebrem.py`/`reline.py`/`prune.py` into the checkpoint
recompute/cleanup modules rather than leaving parallel entry points.

**Accept:** retired entry points alias+warn per D7; no orphaned shim modules;
tests green.

### Slice 5 — vocab controls + deprecation rollout (redesign D5–D7)

Controlled flag/verb vocabularies; deprecation policy with grace windows;
drop compatibility shims on the published schedule.

**Accept:** deprecation warnings carry removal-version metadata; controlled
vocab enforced; docs updated; tests green.

### Slice 6 — content-addressed artifact model (artifact RFC, phase 5)

Deepest, last. Content-addressed immutable store; profile becomes the sole
mutable ref; `energy-grid apply` stops mutating `standard`; campaign lockfile
in the first cut; `gc` grace window; hash-input tuple frozen by test.

**Accept:** dedup + hash-diff staleness/remote-sync; `apply` no longer mutates
`standard` (deprecation shim for old behavior); lockfile emitted per run;
hash-input freeze test; `gc` + grace window; migration shims per D7; tests green.
