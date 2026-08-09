# RFC: package & repository structure

> **2026-08-09 update:** ADR-0006 supersedes P4/P5's document-location
> decision. Tracked agent work now lives only under `agentdocs/`; `docs/` is
> reserved for durable project documentation.

- **Status:** Accepted — 2026-08-01 (P4/P5 landed; P1–P3 scheduled)
- **Author:** Alex Amador
- **Created:** 2026-08-01
- **Companion:** `docs/cli-redesign-rfc.md` (surface) and its sub-RFC
  `docs/cli-artifact-model-rfc.md` (data model). This RFC is the **third layer**:
  where code and docs *live*, independent of what the CLI spells or how derived
  data is stored.
- **Blocks:** `cli-redesign-rfc.md` §4 phase 0 — the command-home consolidation
  (P1) is a prerequisite for the noun→verb reshuffle (D1), so it lands first.
- **Authoritative inventory:** `docs/repo_map.md` (source), this repo's tree.

An RFC to settle **structure**, not surface or physics. The CLI-redesign RFCs
restructure what commands are *spelled* and how derived data is *stored*; this
one restructures where the implementing code and the surrounding planning docs
*sit*. Reordering a command surface (D1) on top of a split, inconsistent module
layout means editing two homes at once and shipping a bigger, riskier diff — so
the layout is settled first, as a pure refactor with zero surface change.

---

## 1. Why

The physics packages (`materials/`, `montecarlo/`, `results/`, `plots/`,
`detectors/`) are well-factored: clear leaf→driver DAG, frozen-export guards,
narrow public APIs. **Those are not in scope and should not move.** The drift is
concentrated in two places:

1. **The command layer** — where a `cxr` subcommand's implementation lives is
   unpredictable, and one concept has three module names.
2. **The planning/docs surface** — durable design docs, ephemeral plans, and
   several parallel TODO/memory trees are interleaved with no stated rule.

Both are cheap to fix mechanically and expensive to leave in place while the CLI
redesign lands on top of them.

---

## 2. Problems (with evidence)

### 2.1 Command implementations have two homes

Command implementations are split between loose `src/cxr_mc/*.py` modules and the
`src/cxr_mc/cli/*.py` package, with no rule predicting which:

| Home | Command modules |
|---|---|
| `src/cxr_mc/cli/` (thin, canonical) | `profile.py`, `material.py`, `energy_grid.py`, `checkpoint.py`, `performance.py`, `app.py`, `completion.py` |
| `src/cxr_mc/` root (loose) | `scan.py`, `blaze.py`, `analyze.py`, `archive.py`, `check.py`, `check_config.py`, `export.py`, `prune.py`, `rebrem.py`, `reline.py`, `slim.py`, `remote.py`, `viewer.py` |

`cli/__init__.py`'s `LazyGroup` dispatches into **both**. Some loose modules are
pure CLI wiring (`export.py`, `slim.py`, `prune.py`); others fuse CLI wiring with
domain logic in one file (`scan.py` carries `main`/`add_subparser` *and* the
headless sweep driver; `blaze.py` likewise). There is no single answer to "where
does a command live," which directly undercuts `cli-redesign-rfc.md` D1: a
noun→verb reshuffle would touch two directories and two conventions at once.

### 2.2 One concept, three module names

The energy-grid feature is spelled three ways in the tree:
`src/cxr_mc/line_grid/` (package), `src/cxr_mc/_energy_grid.py` (module),
`src/cxr_mc/cli/energy_grid.py` (CLI), plus `tests/test_line_grid_*.py` and the
`energy-grid` surface noun. This is the module-level version of exactly the drift
`cli-redesign-rfc.md` D4/D5 fight at the flag level (`energies`/`tilts`).

### 2.3 Overlapping recompute / prune modules

- **Recompute:** `rebrem.py` + `reline.py` + `recompute_defaults.py` + the
  `cxr checkpoint recompute {brem,line}` path all serve one operation.
- **Prune/gc:** `prune.py` + `checkpoint_cleanup.py` + `cli/performance.py`
  pruning + `remote` pruning all serve reclamation.

`cli-redesign-rfc.md` D4 collapses the *verbs* (`recompute`, `rm`, `gc`). Unless
the *modules* fold the same way, the deprecation shims outlive the redesign.

### 2.4 Docs mix durable reference with ephemeral plans

`docs/` (69 tracked files) interleaves durable reference (`tilt-convention.md`,
`channeling-radiation-physics.md`, `physics-validation-ledger.md`) with
single-use planning artifacts (`cli-energy-grid-sweep-rework-plan.md`,
`scan-config-cli-plan.md`, `case-compare-tab-plan.md`,
`viewer-camera-animation-plan.md`, `gpu-perf-fixes-handoff.md`). There is no
signal for which docs are load-bearing forever versus consumed-and-discarded.
**(Fixed by P4, 2026-08-01: the plans/handoffs moved to `docs/plans/`;
`cli-energy-grid-sweep-rework-plan.md` stayed in `docs/` root as a cited decision
record — ADR-0005.)**

### 2.5 No ADR / decision numbering

RFCs are ad-hoc `docs/*-rfc.md` (this file included). There is no durable,
greppable decision log; "why did we decide X" is scattered across RFCs, plan
files, and commit messages. The RFCs already carry Status/Author/Created — one
step from ADRs.

### 2.6 Multiple planning/memory surfaces

Planning and memory are spread across, at last count:

- Three root TODO files: `TODO.md`, `TODO_CLI.md`, `TODO_UI.md`. **(Fixed by P5:
  folded into `TODO.md` `## CLI backlog` / `## UI backlog`, 2026-08-01.)**
- `agentdocs/{plans,reports,specs}` (untracked local scratch; renamed from
  `claudedocs/` for agent neutrality, 2026-08-01).
- `tasks/{feature,fix}/` (tracked; 10 files) with its own `tasks/README.md`.
- `.remember/` (tracked; 1 file).

`AGENTS.md` declares `TODO.md` **authoritative on `main`** and protects it with a
`merge=ours` driver — but `TODO_CLI.md` and `TODO_UI.md` were sibling files that
bypassed that driver, so they carried real merge-conflict risk the canonical file
was designed to avoid. P5 removed them.

---

## 3. Proposals

### P1 — One home for command code (prerequisite for redesign D1)

Rule: **every `cxr` subcommand's argparse/Click wiring lives under
`src/cxr_mc/cli/`; pure domain logic stays in a domain module.** This is the same
principle `AGENTS.md` already states for marimo apps ("keep reusable logic in
`src/cxr_mc/`; keep apps thin") — applied to the CLI.

- Move the loose command modules (`export.py`, `slim.py`, `prune.py`,
  `archive.py`, `rebrem.py`, `reline.py`, `check_config.py`, `remote.py` facade,
  and the app launchers `analyze.py`/`check.py`/`viewer.py`) into `cli/` (app
  launchers → `cli/app/`).
- For fused modules (`scan.py`, `blaze.py`), split: CLI wiring → `cli/`, headless
  driver logic stays in a domain module (`scan.py` keeps `run`/`run_sweep`
  plumbing; `main`/`add_subparser` move). Preserve the `_entry/scan` box shim.
- **Do not move** domain modules that are not commands: `run.py`, `config.py`,
  `profiles.py`, `sweep.py`, `beam_metrics.py`, `_checkpoint_store.py`, etc.
  stay put; their public imports are unchanged.
- Guarded by the existing `docs/cli-reference.md` freeze test plus the
  export-freeze guards — a pure internal move must leave both green.

### P2 — One name for the energy-grid concept

Pick the surface term `energy-grid` and rename the implementation to match:
collapse `line_grid/` + `_energy_grid.py` into one package (`energy_grid/`),
align `tests/test_line_grid_*.py`. Update `docs/repo_map.md`. Mechanical rename;
no behavior change.

### P3 — Fold the recompute / prune module sprawl

Track with `cli-redesign-rfc.md` D4's verb collapse: as `recompute`/`rm`/`gc`
become the canonical verbs, retire `rebrem.py`/`reline.py`/`prune.py` into the
checkpoint recompute/cleanup modules rather than leaving them as parallel entry
points behind shims. Sequenced with the redesign, not ahead of it.

### P4 — Split docs by lifetime; adopt ADRs (location superseded)

**Landed 2026-08-01; ephemeral-plan location superseded by ADR-0006.**

- `docs/` root = durable reference + index only.
- `docs/plans/` = tracked but ephemeral (`*-plan.md`, `*-handoff.md`); moved the
  existing plan/handoff files there (keeps git history, unlike relocating to the
  untracked scratch dir). Exception: `cli-energy-grid-sweep-rework-plan.md` stays
  in `docs/` root — shipped code cites it as a decision source (ADR-0005), so it
  is reference material, not an ephemeral plan.
- `docs/adr/NNNN-title.md` = numbered decision log (MADR-lite). When an RFC is
  accepted, add a short ADR stub that records the decision and links the RFC;
  the RFC stays as the long-form rationale. Seeded with ADR-0001 (adopt ADRs) and
  Proposed stubs for the three RFCs (0002–0004) plus ADR-0005 (energy-grid).
- Both `docs/plans/` and `docs/adr/` are excluded from the Sphinx site
  (`conf.py`) — dev-facing, not published reference.

### P5 — Consolidate planning/memory surfaces (superseded)

**Landed 2026-08-01; superseded by ADR-0006.**

- Folded `TODO_CLI.md` and `TODO_UI.md` into `TODO.md` as `## CLI backlog` /
  `## UI backlog` sections, so the single `merge=ours`-protected file is the only
  backlog on `main`. (Rejected alternative: extend the merge driver to cover all
  three — one file is simpler and matches the stated "TODO.md is authoritative"
  contract.)
- Left `tasks/` (branch-scoped, per `tasks/README.md`) and `.remember/` as-is —
  distinct, documented roles. Renamed the untracked agent scratch
  `claudedocs/` → `agentdocs/` (agent-neutral; non-Claude agents write there
  too). The only structural fix needed was the TODO fan-out.

---

## 4. Non-goals

- **Physics package layout** (`materials/`, `montecarlo/`, `results/`, `plots/`,
  `detectors/`) — out of scope; it is already well-factored.
- **CLI surface** (noun/verb/flag) — owned by `cli-redesign-rfc.md`.
- **Data model** (content-addressed store) — owned by `cli-artifact-model-rfc.md`.
- **Public import paths** — no domain-module public API moves (frozen-export
  guards must stay green); only command-wiring internals relocate.

---

## 5. Sequencing

1. **P1 command-home consolidation** — pure internal refactor, zero surface
   change. Lands **before** `cli-redesign-rfc.md` D1 (it is that RFC's phase 0).
2. **P2 energy-grid rename** — mechanical; can land alongside P1 or independently.
3. **P4 docs split + ADR dir** + **P5 TODO consolidation** — docs-only,
   **landed 2026-08-01**; lowest risk.
4. **P3 recompute/prune fold** — sequenced *with* `cli-redesign-rfc.md` D4, not
   ahead of it (it depends on the canonical verbs existing).

Each source-touching step regenerates `docs/repo_map.md`; P1 additionally
regenerates `docs/cli-reference.md` and must leave its freeze test green.

---

## 6. Open questions

1. **P1 destination shape** — one flat `cli/` (many modules) or `cli/commands/`
   sub-package to keep dispatcher plumbing (`_core`, `_dashboard`, `json`)
   separate from command modules? (Recommend the latter once `cli/` exceeds ~20
   modules.)
2. **P4 ephemeral home** — ~~`docs/plans/` vs untracked scratch?~~ **Resolved:
   `docs/plans/` (tracked, keeps history), landed 2026-08-01.**
3. **P5 TODO fold vs multi-file driver** — ~~collapse or teach the driver about
   all three?~~ **Resolved: collapsed to one `TODO.md`, landed 2026-08-01.**

---

## References

- gh manual — command package layout: <https://cli.github.com/manual/>
- MADR (Markdown Architectural Decision Records): <https://adr.github.io/madr/>
- clig.dev (CLI guidelines): <https://clig.dev/>
- `docs/repo_map.md` — current source inventory.
- `AGENTS.md` — TODO.md authority + `merge=ours` driver; thin-apps rule.
