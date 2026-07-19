# Notebook Usability and Readability Implementation Plan

> **For implementing agents:** Use the `frontend-design`, `notebook-workflow`,
> `repo-orientation`, and `regression-testing` skills. Keep notebook applications
> thin, preserve Marimo cell-boundary rules, and complete one task at a time.

**Goal:** Make the scan, analysis, and validation applications feel like one
coherent scientific instrument: safer to operate, faster to understand, and
clearer when exported or shown without surrounding context. Improve the remaining
legacy validation notebook as provenance, not as a fourth application.

**Architecture:** Add a notebook-only presentation layer for shared visual tokens,
context/status components, and responsive layout helpers. Reorganize each Marimo
application around its user task while leaving reusable science, selection, and
plot-data logic in `src/cxr_mc/`. Preserve lazy rendering and existing runtime
workarounds. Treat the legacy `.ipynb`/MyST pair as an output-free derivation
artifact.

**Visual direction:** A restrained synchrotron workbench: blue photographic
darkroom surfaces, detector-cyan selection, tungsten-amber computation cues, and a
persistent beamline context rail. Avoid generic dashboard cards, gradients, glow,
and decorative status graphics.

**Tech stack:** Python 3.14, Marimo 0.23, Altair/Vega-Lite, Matplotlib, AnyWidget,
CSS, pytest, Jupytext/MyST, uv.

## Baseline Audit

- `notebooks/scan_app.py` is small and understandable, but changing material can
  flow directly into the expensive sweep cell. Case cost, resume state, output
  path, and completion state are mostly console text.
- `notebooks/analysis_app.py` contains eight equal-weight tabs and many repeated
  control groups. Energy, polar tilt, azimuth, thickness, and axis settings are
  recreated per view. Numeric `0 = auto` axis sentinels are compact but unclear.
- The analysis app uses `marimo.App(width="medium")`, which constrains scientific
  charts and dense control rows. It has no visual checkpoint overview before the
  tab set.
- `notebooks/validation_app.py` has the clearest action gating and explanatory
  copy, but reads as one long column. Its audit table dominates the opening, while
  evidence-producing actions appear later. A check mark currently risks implying
  scientific validation when a diagnostic merely completed successfully.
- `checks/cxr_analysis_feranchuk.ipynb` has only 11 cells. One derivation Markdown
  cell is roughly 319 source lines, making assumptions, equations, references,
  and experiment transitions hard to navigate.
- All three Marimo apps currently pass `marimo check`.

## Global Constraints

- Do not change equations, simulation kernels, selection metrics, scientific
  thresholds, detector models, normalization, or plot-data reduction.
- Keep reusable scientific and data logic in `src/cxr_mc/`. Shared notebook-only
  presentation may live in `notebooks/_design.py`; do not make the core package
  import Marimo.
- Preserve catalog-backed material, energy, thickness, and angle options.
- Preserve the analysis app's default cross-material behavior: compare all beam
  energies unless the user explicitly selects one.
- Preserve geometry labels including beam energy, `theta`, and `phi` in
  cross-material plots.
- Preserve lazy chart construction. Do not reintroduce nested `mo.ui.tabs` inside
  the detector view; Marimo issue `marimo-team/marimo#6919` can render nested-tab
  charts blank. Keep lazy accordions there.
- UI values must be read in downstream cells. Never read `.value` in the cell
  creating a Marimo UI element.
- Values exported across Marimo cells must use non-underscore names. Avoid the
  prior `_opts`/`NameError` failure mode.
- Keep `checks/cxr_analysis_feranchuk.ipynb` output-free and synchronized with its
  MyST companion. Do not add another `.ipynb` workflow.
- Do not alter unrelated worktree changes. At plan creation, unrelated edits exist
  under physics source and `docs/validation/`; leave them untouched.
- Use `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache` for every uv command.
- Use real runtime validation for analysis changes. `marimo check` alone cannot
  detect every cross-cell runtime failure; `cxr analyze --smoke` is required.

## Design System

### Color tokens

| Token | Value | Use |
|---|---:|---|
| Beamline navy | `#12202B` | Page background |
| Instrument slate | `#1D303D` | Controls and grouped surfaces |
| X-ray ice | `#DCEEF2` | Primary text and high-contrast lines |
| Detector cyan | `#72C7D5` | Selected state and interactive data focus |
| Tungsten amber | `#D8A657` | Computation, pending work, warning emphasis |
| Validation rose | `#D8787E` | Failure and destructive/mutating warnings only |

Do not communicate status through color alone. Every state needs a visible word:
`Ready`, `Running`, `Cached`, `Passed`, `Failed`, `Completed—interpret`, or
`Skipped`.

### Typography

- Display and section labels: IBM Plex Sans Condensed.
- Body and controls: IBM Plex Sans.
- Measurements, paths, cache keys, and status details: IBM Plex Mono.

Bundle the minimal required WOFF2 files locally with their licenses. Do not fetch
fonts at application runtime. If font bundling complicates Marimo HTML export,
fall back to a deliberate local stack and record the visual difference; do not
introduce a network dependency.

### Signature component: beamline context rail

The rail is the single memorable visual element. It encodes real state and makes
screenshots/exported pages self-describing:

```text
MATERIAL -- CHECKPOINT -- E0 -- theta -- phi -- THICKNESS -- RECORDS
 HOPG       loaded       100    2 deg  45 deg   10 um        864
```

- Use labels plus values, not icons alone.
- Render unavailable dimensions as `not selected`, never a cryptic dash.
- Update the rail for the active task. Do not imply one globally selected geometry
  when a view intentionally compares multiple values.
- Keep the component visually flat: one horizontal rule/track and measured notch
  spacing, not a row of cards.

### Layout and interaction rules

- Desktop-first scientific workspace; responsive down to 768 px without clipped
  controls or horizontal page scrolling.
- Use full-width Marimo layout for analysis and validation. Constrain prose to a
  readable measure inside the wider page.
- One primary action per surface. Secondary actions belong in accordions or a
  clearly subordinate row.
- Use sentence-case action labels with exact verbs: `Run scan`, `Resume scan`,
  `Run selected checks`, `Save repository default`.
- Use explicit `Auto` switches for chart domains. Numeric zero remains a valid
  scientific input, not a hidden UI sentinel.
- Minimum pointer target 44 px, visible keyboard focus, sufficient contrast, and
  reduced-motion support.
- Empty and failure states must say what is absent, why it matters, and the next
  exact action or command.

## Intended Page Structures

### Analysis

```text
[title] [material/checkpoint selector]
[beamline context rail]

[Explore | Optimize | Instrument | Compare]

[task controls]             [result title + interpretation]
[advanced axes/scaling >]   [primary chart]
                            [secondary chart/table]

[checkpoint contents >]
```

### Scan

```text
[title]
[material] [checkpoint/resume state]
[case count] [remaining work] [output path]

[geometry preview]
[penetration exclusions >]

                         [Run scan / Resume scan]
[progress, current case, completion action]
```

### Validation

```text
[title] [evidence summary]
[Anchors | Reproductions | Supplementary | Provenance]

[study controls]            [authority/status badge]
[run action + cost/cache]    [result/table/figure]
[job status]

[method, audit, raw logs >]
```

## File Structure

| File | Planned role |
|---|---|
| `notebooks/_design.py` | Shared CSS, tokens, context rail, status badge, control group, empty-state helpers |
| `notebooks/assets/` | Locally licensed font assets, only if export-safe |
| `notebooks/scan_app.py` | Preview/run separation, checkpoint state, progress and completion UX |
| `notebooks/analysis_app.py` | Task navigation, shared context, control hierarchy, copy cleanup |
| `notebooks/validation_app.py` | Evidence navigation, status semantics, cache/job presentation |
| `src/cxr_mc/plots/_style.py` | Shared renderer color alignment if needed; no plot-data changes |
| `src/cxr_mc/plots/altair_*` | Only title, tooltip, accessibility, or shared theme hooks required by notebook UI |
| `checks/cxr_analysis_feranchuk.ipynb` | Split provenance narrative and code into readable semantic cells |
| `checks/cxr_analysis_feranchuk.md` | Synchronized MyST source/companion |
| `tests/test_scan_app.py` | Explicit run gating and preview assertions |
| `tests/test_analysis_app.py` | Navigation, context rail, Auto controls, Marimo boundary assertions |
| `tests/test_check.py` | Validation status vocabulary and safe run gating |
| `tests/test_notebook_design.py` | Small shared presentation-token/component tests if `_design.py` warrants them |

Do not modify every listed plot module mechanically. Touch only the smallest owner
needed to make renderer output consistent.

---

### Task 1: Add shared notebook presentation primitives

**Files:** Create `notebooks/_design.py`; optionally create
`notebooks/assets/`; create `tests/test_notebook_design.py` only for behavior worth
freezing.

1. Define named color, spacing, typography, border, focus, and width tokens.
2. Add one scoped style injector. Avoid broad selectors that override Marimo
   internals unpredictably.
3. Implement small helpers for:
   - page title/intro;
   - beamline context rail;
   - status badge with text and semantic kind;
   - labeled control group;
   - directional empty/error state.
4. Keep helpers presentation-only. They may accept already-resolved labels and
   values; they must not load checkpoints, inspect results, or compute science.
5. Confirm local font behavior in both interactive Marimo and exported HTML. If
   WOFF2 URLs are not self-contained in export, use local stacks rather than
   broken fonts.
6. Add focused tests for token completeness, unique status vocabulary, and helper
   output only if those tests remain stable under Marimo upgrades.

**Acceptance:** Each app can import the helper without importing Marimo into
`cxr_mc`; helpers render with keyboard-visible focus and no external requests.

**Focused verification:**

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_notebook_design.py
```

Omit the command if no dedicated test file is warranted; validate through the app
tests instead.

---

### Task 2: Make scan execution explicit and legible

**Files:** Modify `notebooks/scan_app.py` and `tests/test_scan_app.py`.

1. Split current flow into reactive preview cells and a run-gated execution cell.
2. Material selection must build/filter cases and render a preview without calling
   `run_sweep`.
3. Add `Run scan` or `Resume scan` button based on checkpoint state. Read button
   value only in a downstream cell.
4. Before execution, show:
   - selected material and catalog label;
   - total cases and configuration count;
   - dropped penetration cases with a short reason and expandable details;
   - checkpoint path;
   - cached/remaining cases when cheaply discoverable through existing checkpoint
     helpers.
5. Keep `geometry_table(cases)` available as a collapsed preview rather than a
   large unconditional table.
6. Replace `print`-only state with Marimo status/callout output. Preserve useful
   console output from the underlying runner.
7. During execution, show current state and explain checkpointed/resumable
   behavior. Do not invent an ETA unless the runner exposes enough evidence.
8. On completion, show the exact analysis command and selected material.
9. On `EOFError`, distinguish “checkpoint already complete” from generic failure.
10. Keep penetration gating before any sweep execution.

**Required tests:**

- Source/AST test proving `run_sweep` is downstream of explicit run-button value.
- Material discovery still comes from ordered catalog keys.
- Penetration gate still occurs after `build_cases` and before `run_sweep`.
- Preview and analysis-handoff wording remain present.

**Focused verification:**

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_scan_app.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run marimo check notebooks/scan_app.py
```

Do not start a real scan during local UX verification.

---

### Task 3: Reorganize analysis around scientific tasks

**Files:** Modify `notebooks/analysis_app.py`, `tests/test_analysis_app.py`, and
only the smallest required plot-style/Altair modules.

1. Keep checkpoint/material loading behavior and custom material selector.
2. Add top-level checkpoint summary before chart navigation:
   - material;
   - checkpoint state/path;
   - record count;
   - swept dimensions and number of values;
   - explicit empty-checkpoint action.
3. Replace eight equal-weight tab labels with four task groups:
   - `Explore`: compare beam energies, polar angles, or azimuths;
   - `Optimize`: top geometries and geometry/scan maps;
   - `Instrument`: detectors and penetration;
   - `Compare`: cross-material comparison.
4. Use one top-level navigation mechanism. Inside a task, use a radio/dropdown or
   accordion instead of nested tabs. Preserve lazy builders.
5. Standardize view names as actions: `Compare beam energies`, `Compare polar
   angles`, `Compare azimuths`, `Rank geometries`, `Inspect scan maps`, `Model
   detectors`, `Inspect penetration`, `Compare materials`.
6. Build the context rail from the selections actually pinned by the active view.
   For multi-value comparisons, label the varying dimension as `multiple` and do
   not report a misleading single value.
7. Consolidate repeated control presentation:
   - primary slice controls first;
   - line/bremsstrahlung option second;
   - axis/scaling controls in `Axes and scaling` accordion;
   - explicit Auto switches for narrow and broad domains.
8. Keep separate reactive UI objects where Marimo requires them, but render them
   through a consistent control-group layout. Do not force state sharing that
   changes one view when another view changes.
9. Keep narrowband as primary output. Put broadband beneath it or in a lazy
   secondary section; do not compute hidden charts eagerly.
10. Preserve heatmap pixel-selection behavior and its default-transformer
    workaround. Its Altair selection must still reach the kernel.
11. Preserve detector lazy accordions and the known nested-tab workaround.
12. Preserve penetration preset/manual controls, manual thickness in micrometers,
    1 nm lower bound, and downstream conversion to Angstrom.
13. Preserve cross-material all-energy default, quality floor, point-label
    geometry, and renderer-aware label deconfliction.
14. Move checkpoint contents into a clearly labeled metadata/provenance accordion
    near the summary, not below an otherwise disconnected tab set.
15. Fix the mojibake `â€” no data â€”` label. Standardize visible units and symbols:
    `keV`, `eV`, `um` rendered as `µm`, and degrees rendered as `°` where the UI
    supports them reliably.
16. Rewrite captions to answer three questions: what is shown, what varies, what
    remains pinned. Remove uppercase emphasis such as `INTRINSIC` unless it is a
    defined scientific label.
17. Align Altair/Matplotlib presentation only through existing shared style owners.
    Do not alter frame construction, ranking, filtering, or physics.

**Required tests:**

- Existing catalog-backed penetration controls and bounded manual values.
- UI `.value` reads remain downstream of widget creation.
- Cross-cell exported names are non-underscore names.
- Four task groups and standardized action labels exist.
- Auto-domain controls replace `0 = auto` presentation without breaking `None`
  domain behavior.
- Cross-material all-energy default and θ/φ behavior remain unchanged.
- Mojibake label absent.

**Focused verification:**

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_analysis_app.py tests/test_analyze.py tests/test_altair_sweeps.py tests/test_material_comparison.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run marimo check notebooks/analysis_app.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr analyze --smoke
```

Manual acceptance at 1440, 1024, and 768 px:

- no clipped task navigation or control rows;
- empty checkpoint gives exact recovery action;
- single-value dimensions render as context, not disabled-looking widgets;
- multi-value comparisons identify varying and pinned dimensions;
- heatmap clicks update spectra;
- detector charts render inside accordions;
- keyboard focus remains visible;
- exported HTML contains the context needed to interpret each chart.

---

### Task 4: Reframe validation around evidence and authority

**Files:** Modify `notebooks/validation_app.py` and `tests/test_check.py`.

1. Open with a compact evidence summary and four task groups:
   `Anchors`, `Reproductions`, `Supplementary`, `Provenance`.
2. Move the repository audit table and unresolved-method narrative into
   `Provenance`; keep them accessible but not dominant.
3. Assign every runnable artifact one authority label:
   - `Anchor`: has a pass/fail contract;
   - `Diagnostic`: completed output requires scientific interpretation;
   - `Optional oracle`: may skip due to missing dependency;
   - `Provenance`: derivation/reference artifact, not executable evidence.
4. Replace ✓/✗-only headings with exact result words. A zero exit code for a
   diagnostic becomes `Completed—interpret`, not `Passed`.
5. Preserve full subprocess reports in lazy accordions. Lead with a summary table
   containing authority, state, elapsed time, and next action.
6. Keep explicit run buttons and CPU forcing. Present expected workload and cache
   behavior immediately beside the action.
7. Keep Zhai publication-quality defaults, but distinguish quick exploratory
   settings from publication settings in copy. Do not silently change electron
   counts.
8. Consolidate supplementary selection, orientation provenance, cache state,
   remote preparation, and remote/local job status into one workflow surface.
9. Before `Save repository default`, state the exact setting and repository file
   being mutated. Preserve explicit button gating and success confirmation.
10. Before remote-first preparation, state selected counts and fallback behavior.
    Do not start remote or local compute through passive navigation.
11. Give failed, skipped, remote-running, remote-failed, and cache-hit states
    distinct text plus color. Never rely on color/icon alone.
12. Keep supplementary figures centered unless the responsive layout needs a
    bounded full-width wrapper.

**Required tests:**

- Validation app still initializes its default supplementary study.
- No expensive action runs before its explicit button.
- Authority vocabulary distinguishes anchor from diagnostic.
- Diagnostic success is not labeled `Passed`.
- Repository-default save remains explicitly gated.
- Supplementary figure remains centered/readable.

**Focused verification:**

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_check.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run marimo check notebooks/validation_app.py
```

Manual validation should use cached or deliberately small workloads. Do not launch
publication-size or remote jobs merely to inspect layout.

---

### Task 5: Restructure the legacy Feranchuk provenance notebook

**Files:** Modify `checks/cxr_analysis_feranchuk.ipynb` and
`checks/cxr_analysis_feranchuk.md` together.

1. Run the repository's notebook quality check before editing to establish the
   baseline.
2. Preserve scientific content and execution order while splitting the long
   derivation into semantic Markdown cells:
   - coordinate system and source equation;
   - dielectric response and reciprocal-lattice notation;
   - kinematic photon energy;
   - dynamical correction;
   - flux/amplitude derivation;
   - assumptions, validity limits, and unresolved ambiguities;
   - numerical studies and figure interpretation;
   - references.
3. Use one H1 title. Convert the second current H1 line into part of that title or
   a subtitle.
4. Add a compact contents list linked to stable headings.
5. Add clearly styled textual callouts for `Source`, `Assumption`, `Limit`, and
   `Interpretation`. Keep these semantic in MyST/Markdown, not CSS-only boxes.
6. Replace the attachment hash as visible image description with meaningful alt
   text and a scientific caption. Preserve the actual embedded image.
7. Split oversized code cells at experiment boundaries. Factor repeated local
   configuration only when behavior stays identical and execution order remains
   obvious.
8. Convert banner-style comment walls into Markdown explanations adjacent to the
   corresponding code.
9. Do not edit equations or claims during readability cleanup. If an equation
   appears suspect, record it for a separate physics-review task.
10. Synchronize `.ipynb` and `.md`, strip outputs, and audit the final cell order.

**Focused verification:**

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py nbqa
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py nbstrip
```

Then verify:

- no cell outputs or execution counts;
- one H1 title;
- source/assumption/limit sections remain present;
- MyST and notebook remain synchronized;
- no numerical code diff beyond safe extraction/deduplication.

---

### Task 6: Cross-app visual and accessibility review

**Files:** Modify only files touched by Tasks 1–5 to resolve findings.

Review each app at 1440, 1024, and 768 px with representative data and empty-data
states.

Checklist:

- same title scale, background, control language, and focus treatment;
- beamline rail values accurate for active view;
- no horizontal page scroll;
- charts retain readable labels and legends;
- controls appear before the output they affect;
- only one visually primary action per workflow;
- warnings, failures, cache hits, and completion have explicit text;
- keyboard navigation reaches every action in sensible order;
- no gratuitous animation; reduced-motion preference respected;
- empty states name the required scan/checkpoint/action;
- exported analysis HTML is self-describing and uses no network assets.

Take screenshots for review if the environment supports them. Screenshots are
review artifacts; do not commit them unless explicitly requested.

Run focused tests again after visual fixes.

---

### Task 7: Whole-branch verification and handoff

1. Confirm notebook/source changes do not overlap unrelated pre-existing work.
2. Run structural checks:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run marimo check notebooks/scan_app.py notebooks/analysis_app.py notebooks/validation_app.py
```

3. Run real analysis execution:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr analyze --smoke
```

4. Run focused regression tests:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_scan_app.py tests/test_analysis_app.py tests/test_analyze.py tests/test_check.py tests/test_altair_sweeps.py tests/test_material_comparison.py
```

5. Run legacy notebook checks:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py nbqa
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py nbstrip
```

6. Run canonical verification:

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py verify
```

7. Review final diff for:
   - accidental scientific behavior changes;
   - eager compute introduced by layout composition;
   - `.value` reads in widget-creation cells;
   - underscore-prefixed cross-cell exports;
   - network font/assets;
   - notebook outputs;
   - unrelated dirty-worktree changes.
8. Handoff exact commands/results, manual viewport checks, known visual limitations,
   and any deferred physics/readability concerns.

## Recommended Delivery Order

1. Shared presentation primitives.
2. Scan execution safety.
3. Analysis task hierarchy.
4. Validation evidence hierarchy.
5. Legacy provenance cleanup.
6. Cross-app visual review and full verification.

Tasks 2–5 should remain separately reviewable. Do not combine the legacy notebook
rewrite with analysis-app restructuring in one large commit.

## Plan Self-Review

- **Subject-specific:** Beamline context, scientific authority, cached compute,
  scan geometry, and provenance shape the interface. This is not a generic admin
  dashboard skin.
- **One aesthetic risk:** The beamline rail is distinctive but functional. Other
  styling remains quiet.
- **Runtime-safe:** Explicit run gating, lazy builders, downstream UI reads, and
  the detector accordion workaround remain hard constraints.
- **Science-safe:** No equation, metric, selection, normalization, or simulation
  change belongs in this plan.
- **Export-safe:** Local assets only; exported analysis must remain self-contained.
- **Reviewable:** Each application has independent acceptance criteria and focused
  tests before whole-repository verification.
