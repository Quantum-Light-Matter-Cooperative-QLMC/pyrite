# analysis_app — blazed / flat face selector

**Date:** 2026-07-23
**Status:** approved (design)
**Depends on:** `cxr blaze` driver (writes `checkpoints/<material>_blazed.pkl`) —
implemented in the same branch.

## Goal

Let the analysis app (`notebooks/analysis_app.py`) browse a crystal's **blazed**
(sawtooth entrance-face) checkpoint alongside its **flat**-face (`cxr scan`)
checkpoint, chosen by a second dropdown beside the material selector. Today the
app only ever loads `checkpoints/<material>.pkl`; the blazed checkpoint written
by `cxr blaze` is invisible to it.

## Terminology

The face **variant** is "blazed" (vs "flat"); the checkpoint stem is
`<material>_blazed`. The word "groove" is reserved for the **groove spacing**
parameter (`groove_spacing_ang`, `--spacing`, the `groove=<µm>um` case-name tag,
`montecarlo.groove`) and is left unchanged everywhere.

## Non-goals

- No new physics; no change to how checkpoints are written.
- No change to `load_checkpoint` / `checkpoint_path_for`. Passing the stem
  `<material>_blazed` to `load_checkpoint` already loads `<material>_blazed.pkl`
  (it globs `<stem>.pkl`), so the notebook only needs to compute the stem.
- Cross-material summary tab stays **flat-only** — blazed variants are not folded
  into that comparison.
- No new widget class: the existing `MaterialSelect` AnyWidget is a generic
  select and is reused for the face dropdown.
- The material dropdown's availability logic is unchanged (flat-checkpoint glob).
  A material whose only checkpoint is blazed still appears disabled in the
  material menu — out of scope for v1.

## Library surface (`src/cxr_mc/analyze.py`)

Two pure, unit-testable helpers beside `material_menu`:

```python
def face_menu(material, checkpoint_dir) -> tuple[MaterialMenuRow, ...]:
    """Two rows: flat and blazed, each disabled when its checkpoint is absent."""
    # {"value": "flat",   "label": "Flat",   "disabled": not <material>.pkl}
    # {"value": "blazed", "label": "Blazed", "disabled": not <material>_blazed.pkl}
```

```python
def checkpoint_stem(material, face) -> str:
    """`material` for face == "flat", `f"{material}_blazed"` for "blazed"."""
```

`face_menu` uses the same `Path(checkpoint_dir).glob`/existence convention as
`material_menu` so the disabled flags stay consistent with the material menu.
`select_initial_material`'s "first non-disabled" idea is reused to default the
face: **flat** when both exist, else whichever checkpoint is present.

## Notebook wiring (`notebooks/analysis_app.py`)

1. **Face cell** (new, after the material cell): build `face_ui` from
   `face_menu(_DEFAULT_CHECKPOINT_DIR)` for the currently selected `MATERIAL`,
   reusing `MaterialSelect(label="**Face**", ...)`. This cell depends on
   `material_ui.value`, so changing the material rebuilds the face dropdown and
   its disabled flags track that material's available checkpoints. Default value
   = first non-disabled row (flat preferred).
2. **Load cell:** compute
   `stem = checkpoint_stem(MATERIAL, face_ui.value["value"])` and call
   `load_checkpoint(stem)` instead of `load_checkpoint(MATERIAL)`. `MATERIAL`
   stays the catalog key for every other use (labels, `CATALOG.material(MATERIAL)`,
   manifest lookups).
3. **Empty-checkpoint message** (~L820) names the selected face
   (e.g. "No **blazed** checkpoint for `hopg`").
4. **Chart titles** gain a `" (blazed)"` suffix only when the blazed face is
   selected, so a blazed plot is not mistaken for the flat one.

## Testing

`tests/test_analyze.py` (or the existing analyze test module):

1. `face_menu` — with a tmp checkpoint dir holding neither / only flat / only
   blazed / both `.pkl`, the two rows carry the right `disabled` flags.
2. `checkpoint_stem` — `("hopg", "flat") == "hopg"`,
   `("hopg", "blazed") == "hopg_blazed"`.
3. Face defaulting reuses `select_initial_material` semantics (flat preferred
   when both present; the present one when only one exists).

Notebook itself validated with `uvx marimo check notebooks/analysis_app.py`;
reactive behavior is confirmed by launching the app, not asserted in a unit test.

## Files touched

- `src/cxr_mc/analyze.py` — `face_menu`, `checkpoint_stem`.
- `notebooks/analysis_app.py` — face dropdown cell, load-cell stem, title suffix,
  empty-checkpoint wording.
- `tests/test_analyze.py` — `face_menu` / `checkpoint_stem` coverage.
- `docs/repo_map.md` — note the face selector on the `analyze.py` entry.
