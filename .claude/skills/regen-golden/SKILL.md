---
name: regen-golden
description: Use when data/materials.toml or materials/catalog.py changed and the material-catalog golden snapshot must be regenerated and re-verified.
---

# Regenerate Material Catalog Golden

Deterministically rebuild `tests/data/material_catalog_golden.json` from the
on-disk catalog, then prove the golden test is green again. Run after any edit to
`data/materials.toml` or `src/cxr_mc/materials/catalog.py` (thickness_ang,
substrate/stack fields, crystal/material entries) -- the on-edit hook warns when
those files change; this skill is the fix.

Invoke as a normal completion step after an authorized edit to those catalog
sources. Do not run for unrelated work or merely to silence unexplained drift.

## Steps

1. **Confirm what staled the golden.** Check the working tree touched a catalog
   source:

   ```bash
   rtk git status --short data/materials.toml src/cxr_mc/materials/catalog.py
   ```

   If neither shows, stop and ask the user -- regenerating with no source change
   only masks unrelated drift.

2. **Regenerate** (independent re-serialization from the live catalog):

   ```bash
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr energy-grid regen-golden
   ```

3. **Verify green** -- the catalog golden test plus the independence guard:

   ```bash
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test \
     tests/test_material_catalog.py -k golden
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test \
     tests/test_line_grid_golden.py
   ```

4. **Show the diff** so the user reviews exactly what moved before committing:

   ```bash
   rtk git diff --stat tests/data/material_catalog_golden.json
   ```

## Guardrails

- Never hand-edit `tests/data/material_catalog_golden.json` -- only the
  `regen-golden` command may write it; it is the independent oracle.
- If regen produces a large or surprising diff for a small source change,
  surface it to the user rather than committing -- it may signal an unintended
  catalog change.
- Do not stage or commit unless the user asks; leave the regenerated file in the
  working tree for review.
