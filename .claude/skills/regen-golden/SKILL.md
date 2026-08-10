---
name: regen-golden
description: Use when data/materials.toml or materials/catalog.py changes require material-catalog golden snapshot regeneration and verification.
---

# Regenerate Material Catalog Golden

Only regenerate after authorized catalog-source change; otherwise stop because
regen could mask unexplained drift.

```bash
git status --short data/materials.toml src/pyrite/materials/catalog.py
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev regen-golden
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
  tests/materials/test_material_catalog.py -k golden
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
  tests/test_line_grid_golden.py
git diff --stat tests/data/material_catalog_golden.json
```

Never hand-edit golden. Surface surprising diff. Do not stage or commit unless
asked.
