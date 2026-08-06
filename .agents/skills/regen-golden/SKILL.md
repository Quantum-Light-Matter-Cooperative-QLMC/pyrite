---
name: regen-golden
description: Use when data/materials.toml or materials/catalog.py changes require material-catalog golden snapshot regeneration and verification.
---

# Regenerate Material Catalog Golden

Only regenerate after authorized catalog-source change; otherwise stop because
regen could mask unexplained drift.

```bash
rtk git status --short data/materials.toml src/cxr_mc/materials/catalog.py
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr energy-grid regen-golden
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test \
  tests/materials/test_material_catalog.py -k golden
rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test \
  tests/test_line_grid_golden.py
rtk git diff --stat tests/data/material_catalog_golden.json
```

Never hand-edit golden. Surface surprising diff. Do not stage or commit unless
asked.
