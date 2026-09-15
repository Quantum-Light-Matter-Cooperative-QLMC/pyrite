# Issue #124: per-material line grids

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/124
Branch: `issue-124-per-material-line-grids`

Status: approved and implemented locally; awaiting integration.
Worktree: `/home/alex/dev/pyrite/.worktrees/issue-124`.

## Approved decision

Keep explicit per-material artifacts and legacy per-material rows. When neither
covers an energy, resolve through the existing automatic `sinc-nyquist` and
`kinematic-ceiling` policy using that case's material and trajectories. Remove
profile-named and `energy_grids.standard` line-row inheritance and seeding.
Do not copy shared coordinates into 18 new material tables or run regeneration
sweeps as part of this fix.

## Owners and implementation

- `src/pyrite/materials/_parse.py`: replace nested shared lookup with own-material
  lookup. Ensure absent rows reach automatic resolution: an absent mapping
  currently makes `campaign/sweep.py` select the crystal's static `E_grid`,
  whereas an empty mapping reaches automatic policy. Trace `_scan` and
  `campaign/config.py` to preserve that distinction deliberately.
- `src/pyrite/energy_grid/apply.py`: remove `_DEFAULT_MATERIAL`, shared row
  seeding in `_merge_line_rows`, fallback in `_existing_artifact_rows`, and
  fallback in `show_text`. Preserve same-material manual rows and untouched
  profile refs; new material rows start empty.
- `src/pyrite/data/materials.toml`: remove only obsolete shared line-row input if no
  compatibility consumer remains. Preserve own-material rows and artifacts.
- `docs/repo-design/materials-catalog-schema.md`: document own-material
  resolution, automatic missing-energy behavior, and cache identity impact.
- Tests: `tests/materials/test_material_catalog.py`,
  `tests/energy-grid/test_apply.py`, relevant show/case-grid tests, and
  `tests/data/material_catalog_golden.json`.

## Acceptance

- Shared/profile-named tables cannot supply or seed another material's rows.
- Missing own rows and partial coverage reach automatic policy, tested without
  heavy transport. Explicit own rows remain unchanged.
- Change hopg's legacy rows; every other material's resolved coordinates stay
  byte-identical. Include a material formerly using the shared fallback.
- Regenerate hopg's artifact in one profile: its digest changes; every other
  material's coordinates/refs and every untouched profile ref remain unchanged,
  including profiles initially sharing hopg/hbn digests.
- Review golden diff: changes limited to removal of inherited line rows for
  the actual affected materials; enumerate the actual set rather than assume
  the issue's historical count.
- Demonstrate changed case identity for affected formerly inherited grids and
  unchanged identity for preserved explicit rows; do not delete checkpoints.
- No kernel, kinematics, or physics equation changes.

## Validation and dispatch

Worker: `implement-task` with scientific-library, documentation-maintenance,
regen-golden, regression-testing, and CLI skill if display/help changes require
it. Add physics-review only if the implementation changes physics policy.

Start with focused material catalog and energy-grid tests through
`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test <path>`, then required
lint/type/docs/golden checks. No heavy local sweeps; remote GPU skill owns any
later compute. Regenerate CLI reference if CLI contract changes.

Authority at dispatch: task-local checkpoint commits; no implementation push,
PR, issue writes, or delegation unless separately authorized. Stop on overlapping
dirty worktree or a departure from this reviewed policy.

## Implementation evidence

Removed shared/profile-named catalog and apply lookups, legacy seeding, and
display fallback. Empty catalog mappings now reach the existing automatic
policy; explicit fixed grids and own-material stored rows remain supported.
Removed only `energy_grids.standard` from the bundled catalog.

Regression tests reproduced inherited coordinates and shared manual rows blocking
another material's update before the fix. Focused catalog/isolation/apply suite:
160 passed. Neighboring energy-grid/case-building/config suite: 398 passed.
Hopg row and artifact tests preserve every other material's stored coordinates
byte-for-byte; artifact tests also preserve untouched refs in all four profiles.
All 18 formerly inherited materials receive automatic policy and different
case-content keys from their historical shared grid.

Golden comparison against the setup commit found exactly 18 changed materials,
matching the issue inventory. Only `scan.E_grid_line_by_energy` changed, to an
empty mapping; all other golden fields and own-material fingerprints are intact.
Schema docs record checkpoint invalidation. No checkpoints were deleted.

Lint, full typecheck, docs build, golden freshness, CLI reference regeneration/check, and real
human/JSON silicon grid-show probes pass. CLI reference bytes did not change.
