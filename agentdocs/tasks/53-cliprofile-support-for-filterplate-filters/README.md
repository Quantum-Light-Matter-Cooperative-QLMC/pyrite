# CLI/profile support for FilterPlate (issue #53)

Branch: `53-cliprofile-support-for-filterplate-filters`

## Context

`FilterPlate` (rectangular attenuating plate: material, thickness_mm, size_mm,
`PlanarPose` position/orientation) is fully implemented and physics-validated
(`Validation: positioned-filter-attenuation`), but reachable only via the
Python API (`pyrite.simulate(..., filters=(...))`). The repo owner uses the
CLI exclusively — the feature is currently unusable for them. The original
design checkpoint (`agentdocs/tasks/feature/positioned-photon-filters/README.md:307-316`)
deliberately deferred CLI/profile wiring pending "runtime evidence"; that
evidence is now this request.

**Constraint I'm holding hard**: no new physics. `FilterPlate` attenuation
only exists today as part of the `PlanarDetector` + `PixelScorer` pixel-ray
path (`api._simulate_planar`), which is the thing that's actually validated.
I looked at an alternative (a cheaper "nominal single ray" scalar attenuation
bolted onto the existing checkpoint pipeline's scalar `Detector.score()`) and
rejected it — it would be a new, unvalidated physics formula requiring its own
ledger row/validation per `AGENTS.md`, just to avoid configuring one more
object. Reusing the already-validated `PlanarDetector`/`PixelScorer`/
`api.simulate` path verbatim is both less work and the only physics-safe
option.

## Design

### 1. Profile TOML: new `[[profiles.NAME.filters]]` array of tables

One inline table per plate, validated through the real `FilterPlate`
dataclass (`src/pyrite/instrument/model.py:182`) so CLI-entered data gets
identical validation to the Python API:

```toml
[[profiles.NAME.filters]]
name = "half_filter"          # optional, matches FilterPlate.name
material = "silicon"
thickness_mm = 0.1
size_mm = [7.04, 14.08]
distance_mm = 200.0            # -> PlanarPose.from_observation(...)
polar_deg = 90.0
azimuth_deg = 0.0
roll_deg = 0.0
offset_mm = [3.52, 0.0]
```

Pose fields mirror `PlanarPose.from_observation` (`instrument/model.py`)
exactly — no new pose vocabulary invented.

### 2. Profile TOML: new `[profiles.NAME.physical_detector]` table

Required only when the user wants filters actually applied to the output
detector face — filter attenuation only takes effect on the finite pixel
path (`api._simulate_planar`), which needs a real `PlanarDetector` (pose +
`PixelGrid` + `EnergyBins`/response), not the checkpoint pipeline's scalar
`Detector`. Reuses `PlanarPose.from_observation` fields identically to
filters, plus pixel grid shape/pitch (defaulting to
`PixelGrid.timepix3_chip()` via `--shape`/`--pitch-mm` options with sane
defaults so most users only set distance/polar/azimuth).

I confirmed the scalar `Detector` (`detectors/spec.py:165`) has no
`distance_mm`/pose fields to auto-promote from, so this is unavoidably a
distinct, explicit config block — not a flag on the existing detector.

### 3. New CLI: `pyrite profile filter add|rm|list|show`

Mirrors the existing `pyrite beam`/`_beam_shared.py` pattern
(`cli/commands/beam.py`, `cli/commands/_beam_shared.py`) and the existing
`detector_table`/`ACTIVE_DETECTOR_FIELDS` pattern in
`campaign/profile_edit.py:119-172`: Click options per `FilterPlate`/pose
field, `tomlkit`-based atomic writes through `cli/_catalog_io.py`,
`FilterPlate.__post_init__` as the single source of validation truth (no
duplicated validation logic in the CLI layer). `pyrite profile show` gains a
filters section in its existing table/JSON output.

### 4. New CLI: `pyrite material simulate MATERIAL [--profile NAME] [-o table|json|wide]`

New subcommand under the existing `pyrite material` group
(`cli/commands/material.py`), alongside `show|set|blaze|validate`. Resolves
the named profile's beam/target/physical_detector/filters into
`campaign.model.Beam`/`Target`/`PlanarDetector`/`FilterPlate` objects, calls
`api.simulate(...)` directly (no `Sweep`, no checkpoint — this is
deliberately the same "filesystem-free" seam the Python API uses), and
prints the resulting `Result`/`SpatialResult` (spectrum table, or pixel image
summary + optional `--output-file` to dump full arrays). Single case only,
not swept — matches `api.simulate`'s existing single-scene contract; sweeping
this is out of scope.

I'm intentionally not adding a tenth top-level CLI noun (repo_map documents
exactly nine: `run, app, checkpoint, config, remote, job, profile, material,
beam`) — nesting under `pyrite material` keeps the CLI reference stable and
fits existing per-material action verbs.

### Non-goals for this task

- No wiring into `pyrite run`/checkpoints — `SpatialResult` is explicitly not
  checkpoint-schema (`repo_map.md:561`), and `run_sweep`'s scalar `Detector`
  path stays untouched.
- No sweeping of physical-detector/filter cases (single scene per invocation).
- No changes to `Sweep`/`campaign/sweep.py`'s TOML shape.

## Files touched (representative, not exhaustive)

- `src/pyrite/campaign/profile_edit.py` — filter table CRUD + physical
  detector table, mirroring `detector_table`/`ACTIVE_DETECTOR_FIELDS`.
- `src/pyrite/cli/commands/profile.py` (+ possibly a new
  `cli/commands/_filter_shared.py` mirroring `_beam_shared.py`) — `pyrite
  profile filter add|rm|list|show`.
- `src/pyrite/cli/commands/material.py` — `pyrite material simulate`.
- `src/pyrite/cli/_catalog_io.py` — filter row helpers alongside existing
  `beam_rows`.
- `docs/repo-design/cli/cli-reference.md` — regenerate
  (`pyrite-dev repo-map`/CLI reference generation per `AGENTS.md`).
- `docs/guides/` — short guide or extend
  `docs/guides/python-api-workflow.md` cross-reference.
- Tests: `tests/cli/...` for the new commands, `tests/instrument/` unaffected
  (no physics change).

## Verification

- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli`
- `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core`
- Manual: `pyrite profile filter add ... `, `pyrite profile show`, `pyrite
  material simulate <material> --profile <name>` end-to-end against a real
  profile, confirm attenuated spectrum differs from an unfiltered run.
- `pyrite-dev verify` before calling it done (per `AGENTS.md`).

## Implementation progress

- Added profile-schema storage for `filters` and `physical_detector`, profile
  filter CRUD, profile-show JSON/text exposure, and `material simulate` on the
  existing planar `api.simulate` path. No `run`, checkpoint, or `Sweep` code
  changed.
- Added focused CLI/catalog coverage for CRUD, detector preservation/validation,
  JSON output, and the single-scene API seam. CLI reference regenerated; the
  sweep-profile guide now documents the profile TOML and command workflow.
- Remaining verification: CLI/core suites, docs build if its Sphinx dependency
  is available, and the full `pyrite-dev verify` gate.
