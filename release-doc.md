# PyRITE 0.21.0

Changes since `v0.20.4` (36 commits).

## Features

- **optimization**: add fixed-window PXR study comparison (7a02c034)
- add chunk + batch Optuna tuner (002bea73)
- **api**: add transport and radiation facades (25926df4)
- **quantum**: consolidate public imports and diagnostic API map (d594a4b9)
- **apps**: add TR and Cherenkov angular maps (41578de2)
- **materials**: add pinned fused-silica optical reference (1d7a9907)
- **materials**: serialize optical response records (502e9d91)
- **scene**: compose independent executors for scalar and planar simulation (e2468ff4)
- implement configuration API (080b05b8)
- update timepix3 detector object (988c9385)
- implement Detector and Observation API (2deac3b2)
- CLI and app migration to new Scene (#695) (21254512)
- compose independent observations on scenes and sweeps (876ae436)
- normalize units in quantum and profile settings (f026a96c)
- add dimensional unit inputs (bdfb7527)
- define API policy (9902e2a5)

## Fixes

- **perf**: clear tuner lint complexity and docs gates (9a7c347d)
- **perf**: bind NPZ pickle policy explicitly (2038561c)
- **apps**: preserve exact angular-map azimuths (f42d2c51)
- **channeling**: preflight axial consumers before side effects (359180f6)
- **channeling**: refuse raw axial production cases (65df7b83)
- **ci**: sync public API inventories (5e4be71b)
- **api**: preserve valid kernel tuning compatibility policy (4c69d683)

## Refactors

- **scene**: migrate result and directional consumers (b23a9942)
- **scene**: expose independent configuration to result consumers (80b56cd7)
- **api**: curate root exports with compatible domain imports (eae849ac)

## Documentation

- **api**: format advanced facade examples (63370e43)
- **quantum**: format API guide example (0fd267e2)
- **channeling**: sub-MeV planar CR static screen (#585) (b358c46f)
- **scene**: finalize composition contract after config and observation merges (150fc7ba)
- refresh detector configuration dependency graph (91f854e6)
- rewrite simulation composition RFC for physical scenes (e3f214db)
- fix guide command block checks (bcb4368f)

## Tests

- complete beam and geometry facade contracts (a8426f74)

## Chores

- note docs failure quirk in AGENTS.md (2ce1f174)
