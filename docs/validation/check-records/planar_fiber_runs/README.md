# Planar fibre study evidence (#1056)

Harness: `checks/planar_fiber_study.py`; model claim `cr-planar-fiber-unraveling`.
All runs: Si at 293.15 K, fixed energy, TDS strings plus electronic kicks,
outgoing box equal to the window, uniform-position Weyl window (exact
momentum order), tails for every string output outside the window.

## Array 10928 (qlmc, 2026-11-10)

Isolated copy `~/pyrite-1056` from `git archive` of `7cd7d29e7`
(`PYRITE_SOURCE_REVISION` in `submit.sh`; recorded in every output's
`info.source_sha`). 160 tasks, one CPU, 3 GiB each, all `Exit status: 0`
(`/usr/bin/time -v` in the remote logs). Task `id`: configuration
`id // 40` in `(si110_17, si100_17, si100_54, si110_54)`, arm
`(id // 20) % 2` in `(absorbing, reference)`, incidents
`(id % 20) * 100 .. +100`; seed 0, so the two arms are paired per incident.
Absorbing: 300 eV margin. Reference: `P = R`. Windows `M` = 32, 24, 36,
48 respectively. 241 uniform checkpoints over 120 µm (17 MeV) or 500 µm
(54.5 MeV; 2.08 µm spacing). `outputs.sha256` hashes the 160 `.npz`
chunks, which remain at `qlmc:~/pyrite-1056/scratch/runs/`.
`report-<case>.json` is `analyze <absorbing chunks> --reference <reference chunks>`
at commit `986fc3849` (the analysis change since `7cd7d29e7` only removes a
duplicated report depth).

## Local 5 MeV Si(110)

`si110_5-local/`: 4000 incidents per arm, 20 µm, `M` = 16; reference arms
at `M` = 16 (`report-m16.json`) and `M` = 24 (`report-m24.json`), both
paired with the same absorbing arm. Source `df2d79a14`.

## Density-matrix comparison

`dm-300ev-si1{10,00}_17.json`: `checks/kephart_occupation.py --margins 300`
at H = 32/24, 16 cells, dz 62.5 Å (balance defects −0.019/−0.017), run
locally at `986fc3849` for a like-for-like comparison with the fibre's
absorbing arm at the same margin.

## Paired-arm source budgets (`budget/`)

Driver `checks/planar_fiber_budget.py` (arms, observable and statistics in
its docstring); calibration by
`montecarlo/channeling/axial_depth_calibration.py::calibrate_axial_depth`,
unchanged.

- Kephart configurations: remote array 11088 (qlmc, 2026-11-11), isolated
  copy `~/pyrite-1056-budget` from `git archive` of `80393f474`
  (`PYRITE_SOURCE_REVISION` in `submit.sh`). 160 tasks (case `id // 40`,
  incidents `(id % 40) * 100 .. +100`, all seven arms per task), one CPU,
  3 GiB, all `Exit status: 0`, peak RSS 0.29 GiB. `outputs-11088.sha256`
  hashes the 1120 chunks, which remain at
  `qlmc:~/pyrite-1056-budget/scratch/budget/`.
- `channeling_demo` (`demo_17`): local, 200 chunks of 100 incidents.
  Chunks 0–3900 ran from the working tree of `831b7db38` plus the
  then-uncommitted driver, byte-identical to `80393f474`'s run path; chunks
  4000–19900 ran at `80393f474` with only the analysis code edited. Both
  revisions are listed in the report's `source_shas`.
  `outputs-demo_17.sha256` hashes the 1400 chunks (kept in the
  worktree's ignored `scratch/budget-demo/`).
- `report-<case>.json`: `analyze <case>` at `518a92e00` plus the
  checkpoint change committed with this record (`BUDGET_UM`). Each report holds
  every arm's settings, CPU time, the per-interval bounds and audit
  (paired differences, SEs, `t` quantile, drift shifts), and the fate caps
  with their fractions.

## Two-time source vs secular line model (`spectrum/`)

Harness `checks/planar_fiber_spectrum.py`; model `cr-planar-fiber-regression`.
Remote array 11248 (qlmc, 2026-11-11), isolated copy `~/pyrite-1056-spectrum`
from `git archive` of `8179040b0` (`PYRITE_SOURCE_REVISION` in `submit.sh`).
Task `id`: case `id // 50` in `(demo_17, si100_17, si110_5)`; slots 0–39 are
`control` chunks of 100 primaries (incidents `slot * 100 ..`), slots 40–49
the paired half-step `fine` chunks (incidents 0–999). `report-<case>.json`
is `analyze <case> <control chunks> --fine <fine chunks>` run on the remote
copy, so the production reference uses the same revision.
`outputs-11248-*.sha256` hash the chunks, which remain at
`qlmc:~/pyrite-1056-spectrum/scratch/spectrum/`. All 150 tasks have completed with
`Exit status: 0`; peak RSS was 0.75 GiB.
