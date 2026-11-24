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

## Critical-angle-multiple cut scan (`a-scan/`, #522)

Harness `checks/planar_fiber_study.py`, absorbing arm only, with margin
$(a^2-1)D$ ($D$ the well depth: 21.040 eV (110), 11.657 eV (100)) passed
through `--margin`. Seed 0, so incidents are paired with array 10928.
Isolated copy `~/pyrite-522-scan` from `git archive` of `453dd745e`
(`PYRITE_SOURCE_REVISION`), all tasks on qlmc-queen, one CPU, 3 GiB.

- Array 14532 (2026-11-22, `submit-14532.sh`): 400 tasks, case `id // 100`
  in `(si110_17, si100_17, si100_54, si110_54)`, $a$ index `(id // 20) % 5`
  in `(2.5, 3, 3.5, 4, 4.5)`, incidents `(id % 20) * 100 .. +100`.
- Array 14933 (2026-11-22, `submit-14933.sh`): 240 tasks, case `id // 60`;
  `(id // 20) % 3` selects $a = 4$ incidents 2000–3999, or $a = 4.25$
  incidents 0–1999 and 2000–3999.

All 640 tasks ended `COMPLETED` with exit code 0. `outputs-14532-14933.sha256`
hashes the chunks, which remain at `qlmc:~/pyrite-522-scan/scratch/scan/`.
`report-<case>-a<a>.json` is `analyze` of all chunks of that case and $a$,
at the same revision (2000 incidents; 4000 for $a = 4$ and $4.25$).

## Demo budget at the production margin (`budget-v12/`, #522)

`channeling_demo` measured for the provisional `4 psi_c` candidate
(margin 318.6168 eV for Si (110) at 293.15 K with the
bonding-charge profile; matching-surface arm at twice that). Remote array
15183 (qlmc-queen, 2026-11-22, `submit-demo.sh`), isolated copy
`~/pyrite-522-v12` from `git archive` of working-tree snapshot `2875a227c`
(recorded in every chunk's `source_sha`). Its run path differs from the
committed tree only by Ruff formatting of `density.py`. 200 tasks of 100
incidents (all seven arms per task), all `COMPLETED` with exit code 0.
`outputs-15183-demo_17.sha256` hashes the 1400 chunks, which remain at
`qlmc:~/pyrite-522-v12/scratch/budget/`. `report-demo_17.json` is
`analyze demo_17` on the remote copy: ε = 5.38% (4.86% at 300 eV), still
resolution-limited at 20000 incidents. This is historical candidate evidence at the withdrawn `a = 4` margin, not
the admitted entry; the `a = 4.125` admission is measured in
`budget-v12-a4.125/`.

The `a-scan/` margins above used the independent-atom well depth (21.040 and
11.657 eV), i.e. effective $a = 3.9822$ (110) and $3.9835$ (100) at nominal $a=4$
against the production bonding-charge wells (21.241 and 11.760 eV).
These follow from $a_{\mathrm{eff}}=\sqrt{1+(a^2-1)D_{\mathrm{old}}/D}$,
a 0.41–0.45% reduction. Its effect near a tolerance edge must be measured;
the follow-up uses the same well depth on both engines.

## Demo budget at the adopted margin (`budget-v12-a4.125/`, #522)

`channeling_demo` at the production default `4.125 psi_c` (margin 340.1898 eV
for Si (110) at 293.15 K; matching-surface arm at twice that). Remote array
15631 (qlmc-queen, 2026-11-23, `submit-demo.sh`), isolated copy
`~/pyrite-522-v12` from `git archive` of working-tree snapshot `fc4702ece`
(recorded in every chunk's `source_sha`; a dangling snapshot whose run path
equals the committed tree, which only differs in docstrings, provenance text
and records afterwards). 200 tasks of 100 incidents (all seven arms per
task), all `COMPLETED`, exit code 0. `outputs-15631-demo_17.sha256` hashes the
1400 chunks, which remain at `qlmc:~/pyrite-522-v12/scratch/budget/` (the
`a = 4` chunks were moved to `scratch/budget-a4/`). `report-demo_17.json` is
`analyze demo_17` on the remote copy: **ε = 6.04%** (4.86% at 300 eV, 5.38% at
the withdrawn `a = 4`), resolution-limited at 20000 incidents. This is the
admitted entry in `pyrite.quantum._planar_envelope`.

## Production critical-angle cuts (`dm-a-scan/`, #522)

Array 14943 (2026-11-22), source `453dd745e7b7466c6051d043907e898b71674b1c`,
`checks/kephart_occupation.py`, H64, 32 cells, dz15.625 Å. All eight tasks
completed with exit code 0. `submit-14943.sh` records task mapping and settings.
The isolated remote checkout is the same as the fibre scan above.

`run-*.json` retain full output, with the operator host replaced by a placeholder;
`original-outputs-14943.sha256` hashes the original remote bytes, verified after
pulling. Logs retain parameters, resource use and exit status, with home paths
redacted. Those hashes intentionally do not match the redacted JSON copies.
The selected observable is `detector_ratio.bound`; `mean` and unweighted
`ratio` do not decide the acceptance gate.

| Case | a = 4 | a = 4.25 |
|---|---:|---:|
| Si(110), 17 MeV, n=1 | 0.843993 | 0.899633 |
| Si(110), 17 MeV, n=2 | 0.984787 | 1.047612 |
| Si(110), 17 MeV, n=3 | 0.979901 | 1.040600 |
| Si(100), 17 MeV | **0.781648** | 0.837409 |
| Si(100), 54.5 MeV | 1.057675 | 1.133811 |
| Si(110), 54.5 MeV | 0.950285 | 1.014178 |

Bold fails ±20%. At a = 4, absolute balance defects are 0.000705–0.003224,
overflow fractions 0.00809–0.06635; at a = 4.25, 0.000792–0.003588 and
0.00908–0.07508. The Si(100), 17 MeV length is 12.506367 µm at a = 4,
below the 12.8 µm lower bound. At a = 4.25 production passes, but the fibre
scan has Si(100), 54.5 MeV at 1.206; no jointly passing calibration is established.

Follow-up scripts `submit-dm-4125.sh` and `submit-fibre-4125.sh` test a = 4.125
on both engines (arrays 15397/15398, submitted 2026-11-23). The fibre margin is
resolved from the same bonding-charge profile as production, removing the
small well-depth mismatch in the earlier scan. Results are recorded below.

## Joint $a = 4.125$ check (`a-scan/`, `dm-a-scan/`, #522)

Both engines at one cut with the margin derived from the same bonding-charge
profile (340.1898 eV Si(110), 188.3491 eV Si(100)), source
`453dd745e7b7466c6051d043907e898b71674b1c`, isolated copy
`~/pyrite-522-scan` on qlmc-queen.

- Fibre array 15398 (`submit-fibre-4125.sh`): 160 tasks, 4000 incidents per
  case, all `COMPLETED`, exit 0. `a-scan/outputs-15398.sha256` hashes the 160
  chunks; `a-scan/report-<case>-a4.125.json` is `analyze` of each case.
- Production array 15397 (`submit-dm-4125.sh`): 4 tasks, H64, 32 cells,
  dz 15.625 Å, all `COMPLETED`, exit 0 (3.2, 3.0, 9.5, 8.5 h). Public
  `run-*-a4.125.json` replace the operator host with a placeholder;
  `original-outputs-15397.sha256` hashes the original remote bytes, and logs
  have home paths redacted. Selected observable `detector_ratio.bound`.

Result: all twelve central estimates fall within 0.8–1.2 of Kephart
(table in `docs/research/physics/dechanneling-boundary.md`). Si(100),
17 MeV (production 0.808) and Si(100), 54.5 MeV (fibre 1.175 ± 0.031) are
near the edges.

Fibre array 15632 (`a-scan/submit-fibre-4125-extra.sh`, 40 tasks, all
`COMPLETED`, exit 0) doubled the Si(100), 54.5 MeV statistics (incidents
4000–7900, same margin and seed). `a-scan/outputs-15632.sha256` hashes its
chunks and `a-scan/report-si100_54-a4.125-8000.json` is `analyze` of all 80
chunks (8000 incidents): detector length 28.32 ± 0.46 µm, ratio 1.180 ± 0.019
(upper edge 28.8 µm is 1.05 SEM away; 4000 incidents gave 1.175 ± 0.031).

Production sensitivity array 15633 (`dm-a-scan/submit-dm-4125-sens.sh`, both
tasks `COMPLETED`, exit 0, 2.6 h and 2.4 h) for the lower edge, Si(100),
17 MeV at $a = 4.125$: dz halved to 7.8125 Å gives detector `bound` ratio
0.80856 (12.937 µm), and H = 80 at dz 15.625 Å gives 0.80815 (12.930 µm),
against 0.80815 (12.930 µm) at H64/dz 15.625 Å. The lower edge is therefore
numerically converged at this grid, still only 1% above 0.8 (balance defect
−1.9e-4 and −7.5e-4). `run-si100_17-a4.125-{dz7.8125,H80}.json` carry the
records (host redacted), `original-outputs-15633.sha256` the original bytes.

The maintainer adopted $a = 4.125$ as the default
(2026-11-23); see the research note for the interpolated passing interval and
the pending edge-case sensitivity arrays (15632, 15633).
