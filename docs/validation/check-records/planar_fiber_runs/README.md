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
