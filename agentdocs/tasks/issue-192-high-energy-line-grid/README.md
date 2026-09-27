# Issue #192: high-energy h-BN line grid

Branch: `issue-192-high-energy-line-grid`. Canonical plan: GitHub issue #192.

## Observed remote jobs (2026-09-27)

- `high_energy-2`: h-BN failed after 3/117 cases with the reported 1,415,388
  nodes at 1.21575 eV over 50–1,720,800 eV against a 200,000-node budget.
- `high_energy-3`, `-4`, `-5`: h-BN still failed against 600,000 nodes at
  1,011,459 / 621,578 / 633,652 required nodes, respectively. The budget
  increase was a capacity change, not a validated solution.
- `high_energy-6`: h-BN completed 81 cases (48 cached, 33 new); 27 low-energy
  thickness cases were dropped by the penetration watchdog. HoPG failed its
  line-grid budget. The job log reports 33 h-BN cases in 96 seconds, but does
  not record peak GPU memory or line accuracy.
- `high_energy-7`: h-BN completed 72 new cases under the revised profile;
  MoS2 and MoSe2 failed for an unrelated electron-transport step limit.

The original profile contained 30, 100, 1,000, and 5,000 keV h-BN. Commit
`a6458326` changed it to 100, 500, and 1,000 keV, so the current successful
scan does not exercise the reported 5 MeV case. The remote status/log commands
do not identify the precise failing tilt, azimuth, thickness, case key, backend
precision, or peak memory. Existing checkpoint state was partial for jobs
`high_energy-3` through `-6`.

## First implementation slice

Case construction now rejects an automatic grid when the *maximum permitted*
spacing alone needs more nodes than the configured budget. It reports material,
beam energy, bandwidth, minimum node count, and the compute-environment budget
setting before transport. This does not affect accepted case identities or
change the measured-spacing refusal after transport. A focused 5 MeV h-BN
regression covers the guaranteed-impossible 200,000-node configuration.

## Remaining decision

Measure identical 5 MeV trajectories under candidate bandwidths/grids on the
remote host. Record integrated line yield, centroid/shape, detected counts,
node count, wall time, peak memory, and float32/float64 agreement. The current
600,000-node budget still permits the 3 eV backbone across the reported span,
but the observed 1.21575 eV target needs 1,415,388 nodes. Do not narrow the
bandwidth or change the resolution policy until the 0.1% intrinsic-source
error budget has evidence. Keep deliberate refusal when no candidate fits.
