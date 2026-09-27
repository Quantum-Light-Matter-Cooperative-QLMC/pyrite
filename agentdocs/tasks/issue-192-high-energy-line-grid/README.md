# Issue #192: high-energy h-BN line grid

Branch: `issue-192-high-energy-line-grid`. Canonical plan: GitHub issue #192.

## Observed remote jobs (2026-09-27)

- `high_energy-2`: h-BN failed after 3/117 cases with the reported 1,415,388
  nodes at 1.21575 eV over 50–1,720,800 eV against a 200,000-node budget.
  Its manifest identifies the failed fourth case: `hbn 1um pol=10 az=100
  footprint=5x5mm`, 5,000 keV, content key
  `5c5f88c70286474fed0bf1a8d25fc71c79c1cd7772eb70b40368fa926abc1cea`.
  The three lower energies of that configuration completed. Job metadata
  records clean revision `1ac459ce`, full fidelity, and one remote GPU.
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
scan does not exercise the reported 5 MeV case. The failed job did not record
backend dtype or peak GPU memory. Existing checkpoint state was partial for
jobs `high_energy-2` through `-5`.

## First implementation slice

Case construction now rejects an automatic grid when the *maximum permitted*
spacing alone needs more nodes than the configured budget. It reports material,
beam energy, bandwidth, minimum node count, and the compute-environment budget
setting before transport. This does not affect accepted case identities or
change the measured-spacing refusal after transport. A focused 5 MeV h-BN
regression covers the guaranteed-impossible 200,000-node configuration.

After trajectory measurement, a point-budget or precision refusal now includes
the case name, beam energy, and backend dtype alongside the required node
count, spacing, bandwidth, and correction options. This diagnostic change does
not alter policy or case identity.

## Analytic bandwidth comparison

For fixed detector direction `n`, reciprocal vector `g`, and incident speed
`beta`, maximizing the vacuum resonance over *all* scattered electron unit
directions gives `q = (beta*g·n + sqrt(beta²(g·n)² + (1-beta²)|g|²)) /
(1-beta²)` and `E_max = hbar*c*beta*q`. This is a candidate bound, not an
installed policy. It must be checked against the production in-medium
denominator before use.

For the original 5 MeV h-BN reflection set and 90° detector, the current bound
is 1,720,800 eV for every orientation. The joint bound, rounded up to 100 eV,
would be 110,000 eV for the failed 10°/100° case. Across the nine tilt/azimuth
combinations it spans 110,000–1,694,800 eV. At the *one observed* 1.21575 eV
spacing, five orientations fit 600,000 nodes and four still need about
771,000–1,394,000 nodes. This spacing is not measured for the other eight
orientations; these counts are capacity estimates, not convergence evidence.

## Remaining decision

Measure identical 5 MeV trajectories under candidate bandwidths/grids on the
remote host. Record integrated line yield, centroid/shape, detected counts,
node count, wall time, peak memory, and float32/float64 agreement. The current
600,000-node budget still permits the 3 eV backbone across the reported span,
but the observed 1.21575 eV target needs 1,415,388 nodes. Do not narrow the
bandwidth or change the resolution policy until the 0.1% intrinsic-source
error budget has evidence. Keep deliberate refusal when no candidate fits.
