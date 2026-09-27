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

## Local resonance measurement (2026-09-27)

Scratch check on the CPU backend: `build_ladder_case("hbn", 5000, tilt, azim,
thickness)` at seed 0, one `runner._transport_case` with the point budget
lifted, then the vacuum resonance `hbar c v.g / (1 - v.n)` for every
line-electron segment and every case reflection (same `g` construction as
`kinematic_line_seeds`, no mosaic in this case). Weighted by the `t_L**2`
proxy. The automatic grid reproduced the reported failure exactly: 1,415,388
nodes at 1.21575 eV for 1 um, 10 deg/100 deg.

| thickness | tilt/azim | uniform nodes | spacing eV | joint bound eV | max resonance eV | 99.9% eV |
|---|---|---:|---:|---:|---:|---:|
| 1 um (Ne 100) | 10/100 | 1,415,388 | 1.216 | 109,925 | 7,720 | 7,720 |
| 1 um | 10/140 | 1,415,388 | 1.216 | 253,612 | 7,670 | 7,669 |
| 1 um | 10/180 | 1,415,388 | 1.216 | 318,251 | 7,644 | 7,643 |
| 1 um | 45/100 | 1,971,250 | 0.873 | 237,724 | 5,359 | 5,359 |
| 1 um | 45/140 | 1,971,250 | 0.873 | 936,916 | 5,312 | 5,312 |
| 1 um | 45/180 | 1,971,250 | 0.873 | 1,219,392 | 5,242 | 5,242 |
| 1 um | 80/100 | 2,942,791 | 0.585 | 313,993 | 1,444 | 1,427 |
| 1 um | 80/140 | 3,217,986 | 0.535 | 1,300,283 | 1,460 | 1,450 |
| 1 um | 80/180 | 2,925,720 | 0.588 | 1,694,754 | 1,451 | 1,407 |
| 1 mm (Ne 10) | 10/100 | 6,481,315 | 0.265 | 109,925 | 9,370 | 9,289 |

Every orientation needs 1.4-6.5 M uniform nodes, but no segment resonates
above 9.4 keV against a 1.72 MeV ceiling. The joint-geometry bound is
direction-agnostic in the electron and stays loose (110 keV-1.69 MeV), so
it does not rescue five of nine 1 um orientations. The bandwidth is set by the
*measured* resonance population, which is 2-3 orders narrower than any
closed-form bound. Even at the 1 mm spacing, a 50 keV band is about 190,000
nodes.

Caveats: vacuum root (production seeding uses the in-medium root; `Re n < 1`
above the optical region only lowers it); small Ne; `t_L**2` proxy ignores
`|A|**2` and absorption. Content above the population is only sinc-squared
tail (`~2/(pi**2 k)` of a line's yield beyond `k` feature widths), so a
trajectory-derived stop needs an explicit tail margin sized to the 1e-3
budget.

Characteristic lines are deposited as exact Lorentzian bin masses and set
neither spacing nor bandwidth; a separate characteristic grid would not reduce
the node count. h-BN K lines lie below 1 keV.

## Next slice: trajectory-measured bandwidth (approved direction 2026-09-27)

Add a bandwidth policy, working name `resonance-population`, beside
`kinematic-ceiling`. After transport, `runner/line_grid.py` sets `stop_eV`
from the case's own in-medium resonance population (every line segment x
reflection x mosaic orientation, the same roots `kinematic_line_seeds` solves)
plus a sinc-squared tail margin. The closed-form ceiling stays the hard cap;
`start_eV` is unchanged. Spacing stays the measured sinc spacing.

1. Derivation first (physics skill): the upper-tail yield fraction above
   `stop` as a sum over segments of weight x sinc-squared tail beyond
   `(stop - E_i) / width_i`. Solve `stop` so the lost fraction stays within
   the intrinsic-source budget share. Decide the weight: the `t_L**2` proxy
   omits `|A|**2`, `omega` and absorption, so either bound per reflection
   (weight-free) or use the production prefactor. Short, wide segments
   dominate a max-over-segments margin; quantify that before choosing.
   Source equation, assumptions, limiting case, `Validation: <id>`, and ledger
   row, then fresh-context validation.
2. Policy plumbing in `_line_grid_policy.py` (payload, identity, cache key).
   The measured stop joins the resolved record. Case-build pre-transport
   refusal applies only to closed-form bandwidths; the measured policy refuses
   after transport, before the spectrum, if it still exceeds the budget.
3. Opt-in only: `high_energy` profile `line_grid_policy` selects it. The
   default automatic policy and every other profile's identity stay unchanged;
   a default flip is a later, separately evidenced decision.
4. Remote validation (`pyrite remote`): identical 5 MeV h-BN trajectories,
   1 um and 1 mm, extreme orientations, production Ne. Compare the new grid
   against a full-ceiling reference (budget lifted), and a stop with 2x margin:
   integrated line yield, centroid/shape, detector counts, node count,
   wall time, peak GPU memory, float32 vs float64.
5. Restore 5,000 keV (and 30 keV) h-BN to `high_energy` only if step 4
   passes; run `pyrite run high_energy -R -m hbn`.
6. Regressions: stop covers a synthetic straight-flight population plus
   margin, never exceeds the ceiling, identity unchanged for default cases,
   and refusal diagnostics.

## Measured-bandwidth results (2026-09-27)

Implemented opt-in `resonance-population` (commits `8e168d34`, `6196f44c`);
measured with `convergence_job start-bandwidth` (FP64 reference axis to the
ceiling on identical segments, float32 candidate). h-BN, 5 MeV, seed 0.

| thickness | tilt/azim | Ne | nodes measured / ceiling | h eV | stop eV | true loss above stop | line eval s (meas/ref) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 um | 10/100 | 2000 | 42,239 / 1,415,405 | 1.216 | 51,400 | 3.8e-6 | 1.1 / 24.4 |
| 1 um | 80/140 | 2000 | 115,075 / 5,432,472 | 0.317 | 36,500 | 2.2e-6 | 1.2 / 27.7 |
| 1 um | 80/180 | 2000 | 115,154 / 5,436,201 | 0.317 | 36,500 | 2.2e-6 | 0.7 / 27.3 |
| 1 um | 10/100 | 300 | 40,018 / 1,415,401 | 1.216 | 48,700 | 3.9e-6 | 0.9 / 24.4 |
| 1 um | 80/140 | 300 | 65,975 / 3,140,381 | 0.548 | 36,200 | 3.4e-6 | 0.6 / 24.8 |
| 10 um | 10/100 | 300 | 96,974 / 4,176,879 | 0.412 | 40,000 | 2.7e-6 | 0.8 / 25.4 |
| 100 um | 10/100 | 300 | 136,602 / 6,034,819 | 0.285 | 39,000 | 1.8e-6 | 3.2 / 35.1 |
| 1 mm | 80/180 | 100 | 132,164 / 6,361,385 | 0.270 | 35,800 | 2.2e-6 | 12.4 / 80.7 |
| 1 mm | 10/100 | 100 | 512,370 / 5,939,098 | 0.290 | 148,500 | 2e-16 | 16.1 / 59.0 |
| 1 mm | 10/100 | 300 | refused: 1,286,877 at 0.290 eV over [50, 373,000] eV | | | | |

Characteristic loss above stop <= 4.6e-7 everywhere. Line centroid shifts
0.07-0.37 eV (<= 1e-4 relative). float32 vs FP64 on the measured axis:
yield <= 5e-5; centroid <= 0.006 eV except 1 mm 10/100 (3.4 eV on a 148 keV
axis -- a backend-precision finding, not bandwidth). The proxy edge (1e-5)
over-predicts true loss by 2.5-5x; the tail bound itself is 2x the exact far
tail.

1 mm at 10/100 is not a stray-line artifact (the far-line fix did not move
it): forward-scattered segments with small `1 - v.n` resonate Doppler-boosted
up to ~66 keV (Ne=100) and carry real proxy weight, and their lines are wide.
The band there is genuinely ~0.1-0.4 MeV at 1e-5-1e-4, while the global
0.29 eV spacing is set by long straight flights near 5-9 keV. A uniform axis
cannot hold both; energy-dependent spacing (fine where narrow features
resonate, coarse where only wide ones do) is the scalable fix. Proxy safety
2 instead of 10 cuts that stop to 67.9 keV / 234k nodes at Ne=100 but does not
address the Ne scaling. Production runs use Ne=20,000 (per user); not yet
measured. Profile opt-in deferred until #195 (materials/profile rework) lands.

## Handoff (2026-09-27, end of session)

Approved by user: options 1 (energy-dependent spacing) and 3 (proxy safety 2).
Branch rebased onto #195/#198 by user.

Done: proxy safety 2 (`ee20cc76`); opt-in `resolution = "resonance-local"`
(`line_seeds.local_spacing_seeds`, `_measured_line_grid`, generalized
`windowed_coordinates`). 1 mm 10/100 Ne=100: 56,791 nodes local vs 234,183
uniform, same 67.9 keV stop. No dedicated tests or ledger row yet.

Ne=20,000 production mode (job 20260927-094849-5e0f2102, safety 2): 1 um and
10 um pass audit (4-5e-5) at 12-41k nodes; 100 um REFUSED after transport
(audit 4.3e-4): the t_L**2 proxy omits omega |A|^2 T_abs, and absorption shifts
real weight to far lines. 1 mm 45/140 at Ne=300 needs a 935 keV stop (real
forward-scatter Doppler tail): uniform 3.3 M nodes; local spacing required.

Ne=20k 1 mm (same job, done): 80/180 refused at resolution (uniform needs
~3.4 M nodes); 10/100 resolved 271,398 nodes to 75.4 keV (transport 127 s)
but audit refused at 6.3e-4. Both confirm items 1 and local spacing below.

Next (in order):
1. Replace the proxy with exact production weights: run the line kernel once
   on a 2-node axis [start, ceiling] with the audit in a "collect" mode
   (`_accumulate_edge_truncation` appends E_r, a_width, weight); build one
   ResonancePopulation(weight=w, width=pi/a) and feed both
   `resonance_population_stop_eV` and `local_spacing_seeds`. Needs `groove`
   passed into `resolve_line_grid` (runner `_transport_case` l.644). Then drop
   `case_resonance_populations`. Watch host memory at Ne=20k/1 mm (~110 M lines).
2. Tests + ledger row `line-grid-resonance-local-spacing`; budget doc: halo
   share and the lower-edge (~1e-4) finding; write-up page for both rows.
3. Remote: `start-bandwidth` extended to compare local vs uniform on identical
   segments; `--production` at Ne=20,000 incl. 1 mm.
4. float32 3.4 eV centroid shift on 148 keV axis (1 mm) still unexplained;
   new candidate diagnostics (`moment_share_above_20keV`) not yet run. No
   beam-energy float64 switch planned; precision keys on axis top/spacing.
5. Profile opt-in on the post-#195 layout; nonuniform grids are refused by
   EagleXO(resolve_energy=True) and LegacyEDS(convolve=True) -- confirm with
   user before enabling local spacing in `high_energy`.

## Remaining decision

Measure identical 5 MeV trajectories under candidate bandwidths/grids on the
remote host. Record integrated line yield, centroid/shape, detected counts,
node count, wall time, peak memory, and float32/float64 agreement. The current
600,000-node budget still permits the 3 eV backbone across the reported span,
but the observed 1.21575 eV target needs 1,415,388 nodes. Do not narrow the
bandwidth or change the resolution policy until the 0.1% intrinsic-source
error budget has evidence. Keep deliberate refusal when no candidate fits.

## Continuation: production-weight selector

The measured-grid path now runs the existing incoherent line kernels once on
the two-node `[start, ceiling]` axis. Their audit hook collects each emitted
line's resonance, first-zero width, and production coefficient after the same
filters the final spectrum uses. Both kernel routes skip density accumulation
during collection. The bandwidth solver and local spacing planner consume this
population. `groove` and layers are forwarded from transport; the measured
cache revision and B-factor key were updated. Default automatic cases do not
take this path.

Focused tests: 17 bandwidth tests pass, including a two-electron 5 MeV h-BN
transport case. Scoped Ruff passes; Sphinx builds. Full lint stops at the
pre-existing import-order error in `tests/montecarlo/test_multilayer.py`; full
typecheck reports missing optional app dependencies. The ledger rows and two
unverified derivation pages now describe the production-weight selector and
local spacing rule.

Next: extend `start-bandwidth` to compare local and uniform grids on identical
segments; measure Ne=20,000 on the remote host, especially collection peak host
memory (the collector currently holds three arrays per line) and the 1 mm
cases. Then obtain fresh-context physics validation, add the local spacing
regressions, and decide whether any `high_energy` profile can opt in.

## Remote resolution comparison (2026-09-27)

Job `20260927-124515-ff585938` used seed 0, one GPU, FP64 full-ceiling
reference and FP32 replay on identical segments. `--resolution local
--compare-resolution uniform`; line spectra only, no detector response.
The remote h-BN case used the EEDL fallback because BremsLib B/N tables were
absent. Results were pulled to `/tmp/issue192_resolution_compare_20260927.json`.

| thickness | tilt/azim | Ne | local / uniform / ceiling nodes | local / uniform line-yield error | local / uniform centroid shift | FP32 local yield error | FP64 peak host / device MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 um | 10/100 | 300 | 8,536 / 10,324 / 1,415,387 | 2.34e-5 / 2.37e-5 | -0.724 / -0.724 eV | 2.65e-6 | 1,898 / 3,505 |
| 1 mm | 10/100 | 100 | 76,157 / 434,712 / 5,939,087 | -4.78e-5 / 8.7e-10 | 2.86 / 0.000003 eV | -1.07e-6 | 2,281 / 3,505 |
| 1 mm | 80/180 | 100 | 20,359 / 37,155 / 6,361,347 | -6.42e-4 / -5.60e-4 | 0.950 / 0.783 eV | -4.71e-6 | 2,475 / 3,505 |

All line-yield differences are below the 1e-3 intrinsic-source target. The
80/180 difference is mostly grid quadrature/phase error: actual line yield
above the stop is only 2.12e-5, and the uniform truncated grid differs by
5.60e-4 too. FP32 line-yield errors on the local axis are <=4.71e-6.
Device peak is a process-wide high-water mark, not a per-axis comparison.
Reference evaluations took 24-51 s; local evaluations 0.29-6.1 s. These
single runs have no repeat/spread estimate. Detector counts were not measured.

Production job `20260927-124933-d2e44852` uses Ne=20,000, FP32 and local
spacing. Its 1 um 10/100 case passed the truncation audit at 8,891 nodes:
21.64 s transport, 0.22 s lines, 1,319 MiB host and 229 MiB device peak.
The second case, 1 mm 10/100, produced no result through the 30-minute SLURM
limit; status queries slowed to tens of seconds and the SSH proxy intermittently
rejected handshakes. `scontrol show job 189` later reported `TIMEOUT`,
`Reason=TimeLimit`, `RunTime=00:30:15`, `ExitCode=143:0`, and a 12,000 MB
memory allocation. `slurm-189.err` confirms cancellation due to time limit.
No peak RSS was available (`sacct` accounting disabled); the user account
could not read system kernel messages. The third, 1 mm 80/180 case did not
run. This is a confirmed timeout, not a confirmed OOM kill.

Likely pressure: Ne=100 1 mm produced 277,612 segments and 555,224 line
entries. Linear scaling to Ne=20,000 implies roughly 55.5 M segments and
111 M line entries. The exact-weight selector retains three FP32 arrays per
line (~1.24 GiB together), concatenates them into another population, then
the stop solver filters/copies them and makes 60 full-population passes with
FP64 temporaries. This can require many GiB of host memory beyond the
resident trajectories, and many billions of line evaluations. The production
report logs `transport_wall_s` only after grid selection, so it cannot isolate
transport, line collection, stop search, or local-spacing planning. Add
stage-level timings and peak RSS, then use a bounded-memory selector before
another Ne=20,000 1 mm run.

The cluster administrator reports that submissions now default to 1.5 GB per
requested CPU core; jobs may request more with `#SBATCH --mem` or
`--mem-per-cpu`, up to 42 GB per simulation. PyRITE's shared SLURM script
requests 8 CPUs and does not set memory, explaining this job's 12,000 MB
allocation. An explicit memory request is needed for a future controlled
high-memory diagnostic, but the `TIMEOUT` state does not prove that this job
hit its 12 GB allocation. Do not use a larger request as a substitute for
bounded-memory selection and stage-level measurements.
