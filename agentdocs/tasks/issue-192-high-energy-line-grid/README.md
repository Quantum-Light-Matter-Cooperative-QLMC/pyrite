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

## Remote CUDA rerun with higher CPU memory allocation (2026-09-27)

`convergence_job start-bandwidth` now accepts optional `--mem-per-cpu`; absent
the flag, SLURM keeps the site's allocation. The first 5,250 MB/core request (42,000 MB for 8 CPUs) was
rejected because the GPU node reports 40,960 MB total. Retried at 5,000
MB/core: SLURM confirmed `ReqTRES mem=40000M`, 8 CPUs, one GPU (over three
times the prior 12,000 MB allocation). This left only 960 MB below the
reported physical node memory for the OS and other system use; that was too
aggressive and may have contributed to the lab host crash. Do not repeat this
allocation.

Comparison job `20260927-133559-2ca77464` / SLURM 203 completed. Seed 0,
5 MeV h-BN, local resolution compared against uniform and full-ceiling FP64
reference on identical trajectories. For 1 um 10/100 (Ne=2,000), 1 mm 10/100
(Ne=200), and 1 mm 80/180 (Ne=200), local line-yield errors were respectively
`2.35e-5`, `-3.55e-5`, and `2.09e-5`; maximum host RSS was 2,679 MiB and the
reported device high-water mark was 5,972 MiB. Characteristic-yield errors
were below `1.53e-6`. Local grids had 8,744 / 69,360 / 28,244 nodes versus
1,415,387 / 5,899,823 / 6,371,926 full-ceiling points. Report pulled to
`/tmp/issue192_mem_comparison_20260927.json`.

Production job `20260927-133647-a1e30b15` / SLURM 204 requested the same
40,000 MB allocation for two 1 mm, Ne=20,000 configurations (`10/100`,
`80/180`). The user subsequently reported that the production run appeared to
crash the lab computer. SSH is currently unreachable, so its final SLURM state,
exit code, and logs are unconfirmed. Do not submit another production run until
the failure is recovered and the memory profile is improved.

## Job 204 outcome and stage profile (2026-09-27)

The lab host did not reboot (uptime 4 days at 15:43). SLURM 204 ended
`TIMEOUT` at its 2 h limit (`ExitCode=143:0`) without finishing the first
case (1 mm 10/100, Ne=20,000); stdout was empty and no report was written. The
host was concurrently loaded by 16 non-SLURM `pw.x` processes from another
user (load ~37 on 32 cores), which plausibly explains the SSH unresponsiveness.

Stage timings are now also flushed to stderr as they happen
(`line-grid profile: ...`), so a timed-out job still shows its last stage.

Local CPU profile, `bandwidth_check production --resolution local`, h-BN
5 MeV, 1 mm 10/100, seed 0, FP64 NumPy:

| Ne | segments | lines | collect s | stop search s | final line eval s | nodes | stop eV | host peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 54,580 | 109,160 | 21.2 | 0.13 | 52.8 | 33,576 | 13,500 | 3,623 |
| 80 | 222,428 | 444,856 | 20.3 | 0.27 | 330.9 | 53,766 | 41,900 | 3,818 |

Collection cost is nearly flat in Ne; the population itself is 24 bytes/line
(FP64) and the stop search/local planner are sub-second. Final line
evaluation scales as lines x nodes (3.7e9 -> 2.4e10 pair evaluations, 6.5x
-> 6.3x time): both the NumPy path and the CUDA reduction kernel
(`line_jit_kernel._kernel_*`) evaluate every line at every node. At Ne=20,000
this extrapolates to ~1e8 lines x 1e5-3e5 nodes = 1e13-3e13 sinc evaluations,
and the measured stop grows with Ne (rare Doppler-boosted lines), so the
node count grows too. That, not selector memory, is the likely cause of the
2 h timeout. Bounded-memory streaming (`ec8f273d`) stays, but a larger memory
request would not help.

Scalable options (decision needed): per-line windowed evaluation (sorted lines,
each node sums only lines within K first-zero widths; truncated tail
<= 2/(pi**2 K) of each line, K ~ 2e3 for 1e-4) with a derivation, ledger row,
and CPU/GPU kernel; or cap production Ne for 1 mm cases. No production rerun
until one is chosen.

## Ne=4,000 production at 1 mm (2026-09-27)

Job `20260927-171749-f5dfb3c9` / SLURM 205: `--production --resolution local`,
FP32 CUDA (RTX 5080, 16 GB), seed 0, `--mem-per-cpu 3000M` (24,000 MB of the
40,960 MB WSL node, leaving ~17 GB for OS/Windows). Completed in ~2 minutes.
Report pulled to `/tmp/issue192_production_ne4000_20260927.json`.

| tilt/azim | segments | lines | collect s | stop search s | stop eV | nodes | transport s | line eval s | device peak MiB | host peak MiB | audit upper / lower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10/100 | 10,913,900 | 21,501,794 | 22.9 | 5.4 | 35,400 | 56,695 | 33.2 | 3.2 | 8,977 | 1,819 | 5.0e-5 / 2.3e-4 |
| 80/180 | 17,021,822 | 12,947,423 | 1.4 | 3.0 | 798,100 | 304,159 | 13.4 | 42.1 | 13,230 | 2,756 | 2.9e-5 / 4.7e-4 |

Correction to the CPU-profile conclusion above: on CUDA, the dense
lines x nodes reduction is not the bottleneck at this scale (1.2e12 and 3.9e12
pair evaluations in 3.2 s and 42 s). Selector host memory is also small (~0.5 GB
added). The collect time for the first case includes one-time CUDA JIT/table
warm-up. Device memory instead scales with transported segments: ~0.78 GiB per
million (9.0 GB at 10.9 M, 13.2 GB at 17.0 M). Ne=20,000 at 1 mm implies
~55-85 M segments, ~43-66 GB of device memory on a 16 GB card. That is the
likely failure mode of SLURM 189/204. Hypothesis, not yet verified: under WSL
the Windows driver's CUDA sysmem fallback can spill VRAM into shared Windows
RAM outside the SLURM cgroup, which would explain both the stall and the
host-wide unresponsiveness. Next measurement: which transport/spectrum arrays
hold per-segment device memory, and whether segments can be streamed to the
device in bounded blocks.

The 80/180 stop is 798 keV (rare Doppler-boosted forward lines); local spacing
keeps it at 304k nodes. The lower-edge bound (2.3e-4, 4.7e-4) exceeds 1e-4
and is not gated; it matches the earlier lower-edge finding.

## Electron-block streaming (2026-09-27)

Device-scaling job SLURM 207 (pool capped at 11.4 GiB via the runner's
`_ensure_pool_limit`): resident segments cost ~86 B each (1.46 GiB at 17 M), but
population collection at Ne=8,000 OOMed on single ~0.85 GB allocations: the
per-segment setup arrays outside the kernel's `(n_block, N_g)` bound. Fix
(`607031a3`): on a GPU, incoherent `_lines_for_segments` splits into disjoint
electron-aligned blocks sized from pool headroom (1,024 B/segment after SLURM
209 calibration); a block OOM restores the audit and doubles the count. Cases
that fit, CPU runs and coherent sums are unchanged.

SLURM 209, 1 mm, local resolution, FP32, same caps (24 GB host):

| case | Ne | segments | blocks | stop eV | nodes | transport s | line eval s | host peak MiB | upper / lower | line yield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 80/180 | 8k | 34.0 M | 2->4 | 797,400 | 307,346 | 51.4 | 76.1 | 4,157 | 4.5e-5 / 4.5e-4 | 2.66e-8 |
| 10/100 | 8k | 21.8 M | 2 | 672,400 | 270,781 | 22.7 | 97.2 | 4,157 | 5.0e-5 / 2.2e-3 | 2.59e-6 |
| 10/100 | 20k | 54.6 M | 4->8 | 623,300 | 256,073 | 61.4 | 215.3 | 6,731 | 5.0e-5 / 2.0e-3 | 1.18e-6 |
| 80/180 | 20k | 85.7 M | 4->8 | 791,100 | 308,871 | 108.1 | 175.7 | 18,007 | 4.7e-5 / 1.2e-3 | 2.53e-8 |

All Ne=20,000 1 mm cases now complete. At 85.7 M segments the resident
transport fell back to host segments (18 GB host peak). Line-eval throughput is
flat at 1.1-1.3e11 pairs/s; kernel launch tuning is tracked in #200.

Block invariance (SLURM 210, forced 16 blocks): 10/100 Ne=4,000 yield
relative change 2.5e-9, Ne=8,000 5.0e-8, identical nodes/stop/audit. Blocking
is exact to FP32 summation order.

Open finding: 10/100 line yield jumps ~70x from Ne=4,000 (3.74e-8, centroid
7.08 keV, stop 35.4 keV) to Ne=8,000 (2.59e-6, 18.6 keV, 672 keV), then 1.18e-6
at 20,000. 80/180 is stable (2.5-2.8e-8). Electrons 4,000-7,999 therefore
contain rare, very heavy far lines: a heavy-tailed line-weight estimator
(candidate: near-zero `1 - n v.n` denominators on forward-scattered segments),
not a grid effect. The lower-edge bound also reaches 2e-3, above the 1e-3
budget, and is ungated. Needs per-electron yield attribution before any
production conclusion.

## #201 line attribution (2026-09-27, post-#172 rebase)

Branch rebased onto the #172 tip (coupled radiative transport and midpoint
energy now default; the remote box still falls back to uncoupled EEDL for
h-BN). Diagnostic `b7e91ba5`: `bandwidth_check attribute` /
`start-bandwidth --attribute` records the heaviest lines (mass `w pi/a_w`) with
their factors and the per-electron line mass. No weight or spectrum change.

SLURM 213 (`20260927-183307-2dbea117`): 1 mm 10/100, Ne=8,000, seed 0, local
resolution, CUDA FP32 then FP64 on identical transport (21.8 M segments,
43.0 M / 43.6 M lines). Reports: `/tmp/issue201_attribution_ne8000.json`,
`/tmp/issue201_attribution_ne8000.fp64.json`.

- Line mass per electron 2.589e-6 (FP32) vs 2.589e-6 (FP64); every
  per-electron and quarter sum agrees to ~1e-6. Not a precision artifact.
- Two electrons carry 99.6% of the case's line mass: 5025 (65.1%) and 4868
  (33.4%). Electron quarter sums: 9.8e-5, 5.1e-5, **2.05e-2**, 4.1e-5. All
  200 heaviest lines come from those two electrons: long straight flights of
  90-110 consecutive segments on reflections rows 0-3 (|g| = 1.89 A^-1, the
  (0002)-type row set).
- The lines are CBS-dominated (|A_CBS|^2 / |A_PXR|^2 median 11 and 68), at
  5-42 keV, widths 94-200 eV. Detuning is ordinary (3-4.6 A^-2), so the Bragg
  detuning hypothesis is refuted.
- Mechanism: `1 - v.n` = 6.5e-3-7.6e-3 against `1/(2 gamma^2)` = 6.3e-3,
  i.e. the electron travels 15-120 mrad from the detector direction, inside
  its 1/gamma = 112 mrad radiation cone. Its `v.g` is 0.02-0.14 A^-1 against
  1.85 for the ordinary 3.6 keV line population. The CBS factor
  `f_cbs ~ 1/(gamma v.g)` (plus a `1/(v.g)^2` term) and the Doppler-boosted
  `omega` make each line ~1e2-1e3 heavier than an ordinary line.
- The amplitude is inside its stated validity: `|U_g| g^2 / (gamma m c^2
  (v.g)^2)` <= 0.018; angle to the planes 10-72 mrad, above the h-BN
  Lindhard angle (~3 mrad at 5 MeV), so this is not planar channeling.
- Consequence: the 10/100 incoherent line yield is a rare-event estimator.
  Only electrons scattered by ~90 deg into the detector's 1/gamma cone
  contribute appreciably. From the top 20 electrons the relative standard
  error of the mean is ~0.73 at Ne=8,000. The 70x jump between Ne=4,000 and
  8,000 is sampling, not a bug. 80/180 is stable because its geometry has no
  such population, or has not sampled it.

Decision needed (#201 acceptance: "documented variance policy"): report
per-electron standard error and gate on it; add a variance-reduction scheme
(e.g. directional biasing/splitting of electrons scattered toward the
detector cone); or scope 5 MeV incoherent line yields as statistics-limited.
Also open: whether these electrons also dominate the measured stop (672 keV
at Ne=8,000 predates the #172 defaults) and the lower-edge audit.

## Measured-stop attribution and standard-error gate (2026-09-27)

Decision (user): option 1, gate on the per-electron standard error; option 2
(variance reduction) opened as #203.

`2974a801`: resonance-population audits now sum line mass per line electron
(both kernel routes) and `check_line_truncation` refuses a case whose relative
standard error of the mean exceeds `LINE_YIELD_RELATIVE_SE_LIMIT = 0.1`,
recording `line_yield_statistics` (n, relative SE, largest electron share).
The attribution diagnostic also ranks lines/electrons by their tail bound above
the measured stop.

SLURM 215 (`20260927-191234-75a65730`), 1 mm 10/100, seed 0, local
resolution, FP32 and FP64 identical to shown precision. Reports:
`/tmp/issue201_stop_attribution.json`, `/tmp/issue201_stop_attribution.fp64.json`.

| Ne | stop eV | max resonance eV | top-1 / top-10 electron mass share | stop tail by electron |
|---:|---:|---:|---|---|
| 4,000 | 35,400 | 20,705 | 0.51 / 0.76 | 159: 0.69, 3986: 0.13, 2704: 0.08 |
| 8,000 | 672,400 | 77,474 | 0.65 / 0.996 | 5025: 0.63, 4868: 0.37, 4073: 0.001 |

The 672 keV stop reproduces under the #172 defaults and is set entirely by the
two rare detector-cone electrons. No line resonates above 77.5 keV; the edge
sits ~600 keV higher because their lines are both heavy and wide (small
`1 - v.n` gives first-zero widths 0.5-9.5 keV), and the `w/(pi^2 D)` tail bound
decays only as 1/D. Given the sampled population the stop is correct (the bound
is ~2x the exact far tail), so it is a symptom of the rare-event estimator, not
a selector bug. Even Ne=4,000 is dominated by one electron (51% of the mass;
159 is the heaviest electron at Ne=8,000 outside the two cone electrons), so
the new gate refuses both counts. Likely also the source of the ~2e-3
lower-edge bound (wide lines at 5-13 keV spill below `start`); not measured.

Next: a stable 5 MeV 10/100 yield needs #203. Remaining #192 items
(fresh-context validation, `high_energy` opt-in) proceed for cases that pass
the gate; 5 MeV h-BN 10/100 at 1 mm stays refused until #203.

Revision (user, same day): statistics-limited cases must run, with warnings,
not be refused. Above the 0.1 relative standard error the truncation record's
`line_yield_statistics.statistics_limited` is true and
`LineYieldStatisticsWarning` names the error, electron count and largest
electron share. The upper-edge truncation gate is unchanged. 5 MeV h-BN
10/100 at 1 mm therefore runs, flagged, until #203 reduces its variance.

## high_energy opt-in and 5 MeV h-BN (2026-09-28)

After #204 landed (squash `64a4437c`), the branch was reset to main. Profiles
now accept `[profiles.NAME.line_grid_policy]` (`bandwidth`, `resolution`,
validated against the known policies); `material_sweep` applies it as
`Sweep.line_grid_policy`, so it joins the profile's case identity only.
`high_energy` selects `resonance-population` + `resonance-local` and restores
5,000 keV for h-BN via `overrides.hbn.energy_keV`; MoS2/MoSe2 keep
100/500/1000 keV. Statistics-limited cases (e.g. 1 mm 10/100) run with a
`LineYieldStatisticsWarning`. The earlier note that EagleXO/LegacyEDS refuse
nonuniform axes is stale: the EagleXO and Timepix responses accept nonuniform
grids. Not yet run: `pyrite run high_energy -R -m hbn`.

### Remote acceptance: `pyrite run high_energy -R -m hbn` (2026-09-28)

- `high_energy-8`: refused at 32/104 on `hbn 100um 10/140` at 100 keV. The
  measured stop was capped at the 9.1 keV kinematic ceiling (max resonance
  4.4 keV) and the tail bound above it was 1.06e-4 > 1e-4. Fixed in
  `75d7e44f`: at the ceiling the audit warns (`LineGridTruncationWarning`,
  `capped_at_ceiling`), since the automatic bandwidth drops the same tails.
- `high_energy-9`: stopped at 89/104 on `TransportStepLimitError`
  (n_step_limited=177, Ne=300, max_steps=20000). 5 MeV h-BN at 10 mm needs
  ~23-32k steps per electron (local probe: 1 mm ~2.8k). Fixed in `058f2fa6`:
  the runner retries at a doubled budget up to 160k; identity-neutral.
- `high_energy-10`: **104/104 h-BN cases complete** (89 reused, 15 new in
  165 s). Pulled to `checkpoints/hbn@high_energy-b5d0b092df19/`.
  `LineYieldStatisticsWarning` fired for 7 cases, all 10 mm (Ne=300):
  5 MeV 10/140 (RSE 0.155), 10/180 (0.36), 30/140 (0.111), 70/180 (0.24);
  1 MeV 10/180 (0.375), 30/140 (0.183), 30/180 (0.159).
- Gap: scan checkpoints do not persist `line_grid_resolved` (only the Python
  API's provenance does), so `statistics_limited` / `capped_at_ceiling` are
  visible only as log warnings for scans.

## Re-check follow-up (2026-09-28)

Second fresh-context verification of `line-grid-resonance-local-spacing`
(appended to its page) returned `discrepancy` on wording and cited evidence,
not geometry. Addressed (uncommitted at time of writing):

- `resolve_line_grid_policy` refuses `resonance-local` unless `quadrature =
  "bin-mean"`; `bandwidth_check --resolution local` selects bin-mean.
- `local_spacing_seeds` no longer drops narrow-line halos when no line is
  cored (`core_fraction > 1`); regression added.
- The core regression now discriminates (w ~ 3.1 eV, 3 eV below a join:
  node 6.3e-3 without core, ~1.1e-3 with; bin-mean < 1e-4).
- Ledger claim/Notes, derivation page, and budget page: core spacing is
  `max(w/2, floor)` and does not exclude fine-fine joins; node caveat covers
  cored lines; no shape bound is claimed; core parameters are absent from the
  policy payload (identity does not record them).

Still open: identical-trajectory line-shape and detected-count comparison;
fresh-context re-verify
of the revised wording; human sign-off.

## Scan checkpoints keep `line_grid_resolved` (2026-09-28)

`store_result` now copies `line_grid_resolved` into the scan record, and the
legacy CAS-seed whitelist keeps it, so `statistics_limited` /
`capped_at_ceiling` persist in `line.h5` and survive cross-profile CAS replay
(fresh CAS blobs already held the full runner output). Regression:
`tests/scan/test_run.py::test_scan_checkpoint_persists_resolved_line_grid`.
Real CPU smoke (h-BN 100 keV, 1 um, 10/140, Ne=20, local resolution): both
warnings fired and both flags reloaded from the checkpoint; ~2.4 kB/record.
Existing profile checkpoints (e.g. `hbn@high_energy-b5d0b092df19`) predate
this: resume reads their own records, not the CAS, so those records gain the
field only when recomputed (or replayed into a new stem from a CAS blob that
already holds it).

## Second re-check and shape/count comparison (2026-09-28)

Fresh-context second re-check of `line-grid-resonance-local-spacing`
(appended to its page): **`rederived`** as worded; suggested status
`unverified` -> `rederived` (human applies). Follow-ups in `a1c90dd0`:
`Case` refuses a `resonance-local` policy without `line_quadrature =
"bin-mean"` before transport (hand-built / pre-`b01bc2f2` payloads); core
parameters validated; stale trapezoid wording and the `0f78fef1` citation
fixed.

`bandwidth_check reference` now reports, per measured and
`--compare-resolution` axis against the full-ceiling reference on identical
segments (`shape_and_counts`): 100 eV intrinsic bin-mass L1 / worst bin,
dominant-line FWHM, EagleXO counts (full axis), and Timepix3 detected events
over `[start, 60 keV]` through one response matrix on overlap-split channel
masses (grid term only). Timepix3 `native_score` (`apply_native`, #219) assigns each node's
whole cell to the channel holding the node; that axis-dependent term is
reported separately (`node_rebin_counts_rel`). Local smoke (h-BN 100 keV,
1 um 10/140, Ne=20, FP64 CPU): grid term local/uniform -3.5e-6 / -6.9e-6
counts, L1 5.6e-6 / 1.0e-5; node-rebin term 1.5e-4 / 3.8e-4 (above the 2e-4
interpolation share -- a detector-resampling finding, not a line-grid one).

Remote: SLURM 238 (`20260928-093552-b5a224ce`), 5 MeV h-BN, seed 0, local vs
uniform vs full ceiling: 1 um 10/100 and 80/180 at Ne=2,000; 1 mm 10/100 and
80/180 at Ne=200.

### SLURM 238 / 239 results (2026-09-28)

SLURM 238 stalled: prominence peak-finding is quadratic (~minutes per call
at 1.4 M nodes); cancelled after config 1. Fixed in `afc03171` (O(n) FWHM,
cell-snapped intrinsic bins). SLURM 239 (`20260928-103350-a05c09b2`)
completed all four configs in ~25 min. Report: `/tmp/issue192_shape_counts_20260928b.json`.

| case (5 MeV h-BN, seed 0) | nodes local / uniform / ceiling | line yield local | EagleXO counts local | Timepix3 counts local / uniform | Timepix3 detected L1 local / uniform | dominant FWHM local / uniform (ref) | FP32 yield | line eval s local / ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 um 10/100, Ne 2,000 | 8,744 / 10,736 / 1,415,387 | 2.4e-5 | -6.6e-7 | 1.7e-4 / 8.2e-5 | 1.7e-3 / 8.5e-4 | -15.8% / -4.3% (3.05 eV) | 1.4e-7 | 0.5 / 26 |
| 1 um 80/180, Ne 2,000 | 9,631 / 25,433 / 5,436,172 | 2.2e-5 | -1.5e-6 | 6.6e-6 / 1.1e-6 | 1.4e-5 / 1.4e-6 | -4.6% / +0.3% (2.52 eV) | -1.4e-6 | 0.6 / 31 |
| 1 mm 10/100, Ne 200 | 70,080 / 362,923 / 5,899,830 | -3.2e-5 | 1.4e-6 | -5.0e-7 / -6.4e-10 | 1.2e-5 / 6.3e-8 | +0.7% / +0.4% (1.92 eV) | -1.0e-6 | 60 / 191 |
| 1 mm 80/180, Ne 200 | 31,308 / 53,140 / 6,371,941 | 1.9e-5 | -4.5e-7 | 5.0e-7 / 3.1e-7 | 9.2e-7 / 4.0e-7 | -2.4% / -1.5% (0.87 eV) | -3.3e-6 | 40 / 245 |

Line yield differences are the measured-stop truncation (true fraction above the stop 1.9-2.4e-5; the 1 mm 10/100 local value includes -3.2e-5 from the 106 keV stop). Detected counts agree to <=1.7e-4 (Timepix3, grid term) and <=1.5e-6 (EagleXO) -- inside the 2e-4 interpolation share. The Timepix3 detected-shape L1 is <=6e-5 except 1 um 10/100 (1.7e-3, uniform 8.5e-4): there 3 eV lines on a 1.2 eV reference have a 0.35 sub-cell split bound at 100 eV, so part of it is sub-cell ambiguity in the 50 eV channel split, not grid error. The dominant-line FWHM is the one real shape limitation: bin-mean averaging at h ~ w/2 widens a 3 eV line by up to 16% (uniform axis 4%). The detector resolution is far coarser, so detected shape is unaffected, but an intrinsic FWHM finer than a few eV is not preserved by `resonance-local`. FP32 matches FP64 to <=3.3e-6 in yield. Peak host / device memory is 3.2 / 6.0 GiB. Separately, Timepix3 native-bin scoring (`apply_native`, used by physical-detector acquisition; `apply` already splits by overlap) assigns cells to channels by node and moves detected counts by up to 1.8e-3 depending on the axis (`node_rebin_counts_rel`) -- a detector-resampling term above its 2e-4 share, independent of this policy (#219).

## Hybrid bin-mean and the completed h-BN scan (2026-09-28)

`high_energy-12`/`-13`/`-14` died on SLURM TIMEOUT (#220-#222): `bin-mean`
evaluated an FP64 sine integral per (line, bin) pair, ~40x slower than node
sampling on the RTX 5080 (5 MeV 10 mm cases took over an hour). New
`sinc-bin-far-envelope` (rederived after one fixed discrepancy): exact bin
mean within 64 widths, float32 envelope mean `1/(2 pi^2 x_lo x_hi)` beyond,
yield change <= 1.2e-5; coalesced reduction with deterministic near-pair
compaction. RTX 5080, 50k lines x 240k bins: exact 1.2e9, hybrid 1.9e10,
node 1.1e11 pairs/s; CUDA tests 8/8 at `3913f3fc`. Commits `42d360f2`,
`e8554873`, `1a8434e9`, `c5bb3f1e`, `0a3a8e3f`, `3913f3fc`.

`high_energy-15` (SLURM 251, 30-min slices via `start_command`, #220)
completed h-BN 104/104 at 17:41, no timeout; the case that had stalled
(5 MeV 10 mm 10/180) took ~20 min. Pulled to
`checkpoints/hbn@high_energy-18d155e86c96/` (43 MB slimmed). 37 of 104
records carry `line_grid_resolved` (those computed after `5d01f415`; the
100 keV cases and older ones do not): 15 `statistics_limited`, 0
`capped_at_ceiling`. The 67 older records were computed with the all-exact
bin mean (identity unchanged; within 1.2e-5).

Open: backfill/recompute of the 67 records lacking `line_grid_resolved`;
MoS2/MoSe2 members of `high_energy` not rerun; #203 variance reduction for
statistics-limited 5 MeV cases; #219-#222.
